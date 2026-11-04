#include <Python.h>

#include "AllocationProfile.h"
#include "AllocatorPython.h"
#include "ProfileAllocatorRuntime.h"

#include <mutex>

using namespace triton::cuda_allocator;

namespace {
struct ProfileAllocator {
  uintptr_t reserve;
  AllocNode root;
};
std::unique_ptr<ProfileAllocator> allocator;
std::mutex mutex;
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_NONE;
bool configFrozen = false;

CUresult ensureInitialized() {
  if (allocator)
    return CUDA_SUCCESS;
  CUdeviceptr reserve = 0;
  CUresult err = cuMemAddressReserve(&reserve, gsan_profile::kReserveSize,
                                     gsan_profile::kReserveSize, 0, 0);
  if (err != CUDA_SUCCESS)
    return err;
  allocator = std::make_unique<ProfileAllocator>();
  allocator->reserve = reserve;
  allocator->root.virtualAddress = reserve + gsan_profile::kDataSize;
  allocator->root.size = gsan_profile::kDataSize;
  allocator->root.maxFreeBlockSize = gsan_profile::kDataSize;
  return CUDA_SUCCESS;
}

CUmemAllocationHandleType freezeConfig() {
  if (handleType == CU_MEM_HANDLE_TYPE_NONE)
    handleType = defaultShareableHandleType();
  configFrozen = true;
  return handleType;
}

AllocNode *findAllocation(uintptr_t address) {
  if (!allocator || !gsan_profile::isManaged(address, allocator->reserve))
    return nullptr;
  auto *node = findNodeByAddress(&allocator->root, address);
  if (!node || node->maxFreeBlockSize != 0 ||
      address - node->virtualAddress >= node->allocSize)
    return nullptr;
  return node;
}

size_t metadataSize(size_t size) {
  return cdiv(size, gsan_profile::kGranularity) *
         sizeof(gsan_profile::ProfileCell);
}

PyObject *cudaError(CUresult err) {
  const char *message = "unknown CUDA error";
  cuGetErrorString(err, &message);
  PyErr_Format(PyExc_RuntimeError, "profiling allocator: %s", message);
  return nullptr;
}
} // namespace

extern "C" void *gsanProfileMalloc(ssize_t size, int device, void *stream) {
  if (size <= 0 || static_cast<size_t>(size) > gsan_profile::kDataSize)
    return nullptr;
  std::lock_guard guard(mutex);
  CUresult err = ensureContext(device);
  if (err == CUDA_SUCCESS)
    err = ensureInitialized();
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return nullptr;
  }
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.requestedHandleTypes = freezeConfig();
  size_t pageSize = 0;
  err = cuMemGetAllocationGranularity(&pageSize, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return nullptr;
  }
  // Round data so that its quarter-sized metadata is page aligned too.
  size_t allocSize = roundUp(size, pageSize * 4);
  auto *node = allocateNode(&allocator->root, allocSize);
  if (!node)
    return nullptr;
  err = createAllocation(
      node, allocSize, gsan_profile::metadataAddress(node->virtualAddress),
      metadataSize(allocSize), prop, reinterpret_cast<CUstream>(stream));
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    freeNode(node);
    return nullptr;
  }
  return reinterpret_cast<void *>(node->virtualAddress);
}

extern "C" void gsanProfileFree(void *ptr, ssize_t, int, void *stream) {
  if (!ptr)
    return;
  std::lock_guard guard(mutex);
  auto address = reinterpret_cast<uintptr_t>(ptr);
  auto *node = findAllocation(address);
  if (!node || node->virtualAddress != address) {
    fprintf(stderr, "gsanProfileFree called with an invalid pointer\n");
    return;
  }
  releaseAllocation(node, gsan_profile::metadataAddress(address),
                    metadataSize(node->allocSize),
                    reinterpret_cast<CUstream>(stream));
}

bool profileGetLayout(uintptr_t address, uintptr_t &base, size_t &size) {
  std::lock_guard guard(mutex);
  auto *node = findAllocation(address);
  if (!node)
    return false;
  base = node->virtualAddress;
  size = node->allocSize;
  return true;
}

namespace {
PyObject *pyMalloc(PyObject *, PyObject *args) {
  Py_ssize_t size;
  int device;
  unsigned long long stream = 0;
  if (!PyArg_ParseTuple(args, "ni|K", &size, &device, &stream))
    return nullptr;
  auto *ptr = gsanProfileMalloc(size, device, reinterpret_cast<void *>(stream));
  if (!ptr)
    return PyErr_NoMemory();
  return PyLong_FromVoidPtr(ptr);
}

PyObject *pyFree(PyObject *, PyObject *args) {
  unsigned long long ptr, stream = 0;
  int device;
  if (!PyArg_ParseTuple(args, "Ki|K", &ptr, &device, &stream))
    return nullptr;
  gsanProfileFree(reinterpret_cast<void *>(ptr), 0, device,
                  reinterpret_cast<void *>(stream));
  Py_RETURN_NONE;
}

PyObject *pyConfigure(PyObject *, PyObject *arg) {
  CUmemAllocationHandleType requested;
  if (!parseShareableHandleTypeArg(arg, "handle_type", &requested))
    return nullptr;
  std::lock_guard guard(mutex);
  if (configFrozen) {
    PyErr_SetString(PyExc_RuntimeError,
                    "profiling allocator configuration is already frozen");
    return nullptr;
  }
  handleType = requested;
  Py_RETURN_NONE;
}

PyObject *pyReserve(PyObject *, PyObject *) {
  std::lock_guard guard(mutex);
  auto err = ensureInitialized();
  if (err != CUDA_SUCCESS)
    return cudaError(err);
  return PyLong_FromUnsignedLongLong(allocator->reserve);
}

PyObject *pyLayout(PyObject *, PyObject *arg) {
  void *ptr;
  if (!parseVoidPtrArg(arg, &ptr))
    return nullptr;
  std::lock_guard guard(mutex);
  auto address = reinterpret_cast<uintptr_t>(ptr);
  auto *node = findAllocation(address);
  if (!node)
    Py_RETURN_NONE;
  return Py_BuildValue(
      "(KKK)", static_cast<unsigned long long>(node->virtualAddress),
      static_cast<unsigned long long>(node->allocSize),
      static_cast<unsigned long long>(gsan_profile::metadataAddress(address)));
}

PyObject *pyRegions(PyObject *, PyObject *arg) {
  void *ptr;
  if (!parseVoidPtrArg(arg, &ptr))
    return nullptr;
  std::lock_guard guard(mutex);
  auto *node = findAllocation(reinterpret_cast<uintptr_t>(ptr));
  if (!node) {
    PyErr_SetString(PyExc_ValueError, "expected a live profiling allocation");
    return nullptr;
  }
  return Py_BuildValue(
      "(KKKK)", static_cast<unsigned long long>(node->virtualAddress),
      static_cast<unsigned long long>(node->allocSize),
      static_cast<unsigned long long>(
          gsan_profile::metadataAddress(node->virtualAddress)),
      static_cast<unsigned long long>(metadataSize(node->allocSize)));
}

PyObject *pyExport(PyObject *, PyObject *args) {
  PyObject *pointer, *kind;
  if (!PyArg_ParseTuple(args, "OO", &pointer, &kind))
    return nullptr;
  void *ptr;
  CUmemAllocationHandleType type;
  if (!parseVoidPtrArg(pointer, &ptr) ||
      !parseShareableHandleTypeArg(kind, "handle_type", &type))
    return nullptr;
  std::lock_guard guard(mutex);
  auto *node = findAllocation(reinterpret_cast<uintptr_t>(ptr));
  if (!node) {
    PyErr_SetString(PyExc_ValueError, "expected a live profiling allocation");
    return nullptr;
  }
  ShareableHandle real = {}, shadow = {};
  auto err = exportAllocation(node, &real, &shadow, type);
  if (err != CUDA_SUCCESS)
    return cudaError(err);
  return Py_BuildValue("(NNK)", shareableHandleToPyObject(real, type),
                       shareableHandleToPyObject(shadow, type),
                       static_cast<unsigned long long>(node->allocSize));
}

PyObject *pyImport(PyObject *, PyObject *args) {
  PyObject *realObj, *shadowObj, *kind;
  unsigned long long size;
  int device;
  if (!PyArg_ParseTuple(args, "OOKiO", &realObj, &shadowObj, &size, &device,
                        &kind))
    return nullptr;
  CUmemAllocationHandleType type;
  ShareableHandle real = {}, shadow = {};
  if (!parseShareableHandleTypeArg(kind, "handle_type", &type) ||
      !parseShareableHandleArg(realObj, "real_handle", type, &real) ||
      !parseShareableHandleArg(shadowObj, "shadow_handle", type, &shadow))
    return nullptr;
  if (!size || size > gsan_profile::kDataSize ||
      size % gsan_profile::kGranularity) {
    PyErr_SetString(PyExc_ValueError, "invalid profiling allocation size");
    return nullptr;
  }
  std::lock_guard guard(mutex);
  CUresult err = ensureContext(device);
  if (err == CUDA_SUCCESS)
    err = ensureInitialized();
  if (err != CUDA_SUCCESS)
    return cudaError(err);
  freezeConfig();
  auto *node = allocateNode(&allocator->root, size);
  if (!node)
    return PyErr_NoMemory();
  err = importAllocation(node, size,
                         gsan_profile::metadataAddress(node->virtualAddress),
                         metadataSize(size), real, shadow, type, device);
  if (err != CUDA_SUCCESS) {
    freeNode(node);
    return cudaError(err);
  }
  return PyLong_FromUnsignedLongLong(node->virtualAddress);
}

PyObject *pyHasLiveAllocations(PyObject *, PyObject *) {
  std::lock_guard guard(mutex);
  return PyBool_FromLong(allocator && allocator->root.maxFreeBlockSize !=
                                          allocator->root.size);
}

PyObject *pyReset(PyObject *, PyObject *) {
  std::lock_guard guard(mutex);
  if (allocator) {
    if (allocator->root.maxFreeBlockSize != allocator->root.size) {
      PyErr_SetString(PyExc_RuntimeError,
                      "cannot reset profiling allocator with live allocations");
      return nullptr;
    }
    auto err = cuMemAddressFree(allocator->reserve, gsan_profile::kReserveSize);
    if (err != CUDA_SUCCESS)
      return cudaError(err);
    allocator.reset();
  }
  configFrozen = false;
  handleType = CU_MEM_HANDLE_TYPE_NONE;
  Py_RETURN_NONE;
}

PyObject *pyRegisterCUDAAllocator(PyObject *self, PyObject *) {
  if (!registerProfileCUDAAllocatorBindings(self))
    return nullptr;
  Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"_register_cuda_allocator", pyRegisterCUDAAllocator, METH_NOARGS, nullptr},
    {"malloc", pyMalloc, METH_VARARGS, nullptr},
    {"free", pyFree, METH_VARARGS, nullptr},
    {"configure", pyConfigure, METH_O, nullptr},
    {"get_reserve_pointer", pyReserve, METH_NOARGS, nullptr},
    {"get_allocation_profile_layout", pyLayout, METH_O, nullptr},
    {"export_allocation_memhandle_regions", pyRegions, METH_O, nullptr},
    {"export_allocation_handles", pyExport, METH_VARARGS, nullptr},
    {"import_allocation_handles", pyImport, METH_VARARGS, nullptr},
    {"has_live_allocations", pyHasLiveAllocations, METH_NOARGS, nullptr},
    {"reset", pyReset, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "gsan_profile_allocator", nullptr,
                      -1, methods};
} // namespace

PyMODINIT_FUNC PyInit_gsan_profile_allocator() {
  return PyModule_Create(&module);
}
