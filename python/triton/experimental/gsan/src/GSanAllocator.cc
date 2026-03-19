#include <Python.h>
#include <cuda.h>

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>

#include "GSan.h"

// #define GSAN_LOG_ALLOCATIONS
#ifdef GSAN_LOG_ALLOCATIONS
#define LOGF(...) printf(__VA_ARGS__);
#else
#define LOGF(...)
#endif

extern "C" {
void *gsanMalloc(ssize_t size, int device, void *stream);
void gsanFree(void *ptr, ssize_t size, int device, void *stream);
}

namespace {

// We use a tree structure to manage virtual address allocations.
//
// This is a binary tree where each node represents a power of two-sized region
// of memory. Each node tracks the largest free node in its subtree. This
// allows us to allocate best-fit regions in O(log(AddressSpaceSize)), and same
// for deallocation.
//
// Note that we don't really care about being compact/defragmented in any way,
// since we can reserve millions of times more virtual memory than there is
// physical memory.
// We also are based under the PyTorch CUDACachingAllocator which manages most
// of the hard parts for us and only asks us to allocate large blocks that it
// will divide up as needed.
struct AllocNode {
  CUdeviceptr virtualAddress = 0;
  AllocNode *parent = nullptr;
  std::unique_ptr<AllocNode> leftChild;
  std::unique_ptr<AllocNode> rightChild;
  size_t size = 0;
  size_t maxFreeBlockSize = 0;

  // Allocation handles, used only by leaf nodes
  CUmemGenericAllocationHandle realHandle = 0;
  CUmemGenericAllocationHandle shadowHandle = 0;
  size_t allocSize = 0;
};

struct GSanConfig {
  int numGPUs;
  int numSMs;
  int numThreads;
  int clockBufferSize;
  uint32_t rngSeed;
};

struct AllocatorState {
  GSanConfig config;

  // User memory + shadow memory
  CUdeviceptr reserveBaseAddress = 0;
  AllocNode treeRoot;

  // GSan global state
  CUdeviceptr globalStateAddress = 0;
  CUmemGenericAllocationHandle perDeviceHandles[gsan::kMaxGPUs] = {0};
  size_t perDeviceStateSize = 0;
};

void printCUDAError(CUresult err) {
  const char *errs = "<unknown error>";
  cuGetErrorString(err, &errs);
  fprintf(stderr, "gsan allocator encountered an unexpected error: %s\n", errs);
}

static AllocatorState *alloc = nullptr;
static std::mutex mut;

size_t cdiv(size_t num, size_t den) { return (num + (den - 1)) / den; }

size_t roundUp(size_t val, size_t alignment) {
  return cdiv(val, alignment) * alignment;
}

uint32_t roundDownToPowerOfTwo(uint32_t x) {
  if (x == 0)
    return 0;

  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;

  return x - (x >> 1);
}

size_t getShadowSize(size_t realMemSize) {
  auto wordSize = cdiv(realMemSize, gsan::kShadowMemGranularityBytes);
  return wordSize * sizeof(gsan::ShadowCell);
}

bool isLeaf(const AllocNode *node) {
  return node->leftChild == nullptr && node->rightChild == nullptr;
}

void recomputeNodeState(AllocNode *node) {
  assert((node->leftChild == nullptr) == (node->rightChild == nullptr) &&
         "allocator tree node should have both children or none");

  if (isLeaf(node)) {
    assert(
        (node->maxFreeBlockSize == 0 || node->maxFreeBlockSize == node->size) &&
        "leaf nodes should be either fully free or fully allocated");
    return;
  }

  node->maxFreeBlockSize = std::max(node->leftChild->maxFreeBlockSize,
                                    node->rightChild->maxFreeBlockSize);
}

void recomputeToRoot(AllocNode *node) {
  for (AllocNode *curr = node; curr != nullptr; curr = curr->parent)
    recomputeNodeState(curr);
}

void splitNode(AllocNode *node) {
  assert(isLeaf(node));
  assert(node->maxFreeBlockSize == node->size);
  const size_t halfSize = node->size / 2;
  auto left = std::make_unique<AllocNode>();
  auto right = std::make_unique<AllocNode>();

  left->virtualAddress = node->virtualAddress;
  left->size = halfSize;
  left->maxFreeBlockSize = halfSize;
  left->parent = node;

  right->virtualAddress = node->virtualAddress + halfSize;
  right->size = halfSize;
  right->maxFreeBlockSize = halfSize;
  right->parent = node;

  node->leftChild = std::move(left);
  node->rightChild = std::move(right);
  node->maxFreeBlockSize = halfSize;
}

AllocNode *allocateNode(AllocNode *root, size_t allocSize) {
  AllocNode *node = root;
  if (node == nullptr || node->maxFreeBlockSize < allocSize)
    return nullptr;

  if (isLeaf(node)) {
    assert(node->maxFreeBlockSize == node->size);

    while (node->size > 1 && (node->size / 2) >= allocSize) {
      splitNode(node);
      node = node->leftChild.get();
    }
    node->maxFreeBlockSize = 0;
    recomputeToRoot(node->parent);
    return node;
  }

  auto *left = node->leftChild.get();
  auto *right = node->rightChild.get();
  const bool leftFits = left->maxFreeBlockSize >= allocSize;
  const bool rightFits = right->maxFreeBlockSize >= allocSize;

  AllocNode *next = nullptr;
  // Prefer the tighter-fitting subtree to keep larger blocks available.
  if (leftFits && rightFits) {
    next = (left->maxFreeBlockSize <= right->maxFreeBlockSize) ? left : right;
  } else if (leftFits) {
    next = left;
  } else {
    next = right;
  }
  return allocateNode(next, allocSize);
}

AllocNode *findNodeByAddress(AllocNode *root, CUdeviceptr address) {
  AllocNode *node = root;
  while (node != nullptr) {
    if (address < node->virtualAddress ||
        address >= node->virtualAddress + node->size)
      return nullptr;

    if (!node->leftChild && !node->rightChild)
      return node;

    if (node->rightChild && address >= node->rightChild->virtualAddress) {
      node = node->rightChild.get();
    } else {
      node = node->leftChild.get();
    }
  }
  return nullptr;
}

bool canCoalesce(const AllocNode *node) {
  if (node == nullptr)
    return false;
  assert((node->leftChild == nullptr) == (node->rightChild == nullptr) &&
         "allocator tree node should have both children or none");
  if (!node->leftChild)
    return false;

  const auto *left = node->leftChild.get();
  const auto *right = node->rightChild.get();
  const bool leftFree = left->maxFreeBlockSize == left->size;
  const bool rightFree = right->maxFreeBlockSize == right->size;
  return leftFree && rightFree;
}

void coalesceUp(AllocNode *node) {
  if (node == nullptr)
    return;
  while (node != nullptr && canCoalesce(node)) {
    node->leftChild.reset();
    node->rightChild.reset();
    node->maxFreeBlockSize = node->size;
    node = node->parent;
  }
  recomputeToRoot(node);
}

void freeNode(AllocNode *leaf) {
  assert(isLeaf(leaf));
  leaf->allocSize = 0;
  leaf->realHandle = 0;
  leaf->shadowHandle = 0;
  leaf->maxFreeBlockSize = leaf->size;
  coalesceUp(leaf->parent);
}

int gsanEnsureInit() {
  if (alloc)
    return 0;

  CUdeviceptr reserveBase;
  CUresult err = cuMemAddressReserve(&reserveBase, /*size*/ gsan::kReserveSize,
                                     /*alignment*/ gsan::kReserveSize,
                                     /*addr*/ 0, /*flags*/ 0);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return -1;
  }

  CUdeviceptr globalsBase;
  err = cuMemAddressReserve(&globalsBase, /*size*/ gsan::kGlobalsReserveSize,
                            /*alignment*/ gsan::kGlobalsReserveSize,
                            /*addr*/ 0, /*flags*/ 0);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return -1;
  }
  alloc = new AllocatorState();
  alloc->reserveBaseAddress = reserveBase;
  alloc->globalStateAddress = globalsBase;

  auto *root = &alloc->treeRoot;
  root->virtualAddress = gsan::getRealBaseAddress(reserveBase);

  // Choose size so that both shadow memory and real memory definitely fit in
  // the address reservation
  auto shadowSize = gsan::kReserveSize / 2;
  auto realSize = gsan::kShadowMemGranularityBytes *
                  (shadowSize / sizeof(gsan::ShadowCell));
  realSize = std::min(gsan::kReserveSize / 2, realSize);
  realSize = roundDownToPowerOfTwo(realSize);
  root->size = realSize;
  root->maxFreeBlockSize = realSize;
  return 0;
}

CUresult ensureContext(int device) {
  CUcontext ctx = 0;
  CUresult res = cuCtxGetCurrent(&ctx);
  if (res != CUDA_SUCCESS)
    return res;
  if (ctx)
    return res;

  res = cuDevicePrimaryCtxRetain(&ctx, device);
  if (res != CUDA_SUCCESS)
    return res;
  return cuCtxSetCurrent(ctx);
}

CUresult refreshConfigForDevice(int device) {
  if (alloc == nullptr)
    return CUDA_ERROR_NOT_INITIALIZED;

  int numGPUs = 0;
  CUresult err = cuDeviceGetCount(&numGPUs);
  if (err != CUDA_SUCCESS)
    return err;
  if (numGPUs <= 0)
    return CUDA_ERROR_NO_DEVICE;
  if (numGPUs > static_cast<int>(gsan::kMaxGPUs))
    return CUDA_ERROR_NOT_SUPPORTED;
  if (device < 0 || device >= numGPUs)
    return CUDA_ERROR_INVALID_DEVICE;

  CUdevice cuDevice = 0;
  err = cuDeviceGet(&cuDevice, device);
  if (err != CUDA_SUCCESS)
    return err;

  int numSMs = 0;
  err = cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                             cuDevice);
  if (err != CUDA_SUCCESS)
    return err;
  if (numSMs <= 0)
    return CUDA_ERROR_INVALID_VALUE;

  auto &config = alloc->config;
  config.numGPUs = numGPUs;
  config.numSMs = numSMs;
  config.numThreads = config.numGPUs * config.numSMs;

  // Seed rng for stochastic read clocks
  auto userSeed = getenv("TRITON_GSAN_SEED");
  if (userSeed) {
    auto res =
        std::from_chars(userSeed, userSeed + strlen(userSeed), config.rngSeed);
    if (res.ec != std::errc()) {
      auto errc = make_error_code(res.ec);
      auto msg = errc.message();
      fprintf(stderr, "Invalid TRITON_GSAN_SEED value: %s", msg.c_str());
      return CUDA_ERROR_INVALID_VALUE;
    }
  } else {
    std::uniform_int_distribution<uint32_t> dist;
    std::random_device rd{};
    config.rngSeed = dist(rd);
  }

  auto userClockSize = getenv("TRITON_GSAN_CLOCK_BUFFER_SIZE");
  if (userClockSize) {
    auto res =
        std::from_chars(userSeed, userSeed + strlen(userSeed), config.rngSeed);
    if (res.ec != std::errc()) {
      auto errc = make_error_code(res.ec);
      auto msg = errc.message();
      fprintf(stderr, "Invalid TRITON_CLOCK_BUFFER_SIZE value: %s",
              msg.c_str());
      return CUDA_ERROR_INVALID_VALUE;
    }
  } else {
    config.clockBufferSize = 1024;
  }
  return CUDA_SUCCESS;
}

CUresult ensureRuntimeStateMapped(int device) {
  if (alloc == nullptr)
    return CUDA_ERROR_NOT_INITIALIZED;
  CUresult err = refreshConfigForDevice(device);
  if (err != CUDA_SUCCESS)
    return err;
  auto &config = alloc->config;
  if (alloc->perDeviceHandles[device] != 0)
    return CUDA_SUCCESS;

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;

  size_t granularity = 0;
  err = cuMemGetAllocationGranularity(&granularity, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (err != CUDA_SUCCESS)
    return err;

  static_assert(alignof(gsan::GlobalState) >= alignof(gsan::ThreadState));
  static_assert(alignof(gsan::ThreadState) >= alignof(gsan::epoch_t));

  auto numSMs = config.numSMs;
  auto numThreads = config.numThreads;
  assert(numThreads <= gsan::kMaxThreads);
  auto clockSizeBytes = sizeof(gsan::epoch_t) * config.numThreads;
  // 1 local clock + the circular clock buffer
  auto clocksPerThread = 1 + config.clockBufferSize;
  auto perSMStateSize =
      sizeof(gsan::ThreadState) + clockSizeBytes * clocksPerThread;
  perSMStateSize = roundUp(perSMStateSize, alignof(gsan::ThreadState));
  // Each device has a local copy of the constant global state
  auto perDeviceStateSize =
      (sizeof(gsan::GlobalState) + config.numSMs * perSMStateSize);
  size_t allocSize = roundUp(perDeviceStateSize, granularity);
  assert(allocSize <= gsan::kPerDeviceStateStride);

  CUmemGenericAllocationHandle allocHandle = 0;
  bool mapped = false;
  CUmemAccessDesc accessDesc = {};
  gsan::GlobalState globals = {};
  CUdeviceptr deviceAddr =
      alloc->globalStateAddress + device * gsan::kPerDeviceStateStride;

  err = cuMemCreate(&allocHandle, allocSize, &prop, 0);
  if (err != CUDA_SUCCESS)
    goto error;

  err = cuMemMap(deviceAddr, allocSize, /*offset*/ 0, allocHandle, /*flags*/ 0);
  if (err != CUDA_SUCCESS)
    goto error;
  mapped = true;

  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  err = cuMemSetAccess(deviceAddr, allocSize, &accessDesc, 1);
  if (err != CUDA_SUCCESS)
    goto error;

  err = cuMemsetD8(deviceAddr, 0, allocSize);
  if (err != CUDA_SUCCESS)
    goto error;

  globals.reserveBase = static_cast<uintptr_t>(alloc->reserveBaseAddress);
  globals.globalsBase = static_cast<uintptr_t>(alloc->globalStateAddress);
  globals.rngSeed = config.rngSeed;
  globals.numSms = static_cast<gsan::thread_id_t>(config.numSMs);
  globals.numDevices = static_cast<gsan::thread_id_t>(config.numGPUs);
  globals.numThreads = static_cast<gsan::thread_id_t>(config.numThreads);
  globals.clockBufferSize = config.clockBufferSize;
  err = cuMemcpyHtoD(deviceAddr, &globals, sizeof(globals));
  if (err != CUDA_SUCCESS)
    goto error;

  alloc->perDeviceHandles[device] = allocHandle;
  alloc->perDeviceStateSize = allocSize;
  return CUDA_SUCCESS;

error:
  if (mapped)
    cuMemUnmap(deviceAddr, allocSize);
  if (allocHandle != 0)
    cuMemRelease(allocHandle);
  return err;
}

CUresult mapNodeHandles(AllocNode *node,
                        CUmemGenericAllocationHandle realHandle,
                        CUmemGenericAllocationHandle shadowHandle, int device,
                        bool *realMapped, bool *shadowMapped) {
  assert(node != nullptr);
  assert(realMapped != nullptr);
  assert(shadowMapped != nullptr);

  const auto shadowAddress = gsan::getShadowAddress(node->virtualAddress);
  const auto shadowSize = getShadowSize(node->size);

  CUresult err = cuMemMap(node->virtualAddress, node->size, /*offset*/ 0,
                          realHandle, /*flags*/ 0);
  if (err != CUDA_SUCCESS)
    return err;
  *realMapped = true;

  err = cuMemMap(shadowAddress, shadowSize, /*offset*/ 0, shadowHandle,
                 /*flags*/ 0);
  if (err != CUDA_SUCCESS)
    return err;
  *shadowMapped = true;

  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  err = cuMemSetAccess(node->virtualAddress, node->size, &accessDesc, 1);
  if (err != CUDA_SUCCESS)
    return err;

  return cuMemSetAccess(shadowAddress, shadowSize, &accessDesc, 1);
}

void unmapNodeHandles(AllocNode *node, bool realMapped, bool shadowMapped) {
  assert(node != nullptr);
  const auto shadowAddress = gsan::getShadowAddress(node->virtualAddress);
  const auto shadowSize = getShadowSize(node->size);
  if (shadowMapped)
    cuMemUnmap(shadowAddress, shadowSize);
  if (realMapped)
    cuMemUnmap(node->virtualAddress, node->size);
}

} // namespace

// TODO: Handle streams?
extern "C" void *gsanMalloc(ssize_t size, int device,
                            [[maybe_unused]] void *stream) {
  if (size <= 0)
    return nullptr;

  std::lock_guard lg(mut);
  if (gsanEnsureInit() != 0)
    return nullptr;

  CUresult err = ensureContext(device);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return nullptr;
  }
  err = ensureRuntimeStateMapped(device);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return nullptr;
  }

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;

  size_t granularity = 0;
  err = cuMemGetAllocationGranularity(&granularity, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return nullptr;
  }
  size_t allocSize = roundUp(static_cast<size_t>(size), granularity);
  AllocNode *node = allocateNode(&alloc->treeRoot, allocSize);
  if (node == nullptr)
    return nullptr;

  CUmemGenericAllocationHandle realHandle = 0;
  CUmemGenericAllocationHandle shadowHandle = 0;
  bool realMapped = false;
  bool shadowMapped = false;
  auto cuStream = reinterpret_cast<CUstream>(stream);
  auto shadowAddress = gsan::getShadowAddress(node->virtualAddress);
  auto shadowSize = getShadowSize(allocSize);

  err = cuMemCreate(&realHandle, allocSize, &prop, 0);
  if (err != CUDA_SUCCESS)
    goto error;

  err = cuMemCreate(&shadowHandle, shadowSize, &prop, 0);
  if (err != CUDA_SUCCESS)
    goto error;

  err = mapNodeHandles(node, realHandle, shadowHandle, device, &realMapped,
                       &shadowMapped);
  if (err != CUDA_SUCCESS)
    goto error;

  // Zero-initialize shadow memory
  err = cuMemsetD8Async(shadowAddress, 0, shadowSize, cuStream);
  if (err != CUDA_SUCCESS)
    goto error;

  node->allocSize = allocSize;
  node->realHandle = realHandle;
  node->shadowHandle = shadowHandle;
  LOGF("gsanMalloc: %p, 0x%zxu", reinterpret_cast<void *>(node->virtualAddress),
       size);
  return reinterpret_cast<void *>(node->virtualAddress);

error:
  printCUDAError(err);
  unmapNodeHandles(node, realMapped, shadowMapped);
  if (shadowHandle != 0)
    cuMemRelease(shadowHandle);
  if (realHandle != 0)
    cuMemRelease(realHandle);
  freeNode(node);
  return nullptr;
}

extern "C" void gsanFree(void *void_ptr, [[maybe_unused]] ssize_t size,
                         [[maybe_unused]] int device, void *stream) {
  LOGF("gsanFree: %p, 0x%zx", void_ptr, size);
  auto ptr = reinterpret_cast<CUdeviceptr>(void_ptr);
  if (!ptr)
    return;

  std::lock_guard lg(mut);
  if (alloc == nullptr)
    return;

  AllocNode *node = findNodeByAddress(&alloc->treeRoot, ptr);
  if (node == nullptr || node->maxFreeBlockSize != 0 ||
      node->virtualAddress != ptr) {
    fprintf(stderr, "gsanFree called with an invalid pointer\n");
    return;
  }

  // Wait for outstanding work on the deallocation stream, including the
  // allocator's own async shadow memset from gsanMalloc, before unmapping.
  auto cuStream = reinterpret_cast<CUstream>(stream);
  CUresult err = cuStreamSynchronize(cuStream);
  if (err != CUDA_SUCCESS)
    printCUDAError(err);

  const auto shadowAddress = gsan::getShadowAddress(node->virtualAddress);
  const auto shadowSize = getShadowSize(node->allocSize);

  err = cuMemUnmap(node->virtualAddress, node->allocSize);
  if (err != CUDA_SUCCESS)
    printCUDAError(err);

  err = cuMemUnmap(shadowAddress, shadowSize);
  if (err != CUDA_SUCCESS)
    printCUDAError(err);

  err = cuMemRelease(node->realHandle);
  if (err != CUDA_SUCCESS)
    printCUDAError(err);

  err = cuMemRelease(node->shadowHandle);
  if (err != CUDA_SUCCESS)
    printCUDAError(err);

  freeNode(node);
}

void *gsanGetReservePointer() {
  std::lock_guard lg(mut);
  if (gsanEnsureInit() != 0)
    return nullptr;
  return reinterpret_cast<void *>(alloc->reserveBaseAddress);
}

namespace {

constexpr const char *kModuleName = "gsan_allocator";

bool parseIntArg(PyObject *obj, const char *name, int *out) {
  long value = PyLong_AsLong(obj);
  if (value == -1 && PyErr_Occurred())
    return false;
  if (value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    PyErr_Format(PyExc_OverflowError, "%s is out of range for int", name);
    return false;
  }
  *out = static_cast<int>(value);
  return true;
}

bool parseVoidPtrArg(PyObject *obj, void **out) {
  *out = PyLong_AsVoidPtr(obj);
  return !(*out == nullptr && PyErr_Occurred());
}

PyObject *pyMalloc([[maybe_unused]] PyObject *self, PyObject *const *args,
                   Py_ssize_t nargs) {
  if (nargs != 2 && nargs != 3) {
    PyErr_Format(PyExc_TypeError,
                 "%s.malloc expected 2 or 3 positional arguments, got %zd",
                 kModuleName, nargs);
    return nullptr;
  }

  Py_ssize_t size = PyLong_AsSsize_t(args[0]);
  if (size == -1 && PyErr_Occurred())
    return nullptr;

  int device = 0;
  if (!parseIntArg(args[1], "device", &device))
    return nullptr;

  void *stream = nullptr;
  if (nargs == 3 && !parseVoidPtrArg(args[2], &stream))
    return nullptr;

  return PyLong_FromVoidPtr(gsanMalloc(size, device, stream));
}

PyObject *pyFree([[maybe_unused]] PyObject *self, PyObject *const *args,
                 Py_ssize_t nargs) {
  if (nargs < 2 || nargs > 4) {
    PyErr_Format(
        PyExc_TypeError,
        "%s.free expected between 2 and 4 positional arguments, got %zd",
        kModuleName, nargs);
    return nullptr;
  }

  void *ptr = nullptr;
  if (!parseVoidPtrArg(args[0], &ptr))
    return nullptr;

  int device = 0;
  if (!parseIntArg(args[1], "device", &device))
    return nullptr;

  Py_ssize_t size = 0;
  if (nargs >= 3) {
    size = PyLong_AsSsize_t(args[2]);
    if (size == -1 && PyErr_Occurred())
      return nullptr;
  }

  void *stream = nullptr;
  if (nargs == 4 && !parseVoidPtrArg(args[3], &stream))
    return nullptr;

  gsanFree(ptr, size, device, stream);
  Py_RETURN_NONE;
}

PyObject *pyGetReservePointer([[maybe_unused]] PyObject *self,
                              PyObject *const *args, Py_ssize_t nargs) {
  if (nargs != 0) {
    PyErr_Format(
        PyExc_TypeError,
        "%s.get_reserve_pointer expected 0 positional arguments, got %zd",
        kModuleName, nargs);
    return nullptr;
  }
  return PyLong_FromVoidPtr(gsanGetReservePointer());
}

PyObject *pyGetReserveSize(PyObject *self, PyObject *args) {
  return PyLong_FromUnsignedLongLong(gsan::kReserveSize);
}

PyObject *pyGetShadowSizeBytes(PyObject *self, PyObject *args) {
  return PyLong_FromLong(sizeof(gsan::ShadowCell));
}

PyObject *pyGetGlobalStatePointer([[maybe_unused]] PyObject *self,
                                  PyObject *args) {
  std::lock_guard lg(mut);
  if (gsanEnsureInit() != 0) {
    PyErr_SetString(PyExc_RuntimeError, "failed to initialize gsan allocator");
    return nullptr;
  }
  return PyLong_FromUnsignedLongLong(alloc->globalStateAddress);
}

PyMethodDef kGSanAllocatorMethods[] = {
    {"malloc", reinterpret_cast<PyCFunction>(pyMalloc), METH_FASTCALL,
     "Allocate GSan memory. Returns a CUDA pointer as an integer."},
    {"free", reinterpret_cast<PyCFunction>(pyFree), METH_FASTCALL,
     "Free GSan memory by pointer."},
    {"get_reserve_pointer", reinterpret_cast<PyCFunction>(pyGetReservePointer),
     METH_FASTCALL, "Return the reserve base pointer as an integer."},
    {"get_reserve_size", reinterpret_cast<PyCFunction>(pyGetReserveSize),
     METH_NOARGS, "Return the reserve size in bytes."},
    {"get_shadow_size_bytes",
     reinterpret_cast<PyCFunction>(pyGetShadowSizeBytes), METH_NOARGS,
     "Return the shadow cell size in bytes."},
    {"get_global_state_pointer",
     reinterpret_cast<PyCFunction>(pyGetGlobalStatePointer), METH_NOARGS,
     "Return the pointer to the GSan global state region."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kGSanAllocatorModuleDef = {
    PyModuleDef_HEAD_INIT, "gsan_allocator", nullptr, -1, kGSanAllocatorMethods,
};

} // namespace

PyMODINIT_FUNC PyInit_gsan_allocator(void) {
  return PyModule_Create(&kGSanAllocatorModuleDef);
}
