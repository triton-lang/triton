#include <Python.h>
#include <cuda.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>

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
};

struct GSanConfig {
  int numGPUs = 4;
  int numSMs = 152;
  int numThreads = 4 * 152;
  int clockBufferSize = 1024;
  uint32_t rngSeed = 0x12345678u;
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

size_t roundUpToPowerOfTwo(size_t value) {
  if (value <= 1)
    return 1;
  if ((value & (value - 1)) == 0)
    return value;
  size_t rounded = 1;
  while (rounded < value)
    rounded <<= 1;
  return rounded;
}

size_t getShadowSize(size_t realMemSize) {
  auto wordSize = cdiv(realMemSize, gsan::kShadowMemGranularityBytes);
  return wordSize * gsan::kShadowSizeBytes;
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
  root->size = gsan::kReserveSize / 2;
  root->maxFreeBlockSize = root->size;
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
  // Triton may execute more than one instrumented launch on a kernel's first
  // user-visible invocation (e.g. compile/warmup paths). Using 3 slots avoids
  // aliasing back to the same logical thread ID when two launches occur.
  constexpr int kGSanThreadSlotsPerDeviceThread = 3;
  config.numGPUs = numGPUs;
  config.numSMs = numSMs;
  config.numThreads =
      kGSanThreadSlotsPerDeviceThread * config.numGPUs * config.numSMs;
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
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

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
  auto perDeviceStateSize = (
      // Each device has a local copy of the constant global state
      sizeof(gsan::GlobalState) +
      // Plus per-thread state for each SM
      config.numSMs *
          (sizeof(gsan::ThreadState) + clockSizeBytes * clocksPerThread));
  assert(perDeviceStateSize <= gsan::kPerDeviceStateStride);

  size_t allocSize = roundUp(perDeviceStateSize, granularity);

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
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

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

  err = cuMemCreate(&realHandle, node->size, &prop, 0);
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
                         [[maybe_unused]] int device,
                         [[maybe_unused]] void *stream) {
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

  const auto shadowAddress = gsan::getShadowAddress(node->virtualAddress);
  const auto shadowSize = getShadowSize(node->size);

  CUresult err = cuMemUnmap(node->virtualAddress, node->size);
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

int gsanExportAllocationHandles(void *void_ptr, int *realFd, int *shadowFd,
                                size_t *allocSize) {
  if (realFd == nullptr || shadowFd == nullptr || allocSize == nullptr)
    return -1;
  *realFd = -1;
  *shadowFd = -1;
  *allocSize = 0;

  const auto ptr = reinterpret_cast<CUdeviceptr>(void_ptr);
  if (ptr == 0)
    return -1;

  std::lock_guard lg(mut);
  if (alloc == nullptr)
    return -1;

  AllocNode *node = findNodeByAddress(&alloc->treeRoot, ptr);
  if (node == nullptr || node->maxFreeBlockSize != 0 ||
      node->virtualAddress != ptr) {
    fprintf(stderr,
            "gsanExportAllocationHandles called with invalid pointer\n");
    return -1;
  }

  int realFdLocal = -1;
  int shadowFdLocal = -1;
  CUresult err =
      cuMemExportToShareableHandle(&realFdLocal, node->realHandle,
                                   CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return -1;
  }

  err =
      cuMemExportToShareableHandle(&shadowFdLocal, node->shadowHandle,
                                   CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    close(realFdLocal);
    return -1;
  }

  *realFd = realFdLocal;
  *shadowFd = shadowFdLocal;
  *allocSize = node->size;
  return 0;
}

int gsanExportRuntimeStateHandle(int device, int *fd, size_t *allocSize) {
  if (fd == nullptr || allocSize == nullptr)
    return -1;
  *fd = -1;
  *allocSize = 0;

  if (device < 0 || device >= static_cast<int>(gsan::kMaxGPUs))
    return -1;

  std::lock_guard lg(mut);
  if (gsanEnsureInit() != 0)
    return -1;
  CUresult err = ensureContext(device);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return -1;
  }
  err = ensureRuntimeStateMapped(device);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return -1;
  }

  auto handle = alloc->perDeviceHandles[device];
  auto size = alloc->perDeviceStateSize;
  if (handle == 0 || size == 0)
    return -1;

  int fdLocal = -1;
  err = cuMemExportToShareableHandle(
      &fdLocal, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return -1;
  }

  *fd = fdLocal;
  *allocSize = size;
  return 0;
}

void *gsanImportAllocationHandles(int realFd, int shadowFd, size_t allocSize,
                                  int device) {
  if (realFd < 0 || shadowFd < 0 || allocSize == 0)
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

  AllocNode *node = allocateNode(&alloc->treeRoot, allocSize);
  if (node == nullptr)
    return nullptr;

  if (node->size != allocSize) {
    freeNode(node);
    fprintf(stderr, "gsanImportAllocationHandles requires power-of-two size\n");
    return nullptr;
  }

  CUmemGenericAllocationHandle realHandle = 0;
  CUmemGenericAllocationHandle shadowHandle = 0;
  bool realMapped = false;
  bool shadowMapped = false;
  err = cuMemImportFromShareableHandle(
      &realHandle, reinterpret_cast<void *>(static_cast<uintptr_t>(realFd)),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  if (err != CUDA_SUCCESS)
    goto error;

  err = cuMemImportFromShareableHandle(
      &shadowHandle, reinterpret_cast<void *>(static_cast<uintptr_t>(shadowFd)),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  if (err != CUDA_SUCCESS)
    goto error;

  err = mapNodeHandles(node, realHandle, shadowHandle, device, &realMapped,
                       &shadowMapped);
  if (err != CUDA_SUCCESS)
    goto error;

  node->realHandle = realHandle;
  node->shadowHandle = shadowHandle;
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

int gsanImportRuntimeStateHandle(int fd, size_t allocSize, int peerDevice,
                                 int device) {
  if (fd < 0 || allocSize == 0)
    return -1;
  if (peerDevice < 0 || peerDevice >= static_cast<int>(gsan::kMaxGPUs))
    return -1;

  std::lock_guard lg(mut);
  if (gsanEnsureInit() != 0)
    return -1;
  CUresult err = ensureContext(device);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return -1;
  }

  if (device < 0 || device >= alloc->config.numGPUs)
    return -1;
  if (allocSize != alloc->perDeviceStateSize)
    return -1;

  CUmemGenericAllocationHandle importedHandle = 0;
  bool mapped = false;
  CUmemAccessDesc accessDesc = {};
  CUdeviceptr deviceAddr =
      alloc->globalStateAddress + peerDevice * gsan::kPerDeviceStateStride;

  err = cuMemImportFromShareableHandle(
      &importedHandle, reinterpret_cast<void *>(static_cast<uintptr_t>(fd)),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  if (err != CUDA_SUCCESS)
    goto error;

  err = cuMemMap(deviceAddr, allocSize, /*offset*/ 0, importedHandle,
                 /*flags*/ 0);
  if (err != CUDA_SUCCESS)
    goto error;
  mapped = true;

  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  err = cuMemSetAccess(deviceAddr, allocSize, &accessDesc, 1);
  if (err != CUDA_SUCCESS)
    goto error;

  alloc->perDeviceHandles[peerDevice] = importedHandle;
  return 0;

error:
  printCUDAError(err);
  if (mapped)
    cuMemUnmap(deviceAddr, allocSize);
  if (importedHandle != 0)
    cuMemRelease(importedHandle);
  return -1;
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

PyObject *pyMalloc(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
  (void)self;
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

PyObject *pyFree(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
  (void)self;
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

PyObject *pyGetReservePointer(PyObject *self, PyObject *const *args,
                              Py_ssize_t nargs) {
  (void)self;
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

PyObject *pyGetGlobalStatePointer(PyObject *self, PyObject *args) {
  (void)self;
  std::lock_guard lg(mut);
  if (gsanEnsureInit() != 0) {
    PyErr_SetString(PyExc_RuntimeError, "failed to initialize gsan allocator");
    return nullptr;
  }
  return PyLong_FromUnsignedLongLong(alloc->globalStateAddress);
}

PyObject *pyExportAllocationHandles(PyObject *self, PyObject *const *args,
                                    Py_ssize_t nargs) {
  (void)self;
  if (nargs != 1) {
    PyErr_Format(
        PyExc_TypeError,
        "%s.export_allocation_handles expected 1 positional argument, got %zd",
        kModuleName, nargs);
    return nullptr;
  }

  void *ptr = nullptr;
  if (!parseVoidPtrArg(args[0], &ptr))
    return nullptr;

  int realFd = -1;
  int shadowFd = -1;
  size_t allocSize = 0;
  int rc = gsanExportAllocationHandles(ptr, &realFd, &shadowFd, &allocSize);
  if (rc != 0) {
    PyErr_SetString(PyExc_RuntimeError, "gsanExportAllocationHandles failed.");
    return nullptr;
  }

  return Py_BuildValue("(iiK)", realFd, shadowFd,
                       static_cast<unsigned long long>(allocSize));
}

PyObject *pyImportAllocationHandles(PyObject *self, PyObject *const *args,
                                    Py_ssize_t nargs) {
  (void)self;
  if (nargs != 4) {
    PyErr_Format(
        PyExc_TypeError,
        "%s.import_allocation_handles expected 4 positional arguments, got %zd",
        kModuleName, nargs);
    return nullptr;
  }

  int realFd = 0;
  int shadowFd = 0;
  int device = 0;
  if (!parseIntArg(args[0], "real_fd", &realFd) ||
      !parseIntArg(args[1], "shadow_fd", &shadowFd) ||
      !parseIntArg(args[3], "device", &device)) {
    return nullptr;
  }

  size_t allocSize = PyLong_AsSize_t(args[2]);
  if (allocSize == static_cast<size_t>(-1) && PyErr_Occurred())
    return nullptr;

  void *ptr = gsanImportAllocationHandles(realFd, shadowFd, allocSize, device);
  if (ptr == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "gsanImportAllocationHandles failed.");
    return nullptr;
  }

  return PyLong_FromVoidPtr(ptr);
}

PyObject *pyExportRuntimeStateHandle(PyObject *self, PyObject *const *args,
                                     Py_ssize_t nargs) {
  (void)self;
  if (nargs != 1) {
    PyErr_Format(PyExc_TypeError,
                 "%s.export_runtime_state_handle expected 1 positional "
                 "argument, got %zd",
                 kModuleName, nargs);
    return nullptr;
  }

  int device = 0;
  if (!parseIntArg(args[0], "device", &device))
    return nullptr;

  int fd = -1;
  size_t allocSize = 0;
  int rc = gsanExportRuntimeStateHandle(device, &fd, &allocSize);
  if (rc != 0) {
    PyErr_SetString(PyExc_RuntimeError, "gsanExportRuntimeStateHandle failed.");
    return nullptr;
  }

  return Py_BuildValue("(iK)", fd, static_cast<unsigned long long>(allocSize));
}

PyObject *pyImportRuntimeStateHandle(PyObject *self, PyObject *const *args,
                                     Py_ssize_t nargs) {
  (void)self;
  if (nargs != 4) {
    PyErr_Format(PyExc_TypeError,
                 "%s.import_runtime_state_handle expected 4 positional "
                 "arguments, got %zd",
                 kModuleName, nargs);
    return nullptr;
  }

  int fd = 0;
  int peerDevice = 0;
  int device = 0;
  if (!parseIntArg(args[0], "fd", &fd) ||
      !parseIntArg(args[2], "peer_device", &peerDevice) ||
      !parseIntArg(args[3], "device", &device)) {
    return nullptr;
  }

  size_t allocSize = PyLong_AsSize_t(args[1]);
  if (allocSize == static_cast<size_t>(-1) && PyErr_Occurred())
    return nullptr;

  int rc = gsanImportRuntimeStateHandle(fd, allocSize, peerDevice, device);
  if (rc != 0) {
    PyErr_SetString(PyExc_RuntimeError, "gsanImportRuntimeStateHandle failed.");
    return nullptr;
  }

  Py_RETURN_NONE;
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
    {"export_allocation_handles",
     reinterpret_cast<PyCFunction>(pyExportAllocationHandles), METH_FASTCALL,
     "Export allocation handles for an existing allocation pointer."},
    {"import_allocation_handles",
     reinterpret_cast<PyCFunction>(pyImportAllocationHandles), METH_FASTCALL,
     "Import allocation handles and map into this process's VA space."},
    {"export_runtime_state_handle",
     reinterpret_cast<PyCFunction>(pyExportRuntimeStateHandle), METH_FASTCALL,
     "Export a runtime-state handle for a local device."},
    {"import_runtime_state_handle",
     reinterpret_cast<PyCFunction>(pyImportRuntimeStateHandle), METH_FASTCALL,
     "Import a peer runtime-state handle into this process's global-state VA."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kGSanAllocatorModuleDef = {
    PyModuleDef_HEAD_INIT, "gsan_allocator", nullptr, -1, kGSanAllocatorMethods,
};

} // namespace

PyMODINIT_FUNC PyInit_gsan_allocator(void) {
  return PyModule_Create(&kGSanAllocatorModuleDef);
}
