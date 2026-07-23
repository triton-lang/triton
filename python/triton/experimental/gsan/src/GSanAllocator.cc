#include <Python.h>
#include <cuda.h>

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
constexpr size_t kThreadStateHeaderSize =
    offsetof(gsan::ThreadState, vectorClock);

union GSanShareableHandle {
  int fd;
  CUmemFabricHandle fabricHandle;
};

bool isSupportedShareableHandleType(CUmemAllocationHandleType handleType) {
  return handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR ||
         handleType == CU_MEM_HANDLE_TYPE_FABRIC;
}

void *getShareableHandleImportArg(const GSanShareableHandle *handle,
                                  CUmemAllocationHandleType handleType) {
  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    return reinterpret_cast<void *>(static_cast<uintptr_t>(handle->fd));
  if (handleType == CU_MEM_HANDLE_TYPE_FABRIC)
    return const_cast<CUmemFabricHandle *>(&handle->fabricHandle);
  return nullptr;
}

void *getShareableHandleExportArg(GSanShareableHandle *handle,
                                  CUmemAllocationHandleType handleType) {
  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    return &handle->fd;
  if (handleType == CU_MEM_HANDLE_TYPE_FABRIC)
    return &handle->fabricHandle;
  return nullptr;
}

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
  int numGPUs = 1;
  int numSMs = 0;
  int numThreads = 0;
  int clockBufferSize = 0;
  uint32_t rngSeed = 0;
  CUmemAllocationHandleType shareableHandleType =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  bool clockBufferSizeConfigured = false;
  bool rngSeedConfigured = false;
  bool shareableHandleTypeConfigured = false;
  int deviceRanks[gsan::kMaxGPUs] = {};
  bool configuredDeviceRanks[gsan::kMaxGPUs] = {};
  bool topologyConfigured = false;
  bool topologyFrozen = false;
};

struct AllocatorState {
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
static GSanConfig config;
static std::mutex mut;

CUmemAllocationHandleType getRequestedShareableHandleType() {
  if (config.shareableHandleTypeConfigured)
    return config.shareableHandleType;

  const auto *allocConf = getenv("PYTORCH_CUDA_ALLOC_CONF");
  if (allocConf != nullptr &&
      strstr(allocConf, "fabric_handles:True") != nullptr) {
    return CU_MEM_HANDLE_TYPE_FABRIC;
  }
  return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
}

int getDeviceRankForCudaDevice(int device) {
  if (!config.topologyConfigured)
    return -1;
  if (device < 0 || device >= static_cast<int>(gsan::kMaxGPUs))
    return -1;
  if (!config.configuredDeviceRanks[device])
    return -1;
  return config.deviceRanks[device];
}

CUresult ensureTopologyConfigured() {
  if (config.topologyConfigured)
    return CUDA_SUCCESS;

  // Default topology assumes a single node with 1:1 mapping of device index to
  // GSan device ID.
  int cudaDeviceCount = 0;
  CUresult err = cuDeviceGetCount(&cudaDeviceCount);
  if (err != CUDA_SUCCESS)
    return err;
  if (cudaDeviceCount <= 0)
    return CUDA_ERROR_NO_DEVICE;
  if (cudaDeviceCount > static_cast<int>(gsan::kMaxGPUs))
    return CUDA_ERROR_NOT_SUPPORTED;

  config.numGPUs = cudaDeviceCount;
  for (int cudaDevice = 0; cudaDevice < cudaDeviceCount; ++cudaDevice) {
    config.deviceRanks[cudaDevice] = cudaDevice;
    config.configuredDeviceRanks[cudaDevice] = true;
  }
  config.topologyConfigured = true;
  return CUDA_SUCCESS;
}

size_t cdiv(size_t num, size_t den) { return (num + (den - 1)) / den; }

size_t roundUp(size_t val, size_t alignment) {
  return cdiv(val, alignment) * alignment;
}

size_t roundDownToPowerOfTwo(size_t x) {
  if (x == 0)
    return 0;

  for (size_t shift = 1; shift < sizeof(x) * 8; shift <<= 1)
    x |= x >> shift;

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

  config.topologyFrozen = true;
  CUresult err = ensureTopologyConfigured();
  if (err != CUDA_SUCCESS)
    return err;

  int deviceRank = getDeviceRankForCudaDevice(device);
  if (device < 0)
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

  config.numSMs = numSMs;
  config.numThreads = config.numGPUs * config.numSMs;
  if (config.numThreads > gsan::kMaxThreads)
    return CUDA_ERROR_NOT_SUPPORTED;

  // Seed rng for stochastic read clocks.
  if (!config.rngSeedConfigured) {
    auto userSeed = getenv("TRITON_GSAN_SEED");
    if (userSeed) {
      const char *userSeedEnd = userSeed + strlen(userSeed);
      auto res = std::from_chars(userSeed, userSeedEnd, config.rngSeed);
      if (res.ec != std::errc() || res.ptr != userSeedEnd) {
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
    config.rngSeedConfigured = true;
  }

  if (!config.clockBufferSizeConfigured) {
    auto userClockSize = getenv("TRITON_GSAN_CLOCK_BUFFER_SIZE");
    if (userClockSize) {
      int clockBufferSize = 0;
      const char *userClockSizeEnd = userClockSize + strlen(userClockSize);
      auto res =
          std::from_chars(userClockSize, userClockSizeEnd, clockBufferSize);
      if (res.ec != std::errc() || res.ptr != userClockSizeEnd ||
          clockBufferSize <= 0) {
        auto errc = make_error_code(res.ec);
        auto msg = errc.message();
        fprintf(stderr, "Invalid TRITON_GSAN_CLOCK_BUFFER_SIZE value: %s",
                msg.c_str());
        return CUDA_ERROR_INVALID_VALUE;
      }
      config.clockBufferSize = clockBufferSize;
    } else {
      config.clockBufferSize = 1024;
    }
    config.clockBufferSizeConfigured = true;
  }
  if (!config.shareableHandleTypeConfigured) {
    config.shareableHandleType = getRequestedShareableHandleType();
    config.shareableHandleTypeConfigured = true;
  }
  return CUDA_SUCCESS;
}

CUresult ensureRuntimeStateMapped(int device) {
  if (alloc == nullptr)
    return CUDA_ERROR_NOT_INITIALIZED;
  CUresult err = refreshConfigForDevice(device);
  if (err != CUDA_SUCCESS)
    return err;
  int deviceRank = getDeviceRankForCudaDevice(device);
  if (alloc->perDeviceHandles[deviceRank] != 0)
    return CUDA_SUCCESS;

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.requestedHandleTypes = getRequestedShareableHandleType();

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
      alloc->globalStateAddress + deviceRank * gsan::kPerDeviceStateStride;

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

  alloc->perDeviceHandles[deviceRank] = allocHandle;
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
                        CUmemGenericAllocationHandle shadowHandle,
                        size_t allocSize, int device, bool *realMapped,
                        bool *shadowMapped) {
  assert(node != nullptr);
  assert(realMapped != nullptr);
  assert(shadowMapped != nullptr);

  const auto shadowAddress = gsan::getShadowAddress(node->virtualAddress);
  const auto shadowSize = getShadowSize(allocSize);

  CUresult err = cuMemMap(node->virtualAddress, allocSize, /*offset*/ 0,
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

  err = cuMemSetAccess(node->virtualAddress, allocSize, &accessDesc, 1);
  if (err != CUDA_SUCCESS)
    return err;

  err = cuMemSetAccess(shadowAddress, shadowSize, &accessDesc, 1);
  if (err != CUDA_SUCCESS)
    return err;

  node->allocSize = allocSize;
  node->realHandle = realHandle;
  node->shadowHandle = shadowHandle;
  return CUDA_SUCCESS;
}

void unmapNodeHandles(AllocNode *node, bool realMapped, bool shadowMapped) {
  assert(node != nullptr);
  const auto shadowAddress = gsan::getShadowAddress(node->virtualAddress);
  const auto shadowSize = getShadowSize(node->allocSize);
  if (shadowMapped)
    cuMemUnmap(shadowAddress, shadowSize);
  if (realMapped)
    cuMemUnmap(node->virtualAddress, node->allocSize);
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
  prop.requestedHandleTypes = getRequestedShareableHandleType();

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

  err = mapNodeHandles(node, realHandle, shadowHandle, allocSize, device,
                       &realMapped, &shadowMapped);
  if (err != CUDA_SUCCESS)
    goto error;

  // Zero-initialize shadow memory
  err = cuMemsetD8Async(shadowAddress, 0, shadowSize, cuStream);
  if (err != CUDA_SUCCESS)
    goto error;

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

int gsanExportAllocationHandles(void *void_ptr,
                                GSanShareableHandle *realShareableHandle,
                                GSanShareableHandle *shadowShareableHandle,
                                size_t *allocSize,
                                CUmemAllocationHandleType handleType) {
  if (realShareableHandle == nullptr || shadowShareableHandle == nullptr ||
      allocSize == nullptr || !isSupportedShareableHandleType(handleType)) {
    return -1;
  }
  *realShareableHandle = {};
  *shadowShareableHandle = {};
  *allocSize = 0;

  const auto ptr = reinterpret_cast<CUdeviceptr>(void_ptr);
  if (ptr == 0)
    return -1;

  std::lock_guard lg(mut);
  if (alloc == nullptr)
    return -1;

  AllocNode *node = findNodeByAddress(&alloc->treeRoot, ptr);
  if (node == nullptr || node->maxFreeBlockSize != 0 ||
      ptr < node->virtualAddress ||
      ptr >= node->virtualAddress + node->allocSize) {
    fprintf(stderr,
            "gsanExportAllocationHandles called with invalid pointer\n");
    return -1;
  }

  CUresult err = cuMemExportToShareableHandle(
      getShareableHandleExportArg(realShareableHandle, handleType),
      node->realHandle, handleType, 0);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return -1;
  }

  err = cuMemExportToShareableHandle(
      getShareableHandleExportArg(shadowShareableHandle, handleType),
      node->shadowHandle, handleType, 0);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
      close(realShareableHandle->fd);
    return -1;
  }

  *allocSize = node->allocSize;
  return 0;
}

int gsanExportAllocationMemhandleRegions(void *void_ptr, uintptr_t *realPtr,
                                         size_t *realSize, uintptr_t *shadowPtr,
                                         size_t *shadowSize) {
  if (realPtr == nullptr || realSize == nullptr || shadowPtr == nullptr ||
      shadowSize == nullptr) {
    return -1;
  }
  *realPtr = 0;
  *realSize = 0;
  *shadowPtr = 0;
  *shadowSize = 0;

  const auto ptr = reinterpret_cast<CUdeviceptr>(void_ptr);
  if (ptr == 0)
    return -1;

  std::lock_guard lg(mut);
  if (alloc == nullptr)
    return -1;

  AllocNode *node = findNodeByAddress(&alloc->treeRoot, ptr);
  if (node == nullptr || node->maxFreeBlockSize != 0 ||
      ptr < node->virtualAddress ||
      ptr >= node->virtualAddress + node->allocSize) {
    fprintf(
        stderr,
        "gsanExportAllocationMemhandleRegions called with invalid pointer\n");
    return -1;
  }

  *realPtr = static_cast<uintptr_t>(node->virtualAddress);
  *realSize = node->allocSize;
  *shadowPtr =
      static_cast<uintptr_t>(gsan::getShadowAddress(node->virtualAddress));
  *shadowSize = getShadowSize(node->allocSize);
  return 0;
}

int gsanExportRuntimeStateHandle(int device,
                                 GSanShareableHandle *shareableHandle,
                                 size_t *allocSize,
                                 CUmemAllocationHandleType handleType) {
  if (shareableHandle == nullptr || allocSize == nullptr ||
      !isSupportedShareableHandleType(handleType)) {
    return -1;
  }
  *shareableHandle = {};
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

  int deviceRank = getDeviceRankForCudaDevice(device);
  auto handle = alloc->perDeviceHandles[deviceRank];
  auto size = alloc->perDeviceStateSize;
  if (handle == 0 || size == 0)
    return -1;

  err = cuMemExportToShareableHandle(
      getShareableHandleExportArg(shareableHandle, handleType), handle,
      handleType, 0);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return -1;
  }

  *allocSize = size;
  return 0;
}

void *
gsanImportAllocationHandles(const GSanShareableHandle *realShareableHandle,
                            const GSanShareableHandle *shadowShareableHandle,
                            CUmemAllocationHandleType handleType,
                            size_t allocSize, int device) {
  if (realShareableHandle == nullptr || shadowShareableHandle == nullptr ||
      !isSupportedShareableHandleType(handleType) || allocSize == 0)
    return nullptr;
  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
      (realShareableHandle->fd < 0 || shadowShareableHandle->fd < 0)) {
    return nullptr;
  }

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

  CUmemGenericAllocationHandle realHandle = 0;
  CUmemGenericAllocationHandle shadowHandle = 0;
  bool realMapped = false;
  bool shadowMapped = false;
  err = cuMemImportFromShareableHandle(
      &realHandle, getShareableHandleImportArg(realShareableHandle, handleType),
      handleType);
  if (err != CUDA_SUCCESS)
    goto error;

  err = cuMemImportFromShareableHandle(
      &shadowHandle,
      getShareableHandleImportArg(shadowShareableHandle, handleType),
      handleType);
  if (err != CUDA_SUCCESS)
    goto error;

  err = mapNodeHandles(node, realHandle, shadowHandle, allocSize, device,
                       &realMapped, &shadowMapped);
  if (err != CUDA_SUCCESS)
    goto error;

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

int gsanImportRuntimeStateHandle(const GSanShareableHandle *shareableHandle,
                                 CUmemAllocationHandleType handleType,
                                 size_t allocSize, int peerDevice, int device) {
  if (shareableHandle == nullptr ||
      !isSupportedShareableHandleType(handleType) || allocSize == 0)
    return -1;
  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
      shareableHandle->fd < 0) {
    return -1;
  }
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

  if (peerDevice < 0 || peerDevice >= config.numGPUs)
    return -1;
  if (allocSize != alloc->perDeviceStateSize)
    return -1;

  CUmemGenericAllocationHandle importedHandle = 0;
  bool mapped = false;
  CUmemAccessDesc accessDesc = {};
  CUdeviceptr deviceAddr =
      alloc->globalStateAddress + peerDevice * gsan::kPerDeviceStateStride;

  err = cuMemImportFromShareableHandle(
      &importedHandle, getShareableHandleImportArg(shareableHandle, handleType),
      handleType);
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

bool parseShareableHandleTypeArg(PyObject *obj, const char *name,
                                 CUmemAllocationHandleType *out) {
  int handleType = 0;
  if (!parseIntArg(obj, name, &handleType))
    return false;
  if (handleType != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
      handleType != CU_MEM_HANDLE_TYPE_FABRIC) {
    PyErr_Format(PyExc_ValueError, "%s has unsupported value %d", name,
                 handleType);
    return false;
  }
  *out = static_cast<CUmemAllocationHandleType>(handleType);
  return true;
}

bool parseShareableHandleArg(PyObject *obj, const char *name,
                             CUmemAllocationHandleType handleType,
                             GSanShareableHandle *out) {
  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    return parseIntArg(obj, name, &out->fd);

  if (!PyBytes_Check(obj)) {
    PyErr_Format(PyExc_TypeError, "%s must be bytes", name);
    return false;
  }

  char *data = nullptr;
  Py_ssize_t size = 0;
  if (PyBytes_AsStringAndSize(obj, &data, &size) != 0)
    return false;
  if (size != static_cast<Py_ssize_t>(sizeof(out->fabricHandle))) {
    PyErr_Format(PyExc_ValueError, "%s must contain exactly %zu bytes, got %zd",
                 name, sizeof(out->fabricHandle), size);
    return false;
  }

  memcpy(&out->fabricHandle, data, sizeof(out->fabricHandle));
  return true;
}

PyObject *shareableHandleToPyObject(const GSanShareableHandle &handle,
                                    CUmemAllocationHandleType handleType) {
  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    return PyLong_FromLong(handle.fd);
  return PyBytes_FromStringAndSize(
      reinterpret_cast<const char *>(&handle.fabricHandle),
      sizeof(handle.fabricHandle));
}

bool parseUInt32Arg(PyObject *obj, const char *name, uint32_t *out) {
  unsigned long long value = PyLong_AsUnsignedLongLong(obj);
  if (value == std::numeric_limits<unsigned long long>::max() &&
      PyErr_Occurred())
    return false;
  if (value > std::numeric_limits<uint32_t>::max()) {
    PyErr_Format(PyExc_OverflowError, "%s is out of range for uint32", name);
    return false;
  }
  *out = static_cast<uint32_t>(value);
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

PyObject *pyConfigure([[maybe_unused]] PyObject *self, PyObject *const *args,
                      Py_ssize_t nargs) {
  if (nargs != 5) {
    PyErr_Format(PyExc_TypeError,
                 "%s.configure expected 5 positional arguments, got %zd",
                 kModuleName, nargs);
    return nullptr;
  }

  PyObject *deviceRankMap = args[0];
  PyObject *numDevicesArg = args[1];
  const bool topologyRequested =
      deviceRankMap != Py_None || numDevicesArg != Py_None;
  if ((deviceRankMap == Py_None) != (numDevicesArg == Py_None)) {
    PyErr_SetString(PyExc_ValueError,
                    "device_ranks and num_devices must be configured "
                    "together");
    return nullptr;
  }

  int requestedNumDevices = 0;
  int requestedDeviceRanks[gsan::kMaxGPUs] = {};
  bool requestedConfiguredDeviceRanks[gsan::kMaxGPUs] = {};
  if (topologyRequested) {
    if (!PyDict_Check(deviceRankMap)) {
      PyErr_SetString(PyExc_TypeError, "device_ranks must be a dict[int, int]");
      return nullptr;
    }
    if (!parseIntArg(numDevicesArg, "num_devices", &requestedNumDevices))
      return nullptr;
    if (requestedNumDevices <= 0 ||
        requestedNumDevices > static_cast<int>(gsan::kMaxGPUs)) {
      PyErr_Format(PyExc_ValueError, "num_devices must be in [1, %zu], got %d",
                   static_cast<size_t>(gsan::kMaxGPUs), requestedNumDevices);
      return nullptr;
    }
    if (PyDict_Size(deviceRankMap) <= 0) {
      PyErr_SetString(PyExc_ValueError, "device_ranks must not be empty");
      return nullptr;
    }

    bool requestedGlobalDeviceIds[gsan::kMaxGPUs] = {};
    Py_ssize_t pos = 0;
    PyObject *key = nullptr;
    PyObject *value = nullptr;
    while (PyDict_Next(deviceRankMap, &pos, &key, &value)) {
      int cudaDevice = 0;
      int globalDeviceId = 0;
      if (!parseIntArg(key, "cuda_device", &cudaDevice) ||
          !parseIntArg(value, "global_device_id", &globalDeviceId)) {
        return nullptr;
      }
      if (cudaDevice < 0 || cudaDevice >= static_cast<int>(gsan::kMaxGPUs)) {
        PyErr_Format(PyExc_ValueError,
                     "cuda_device must be in [0, %zu), got %d",
                     static_cast<size_t>(gsan::kMaxGPUs), cudaDevice);
        return nullptr;
      }
      if (globalDeviceId < 0 || globalDeviceId >= requestedNumDevices) {
        PyErr_Format(PyExc_ValueError,
                     "global_device_id must be in [0, %d), got %d",
                     requestedNumDevices, globalDeviceId);
        return nullptr;
      }
      if (requestedGlobalDeviceIds[globalDeviceId]) {
        PyErr_Format(PyExc_ValueError,
                     "global_device_id %d is assigned to more than one CUDA "
                     "device",
                     globalDeviceId);
        return nullptr;
      }
      requestedDeviceRanks[cudaDevice] = globalDeviceId;
      requestedConfiguredDeviceRanks[cudaDevice] = true;
      requestedGlobalDeviceIds[globalDeviceId] = true;
    }
  }

  const bool rngSeedRequested = args[2] != Py_None;
  uint32_t requestedRngSeed = 0;
  if (rngSeedRequested &&
      !parseUInt32Arg(args[2], "rng_seed", &requestedRngSeed))
    return nullptr;

  const bool clockBufferSizeRequested = args[3] != Py_None;
  int requestedClockBufferSize = 0;
  if (clockBufferSizeRequested) {
    if (!parseIntArg(args[3], "clock_buffer_size", &requestedClockBufferSize))
      return nullptr;
    if (requestedClockBufferSize <= 0) {
      PyErr_Format(PyExc_ValueError,
                   "clock_buffer_size must be positive, got %d",
                   requestedClockBufferSize);
      return nullptr;
    }
  }

  const bool shareableHandleTypeRequested = args[4] != Py_None;
  CUmemAllocationHandleType requestedShareableHandleType =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  if (shareableHandleTypeRequested &&
      !parseShareableHandleTypeArg(args[4], "handle_type",
                                   &requestedShareableHandleType)) {
    return nullptr;
  }

  std::lock_guard lg(mut);
  if (config.topologyFrozen) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GSan allocator configuration is already frozen and "
                    "cannot be changed");
    return nullptr;
  }

  if (topologyRequested) {
    config.numGPUs = requestedNumDevices;
    for (size_t i = 0; i < gsan::kMaxGPUs; ++i) {
      config.deviceRanks[i] = requestedDeviceRanks[i];
      config.configuredDeviceRanks[i] = requestedConfiguredDeviceRanks[i];
    }
    config.topologyConfigured = true;
  }
  if (rngSeedRequested) {
    config.rngSeed = requestedRngSeed;
    config.rngSeedConfigured = true;
  }
  if (clockBufferSizeRequested) {
    config.clockBufferSize = requestedClockBufferSize;
    config.clockBufferSizeConfigured = true;
  }
  if (shareableHandleTypeRequested) {
    config.shareableHandleType = requestedShareableHandleType;
    config.shareableHandleTypeConfigured = true;
  }
  Py_RETURN_NONE;
}

PyObject *pyFreezeConfig([[maybe_unused]] PyObject *self, PyObject *const *args,
                         Py_ssize_t nargs) {
  if (nargs != 0) {
    PyErr_Format(PyExc_TypeError,
                 "%s.freeze_config expected 0 positional arguments, got %zd",
                 kModuleName, nargs);
    return nullptr;
  }

  std::lock_guard lg(mut);
  CUresult err = ensureTopologyConfigured();
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    PyErr_SetString(PyExc_RuntimeError,
                    "failed to configure the default GSan topology");
    return nullptr;
  }
  config.topologyFrozen = true;
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

PyObject *pyGetDeviceRank([[maybe_unused]] PyObject *self,
                          PyObject *const *args, Py_ssize_t nargs) {
  if (nargs != 1) {
    PyErr_Format(PyExc_TypeError,
                 "%s.get_device_rank expected 1 positional argument, got %zd",
                 kModuleName, nargs);
    return nullptr;
  }

  int device = 0;
  if (!parseIntArg(args[0], "device", &device))
    return nullptr;

  std::lock_guard lg(mut);
  CUresult err = ensureTopologyConfigured();
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    PyErr_SetString(PyExc_RuntimeError,
                    "failed to configure the default GSan topology");
    return nullptr;
  }

  int deviceRank = getDeviceRankForCudaDevice(device);
  if (deviceRank < 0) {
    PyErr_Format(PyExc_ValueError,
                 "no GSan device rank configured for CUDA device %d", device);
    return nullptr;
  }
  return PyLong_FromLong(deviceRank);
}

PyObject *pyGetRuntimeStateLayout([[maybe_unused]] PyObject *self,
                                  PyObject *const *args, Py_ssize_t nargs) {
  if (nargs != 1) {
    PyErr_Format(PyExc_TypeError,
                 "%s.get_runtime_state_layout expected 1 positional argument, "
                 "got %zd",
                 kModuleName, nargs);
    return nullptr;
  }

  int device = 0;
  if (!parseIntArg(args[0], "device", &device))
    return nullptr;

  std::lock_guard lg(mut);
  if (gsanEnsureInit() != 0) {
    PyErr_SetString(PyExc_RuntimeError, "failed to initialize gsan allocator");
    return nullptr;
  }
  if (device < 0 || device >= config.numGPUs) {
    PyErr_Format(PyExc_ValueError, "device must be in [0, %d), got %d",
                 config.numGPUs, device);
    return nullptr;
  }
  if (alloc->perDeviceHandles[device] == 0) {
    PyErr_Format(PyExc_RuntimeError,
                 "GSan runtime state for device %d has not been mapped",
                 device);
    return nullptr;
  }

  uintptr_t globalStateAddress =
      alloc->globalStateAddress + device * gsan::kPerDeviceStateStride;
  uintptr_t threadStateBase =
      roundUp(globalStateAddress + sizeof(gsan::GlobalState),
              alignof(gsan::ThreadState));
  size_t threadStateStride =
      sizeof(gsan::ThreadState) +
      sizeof(gsan::epoch_t) * config.numThreads * (1 + config.clockBufferSize);
  threadStateStride = roundUp(threadStateStride, alignof(gsan::ThreadState));

  return Py_BuildValue(
      "{s:K,s:K,s:K,s:K,s:i,s:i,s:i}", "global_state_ptr",
      static_cast<unsigned long long>(globalStateAddress),
      "thread_state_base_ptr", static_cast<unsigned long long>(threadStateBase),
      "thread_state_stride_bytes",
      static_cast<unsigned long long>(threadStateStride),
      "thread_state_header_size_bytes",
      static_cast<unsigned long long>(kThreadStateHeaderSize), "num_sms",
      config.numSMs, "num_threads", config.numThreads, "clock_buffer_size",
      config.clockBufferSize);
}

PyObject *pyExportAllocationHandles(PyObject *self, PyObject *const *args,
                                    Py_ssize_t nargs) {
  (void)self;
  if (nargs != 2) {
    PyErr_Format(
        PyExc_TypeError,
        "%s.export_allocation_handles expected 2 positional arguments, got %zd",
        kModuleName, nargs);
    return nullptr;
  }

  void *ptr = nullptr;
  CUmemAllocationHandleType handleType =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  if (!parseVoidPtrArg(args[0], &ptr) ||
      !parseShareableHandleTypeArg(args[1], "handle_type", &handleType))
    return nullptr;

  GSanShareableHandle realShareableHandle = {};
  GSanShareableHandle shadowShareableHandle = {};
  size_t allocSize = 0;
  int rc = gsanExportAllocationHandles(ptr, &realShareableHandle,
                                       &shadowShareableHandle, &allocSize,
                                       handleType);
  if (rc != 0) {
    PyErr_SetString(PyExc_RuntimeError, "gsanExportAllocationHandles failed.");
    return nullptr;
  }

  return Py_BuildValue(
      "(NNK)", shareableHandleToPyObject(realShareableHandle, handleType),
      shareableHandleToPyObject(shadowShareableHandle, handleType),
      static_cast<unsigned long long>(allocSize));
}

PyObject *pyExportAllocationMemhandleRegions(PyObject *self,
                                             PyObject *const *args,
                                             Py_ssize_t nargs) {
  (void)self;
  if (nargs != 1) {
    PyErr_Format(PyExc_TypeError,
                 "%s.export_allocation_memhandle_regions expected 1 positional "
                 "argument, got %zd",
                 kModuleName, nargs);
    return nullptr;
  }

  void *ptr = nullptr;
  if (!parseVoidPtrArg(args[0], &ptr))
    return nullptr;

  uintptr_t realPtr = 0;
  uintptr_t shadowPtr = 0;
  size_t realSize = 0;
  size_t shadowSize = 0;
  int rc = gsanExportAllocationMemhandleRegions(ptr, &realPtr, &realSize,
                                                &shadowPtr, &shadowSize);
  if (rc != 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "gsanExportAllocationMemhandleRegions failed.");
    return nullptr;
  }

  return Py_BuildValue("(KKKK)", static_cast<unsigned long long>(realPtr),
                       static_cast<unsigned long long>(realSize),
                       static_cast<unsigned long long>(shadowPtr),
                       static_cast<unsigned long long>(shadowSize));
}

PyObject *pyImportAllocationHandles(PyObject *self, PyObject *const *args,
                                    Py_ssize_t nargs) {
  (void)self;
  if (nargs != 5) {
    PyErr_Format(
        PyExc_TypeError,
        "%s.import_allocation_handles expected 5 positional arguments, got %zd",
        kModuleName, nargs);
    return nullptr;
  }

  GSanShareableHandle realShareableHandle = {};
  GSanShareableHandle shadowShareableHandle = {};
  int device = 0;
  CUmemAllocationHandleType handleType =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  if (!parseIntArg(args[3], "device", &device) ||
      !parseShareableHandleTypeArg(args[4], "handle_type", &handleType) ||
      !parseShareableHandleArg(args[0], "real_handle", handleType,
                               &realShareableHandle) ||
      !parseShareableHandleArg(args[1], "shadow_handle", handleType,
                               &shadowShareableHandle)) {
    return nullptr;
  }

  size_t allocSize = PyLong_AsSize_t(args[2]);
  if (allocSize == static_cast<size_t>(-1) && PyErr_Occurred())
    return nullptr;

  void *ptr =
      gsanImportAllocationHandles(&realShareableHandle, &shadowShareableHandle,
                                  handleType, allocSize, device);
  if (ptr == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "gsanImportAllocationHandles failed.");
    return nullptr;
  }

  return PyLong_FromVoidPtr(ptr);
}

PyObject *pyExportRuntimeStateHandle(PyObject *self, PyObject *const *args,
                                     Py_ssize_t nargs) {
  (void)self;
  if (nargs != 2) {
    PyErr_Format(PyExc_TypeError,
                 "%s.export_runtime_state_handle expected 2 positional "
                 "arguments, got %zd",
                 kModuleName, nargs);
    return nullptr;
  }

  int device = 0;
  CUmemAllocationHandleType handleType =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  if (!parseIntArg(args[0], "device", &device) ||
      !parseShareableHandleTypeArg(args[1], "handle_type", &handleType))
    return nullptr;

  GSanShareableHandle shareableHandle = {};
  size_t allocSize = 0;
  int rc = gsanExportRuntimeStateHandle(device, &shareableHandle, &allocSize,
                                        handleType);
  if (rc != 0) {
    PyErr_SetString(PyExc_RuntimeError, "gsanExportRuntimeStateHandle failed.");
    return nullptr;
  }

  return Py_BuildValue("(NK)",
                       shareableHandleToPyObject(shareableHandle, handleType),
                       static_cast<unsigned long long>(allocSize));
}

PyObject *pyImportRuntimeStateHandle(PyObject *self, PyObject *const *args,
                                     Py_ssize_t nargs) {
  (void)self;
  if (nargs != 5) {
    PyErr_Format(PyExc_TypeError,
                 "%s.import_runtime_state_handle expected 5 positional "
                 "arguments, got %zd",
                 kModuleName, nargs);
    return nullptr;
  }

  GSanShareableHandle shareableHandle = {};
  int peerDevice = 0;
  int device = 0;
  CUmemAllocationHandleType handleType =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  if (!parseIntArg(args[2], "peer_device", &peerDevice) ||
      !parseIntArg(args[3], "device", &device) ||
      !parseShareableHandleTypeArg(args[4], "handle_type", &handleType) ||
      !parseShareableHandleArg(args[0], "handle", handleType,
                               &shareableHandle)) {
    return nullptr;
  }

  size_t allocSize = PyLong_AsSize_t(args[1]);
  if (allocSize == static_cast<size_t>(-1) && PyErr_Occurred())
    return nullptr;

  int rc = gsanImportRuntimeStateHandle(&shareableHandle, handleType, allocSize,
                                        peerDevice, device);
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
    {"configure", reinterpret_cast<PyCFunction>(pyConfigure), METH_FASTCALL,
     "Configure GSan topology and runtime tuning fields."},
    {"freeze_config", reinterpret_cast<PyCFunction>(pyFreezeConfig),
     METH_FASTCALL, "Prevent later changes to the GSan allocator config."},
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
    {"get_device_rank", reinterpret_cast<PyCFunction>(pyGetDeviceRank),
     METH_FASTCALL, "Return the configured logical GSan device rank."},
    {"get_runtime_state_layout",
     reinterpret_cast<PyCFunction>(pyGetRuntimeStateLayout), METH_FASTCALL,
     "Return the per-device GSan runtime state layout."},
    {"export_allocation_handles",
     reinterpret_cast<PyCFunction>(pyExportAllocationHandles), METH_FASTCALL,
     "Export allocation handles for an existing allocation pointer."},
    {"export_allocation_memhandle_regions",
     reinterpret_cast<PyCFunction>(pyExportAllocationMemhandleRegions),
     METH_FASTCALL,
     "Export real and shadow allocation regions for an existing pointer."},
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
