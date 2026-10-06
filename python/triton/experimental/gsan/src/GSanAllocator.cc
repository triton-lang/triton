#include <Python.h>

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
#include <numeric>
#include <random>

#include "GSan.h"
#include "GSanDriver.h"

namespace drv = gsan::drv;

// #define GSAN_LOG_ALLOCATIONS
#ifdef GSAN_LOG_ALLOCATIONS
#define LOGF(...) printf(__VA_ARGS__);
#else
#define LOGF(...)
#endif

extern "C" {
void *gsanMalloc(ssize_t size, int device, void *stream);
void *gsanMallocWriteOnce(ssize_t size, int device, void *stream);
void gsanFree(void *ptr, ssize_t size, int device, void *stream);
}

namespace {
constexpr size_t kThreadStateHeaderSize =
    offsetof(gsan::ThreadState, vectorClock);

union GSanShareableHandle {
  int fd;
  drv::FabricHandle fabricHandle;
};

void *getShareableHandleImportArg(const GSanShareableHandle *handle,
                                  drv::HandleType handleType) {
  if (handleType == drv::HandleType::PosixFileDescriptor)
    return reinterpret_cast<void *>(static_cast<uintptr_t>(handle->fd));
  if (handleType == drv::HandleType::Fabric)
    return const_cast<drv::FabricHandle *>(&handle->fabricHandle);
  return nullptr;
}

void *getShareableHandleExportArg(GSanShareableHandle *handle,
                                  drv::HandleType handleType) {
  if (handleType == drv::HandleType::PosixFileDescriptor)
    return &handle->fd;
  if (handleType == drv::HandleType::Fabric)
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
  uintptr_t virtualAddress = 0;
  AllocNode *parent = nullptr;
  std::unique_ptr<AllocNode> leftChild;
  std::unique_ptr<AllocNode> rightChild;
  size_t size = 0;
  size_t maxFreeBlockSize = 0;

  // Allocation handles, used only by leaf nodes
  drv::Handle realHandle = 0;
  drv::Handle shadowHandle = 0;
  size_t allocSize = 0;
};

struct GSanConfig {
  int numGPUs = 1;
  int numSMs = 0;
  int numThreads = 0;
  int clockBufferSize = 0;
  uint32_t rngSeed = 0;
  drv::HandleType shareableHandleType = drv::HandleType::PosixFileDescriptor;
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
  uintptr_t reserveBaseAddress = 0;
  AllocNode treeRoots[gsan::kNumPools];

  // GSan global state
  uintptr_t globalStateAddress = 0;
  drv::Handle perDeviceHandles[gsan::kMaxGPUs] = {0};
  size_t perDeviceStateSize = 0;
};

void printDriverError(drv::Result err) {
  fprintf(stderr, "gsan allocator encountered an unexpected error: %s\n",
          drv::errorString(err));
}

static AllocatorState *alloc = nullptr;
static GSanConfig config;
static std::mutex mut;

bool hasLiveAllocations() {
  if (alloc == nullptr)
    return false;
  for (const auto &root : alloc->treeRoots)
    if (root.maxFreeBlockSize != root.size)
      return true;
  return false;
}

drv::HandleType getRequestedShareableHandleType() {
  if (config.shareableHandleTypeConfigured)
    return config.shareableHandleType;

  const auto *allocConf = getenv("PYTORCH_CUDA_ALLOC_CONF");
  if (drv::kHasFabricHandles && allocConf != nullptr &&
      strstr(allocConf, "fabric_handles:True") != nullptr) {
    return drv::HandleType::Fabric;
  }
  return drv::HandleType::PosixFileDescriptor;
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

drv::Result ensureTopologyConfigured() {
  if (config.topologyConfigured)
    return drv::kSuccess;

  // Default topology assumes a single node with 1:1 mapping of device index to
  // GSan device ID.
  int cudaDeviceCount = 0;
  drv::Result err = drv::deviceCount(&cudaDeviceCount);
  if (err != drv::kSuccess)
    return err;
  if (cudaDeviceCount <= 0)
    return drv::kErrorNoDevice;
  if (cudaDeviceCount > static_cast<int>(gsan::kMaxGPUs))
    return drv::kErrorNotSupported;

  config.numGPUs = cudaDeviceCount;
  for (int cudaDevice = 0; cudaDevice < cudaDeviceCount; ++cudaDevice) {
    config.deviceRanks[cudaDevice] = cudaDevice;
    config.configuredDeviceRanks[cudaDevice] = true;
  }
  config.topologyConfigured = true;
  return drv::kSuccess;
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

size_t getShadowSize(size_t realMemSize, uintptr_t realAddress) {
  auto wordSize = cdiv(realMemSize, gsan::getShadowGranularity(realAddress));
  return wordSize * gsan::getShadowCellSize(realAddress);
}

// GlobalState stores the clock buffer size as a u16.
constexpr int kMaxClockBufferSize = std::numeric_limits<uint16_t>::max();
constexpr int kDefaultClockBufferSize = 1024;

size_t getPerSMStateSize(int numThreads, int clockBufferSize) {
  auto clockSizeBytes = sizeof(gsan::epoch_t) * numThreads;
  // 1 local clock + the circular clock buffer
  auto clocksPerThread = 1 + static_cast<size_t>(clockBufferSize);
  return roundUp(sizeof(gsan::ThreadState) + clockSizeBytes * clocksPerThread,
                 alignof(gsan::ThreadState));
}

// Each device has a local copy of the constant global state, followed by one
// thread state per SM.
size_t getPerDeviceStateSize(int numSMs, int numThreads, int clockBufferSize) {
  static_assert(alignof(gsan::GlobalState) >= alignof(gsan::ThreadState));
  static_assert(alignof(gsan::ThreadState) >= alignof(gsan::epoch_t));
  return sizeof(gsan::GlobalState) +
         numSMs * getPerSMStateSize(numThreads, clockBufferSize);
}

// Largest clock buffer whose per-device state fits in its fixed-stride slot of
// the globals reservation, or 0 if not even one entry fits.
int getMaxClockBufferSize(int numSMs, int numThreads) {
  size_t perSMBudget =
      (gsan::kPerDeviceStateStride - sizeof(gsan::GlobalState)) / numSMs;
  perSMBudget -= perSMBudget % alignof(gsan::ThreadState);
  if (perSMBudget < sizeof(gsan::ThreadState))
    return 0;
  size_t clocksPerThread = (perSMBudget - sizeof(gsan::ThreadState)) /
                           (sizeof(gsan::epoch_t) * numThreads);
  if (clocksPerThread < 2)
    return 0;
  return static_cast<int>(
      std::min<size_t>(clocksPerThread - 1, kMaxClockBufferSize));
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

AllocNode *findNodeByAddress(AllocNode *root, uintptr_t address) {
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

AllocNode *findAllocation(uintptr_t address) {
  if (!gsan::isGsanManaged(address, alloc->reserveBaseAddress))
    return nullptr;
  return findNodeByAddress(&alloc->treeRoots[gsan::getPoolIndex(address)],
                           address);
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

  uintptr_t reserveBase;
  drv::Result err =
      drv::addressReserve(&reserveBase, /*size*/ gsan::kReserveSize,
                          /*alignment*/ gsan::kReserveSize);
  if (err != drv::kSuccess) {
    printDriverError(err);
    return -1;
  }

  uintptr_t globalsBase;
  err = drv::addressReserve(&globalsBase, /*size*/ gsan::kGlobalsReserveSize,
                            /*alignment*/ gsan::kGlobalsReserveSize);
  if (err != drv::kSuccess) {
    printDriverError(err);
    return -1;
  }
  alloc = new AllocatorState();
  alloc->reserveBaseAddress = reserveBase;
  alloc->globalStateAddress = globalsBase;

  for (int i = 0; i < gsan::kNumPools; ++i) {
    bool writeOnce = (i & 1) != 0;
    int granularity = 1 << (i >> 1);
    auto *root = &alloc->treeRoots[i];
    root->virtualAddress =
        gsan::getRealBaseAddress(reserveBase, granularity, writeOnce);

    // Both physical mappings must fit in their respective half of the pool.
    auto shadowSize = gsan::kPoolReserveSize / 2;
    auto realSize =
        granularity * (shadowSize / gsan::getShadowCellSize(writeOnce));
    realSize = roundDownToPowerOfTwo(std::min(shadowSize, realSize));
    root->size = realSize;
    root->maxFreeBlockSize = realSize;
  }
  return 0;
}

drv::Result refreshConfigForDevice(int device) {
  if (alloc == nullptr)
    return drv::kErrorNotInitialized;

  config.topologyFrozen = true;
  drv::Result err = ensureTopologyConfigured();
  if (err != drv::kSuccess)
    return err;

  int deviceRank = getDeviceRankForCudaDevice(device);
  if (device < 0)
    return drv::kErrorInvalidDevice;

  int numSMs = 0;
  err = drv::multiprocessorCount(device, &numSMs);
  if (err != drv::kSuccess)
    return err;
  if (numSMs <= 0)
    return drv::kErrorInvalidValue;

  config.numSMs = numSMs;
  config.numThreads = config.numGPUs * config.numSMs;
  if (config.numThreads > gsan::kMaxThreads)
    return drv::kErrorNotSupported;
  const int maxClockBufferSize =
      getMaxClockBufferSize(config.numSMs, config.numThreads);
  if (maxClockBufferSize == 0) {
    fprintf(stderr,
            "GSan runtime state for %d threads on %d SMs does not fit in "
            "%zu MiB per device\n",
            config.numThreads, config.numSMs,
            static_cast<size_t>(gsan::kPerDeviceStateStride >> 20));
    return drv::kErrorNotSupported;
  }

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
        return drv::kErrorInvalidValue;
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
          clockBufferSize <= 0 || clockBufferSize > kMaxClockBufferSize) {
        fprintf(stderr,
                "Invalid TRITON_GSAN_CLOCK_BUFFER_SIZE value '%s': must be an "
                "integer in [1, %d]\n",
                userClockSize, kMaxClockBufferSize);
        return drv::kErrorInvalidValue;
      }
      config.clockBufferSize = clockBufferSize;
    } else {
      // Large topologies cannot afford the default for every SM, so shrink
      // it to what fits rather than failing out of the box.
      config.clockBufferSize =
          std::min(kDefaultClockBufferSize, maxClockBufferSize);
    }
    config.clockBufferSizeConfigured = true;
  }
  if (config.clockBufferSize > maxClockBufferSize) {
    fprintf(stderr,
            "GSan clock_buffer_size %d does not fit in %zu MiB of runtime "
            "state per device for %d threads on %d SMs; use at most %d\n",
            config.clockBufferSize,
            static_cast<size_t>(gsan::kPerDeviceStateStride >> 20),
            config.numThreads, config.numSMs, maxClockBufferSize);
    return drv::kErrorInvalidValue;
  }
  if (!config.shareableHandleTypeConfigured) {
    config.shareableHandleType = getRequestedShareableHandleType();
    config.shareableHandleTypeConfigured = true;
  }
  return drv::kSuccess;
}

drv::Result initializeRuntimeState(uintptr_t deviceAddr, size_t allocSize) {
  drv::Result err = drv::memsetD8(deviceAddr, allocSize);
  if (err != drv::kSuccess)
    return err;

  gsan::GlobalState globals = {};
  globals.reserveBase = static_cast<uintptr_t>(alloc->reserveBaseAddress);
  globals.globalsBase = static_cast<uintptr_t>(alloc->globalStateAddress);
  globals.rngSeed = config.rngSeed;
  globals.numSms = static_cast<gsan::thread_id_t>(config.numSMs);
  globals.numDevices = static_cast<gsan::thread_id_t>(config.numGPUs);
  globals.numThreads = static_cast<gsan::thread_id_t>(config.numThreads);
  globals.clockBufferSize = config.clockBufferSize;
  return drv::memcpyHtoD(deviceAddr, &globals, sizeof(globals));
}

drv::Result ensureRuntimeStateMapped(int device) {
  if (alloc == nullptr)
    return drv::kErrorNotInitialized;
  drv::Result err = refreshConfigForDevice(device);
  if (err != drv::kSuccess)
    return err;
  int deviceRank = getDeviceRankForCudaDevice(device);
  if (alloc->perDeviceHandles[deviceRank] != 0)
    return drv::kSuccess;

  const auto handleType = getRequestedShareableHandleType();
  size_t granularity = 0;
  err = drv::allocationGranularity(device, handleType, &granularity);
  if (err != drv::kSuccess)
    return err;

  // refreshConfigForDevice bounds the clock buffer so this fits.
  size_t allocSize =
      roundUp(getPerDeviceStateSize(config.numSMs, config.numThreads,
                                    config.clockBufferSize),
              granularity);
  if (allocSize > gsan::kPerDeviceStateStride)
    return drv::kErrorNotSupported;

  drv::Handle allocHandle = 0;
  bool mapped = false;
  uintptr_t deviceAddr =
      alloc->globalStateAddress + deviceRank * gsan::kPerDeviceStateStride;

  err = drv::memCreate(&allocHandle, allocSize, device, handleType);
  if (err != drv::kSuccess)
    goto error;

  err = drv::memMap(deviceAddr, allocSize, allocHandle);
  if (err != drv::kSuccess)
    goto error;
  mapped = true;

  err = drv::memSetAccess(deviceAddr, allocSize, device);
  if (err != drv::kSuccess)
    goto error;

  err = initializeRuntimeState(deviceAddr, allocSize);
  if (err != drv::kSuccess)
    goto error;

  alloc->perDeviceHandles[deviceRank] = allocHandle;
  alloc->perDeviceStateSize = allocSize;
  return drv::kSuccess;

error:
  if (mapped)
    (void)drv::memUnmap(deviceAddr, allocSize);
  if (allocHandle != 0)
    (void)drv::memRelease(allocHandle);
  return err;
}

drv::Result mapNodeHandles(AllocNode *node, drv::Handle realHandle,
                           drv::Handle shadowHandle, size_t allocSize,
                           int device, bool *realMapped, bool *shadowMapped) {
  assert(node != nullptr);
  assert(realMapped != nullptr);
  assert(shadowMapped != nullptr);

  const auto shadowAddress = gsan::getShadowAddress(node->virtualAddress);
  const auto shadowSize = getShadowSize(allocSize, node->virtualAddress);

  drv::Result err = drv::memMap(node->virtualAddress, allocSize, realHandle);
  if (err != drv::kSuccess)
    return err;
  *realMapped = true;

  err = drv::memMap(shadowAddress, shadowSize, shadowHandle);
  if (err != drv::kSuccess)
    return err;
  *shadowMapped = true;

  err = drv::memSetAccess(node->virtualAddress, allocSize, device);
  if (err != drv::kSuccess)
    return err;

  err = drv::memSetAccess(shadowAddress, shadowSize, device);
  if (err != drv::kSuccess)
    return err;

  node->allocSize = allocSize;
  node->realHandle = realHandle;
  node->shadowHandle = shadowHandle;
  return drv::kSuccess;
}

void unmapNodeHandles(AllocNode *node, bool realMapped, bool shadowMapped) {
  assert(node != nullptr);
  const auto shadowAddress = gsan::getShadowAddress(node->virtualAddress);
  const auto shadowSize = getShadowSize(node->allocSize, node->virtualAddress);
  if (shadowMapped)
    (void)drv::memUnmap(shadowAddress, shadowSize);
  if (realMapped)
    (void)drv::memUnmap(node->virtualAddress, node->allocSize);
}

} // namespace

// TODO: Handle streams?
void *gsanMallocWithGranularity(ssize_t size, int device, void *stream,
                                int shadowGranularity, bool writeOnce = false) {
  if (size <= 0 || !gsan::isValidShadowGranularity(shadowGranularity))
    return nullptr;

  std::lock_guard lg(mut);
  if (gsanEnsureInit() != 0)
    return nullptr;

  drv::Result err = drv::ensureCurrentDevice(device);
  if (err != drv::kSuccess) {
    printDriverError(err);
    return nullptr;
  }
  err = ensureRuntimeStateMapped(device);
  if (err != drv::kSuccess) {
    printDriverError(err);
    return nullptr;
  }

  const auto handleType = getRequestedShareableHandleType();
  size_t granularity = 0;
  err = drv::allocationGranularity(device, handleType, &granularity);
  if (err != drv::kSuccess) {
    printDriverError(err);
    return nullptr;
  }
  // Scale real allocation alignment so fractional shadow-to-real ratios
  // still produce page-aligned shadow addresses and sizes.
  size_t alignment =
      granularity * shadowGranularity /
      std::gcd(shadowGranularity, gsan::getShadowCellSize(writeOnce));
  size_t allocSize = roundUp(static_cast<size_t>(size), alignment);
  auto *root = &alloc->treeRoots[gsan::getPoolIndexForGranularity(
      shadowGranularity, writeOnce)];
  AllocNode *node = allocateNode(root, allocSize);
  if (node == nullptr)
    return nullptr;

  drv::Handle realHandle = 0;
  drv::Handle shadowHandle = 0;
  bool realMapped = false;
  bool shadowMapped = false;
  auto drvStream = reinterpret_cast<drv::Stream>(stream);
  auto shadowAddress = gsan::getShadowAddress(node->virtualAddress);
  auto shadowSize = getShadowSize(allocSize, node->virtualAddress);
  err = drv::memCreate(&realHandle, allocSize, device, handleType);
  if (err != drv::kSuccess)
    goto error;

  err = drv::memCreate(&shadowHandle, shadowSize, device, handleType);
  if (err != drv::kSuccess)
    goto error;

  err = mapNodeHandles(node, realHandle, shadowHandle, allocSize, device,
                       &realMapped, &shadowMapped);
  if (err != drv::kSuccess)
    goto error;

  // Zero-initialize shadow memory
  err = drv::memsetD8Async(shadowAddress, shadowSize, drvStream);
  if (err != drv::kSuccess)
    goto error;

  LOGF("gsanMalloc: %p, 0x%zxu", reinterpret_cast<void *>(node->virtualAddress),
       size);
  return reinterpret_cast<void *>(node->virtualAddress);

error:
  printDriverError(err);
  unmapNodeHandles(node, realMapped, shadowMapped);
  if (shadowHandle != 0)
    (void)drv::memRelease(shadowHandle);
  if (realHandle != 0)
    (void)drv::memRelease(realHandle);
  freeNode(node);
  return nullptr;
}

extern "C" void *gsanMalloc(ssize_t size, int device, void *stream) {
  return gsanMallocWithGranularity(size, device, stream, 4);
}

extern "C" void *gsanMallocWriteOnce(ssize_t size, int device, void *stream) {
  return gsanMallocWithGranularity(size, device, stream, 1, true);
}

extern "C" void *gsanMallocWriteOnce2(ssize_t size, int device, void *stream) {
  return gsanMallocWithGranularity(size, device, stream, 2, true);
}

extern "C" void *gsanMallocWriteOnce4(ssize_t size, int device, void *stream) {
  return gsanMallocWithGranularity(size, device, stream, 4, true);
}

extern "C" void *gsanMallocWriteOnce8(ssize_t size, int device, void *stream) {
  return gsanMallocWithGranularity(size, device, stream, 8, true);
}

extern "C" void *gsanMallocWriteOnce16(ssize_t size, int device, void *stream) {
  return gsanMallocWithGranularity(size, device, stream, 16, true);
}

extern "C" void *gsanMalloc1(ssize_t size, int device, void *stream) {
  return gsanMallocWithGranularity(size, device, stream, 1);
}

extern "C" void *gsanMalloc2(ssize_t size, int device, void *stream) {
  return gsanMallocWithGranularity(size, device, stream, 2);
}

extern "C" void *gsanMalloc8(ssize_t size, int device, void *stream) {
  return gsanMallocWithGranularity(size, device, stream, 8);
}

extern "C" void *gsanMalloc16(ssize_t size, int device, void *stream) {
  return gsanMallocWithGranularity(size, device, stream, 16);
}

extern "C" void gsanFree(void *void_ptr, [[maybe_unused]] ssize_t size,
                         [[maybe_unused]] int device, void *stream) {
  LOGF("gsanFree: %p, 0x%zx", void_ptr, size);
  auto ptr = reinterpret_cast<uintptr_t>(void_ptr);
  if (!ptr)
    return;

  std::lock_guard lg(mut);
  if (alloc == nullptr)
    return;

  AllocNode *node = findAllocation(ptr);
  if (node == nullptr || node->maxFreeBlockSize != 0 ||
      node->virtualAddress != ptr) {
    fprintf(stderr, "gsanFree called with an invalid pointer\n");
    return;
  }

  // Wait for outstanding work on the deallocation stream, including the
  // allocator's own async shadow memset from gsanMalloc, before unmapping.
  auto drvStream = reinterpret_cast<drv::Stream>(stream);
  drv::Result err = drv::streamSynchronize(drvStream);
  if (err != drv::kSuccess)
    printDriverError(err);

  const auto shadowAddress = gsan::getShadowAddress(node->virtualAddress);
  const auto shadowSize = getShadowSize(node->allocSize, node->virtualAddress);

  err = drv::memUnmap(node->virtualAddress, node->allocSize);
  if (err != drv::kSuccess)
    printDriverError(err);

  err = drv::memUnmap(shadowAddress, shadowSize);
  if (err != drv::kSuccess)
    printDriverError(err);

  err = drv::memRelease(node->realHandle);
  if (err != drv::kSuccess)
    printDriverError(err);

  err = drv::memRelease(node->shadowHandle);
  if (err != drv::kSuccess)
    printDriverError(err);

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
                                size_t *allocSize, drv::HandleType handleType,
                                bool includeGranularity) {
  if (realShareableHandle == nullptr || shadowShareableHandle == nullptr ||
      allocSize == nullptr || !drv::isSupportedHandleType(handleType)) {
    return -1;
  }
  *realShareableHandle = {};
  *shadowShareableHandle = {};
  *allocSize = 0;

  const auto ptr = reinterpret_cast<uintptr_t>(void_ptr);
  if (ptr == 0)
    return -1;

  std::lock_guard lg(mut);
  if (alloc == nullptr)
    return -1;

  AllocNode *node = findAllocation(ptr);
  if (node == nullptr || node->maxFreeBlockSize != 0 ||
      ptr < node->virtualAddress ||
      ptr >= node->virtualAddress + node->allocSize) {
    fprintf(stderr,
            "gsanExportAllocationHandles called with invalid pointer\n");
    return -1;
  }
  if (!includeGranularity && gsan::getShadowGranularity(ptr) !=
                                 (gsan::isWriteOnceAddress(ptr) ? 1 : 4))
    return -2;

  drv::Result err = drv::exportHandle(
      getShareableHandleExportArg(realShareableHandle, handleType),
      node->realHandle, handleType);
  if (err != drv::kSuccess) {
    printDriverError(err);
    return -1;
  }

  err = drv::exportHandle(
      getShareableHandleExportArg(shadowShareableHandle, handleType),
      node->shadowHandle, handleType);
  if (err != drv::kSuccess) {
    printDriverError(err);
    if (handleType == drv::HandleType::PosixFileDescriptor)
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

  const auto ptr = reinterpret_cast<uintptr_t>(void_ptr);
  if (ptr == 0)
    return -1;

  std::lock_guard lg(mut);
  if (alloc == nullptr)
    return -1;

  AllocNode *node = findAllocation(ptr);
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
  *shadowSize = getShadowSize(node->allocSize, node->virtualAddress);
  return 0;
}

int gsanExportRuntimeStateHandle(int device,
                                 GSanShareableHandle *shareableHandle,
                                 size_t *allocSize,
                                 drv::HandleType handleType) {
  if (shareableHandle == nullptr || allocSize == nullptr ||
      !drv::isSupportedHandleType(handleType)) {
    return -1;
  }
  *shareableHandle = {};
  *allocSize = 0;

  if (device < 0 || device >= static_cast<int>(gsan::kMaxGPUs))
    return -1;

  std::lock_guard lg(mut);
  if (gsanEnsureInit() != 0)
    return -1;
  drv::Result err = drv::ensureCurrentDevice(device);
  if (err != drv::kSuccess) {
    printDriverError(err);
    return -1;
  }
  err = ensureRuntimeStateMapped(device);
  if (err != drv::kSuccess) {
    printDriverError(err);
    return -1;
  }

  int deviceRank = getDeviceRankForCudaDevice(device);
  auto handle = alloc->perDeviceHandles[deviceRank];
  auto size = alloc->perDeviceStateSize;
  if (handle == 0 || size == 0)
    return -1;

  err = drv::exportHandle(
      getShareableHandleExportArg(shareableHandle, handleType), handle,
      handleType);
  if (err != drv::kSuccess) {
    printDriverError(err);
    return -1;
  }

  *allocSize = size;
  return 0;
}

void *
gsanImportAllocationHandles(const GSanShareableHandle *realShareableHandle,
                            const GSanShareableHandle *shadowShareableHandle,
                            drv::HandleType handleType, size_t allocSize,
                            int device, int shadowGranularity,
                            bool writeOnce = false) {
  if (realShareableHandle == nullptr || shadowShareableHandle == nullptr ||
      !drv::isSupportedHandleType(handleType) || allocSize == 0 ||
      !gsan::isValidShadowGranularity(shadowGranularity))
    return nullptr;
  if (handleType == drv::HandleType::PosixFileDescriptor &&
      (realShareableHandle->fd < 0 || shadowShareableHandle->fd < 0)) {
    return nullptr;
  }

  std::lock_guard lg(mut);
  if (gsanEnsureInit() != 0)
    return nullptr;
  drv::Result err = drv::ensureCurrentDevice(device);
  if (err != drv::kSuccess) {
    printDriverError(err);
    return nullptr;
  }
  err = ensureRuntimeStateMapped(device);
  if (err != drv::kSuccess) {
    printDriverError(err);
    return nullptr;
  }

  auto *root = &alloc->treeRoots[gsan::getPoolIndexForGranularity(
      shadowGranularity, writeOnce)];
  AllocNode *node = allocateNode(root, allocSize);
  if (node == nullptr)
    return nullptr;

  drv::Handle realHandle = 0;
  drv::Handle shadowHandle = 0;
  bool realMapped = false;
  bool shadowMapped = false;
  err = drv::importHandle(
      &realHandle, getShareableHandleImportArg(realShareableHandle, handleType),
      handleType);
  if (err != drv::kSuccess)
    goto error;

  err = drv::importHandle(
      &shadowHandle,
      getShareableHandleImportArg(shadowShareableHandle, handleType),
      handleType);
  if (err != drv::kSuccess)
    goto error;

  err = mapNodeHandles(node, realHandle, shadowHandle, allocSize, device,
                       &realMapped, &shadowMapped);
  if (err != drv::kSuccess)
    goto error;

  return reinterpret_cast<void *>(node->virtualAddress);

error:
  printDriverError(err);
  unmapNodeHandles(node, realMapped, shadowMapped);
  if (shadowHandle != 0)
    (void)drv::memRelease(shadowHandle);
  if (realHandle != 0)
    (void)drv::memRelease(realHandle);
  freeNode(node);
  return nullptr;
}

int gsanImportRuntimeStateHandle(const GSanShareableHandle *shareableHandle,
                                 drv::HandleType handleType, size_t allocSize,
                                 int peerDevice, int device) {
  if (shareableHandle == nullptr || !drv::isSupportedHandleType(handleType) ||
      allocSize == 0)
    return -1;
  if (handleType == drv::HandleType::PosixFileDescriptor &&
      shareableHandle->fd < 0) {
    return -1;
  }
  if (peerDevice < 0 || peerDevice >= static_cast<int>(gsan::kMaxGPUs))
    return -1;

  std::lock_guard lg(mut);
  if (gsanEnsureInit() != 0)
    return -1;
  drv::Result err = drv::ensureCurrentDevice(device);
  if (err != drv::kSuccess) {
    printDriverError(err);
    return -1;
  }

  if (peerDevice < 0 || peerDevice >= config.numGPUs)
    return -1;
  if (allocSize != alloc->perDeviceStateSize)
    return -1;

  drv::Handle importedHandle = 0;
  bool mapped = false;
  uintptr_t deviceAddr =
      alloc->globalStateAddress + peerDevice * gsan::kPerDeviceStateStride;

  err = drv::importHandle(
      &importedHandle, getShareableHandleImportArg(shareableHandle, handleType),
      handleType);
  if (err != drv::kSuccess)
    goto error;

  err = drv::memMap(deviceAddr, allocSize, importedHandle);
  if (err != drv::kSuccess)
    goto error;
  mapped = true;

  err = drv::memSetAccess(deviceAddr, allocSize, device);
  if (err != drv::kSuccess)
    goto error;

  alloc->perDeviceHandles[peerDevice] = importedHandle;
  return 0;

error:
  printDriverError(err);
  if (mapped)
    (void)drv::memUnmap(deviceAddr, allocSize);
  if (importedHandle != 0)
    (void)drv::memRelease(importedHandle);
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
                                 drv::HandleType *out) {
  int handleType = 0;
  if (!parseIntArg(obj, name, &handleType))
    return false;
  if (!drv::isSupportedHandleType(static_cast<drv::HandleType>(handleType))) {
    PyErr_Format(PyExc_ValueError, "%s has unsupported value %d", name,
                 handleType);
    return false;
  }
  *out = static_cast<drv::HandleType>(handleType);
  return true;
}

bool parseShareableHandleArg(PyObject *obj, const char *name,
                             drv::HandleType handleType,
                             GSanShareableHandle *out) {
  if (handleType == drv::HandleType::PosixFileDescriptor)
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
                                    drv::HandleType handleType) {
  if (handleType == drv::HandleType::PosixFileDescriptor)
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

bool parseShadowGranularityArg(PyObject *obj, int *out) {
  if (PyBool_Check(obj) || !parseIntArg(obj, "shadow_granularity", out)) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_ValueError,
                      "shadow_granularity must be 1, 2, 4, 8, or 16");
    return false;
  }
  if (!gsan::isValidShadowGranularity(*out)) {
    PyErr_SetString(PyExc_ValueError,
                    "shadow_granularity must be 1, 2, 4, 8, or 16");
    return false;
  }
  return true;
}

PyObject *pyMalloc([[maybe_unused]] PyObject *self, PyObject *const *args,
                   Py_ssize_t nargs) {
  if (nargs < 2 || nargs > 5) {
    PyErr_Format(PyExc_TypeError,
                 "%s.malloc expected 2 to 5 positional arguments, got %zd",
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
  if (nargs >= 3 && !parseVoidPtrArg(args[2], &stream))
    return nullptr;

  int writeOnce = nargs >= 4 ? PyObject_IsTrue(args[3]) : 0;
  if (writeOnce < 0)
    return nullptr;
  int shadowGranularity = writeOnce ? 1 : gsan::kShadowMemGranularityBytes;
  if (nargs == 5 && !parseShadowGranularityArg(args[4], &shadowGranularity))
    return nullptr;
  return PyLong_FromVoidPtr(gsanMallocWithGranularity(
      size, device, stream, shadowGranularity, writeOnce));
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
    if (requestedClockBufferSize <= 0 ||
        requestedClockBufferSize > kMaxClockBufferSize) {
      PyErr_Format(PyExc_ValueError,
                   "clock_buffer_size must be in [1, %d], got %d",
                   kMaxClockBufferSize, requestedClockBufferSize);
      return nullptr;
    }
  }

  const bool shareableHandleTypeRequested = args[4] != Py_None;
  drv::HandleType requestedShareableHandleType =
      drv::HandleType::PosixFileDescriptor;
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
  drv::Result err = ensureTopologyConfigured();
  if (err != drv::kSuccess) {
    printDriverError(err);
    PyErr_SetString(PyExc_RuntimeError,
                    "failed to configure the default GSan topology");
    return nullptr;
  }
  config.topologyFrozen = true;
  Py_RETURN_NONE;
}

PyObject *pyHasLiveAllocations([[maybe_unused]] PyObject *self,
                               [[maybe_unused]] PyObject *args) {
  std::lock_guard lg(mut);
  return PyBool_FromLong(hasLiveAllocations());
}

PyObject *pySupportsFabricHandles([[maybe_unused]] PyObject *self,
                                  PyObject *const *args, Py_ssize_t nargs) {
  if (nargs != 1) {
    PyErr_Format(PyExc_TypeError,
                 "%s.supports_fabric_handles expected 1 positional argument, "
                 "got %zd",
                 kModuleName, nargs);
    return nullptr;
  }

  int device = 0;
  if (!parseIntArg(args[0], "device", &device))
    return nullptr;

  bool supported = false;
  drv::Result err = drv::supportsFabricHandles(device, &supported);
  if (err != drv::kSuccess) {
    printDriverError(err);
    PyErr_SetString(PyExc_RuntimeError,
                    "failed to query fabric handle support.");
    return nullptr;
  }
  return PyBool_FromLong(supported);
}

PyObject *pyReset([[maybe_unused]] PyObject *self, PyObject *const *args,
                  Py_ssize_t nargs) {
  if (nargs != 0) {
    PyErr_Format(PyExc_TypeError,
                 "%s.reset expected 0 positional arguments, got %zd",
                 kModuleName, nargs);
    return nullptr;
  }

  std::lock_guard lg(mut);
  if (alloc == nullptr)
    Py_RETURN_NONE;

  if (hasLiveAllocations()) {
    PyErr_SetString(PyExc_AssertionError,
                    "cannot reset GSan while GSan allocations are still live");
    return nullptr;
  }

  for (int device = 0; device < static_cast<int>(gsan::kMaxGPUs); ++device) {
    if (!config.configuredDeviceRanks[device])
      continue;

    int deviceRank = config.deviceRanks[device];
    if (alloc->perDeviceHandles[deviceRank] == 0)
      continue;

    drv::DeviceGuard guard;
    drv::Result err = guard.enter(device);
    if (err == drv::kSuccess)
      err = guard.synchronize();
    if (err == drv::kSuccess) {
      uintptr_t deviceAddr =
          alloc->globalStateAddress + deviceRank * gsan::kPerDeviceStateStride;
      err = initializeRuntimeState(deviceAddr, alloc->perDeviceStateSize);
    }

    drv::Result exitErr = guard.exit();
    if (err == drv::kSuccess)
      err = exitErr;

    if (err != drv::kSuccess) {
      printDriverError(err);
      PyErr_SetString(PyExc_RuntimeError,
                      "failed to reinitialize GSan runtime state");
      return nullptr;
    }
  }

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
  drv::Result err = ensureTopologyConfigured();
  if (err != drv::kSuccess) {
    printDriverError(err);
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
      getPerSMStateSize(config.numThreads, config.clockBufferSize);

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

PyObject *pyIsWriteOnceAllocation(PyObject *self, PyObject *arg) {
  void *ptr = nullptr;
  if (!parseVoidPtrArg(arg, &ptr))
    return nullptr;
  uintptr_t realPtr = 0, shadowPtr = 0;
  size_t realSize = 0, shadowSize = 0;
  if (gsanExportAllocationMemhandleRegions(ptr, &realPtr, &realSize, &shadowPtr,
                                           &shadowSize) != 0) {
    PyErr_SetString(PyExc_ValueError,
                    "expected a live GSan allocation pointer");
    return nullptr;
  }
  return PyBool_FromLong(gsan::isWriteOnceAddress(realPtr));
}

PyObject *pyExportAllocationHandles(PyObject *self, PyObject *const *args,
                                    Py_ssize_t nargs) {
  (void)self;
  if (nargs < 2 || nargs > 4) {
    PyErr_Format(PyExc_TypeError,
                 "%s.export_allocation_handles expected 2 to 4 positional "
                 "arguments, got %zd",
                 kModuleName, nargs);
    return nullptr;
  }

  void *ptr = nullptr;
  drv::HandleType handleType = drv::HandleType::PosixFileDescriptor;
  if (!parseVoidPtrArg(args[0], &ptr) ||
      !parseShareableHandleTypeArg(args[1], "handle_type", &handleType))
    return nullptr;

  int writeOnce = nargs >= 3 ? PyObject_IsTrue(args[2]) : 0;
  if (writeOnce < 0)
    return nullptr;
  if (gsan::isWriteOnceAddress(reinterpret_cast<uintptr_t>(ptr)) !=
      static_cast<bool>(writeOnce)) {
    PyErr_SetString(PyExc_ValueError,
                    "write_once does not match the allocation mode");
    return nullptr;
  }
  int includeGranularity = nargs == 4 ? PyObject_IsTrue(args[3]) : 0;
  if (includeGranularity < 0)
    return nullptr;

  GSanShareableHandle realShareableHandle = {};
  GSanShareableHandle shadowShareableHandle = {};
  size_t allocSize = 0;
  int rc = gsanExportAllocationHandles(ptr, &realShareableHandle,
                                       &shadowShareableHandle, &allocSize,
                                       handleType, includeGranularity != 0);
  if (rc == -2) {
    PyErr_SetString(PyExc_ValueError,
                    "Non-default pools require include_granularity=True");
    return nullptr;
  }
  if (rc != 0) {
    PyErr_SetString(PyExc_RuntimeError, "gsanExportAllocationHandles failed.");
    return nullptr;
  }

  if (!includeGranularity)
    return Py_BuildValue(
        "(NNK)", shareableHandleToPyObject(realShareableHandle, handleType),
        shareableHandleToPyObject(shadowShareableHandle, handleType),
        static_cast<unsigned long long>(allocSize));
  return Py_BuildValue(
      "(NNKi)", shareableHandleToPyObject(realShareableHandle, handleType),
      shareableHandleToPyObject(shadowShareableHandle, handleType),
      static_cast<unsigned long long>(allocSize),
      gsan::getShadowGranularity(reinterpret_cast<uintptr_t>(ptr)));
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
  if (nargs < 5 || nargs > 7) {
    PyErr_Format(PyExc_TypeError,
                 "%s.import_allocation_handles expected 5 to 7 positional "
                 "arguments, got %zd",
                 kModuleName, nargs);
    return nullptr;
  }

  GSanShareableHandle realShareableHandle = {};
  GSanShareableHandle shadowShareableHandle = {};
  int device = 0;
  drv::HandleType handleType = drv::HandleType::PosixFileDescriptor;
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

  int writeOnce = nargs >= 6 ? PyObject_IsTrue(args[5]) : 0;
  if (writeOnce < 0)
    return nullptr;
  int shadowGranularity = writeOnce ? 1 : gsan::kShadowMemGranularityBytes;
  if (nargs == 7 && !parseShadowGranularityArg(args[6], &shadowGranularity))
    return nullptr;

  void *ptr = gsanImportAllocationHandles(
      &realShareableHandle, &shadowShareableHandle, handleType, allocSize,
      device, shadowGranularity, writeOnce);
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
  drv::HandleType handleType = drv::HandleType::PosixFileDescriptor;
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
  drv::HandleType handleType = drv::HandleType::PosixFileDescriptor;
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
    {"has_live_allocations",
     reinterpret_cast<PyCFunction>(pyHasLiveAllocations), METH_NOARGS,
     "Return whether the GSan allocation reserve has live allocations."},
    {"supports_fabric_handles",
     reinterpret_cast<PyCFunction>(pySupportsFabricHandles), METH_FASTCALL,
     "Return whether a CUDA device supports fabric allocation handles."},
    {"reset", reinterpret_cast<PyCFunction>(pyReset), METH_FASTCALL,
     "Reset GSan runtime state when there are no live allocations."},
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
    {"is_write_once_allocation",
     reinterpret_cast<PyCFunction>(pyIsWriteOnceAllocation), METH_O,
     "Return the mode of a live allocation, accepting interior pointers."},
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
