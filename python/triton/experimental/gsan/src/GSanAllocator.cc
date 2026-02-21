#include <cuda.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>

// #define GSAN_LOG_ALLOCATIONS
#ifdef GSAN_LOG_ALLOCATIONS
#define LOGF(...) printf(__VA_ARGS__);
#else
#define LOGF(...)
#endif

extern "C" {
void *gsanMalloc(ssize_t size, int device, void *stream);
void gsanFree(void *ptr, ssize_t size, int device, void *stream);
void *gsanGetReservePointer();
size_t gsanGetReserveSize();
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

struct AllocatorState {
  CUdeviceptr reserveBaseAddress;
  AllocNode treeRoot;
};

void printCUDAError(CUresult err) {
  const char *errs = "<unknown error>";
  cuGetErrorString(err, &errs);
  fprintf(stderr, "gsan allocator encountered an unexpected error: %s\n", errs);
}

static AllocatorState *alloc = nullptr;
static std::mutex mut;

// Reserve 1 PiB, should be enough for now :)
constexpr size_t kReserveSize = 1ull << 40;
constexpr int kShadowSizeBytes = 8;
constexpr int kShadowMemGranularityBytes = 4;

static_assert((kReserveSize & (kReserveSize - 1)) == 0,
              "kReserveSize must be a power of 2");

size_t cdiv(size_t num, size_t den) { return (num + (den - 1)) / den; }

size_t roundUp(size_t val, size_t alignment) {
  return cdiv(val, alignment) * alignment;
}

CUdeviceptr getShadowAddress(CUdeviceptr virtualAddress) {
  auto shadowBase = alloc->reserveBaseAddress;
  auto realBase = shadowBase + kReserveSize / 2;
  auto byteOffset = virtualAddress - realBase;
  auto wordOffset = byteOffset / kShadowMemGranularityBytes;
  return shadowBase + kShadowSizeBytes * wordOffset;
}

size_t getShadowSize(size_t realMemSize) {
  auto wordSize = cdiv(realMemSize, kShadowMemGranularityBytes);
  return wordSize * kShadowSizeBytes;
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
  CUresult err =
      cuMemAddressReserve(&reserveBase, /*size*/ kReserveSize,
                          /*alignment*/ kReserveSize, /*addr*/ 0, /*flags*/ 0);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return -1;
  }
  alloc = new AllocatorState();
  alloc->reserveBaseAddress = reserveBase;
  auto *root = &alloc->treeRoot;
  root->virtualAddress = reserveBase + (kReserveSize / 2);
  root->size = kReserveSize / 2;
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

} // namespace

// TODO: Handle streams?
void *gsanMalloc(ssize_t size, int device, [[maybe_unused]] void *stream) {
  if (size <= 0)
    return nullptr;

  std::lock_guard lg(mut);
  if (gsanEnsureInit() != 0)
    return nullptr;

  if (!stream) {
    CUresult err = ensureContext(device);
    if (err != CUDA_SUCCESS) {
      printCUDAError(err);
      return nullptr;
    }
  }

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;

  size_t granularity = 0;
  CUresult err = cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (err != CUDA_SUCCESS) {
    printCUDAError(err);
    return nullptr;
  }
  size_t allocSize = roundUp(static_cast<size_t>(size), granularity);
  AllocNode *node = allocateNode(&alloc->treeRoot, allocSize);
  if (node == nullptr)
    return nullptr;
  auto shadowSize = getShadowSize(node->size);

  CUmemGenericAllocationHandle realHandle = 0;
  CUmemGenericAllocationHandle shadowHandle = 0;
  bool realMapped = false;
  bool shadowMapped = false;
  CUmemAccessDesc accessDesc = {};
  auto cuStream = reinterpret_cast<CUstream>(stream);

  CUdeviceptr shadowAddress = getShadowAddress(node->virtualAddress);

  err = cuMemCreate(&realHandle, node->size, &prop, 0);
  if (err != CUDA_SUCCESS)
    goto error;

  err = cuMemCreate(&shadowHandle, getShadowSize(node->size), &prop, 0);
  if (err != CUDA_SUCCESS)
    goto error;

  err = cuMemMap(node->virtualAddress, node->size, /*offset*/ 0, realHandle,
                 /*flags*/ 0);
  if (err != CUDA_SUCCESS)
    goto error;
  realMapped = true;

  err = cuMemMap(shadowAddress, shadowSize, /*offset*/ 0, shadowHandle,
                 /*flags*/ 0);
  if (err != CUDA_SUCCESS)
    goto error;
  shadowMapped = true;

  accessDesc.location = prop.location;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  err = cuMemSetAccess(node->virtualAddress, node->size, &accessDesc, 1);
  if (err != CUDA_SUCCESS)
    goto error;

  err = cuMemSetAccess(shadowAddress, shadowSize, &accessDesc, 1);
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
  if (shadowMapped)
    cuMemUnmap(shadowAddress, shadowSize);
  if (realMapped)
    cuMemUnmap(node->virtualAddress, node->size);
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

  const auto shadowAddress = getShadowAddress(node->virtualAddress);
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

extern "C" void *gsanGetReservePointer() {
  std::lock_guard lg(mut);
  if (gsanEnsureInit() != 0)
    return nullptr;
  return reinterpret_cast<void *>(alloc->reserveBaseAddress);
}

extern "C" size_t gsanGetReserveSize() { return kReserveSize; }
