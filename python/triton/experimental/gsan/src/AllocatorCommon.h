#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <memory>
#include <unistd.h>

// CUDA VMM mechanics shared by the independent race and profiling allocators.
// No reservation, configuration, or allocation state is shared here.
namespace triton::cuda_allocator {

union ShareableHandle {
  int fd;
  CUmemFabricHandle fabricHandle;
};

inline bool
isSupportedShareableHandleType(CUmemAllocationHandleType handleType) {
  return handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR ||
         handleType == CU_MEM_HANDLE_TYPE_FABRIC;
}

inline void *getShareableHandleImportArg(const ShareableHandle *handle,
                                         CUmemAllocationHandleType handleType) {
  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    return reinterpret_cast<void *>(static_cast<uintptr_t>(handle->fd));
  if (handleType == CU_MEM_HANDLE_TYPE_FABRIC)
    return const_cast<CUmemFabricHandle *>(&handle->fabricHandle);
  return nullptr;
}

inline void *getShareableHandleExportArg(ShareableHandle *handle,
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

inline void printCUDAError(CUresult err) {
  const char *errs = "<unknown error>";
  cuGetErrorString(err, &errs);
  fprintf(stderr, "CUDA allocator encountered an unexpected error: %s\n", errs);
}

inline size_t cdiv(size_t num, size_t den) { return (num + (den - 1)) / den; }

inline size_t roundUp(size_t val, size_t alignment) {
  return cdiv(val, alignment) * alignment;
}

inline size_t roundDownToPowerOfTwo(size_t x) {
  if (x == 0)
    return 0;

  for (size_t shift = 1; shift < sizeof(x) * 8; shift <<= 1)
    x |= x >> shift;

  return x - (x >> 1);
}

inline bool isLeaf(const AllocNode *node) {
  return node->leftChild == nullptr && node->rightChild == nullptr;
}

inline void recomputeNodeState(AllocNode *node) {
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

inline void recomputeToRoot(AllocNode *node) {
  for (AllocNode *curr = node; curr != nullptr; curr = curr->parent)
    recomputeNodeState(curr);
}

inline void splitNode(AllocNode *node) {
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

inline AllocNode *allocateNode(AllocNode *root, size_t allocSize) {
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

inline AllocNode *findNodeByAddress(AllocNode *root, CUdeviceptr address) {
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

inline bool canCoalesce(const AllocNode *node) {
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

inline void coalesceUp(AllocNode *node) {
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

inline void freeNode(AllocNode *leaf) {
  assert(isLeaf(leaf));
  leaf->allocSize = 0;
  leaf->realHandle = 0;
  leaf->shadowHandle = 0;
  leaf->maxFreeBlockSize = leaf->size;
  coalesceUp(leaf->parent);
}

inline CUresult ensureContext(int device) {
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

inline CUresult mapNodeHandles(AllocNode *node,
                               CUmemGenericAllocationHandle realHandle,
                               CUmemGenericAllocationHandle shadowHandle,
                               size_t allocSize, uintptr_t shadowAddress,
                               size_t shadowSize, int device, bool *realMapped,
                               bool *shadowMapped) {
  assert(node != nullptr);
  assert(realMapped != nullptr);
  assert(shadowMapped != nullptr);

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

inline void unmapNodeHandles(AllocNode *node, size_t allocSize,
                             uintptr_t shadowAddress, size_t shadowSize,
                             bool realMapped, bool shadowMapped) {
  assert(node != nullptr);
  if (shadowMapped)
    cuMemUnmap(shadowAddress, shadowSize);
  if (realMapped)
    cuMemUnmap(node->virtualAddress, allocSize);
}

inline CUmemAllocationHandleType defaultShareableHandleType() {
  const auto *conf = getenv("PYTORCH_CUDA_ALLOC_CONF");
  return conf && strstr(conf, "fabric_handles:True")
             ? CU_MEM_HANDLE_TYPE_FABRIC
             : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
}

inline CUresult createAllocation(AllocNode *node, size_t allocSize,
                                 uintptr_t shadowAddress, size_t shadowSize,
                                 const CUmemAllocationProp &prop,
                                 CUstream stream) {
  CUmemGenericAllocationHandle realHandle = 0, shadowHandle = 0;
  bool realMapped = false, shadowMapped = false;
  CUresult err = cuMemCreate(&realHandle, allocSize, &prop, 0);
  if (err == CUDA_SUCCESS)
    err = cuMemCreate(&shadowHandle, shadowSize, &prop, 0);
  if (err == CUDA_SUCCESS)
    err = mapNodeHandles(node, realHandle, shadowHandle, allocSize,
                         shadowAddress, shadowSize, prop.location.id,
                         &realMapped, &shadowMapped);
  if (err == CUDA_SUCCESS)
    err = cuMemsetD8Async(shadowAddress, 0, shadowSize, stream);
  if (err != CUDA_SUCCESS) {
    unmapNodeHandles(node, allocSize, shadowAddress, shadowSize, realMapped,
                     shadowMapped);
    if (shadowHandle)
      cuMemRelease(shadowHandle);
    if (realHandle)
      cuMemRelease(realHandle);
  }
  return err;
}

inline CUresult importAllocation(AllocNode *node, size_t allocSize,
                                 uintptr_t shadowAddress, size_t shadowSize,
                                 const ShareableHandle &real,
                                 const ShareableHandle &shadow,
                                 CUmemAllocationHandleType handleType,
                                 int device) {
  CUmemGenericAllocationHandle realHandle = 0, shadowHandle = 0;
  bool realMapped = false, shadowMapped = false;
  CUresult err = cuMemImportFromShareableHandle(
      &realHandle, getShareableHandleImportArg(&real, handleType), handleType);
  if (err == CUDA_SUCCESS)
    err = cuMemImportFromShareableHandle(
        &shadowHandle, getShareableHandleImportArg(&shadow, handleType),
        handleType);
  if (err == CUDA_SUCCESS)
    err =
        mapNodeHandles(node, realHandle, shadowHandle, allocSize, shadowAddress,
                       shadowSize, device, &realMapped, &shadowMapped);
  if (err != CUDA_SUCCESS) {
    unmapNodeHandles(node, allocSize, shadowAddress, shadowSize, realMapped,
                     shadowMapped);
    if (shadowHandle)
      cuMemRelease(shadowHandle);
    if (realHandle)
      cuMemRelease(realHandle);
  }
  return err;
}

inline CUresult exportAllocation(const AllocNode *node, ShareableHandle *real,
                                 ShareableHandle *shadow,
                                 CUmemAllocationHandleType handleType) {
  CUresult err = cuMemExportToShareableHandle(
      getShareableHandleExportArg(real, handleType), node->realHandle,
      handleType, 0);
  if (err != CUDA_SUCCESS)
    return err;
  err = cuMemExportToShareableHandle(
      getShareableHandleExportArg(shadow, handleType), node->shadowHandle,
      handleType, 0);
  if (err != CUDA_SUCCESS &&
      handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    close(real->fd);
  return err;
}

inline void releaseAllocation(AllocNode *node, uintptr_t shadowAddress,
                              size_t shadowSize, CUstream stream) {
  // Wait for the allocator's memset and the caller's work before unmapping.
  auto check = [](CUresult err) {
    if (err != CUDA_SUCCESS)
      printCUDAError(err);
  };
  check(cuStreamSynchronize(stream));
  check(cuMemUnmap(node->virtualAddress, node->allocSize));
  check(cuMemUnmap(shadowAddress, shadowSize));
  check(cuMemRelease(node->realHandle));
  check(cuMemRelease(node->shadowHandle));
  freeNode(node);
}

} // namespace triton::cuda_allocator
