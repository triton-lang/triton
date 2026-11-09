#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unistd.h>

#include "GSanDriver.h"

// VMM mechanics shared by the independent race and profiling allocators.
// No reservation, configuration, or allocation state is shared here.
namespace triton::cuda_allocator {

namespace drv = gsan::drv;

union ShareableHandle {
  int fd;
  drv::FabricHandle fabricHandle;
};

inline void *getShareableHandleImportArg(const ShareableHandle *handle,
                                         drv::HandleType handleType) {
  if (handleType == drv::HandleType::PosixFileDescriptor)
    return reinterpret_cast<void *>(static_cast<uintptr_t>(handle->fd));
  if (handleType == drv::HandleType::Fabric)
    return const_cast<drv::FabricHandle *>(&handle->fabricHandle);
  return nullptr;
}

inline void *getShareableHandleExportArg(ShareableHandle *handle,
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

inline void printDriverError(drv::Result err) {
  fprintf(stderr, "allocator encountered an unexpected error: %s\n",
          drv::errorString(err));
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

inline AllocNode *findNodeByAddress(AllocNode *root, uintptr_t address) {
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

#ifndef GSAN_BACKEND_HIP
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
#endif

inline drv::Result mapNodeHandles(AllocNode *node, drv::Handle realHandle,
                                  drv::Handle shadowHandle, size_t allocSize,
                                  uintptr_t shadowAddress, size_t shadowSize,
                                  int device, bool *realMapped,
                                  bool *shadowMapped) {
  assert(node != nullptr);
  assert(realMapped != nullptr);
  assert(shadowMapped != nullptr);

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

inline void unmapNodeHandles(AllocNode *node, size_t allocSize,
                             uintptr_t shadowAddress, size_t shadowSize,
                             bool realMapped, bool shadowMapped) {
  assert(node != nullptr);
  if (shadowMapped)
    (void)drv::memUnmap(shadowAddress, shadowSize);
  if (realMapped)
    (void)drv::memUnmap(node->virtualAddress, allocSize);
}

inline drv::HandleType defaultShareableHandleType() {
  const auto *conf = getenv("PYTORCH_CUDA_ALLOC_CONF");
  return drv::kHasFabricHandles && conf && strstr(conf, "fabric_handles:True")
             ? drv::HandleType::Fabric
             : drv::HandleType::PosixFileDescriptor;
}

inline drv::Result createAllocation(AllocNode *node, size_t allocSize,
                                    uintptr_t shadowAddress, size_t shadowSize,
                                    int device, drv::HandleType handleType,
                                    drv::Stream stream) {
  drv::Handle realHandle = 0, shadowHandle = 0;
  bool realMapped = false, shadowMapped = false;
  drv::Result err = drv::memCreate(&realHandle, allocSize, device, handleType);
  if (err == drv::kSuccess)
    err = drv::memCreate(&shadowHandle, shadowSize, device, handleType);
  if (err == drv::kSuccess)
    err =
        mapNodeHandles(node, realHandle, shadowHandle, allocSize, shadowAddress,
                       shadowSize, device, &realMapped, &shadowMapped);
  if (err == drv::kSuccess)
    err = drv::memsetD8Async(shadowAddress, shadowSize, stream);
  if (err != drv::kSuccess) {
    unmapNodeHandles(node, allocSize, shadowAddress, shadowSize, realMapped,
                     shadowMapped);
    if (shadowHandle)
      (void)drv::memRelease(shadowHandle);
    if (realHandle)
      (void)drv::memRelease(realHandle);
  }
  return err;
}

inline drv::Result importAllocation(AllocNode *node, size_t allocSize,
                                    uintptr_t shadowAddress, size_t shadowSize,
                                    const ShareableHandle &real,
                                    const ShareableHandle &shadow,
                                    drv::HandleType handleType, int device) {
  drv::Handle realHandle = 0, shadowHandle = 0;
  bool realMapped = false, shadowMapped = false;
  drv::Result err = drv::importHandle(
      &realHandle, getShareableHandleImportArg(&real, handleType), handleType);
  if (err == drv::kSuccess)
    err = drv::importHandle(&shadowHandle,
                            getShareableHandleImportArg(&shadow, handleType),
                            handleType);
  if (err == drv::kSuccess)
    err =
        mapNodeHandles(node, realHandle, shadowHandle, allocSize, shadowAddress,
                       shadowSize, device, &realMapped, &shadowMapped);
  if (err != drv::kSuccess) {
    unmapNodeHandles(node, allocSize, shadowAddress, shadowSize, realMapped,
                     shadowMapped);
    if (shadowHandle)
      (void)drv::memRelease(shadowHandle);
    if (realHandle)
      (void)drv::memRelease(realHandle);
  }
  return err;
}

inline drv::Result exportAllocation(const AllocNode *node,
                                    ShareableHandle *real,
                                    ShareableHandle *shadow,
                                    drv::HandleType handleType) {
  drv::Result err =
      drv::exportHandle(getShareableHandleExportArg(real, handleType),
                        node->realHandle, handleType);
  if (err != drv::kSuccess)
    return err;
  err = drv::exportHandle(getShareableHandleExportArg(shadow, handleType),
                          node->shadowHandle, handleType);
  if (err != drv::kSuccess &&
      handleType == drv::HandleType::PosixFileDescriptor)
    close(real->fd);
  return err;
}

inline void releaseAllocation(AllocNode *node, uintptr_t shadowAddress,
                              size_t shadowSize, drv::Stream stream) {
  // Wait for the allocator's memset and the caller's work before unmapping.
  auto check = [](drv::Result err) {
    if (err != drv::kSuccess)
      printDriverError(err);
  };
  check(drv::streamSynchronize(stream));
  check(drv::memUnmap(node->virtualAddress, node->allocSize));
  check(drv::memUnmap(shadowAddress, shadowSize));
  check(drv::memRelease(node->realHandle));
  check(drv::memRelease(node->shadowHandle));
  freeNode(node);
}

} // namespace triton::cuda_allocator
