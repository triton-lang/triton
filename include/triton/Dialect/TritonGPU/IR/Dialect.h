#ifndef TRITON_DIALECT_TRITONGPU_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONGPU_IR_DIALECT_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

// TritonGPU depends on Triton
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h.inc"
#include "triton/Dialect/TritonGPU/IR/Traits.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonGPU/IR/Ops.h.inc"

namespace mlir {
namespace triton {
namespace gpu {

struct SharedMemory : public SideEffects::Resource::Base<SharedMemory> {
  StringRef getName() final { return "<SharedMemory>"; }
};

unsigned getTotalElemsPerThread(Type type);

unsigned getTotalElemsPerThread(Attribute layout, ArrayRef<int64_t> shape,
                                Type eltTy);

SmallVector<unsigned> getElemsPerThread(Type type);

// Returns the number of threads per warp that may have access to replicated
// elements. If you want non-replicated threads, use
// getThreadsPerWarpWithUniqueData.
SmallVector<unsigned> getThreadsPerWarp(Attribute layout);

unsigned getWarpSize(Attribute layout);

// Returns the number of warps per CTA that may have access to replicated
// elements. If you want non-replicated warps, use getWarpsPerCTAWithUniqueData.
SmallVector<unsigned> getWarpsPerCTA(Attribute layout);

SmallVector<unsigned> getSizePerThread(Attribute layout);

// Returns the number of contiguous elements that each thread
// has access to, on each dimension of the tensor. E.g.
// for a blocked layout with sizePerThread = [1, 4], returns [1, 4],
// regardless of the shape of the tensor.
SmallVector<unsigned> getContigPerThread(Attribute layout);

// Returns the number of non-replicated contiguous elements that each thread
// has access to, on each dimension of the tensor. For a blocked layout
// with sizePerThread = [1, 4] and tensor shape = [128, 1], the elements
// for thread 0 would be [A_{0, 0}, A_{0, 0}, A_{0, 0}, A_{0, 0}], returns [1,
// 1]. Whereas for a tensor shape [128, 128], the elements for thread 0 would be
// [A_{0, 0}, A_{0, 1}, A_{0, 2}, A_{0, 3}], returns [1, 4].
SmallVector<unsigned> getUniqueContigPerThread(Attribute layout,
                                               ArrayRef<int64_t> tensorShape);

// Returns the number of threads per warp that have access to non-replicated
// elements of the tensor. E.g. for a blocked layout with sizePerThread = [1,
// 1], threadsPerWarp = [2, 16] and tensor shape = [2, 2], threads 0, 1, 16, 17
// have access to the full tensor, whereas the other threads have access to
// replicated elements, so this function returns [2, 2].
SmallVector<unsigned>
getThreadsPerWarpWithUniqueData(Attribute layout,
                                ArrayRef<int64_t> tensorShape);

// Returns the number of warps per CTA that have access to non-replicated
// elements of the tensor. E.g. for a blocked layout with sizePerThread = [1,
// 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4] and tensor shape = [2, 2],
// returns [1, 1], since the first warp has access to the full tensor, whereas
// the other warps have access to replicated elements.
SmallVector<unsigned>
getWarpsPerCTAWithUniqueData(Attribute layout, ArrayRef<int64_t> tensorShape);

SmallVector<unsigned> getOrder(Attribute layout);

CTALayoutAttr getCTALayout(Attribute layout);

SmallVector<unsigned> getCTAsPerCGA(Attribute layout);

SmallVector<unsigned> getCTASplitNum(Attribute layout);

SmallVector<unsigned> getCTAOrder(Attribute layout);

/* The difference between ShapePerCTATile and ShapePerCTA:
 * (1) ShapePerCTATile is defined by SizePerThread * ThreadsPerWarp *
 *     WarpsPerCTA in each dimension and is independent from the tensor shape.
 * (2) ShapePerCTA is defined by shape / CTASplitNum in each dimension.
 * (3) In the implementation of emitIndices, ShapePerCTATile will
 *     be replicated or wrapped to fit ShapePerCTA.
 */
SmallVector<unsigned>
getShapePerCTATile(Attribute layout,
                   ArrayRef<int64_t> tensorShape = ArrayRef<int64_t>());

SmallVector<int64_t> getShapePerCTA(ArrayRef<unsigned> CTASplitNum,
                                    ArrayRef<int64_t> shape);
SmallVector<int64_t> getShapePerCTA(Attribute layout, ArrayRef<int64_t> shape);
SmallVector<int64_t> getShapePerCTA(Type type);

unsigned getNumWarpsPerCTA(Attribute layout);

unsigned getNumCTAs(Attribute layout);

bool isaDistributedLayout(Attribute layout);

bool hasSharedEncoding(Value value);

bool isExpensiveCat(CatOp cat, Attribute targetEncoding);

// Return true if a view between the two types cannot be implemented as a no-op.
bool isExpensiveView(Type srcType, Type dstType);

// Return a blocked encoding where the shape is distributed contiguously amongst
// the threads, warps, CTAs with 1 element per threads.
triton::gpu::BlockedEncodingAttr
getDefaultBlockedEncoding(MLIRContext *context, ArrayRef<int64_t> shape,
                          int numWarps, int threadsPerWarp, int numCTAs);

} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_IR_DIALECT_H_
