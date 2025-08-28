// Conversions from TritonGPU layouts (e.g. BlockedEncodingAttr) to
// LinearLayout.

#ifndef TRITON_DIALECT_TRITONGPU_IR_LINEARLAYOUTCONVERSIONS_H
#define TRITON_DIALECT_TRITONGPU_IR_LINEARLAYOUTCONVERSIONS_H

#include <optional>

#include "triton/Tools/LinearLayout.h"

namespace mlir::triton {
enum class ScaleDotElemType : uint32_t;
} // namespace mlir::triton

namespace mlir::triton::gpu {
class SwizzledSharedEncodingAttr;
class NVMMASharedEncodingAttr;
class AMDRotatingSharedEncodingAttr;
class AMDMfmaEncodingAttr;
class TensorOrMemDesc;
class MemDescType;

// - BlockedEncodingAttrs have the following input dimensions.
//
//   "register": elements in one thread
//   "lane": threads in a warp
//   "warp": warps in a block/CTA
//   "block": blocks in a cluster
//
// - An n-dimensional SwizzledSharedEncodingAttr has the following input
// dimensions.
//
//   "offset": the n'th element in the allocation, within a particular thread
//      block (i.e. within a CTA).  The offset is measured in elements, not
//      bytes.
//   "block": blocks in a cluster
//
// All layouts have the following output dimensions.
//
//  "dimi" for i in 0..n-1: the location in the n'th logical dimension of the
//  output tensor.  These also are not reordered according to the layout's
//  `order`.
//
// You can flatten the input or output dimensions into a single dimension using
// LinearLayout::flattenIns/Outs().
//
// elemBitWidth is the bit width of one element in the layout.  This is required
// to compute the linear layout for MMAv3 (i.e. Hopper) shared layouts (i.e.
// shared layouts with nvmma_shared layout) but is otherwise unused.
LinearLayout toLinearLayout(RankedTensorType type);
LinearLayout toLinearLayout(MemDescType type);
LinearLayout toLinearLayout(TensorOrMemDesc type);
// UNSAFE OVERLOAD!
// If you call this with a SharedMemoryEncodingAttr, you should call it
// with the allocShape as the shape, otherwise the layout will be incorrect!
LinearLayout toLinearLayout(ArrayRef<int64_t> shape, Attribute layout);

// Convert the shared encoding of a tensor with `nvmma_shared` layout to a
// LinearLayout that maps from a linear shared memory offset to tensor index.
//
// If `disableSwizzle` is set, then the resulting layout does not include
// swizzling.
LinearLayout nvmmaSharedToLinearLayout(ArrayRef<int64_t> shape,
                                       NVMMASharedEncodingAttr shared,
                                       bool disableSwizzle = false);

// Given a linear layout where the input dimensions contain a "block" dimension,
// this method sets the "block" dimension to 0 and removes the corresponding
// output dimensions.
//
// Note that this behavior differs from calling
// `LinearLayout::sublayout(inDimNames, outDimNames)` when "block" is not in
// `inDimNames`. The latter does not modify the output sizes.
LinearLayout getLayoutWithinBlock(const LinearLayout &layout);

// In this function, we construct a linear layout representing the
// <shared memory offset, iteration, block> -> <tensor element index> mapping
// for entire `src` and `dst` tensors.  We determine the shape of the
// intermediate shared memory buffer needed for a register-to-register
// conversion using the maximum size accessed in each dimension from `src`'s
// layout and `dst`'s layout.  See the getRepShapeForCvt function in
// Allocation.cpp for details. Note that the buffer might be smaller than the
// tensor being converted, so we need multiple "iterations" to move a subregion
// of the `src` tensor to the corresponding subregion of the `dst` tensor.  The
// pesudo code of layout conversion is as follows:
//
// for iter in 0..numIterations:
//   sync threads
//   for vecIdx in [0..numRegisters/storeVec]:
//     registers <- get registers used in iter
//     offsets <- get offsets using the intermediate linear layout
//     store registers[vecIdx * storeVec, (vecIdx + 1) * storeVec)] to shared
//     memory
//   sync threads
//   for vecIdx in [0..numRegisters/loadVec]:
//     registers <- get registers used in iter
//     offsets <- get offsets using the intermediate linear layout
//     load registers[vecIdx * loadVec, (vecIdx + 1) * loadVec)] from shared
//     memory
LinearLayout chooseShemLayoutForRegToRegConversion(
    MLIRContext *ctx, ArrayRef<unsigned> tensorShape,
    ArrayRef<unsigned> repShape, ArrayRef<unsigned> order);

// The primary goal of this function is to efficiently load 2D tiles of a
// tensor from shared memory using the `ds_read_tr` instruction for AMD GPUs.
LinearLayout chooseDsReadB64TrLayout(Attribute enc, ArrayRef<int64_t> shape,
                                     int32_t elemBitWidth);

LinearLayout getScaleTMEMStoreLinearLayout(RankedTensorType scaleType,
                                           int numWarps);

std::optional<LinearLayout>
getTmemLoadStoreLayout16x256(int M, int N, RankedTensorType oldType,
                             int numWarps);

// Return a layout valid for TMemLoad op for a tmem layout of block MxN that
// distribute the data long M for the warp groups. This doesn't affect the TMem
// layout it just returns a distributed layout compatible for tmem_load.
LinearLayout getTmemLoadLayoutSplitLongM(int M, int N, RankedTensorType oldType,
                                         int numWarps);

// Create LinearLayout for scale in scaled mfma.
LinearLayout chooseScaledMfmaScaleLayout(MLIRContext *ctx, int dotOperandIdx,
                                         ArrayRef<int64_t> dotOperandShape,
                                         unsigned mfmaMDim,
                                         ArrayRef<unsigned> tilesPerWarp,
                                         ArrayRef<unsigned> warpsPerCTA);

// Create LinearLayout for nvidia mma tile.
LinearLayout nvidiaMmaTile(MLIRContext *ctx, ArrayRef<unsigned> tileShape,
                           unsigned kWidth, ArrayRef<unsigned> order,
                           ArrayRef<unsigned> repOrder);

// Create a LinearLayout similar to mfmaLayout, but changing each thread to hold
// 8 elements. This layout is useful for emitting the widest 128-bit global
// store instructions. Since it closely resembles mfmaLayout, conversion between
// the two can be done using transferWithinWarp, without involving LDS
std::optional<LinearLayout> chooseMfmaLikeStoreLayout(RankedTensorType valType);

} // namespace mlir::triton::gpu
#endif // TRITON_DIALECT_TRITONGPU_IR_LINEARLAYOUTCONVERSIONS_H
