// Conversions from TritonGPU layouts (e.g. BlockedEncodingAttr) to
// LinearLayout.

#ifndef TRITON_DIALECT_TRITONGPU_IR_LINEARLAYOUTCONVERSIONS_H
#define TRITON_DIALECT_TRITONGPU_IR_LINEARLAYOUTCONVERSIONS_H

#include <optional>

#include "triton/Tools/LinearLayout.h"

namespace mlir::triton::gpu {

// - BlockedEncodingAttrs have the following input dimensions.
//
//   "register": elements in one thread
//   "lane": threads in a warp
//   "warp": warps in a block/CTA
//   "block": blocks in a cluster
//
// - An n-dimensional SharedEncodingAttr has the following input dimensions.
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
// shared layouts with hasLeadingOffset == true) but is otherwise unused.
//
// Returns std::nullopt if the given layout can't be converted to an LL.
// TODO(jlebar): Remove the std::optional once all layouts are supported.
//
std::optional<LinearLayout>
toLinearLayout(ArrayRef<int64_t> shape, Attribute layout,
               std::optional<int32_t> elemBitWidth = std::nullopt);

// Given a linear layout with input dims and output dims containing a "block"
// dimension, determines if the layout moves data across block boundaries.
bool isCrossCTAConversion(const LinearLayout &layout);

// In this function, we construct a linear layout representing the
// <shared memory offset, iteration> -> <tensor element index> mapping for
// entire `src` and `dst` tensors.  We determine the shape of the intermediate
// shared memory buffer needed for a register-to-register conversion using the
// maximum size accessed in each dimension from `src`'s layout and `dst`'s
// layout.  See the getRepShapeForCvt function in Allocation.cpp for details.
// Note that the buffer might be smaller than the tensor being converted, so we
// need multiple "iterations" to move a subregion of the `src` tensor to the
// corresponding subregion of the `dst` tensor.  The pesudo code of layout
// conversion is as follows:
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

// This function constructs a linear layout that maps the
// <register, lane, warp, block> hierarchy to <shared memory offset, iteration>.
// The primary goal is to efficiently store 2D tiles of a tensor into shared
// memory, with each thread responsible for storing `N` elements.
//
// Unlike standard vectorized stores, such as `st.shared.v4 %vec_reg,
// [%offset]`, where `%vec_reg` contains 4 registers that store consecutive data
// elements, the `stmatrix` instruction allows `N` registers to point to
// different locations within a tensor block.
//
// For instance, the `stmatrix %mat_reg, [%offset]` instruction on NVIDIA GPUs
// enables `%mat_reg` to store `N` elements that do not need to be consecutive.
// However, it is crucial that the starting address (`%offset`) is aligned to
// `N` * `elemBitWidth`. The `%offsets` for each thread are calculated based on
// the provided tensor encoding. Currently, this implementation supports only
// the NVIDIA MMAv3 encoding format.
//
// The `stmatrix.x4` instruction is a specific example that repeats the data
// layout four times for each warp. In this format, each `stmatrix.x4`
// instruction stores 8 16-bit elements per thread, resulting in a total of 8 *
// 32 = 256 elements per warp, or 16 * 16 elements per warp when distributed
// across four 8x8 tiles. Each thread’s `%offset` points to an address aligned
// with 8 * 16 bits. However, the values in `%mat_reg` are a vector of
// non-consecutive elements, composed of 4 pairs of consecutive elements. These
// matrix addresses are distributed as follows:
//
//              col[0-7]     col[8-15]
//   row[0-7]  lane[0-7]    lane[16-23]
//   row[8-15] lane[8-15]   lane[24-31]
//
// The matrix elements for thread 0 are distributed in the following pattern:
//
//           col0       col8
//   row0  reg[0-1]   reg[4-5]
//   row8  reg[2-3]   reg[6-7]
std::optional<LinearLayout> chooseStMatrixLayoutForRegToRegConversion(
    MLIRContext *ctx, RankedTensorType tensorTy, ArrayRef<unsigned> repShape,
    ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> order);
} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_IR_LINEARLAYOUTCONVERSIONS_H
