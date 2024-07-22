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

// Chooses a good shared layout for use when converting between the two given
// register layouts.
//
// src and dst must have the following dimensions.
//   input: [register, lane, warp, block]
//   output: [dimi..] and [dimj..] where i and j are permutations over 0..N.
//
// The result has the following dimensions.
//   input: [offset, block, iteration]
//   output: matches src's output dims, so [dimi..].
//
// Note that a regular shared memory layout only has [offset, block] input dims.
// The `iteration` dimension represents the fact that we can do a shmem transfer
// in multiple stages to reduce the total amount of shmem needed.
//
// For example, imagine we have eight 32-bit registers per thread in src and
// dst.  We could write all eight at once to shmem and then read them all back
// in, but that requires a lot of shmem.  At the other extreme, we could write
// one register to shmem and read it back in before writing the next one.  That
// requires 8x less shmem but may prevent us from doing vectorized memory
// accesses.
//
// As a compromise, this function splits the transfer so that we do one
// vectorized load/store per iteration.  (In the example, we'd try to do two
// stages of size 4 32-bit values each, assuming the src/dst layouts allow us to
// vectorize the stores or loads as 4xi32.)
LinearLayout chooseShemLayoutForRegToRegConversion(const LinearLayout &src,
                                                   const LinearLayout &dst,
                                                   int maxVecElems);

// The legacy version of the linear layout conversion function.  We determined
// the intermediate shared memory needed for a register-to-register conversion
// using a legacy heuristic (i.e., repShape), which uses the maximum accessing
// size of each dimension from srcLayout and dstLayout.   See Allocation.cpp for
// details.  Then, we construct an intermediate linear layout representing the
// shared memory -> tensor element index mapping for entire src and dst tensors.
// The pesudo code of layout conversion is as follows:
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
LinearLayout
chooseShemLayoutForRegToRegConversion(MLIRContext *ctx,
                                      ArrayRef<unsigned> tensorShape,
                                      ArrayRef<unsigned> repShape);
} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_IR_LINEARLAYOUTCONVERSIONS_H
