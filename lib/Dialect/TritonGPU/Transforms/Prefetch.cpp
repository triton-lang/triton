//===----------------------------------------------------------------------===//
//
// This pass tries to prefetch operands (a and b) of tt.dot.
// Those ConvertLayoutOps will be lowered to shared memory loads.
//
// For example:
// %a: tensor<128x32xf16, #enc>
// scf.for %iv = ... iter_args(%a_arg = %a, ...) {
//   %d = tt.dot %a_arg, %b, %c
//   ...
//   scf.yield %a_next, ...
// }
//
// will be translated to
//
// %a: tensor<128x32xf16, #enc>
// %a_tmp = tensor.subview %a[0, 0] [128, 16]
// %a_prefetch = ttg.local_load %a_tmp
// scf.for %iv = ... iter_args(%a_buf = %a, ..., %a_prefetch_arg = %a_prefetch)
// {
//   %x = tt.dot %a_prefetch_arg, %b, %c
//   %a_tmp_rem = tensor.subview %a_buf[0, 16] [128, 16]
//   %a_prefetch_next = ttg.local_load %a_tmp_rem
//   ...
//   scf.yield %next_a, ..., %a_prefetch_next
// }
//===----------------------------------------------------------------------===//

#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritongpu-prefetch"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPREFETCH
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

SmallVector<RankedTensorType> typesBeforeSplitting;

// Helper function to split a value along a specific axis into numSlices pieces
// SplitOp only splits along the last dimension, so we need to transpose, reshape, split, reshape, and transpose back
// This function recursively splits in half until numSlices pieces are created
static SmallVector<Value> splitValueAlongAxis(Value input, int32_t numSlices, int axis, Location loc, OpBuilder &builder) {
  // Base case: if numSlices is 1, return the input as-is
  if (numSlices == 1) {
    return {input};
  }
  
  // Store the original encoding to convert back after splitting
  RankedTensorType originalInputType = cast<RankedTensorType>(input.getType());
  Attribute originalEncoding = originalInputType.getEncoding();
  
  // Lambda to perform a single binary split
  auto splitOnce = [&](Value val) -> std::pair<Value, Value> {
    //LDBG("\n\nsplitOnce(): a=" << axis);
    //LDBG("val: " << val);
    RankedTensorType inputType = cast<RankedTensorType>(val.getType());
    //LDBG("inputType: " << inputType);

    auto inputLL = toLinearLayout(inputType);
    //LDBG("inputLL: " << inputLL);

    auto shape = inputType.getShape();
    int rank = shape.size();
    
    // Reshape to split the target axis into [N/2, 2]
    SmallVector<int64_t> newShape;
    int splitDimPos = axis;  // Position of the "2" dimension after reshape
    for (int i = 0; i < rank; ++i) {
      if (i == axis) {
        newShape.push_back(2);
        newShape.push_back(shape[i] / 2);
      } else {
        newShape.push_back(shape[i]);
      }
    }
    //LDBG("newShape: " << newShape[0] << ", " << newShape[1] << ", " << newShape[2]);

    // Use ReshapeOp builder that infers encoding automatically
    // When rank changes, some encodings (like AMDMfmaEncodingAttr) may not support it,
    // so we need to handle encoding inference carefully
    Value reshaped = triton::ReshapeOp::create(builder, loc, newShape, val);
    //LDBG("reshaped: " << reshaped);
    auto reshapedLL = toLinearLayout(cast<RankedTensorType>(reshaped.getType()));
    //LDBG("reshapedLL: " << reshapedLL);    
    // Verify the ReshapeOp is valid
    auto reshapeOp = cast<triton::ReshapeOp>(reshaped.getDefiningOp());
    if (failed(reshapeOp.verify())) {
      reshapeOp.emitError("ReshapeOp verification failed in splitOnce");
      assert(false && "ReshapeOp verification failed");
    }
    
    // Permute to move the "2" dimension to last position if needed
    Value transposed = reshaped;
    int newRank = rank + 1;  // After reshape, we have one more dimension
    if (axis != newRank - 1) {
      //LDBG("transposing");
      SmallVector<int32_t> trans;
      for (int i = 0; i < newRank; ++i) {
        if (i != axis) trans.push_back(i);
      }
      trans.push_back(axis);  // Move "2" dimension to end
      //LDBG("trans: " << trans[0] << "," << trans[1] << "," << trans[2]);
      transposed = triton::TransOp::create(builder, loc, reshaped, trans);
      //LDBG("transposed: " << transposed);
      auto transposedLL = toLinearLayout(cast<RankedTensorType>(transposed.getType()));
      //LDBG("transposedLL: " << transposedLL);      
      // Verify the TransOp is valid
      auto transOp = cast<triton::TransOp>(transposed.getDefiningOp());
      if (failed(transOp.verify())) {
        transOp.emitError("TransOp verification failed in splitOnce");
        assert(false && "TransOp verification failed");
      }
    }

    // Split along the last dimension
    triton::SplitOp split = triton::SplitOp::create(builder, loc, transposed);
    Value left = split.getResult(0);
    Value right = split.getResult(1);
    //LDBG("left: " << left);
    auto leftLL = toLinearLayout(cast<RankedTensorType>(left.getType()));
    //LDBG("leftLL: " << leftLL);
    // Verify the SplitOp is valid
    auto splitOp = cast<triton::SplitOp>(split.getOperation());
    if (failed(splitOp.verify())) {
      splitOp.emitError("SplitOp verification failed in splitOnce");
      assert(false && "SplitOp verification failed");
    }
    
    // Convert back to original encoding (e.g. MfmaEncodingAttr) rather than LinearEncodingAttr so the dots will work
    auto leftType = cast<RankedTensorType>(left.getType());
    auto rightType = cast<RankedTensorType>(right.getType());
    
    // Create target types with original encoding
    auto leftTargetType = RankedTensorType::get(
        leftType.getShape(), leftType.getElementType(), originalEncoding);
    //LDBG("leftTargetType: " << leftTargetType);
    auto leftTargetTypeLL = toLinearLayout(cast<RankedTensorType>(leftTargetType));
    //LDBG("leftTargetTypeLL: " << leftTargetTypeLL);
    auto rightTargetType = RankedTensorType::get(
        rightType.getShape(), rightType.getElementType(), originalEncoding);
    
    // Convert from LinearEncodingAttr back to original encoding
    left = triton::gpu::ConvertLayoutOp::create(builder, loc, leftTargetType, left);
    right = triton::gpu::ConvertLayoutOp::create(builder, loc, rightTargetType, right);
    //LDBG("left after convert: " << left);    
    return {left, right};
  };
  
  // Iteratively split in half until we have numSlices pieces
  SmallVector<Value> tiles;
  tiles.push_back(input);
  int32_t currentCount = 1;
  while (currentCount < numSlices) {
    //LDBG("while " << currentCount << " < " << numSlices);
    RankedTensorType tileType = cast<RankedTensorType>(tiles[0].getType());
    //LDBG("tileType: " << tileType);
    auto tileTypeLL = toLinearLayout(tileType);
    //LDBG("tileTypeLL: " << tileTypeLL);
    typesBeforeSplitting.push_back(tileType);
    SmallVector<Value> nextTiles;
    for (Value tile : tiles) {
      auto [left, right] = splitOnce(tile);
      nextTiles.push_back(left);
      nextTiles.push_back(right);
    }
    tiles = std::move(nextTiles);
    currentCount *= 2;
  }
  //LDBG("typesBeforeSplitting.size(): " << typesBeforeSplitting.size());
  return tiles;
}

// Helper function to join tensors along a specific axis (inverse of splitAlongAxis)
// JoinOp only joins along the last dimension, so we need to join, transpose, and reshape
static Value joinValuesAlongAxis(SmallVector<Value> tiles, int axis, Location loc, OpBuilder &builder) {
  if (tiles.size() == 1) {
    return tiles[0];
  }
  
  // Lambda to perform a single binary join
  auto joinOnce = [&](Value left, Value right, RankedTensorType dstType) -> Value {
    //LDBG("\n\njoinOnce(): a=" << axis);
    //LDBG("left: " << left);
    //LDBG("right: " << right);

    auto leftType = cast<RankedTensorType>(left.getType());
    auto shape = leftType.getShape();
    int rank = shape.size();
    //LDBG("rank: " << rank);
    assert(axis < rank);

    // Join creates a new trailing dimension of size 2
    // 64x64 + 64x64 -> 64x64x2 (where 2 is fastest changing dim)
    Value joined = triton::JoinOp::create(builder, loc, left, right);
    //LDBG("joined: " << joined);
     auto joinedLL = toLinearLayout(cast<RankedTensorType>(joined.getType()));
    //LDBG("joinedLL: " << joinedLL);   
    // Verify the JoinOp is valid
    auto joinOp = cast<triton::JoinOp>(joined.getDefiningOp());
    if (failed(joinOp.verify())) {
      joinOp.emitError("JoinOp verification failed in joinOnce");
      assert(false && "JoinOp verification failed");
    }

    // Transpose 64x64x2 -> 2x64x64
    // for axis=0, trans=2, 0, 1
    // for axis=1, trans=0, 2, 1
    SmallVector<int32_t> trans(rank+1);
    //LDBG("trans.size(): " << trans.size());
    for (int j = 0; j < rank; ++j) {
      trans[j < axis ? j : j + 1] = j;
    }
    trans[axis] = rank;
    //LDBG("trans: " << trans[0] << "," << trans[1] << "," << trans[2]);
    Value transposed = triton::TransOp::create(builder, loc, joined, trans);
    //LDBG("transposed: " << transposed);
    auto transposedLL = toLinearLayout(cast<RankedTensorType>(transposed.getType()));
    //LDBG("transposedLL: " << transposedLL);    
    // Verify the TransOp is valid
    auto transOp = cast<triton::TransOp>(transposed.getDefiningOp());
    if (failed(transOp.verify())) {
      transOp.emitError("TransOp verification failed in joinOnce");
      assert(false && "TransOp verification failed");
    }

    // Reshape 2x64x64 -> 128x64
    auto transposedType = cast<RankedTensorType>(transposed.getType());
    auto transposedShape = transposedType.getShape();
    //LDBG("transposedShape.size(): " << transposedShape.size());
    SmallVector<int64_t> newShape(shape.begin(), shape.end());
    //LDBG("newShape.size(): " << newShape.size());
    newShape[axis] *= 2;
    //LDBG("newShape: " << newShape[0] << "," << newShape[1]);
    
    // Convert dstType (AMDMfmaEncodingAttr) to LinearEncodingAttr for the reshape.
    /* TODO(dtanner) There may be an error here. LLM says:
    The error is more likely from ReshapeOp, not ConvertLayoutOp.
Reasons:
The error format "Expected result encoding ... but was ..." matches verifyLayoutsAreEqual() called by ReshapeOp::verify() (as we saw earlier).
Both encodings are LinearEncodingAttr, but with different LinearLayouts (register bases reordered). This suggests a mismatch during ReshapeOp verification.
ConvertLayoutOp converts between encodings and doesn't verify structural equality like this.
What's happening:
linearType is created from dstType (AMDMfmaEncodingAttr) → LinearLayout A
transposed has LinearEncodingAttr → LinearLayout B
ReshapeOp infers the destination encoding from transposed → LinearLayout B
ReshapeOp::verify() compares LinearLayout A (expected) vs LinearLayout B (inferred) → mismatch
The issue: we're using the LinearLayout from dstType (AMDMfmaEncodingAttr), but ReshapeOp infers from transposed (LinearEncodingAttr), producing a different LinearLayout.
Fix: reshape the LinearLayout from transposed to the target shape, then create the LinearEncodingAttr from that, rather than using the LinearLayout from dstType. This ensures ReshapeOp infers the same encoding we provide.
    */
    auto linearEnc = triton::gpu::toLinearEncoding(dstType);
    //LDBG("linearEnc: " << linearEnc);
    auto linearType = RankedTensorType::get(
        dstType.getShape(), dstType.getElementType(), linearEnc);
    //LDBG("linearType: " << linearType);
    // First we Reshape with just the new shape; second we convert layout to the dstType encoding.
    Value reshaped = triton::ReshapeOp::create(builder, loc, newShape, transposed);
    //LDBG("reshaped: " << reshaped);
    auto reshapedLL = toLinearLayout(cast<RankedTensorType>(reshaped.getType()));
    //LDBG("reshapedLL: " << reshapedLL);
    // Verify the ReshapeOp is valid
    auto reshapeOp = cast<triton::ReshapeOp>(reshaped.getDefiningOp());
    if (failed(reshapeOp.verify())) {
      reshapeOp.emitError("ReshapeOp verification failed");
      assert(false && "ReshapeOp verification failed");
    }
    
    // Convert back to original encoding (AMDMfmaEncodingAttr)
    Value converted = triton::gpu::ConvertLayoutOp::create(builder, loc, dstType, reshaped);
    //LDBG("converted: " << converted);
    auto convertedLL = toLinearLayout(cast<RankedTensorType>(converted.getType()));
    //LDBG("convertedLL: " << convertedLL);
    return converted;
  };
  
  // Iteratively join pairs using log2 iterations
  while (tiles.size() > 1) {
    //LDBG("while " << tiles.size() << " > 1");
    RankedTensorType dstType = typesBeforeSplitting.pop_back_val();
    //LDBG("dstType: " << dstType);
    auto dstTypeLL = toLinearLayout(dstType);
    //LDBG("dstTypeLL: " << dstTypeLL);
    SmallVector<Value> nextTiles;
    for (size_t i = 0; i < tiles.size(); i += 2) {
      Value joined = joinOnce(tiles[i], tiles[i + 1], dstType);
      nextTiles.push_back(joined);
    }
    tiles = std::move(nextTiles);
  }
  
  return tiles[0];
}

class Prefetcher {
  /// cache the ForOp we are working on
  scf::ForOp forOp;
  /// cache the YieldOp of this ForOp
  scf::YieldOp yieldOp;
  ///
  // TODO: add a hook to infer prefetchWidthK
  unsigned prefetchWidthM = 64;
  unsigned prefetchWidthN = 64;
  unsigned prefetchWidthK = 32;
  unsigned kWidth = 8;

  /// dots to be prefetched
  SetVector<triton::DotOp> dots;
  /// dot => dot operand
  DenseMap<Value, Value> dot2aLoopArg;
  DenseMap<Value, Value> dot2aHeaderDef;
  DenseMap<Value, Value> dot2bLoopArg;
  DenseMap<Value, Value> dot2bHeaderDef;
  DenseMap<Value, Value> dot2aYield;
  DenseMap<Value, Value> dot2bYield;
  DenseMap<Value, SmallVector<Value>> dot2aVals;
  DenseMap<Value, SmallVector<Value>> dot2bVals;
  /// operand => defining
  DenseMap<Value, Value> operand2headPrefetch;

  LogicalResult isForOpOperand(Value v);

  FailureOr<Value> getAsyncWaitTokenForLocalLoad(Operation *cvt,
                                                 bool fromPriorIter,
                                                 OpBuilder &builder,
                                                 IRMapping *mapping = nullptr);

  Value generateLocalLoadSlice(Value v, unsigned opIdx, bool isPrologue,
                         Attribute dotEncoding, OpBuilder &builder,
                         std::optional<Value> asyncWaitToken = std::nullopt,
                         std::optional<int64_t> offsetM = std::nullopt,
                         std::optional<int64_t> shapeM = std::nullopt,
                         std::optional<int64_t> offsetN = std::nullopt,
                         std::optional<int64_t> shapeN = std::nullopt,
                         std::optional<int64_t> offsetK = std::nullopt,
                         std::optional<int64_t> shapeK = std::nullopt);

  void cloneElementwiseOps(Value &bRem, const SmallVector<Value> &vals,
                           OpBuilder &builder);

  Operation *generateDotsAndNonPrefetchingLocalLoads(triton::DotOp dot,
                                        Attribute dotEncoding,
                                        OpBuilder &builder,
                                        IRMapping &mapping,
                                        scf::ForOp newForOp);

  void generatePrefetchingLocalLoads(triton::DotOp dot, OpBuilder &builder,
                              IRMapping &mapping,
                              SmallVector<Value> &yieldValues);

public:
  Prefetcher() = delete;

  Prefetcher(scf::ForOp forOp) : forOp(forOp) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  LogicalResult initialize();

  void emitPrologue();

  scf::ForOp createNewForOp();
};

void Prefetcher::cloneElementwiseOps(Value &ret, const SmallVector<Value> &vals,
                                     OpBuilder &builder) {
  IRMapping mapping;
  mapping.map(vals[1], ret);
  for (int i = 2; i < vals.size(); i++) {
    Value v = vals[i];
    Value curr = builder.clone(*v.getDefiningOp(), mapping)->getResult(0);
    if (isa<RankedTensorType>(curr.getType())) {
      auto retType = RankedTensorType::get(
          cast<RankedTensorType>(ret.getType()).getShape(),
          cast<RankedTensorType>(curr.getType()).getElementType(),
          cast<RankedTensorType>(curr.getDefiningOp()->getOperand(0).getType())
              .getEncoding());
      curr.setType(retType);
    }
    mapping.map(v, curr);
  }
  if (vals.size() > 1)
    ret = mapping.lookup(vals.back());
}

// Bitmask that encodes instruction types for LLVM AMD scheduling hints.
enum InstructionKindMask {
  NONE =        0x0000,
  ALL_ALU =     0x0001,
  VALU =        0x0002,
  SALU =        0x0004,
  MFMA =        0x0008,
  ALL_VMEM =    0x0010,
  VMEM_READ =   0x0020,
  VMEM_WRITE =  0x0040,
  ALL_DS =      0x0080,
  DS_READ =     0x0100,
  DS_WRITE =    0x0200,
  TRANSCEND =   0x0400
};

void insertDotLocalLoadSchedBarrier(OpBuilder &builder, Location loc) {
  int32_t mask = 0
      | InstructionKindMask::VALU
      | InstructionKindMask::SALU
      | InstructionKindMask::ALL_VMEM
      | InstructionKindMask::VMEM_READ
      | InstructionKindMask::VMEM_WRITE
      | InstructionKindMask::TRANSCEND;
  ROCDL::SchedBarrier::create(builder, loc, mask);
}

// Generates all dots and first N-1 local_loads.
Operation *Prefetcher::generateDotsAndNonPrefetchingLocalLoads(triton::DotOp dot,
                                                  Attribute dotEncoding,
                                                  OpBuilder &builder,
                                                  IRMapping &mapping,
                                                  scf::ForOp newForOp) {
  // Get total dimensions from operands
  auto aType = dot.getA().getType();
  auto bType = dot.getB().getType();
  int64_t totalM = aType.getShape()[0];
  int64_t totalK = aType.getShape().back();
  int64_t totalN = bType.getShape().back();
  auto dotAttrs = dot->getAttrs();
  bool insertSchedBarriers = tools::getBoolEnv("TRITON_PREFETCH_INSERT_SCHED_BARRIER");
  if (insertSchedBarriers) {
    LDBG("Inserting sched barrier");
  }

  // Map from (M, N) offsets to accumulated dot Values
  DenseMap<std::pair<int32_t, int32_t>, Value> mnToDot;

  // Assert that dimensions are evenly divisible by prefetch widths
  assert(totalM % prefetchWidthM == 0 && "totalM must be divisible by prefetchWidthM");
  assert(totalN % prefetchWidthN == 0 && "totalN must be divisible by prefetchWidthN");
  assert(totalK % prefetchWidthK == 0 && "totalK must be divisible by prefetchWidthK");

  Value cOperand = mapping.lookup(dot.getC());

  int mAxis = 0;
  int nAxis = 1;

  // TODO(dtanner) whether m or n maps to axis=0, 1 may depend on isTransposed
  // Slice M.
  int32_t numSlicesM = totalM / prefetchWidthM;
  //LDBG("numSlicesM: " << numSlicesM);
  SmallVector<Value> mSlices = splitValueAlongAxis(cOperand, numSlicesM, mAxis, dot.getLoc(), builder);
  
  // Slice N.
  int32_t numSlicesN = totalN / prefetchWidthN;
  //LDBG("numSlicesN: " << numSlicesN);
  for (int32_t mIdx = 0; mIdx < numSlicesM; ++mIdx) {
    int32_t mOff = mIdx * prefetchWidthM;
    SmallVector<Value> mnSlices = splitValueAlongAxis(mSlices[mIdx], numSlicesN, nAxis, dot.getLoc(), builder);    
    // Store tiles in map
    for (int32_t nIdx = 0; nIdx < numSlicesN; ++nIdx) {
      int32_t nOff = nIdx * prefetchWidthN;
      mnToDot[{mOff, nOff}] = mnSlices[nIdx];
    }
  }
  // Generate dots[m, n, k] and local_loads[m, n, k] for all slices
  // Triple nested loop over K, M, N dimensions (K outermost)
  Operation *lastDotOp = nullptr;

  for (int32_t kOff = 0; kOff < totalK; kOff += prefetchWidthK) {
    DenseMap<int32_t, Value> aSlices;
    DenseMap<int32_t, Value> bSlices;

    // Outer loop is M dimension
    for (int32_t mOff = 0; mOff < totalM; mOff += prefetchWidthM) {
      for (int32_t nOff = 0; nOff < totalN; nOff += prefetchWidthN) {
        //LDBG("MNK: " << mOff << ", " << nOff << ", " << kOff);

        // local_loads should be inserted before the previous dot.
        if (lastDotOp)
          builder.setInsertionPoint(lastDotOp);

        Value aSlice;
        if (kOff == 0 && mOff == 0) {
          // Prefetched in prior iter.
          if (Value a = operand2headPrefetch.lookup(dot.getA()))
            aSlice = newForOp.getTiedLoopRegionIterArg(&*a.use_begin());
          else
            aSlice = mapping.lookup(dot.getA());
          aSlices[mOff] = aSlice;
          //LDBG("aSlice0: " << aSlice);
        } else {
          if (nOff == 0) {
            // Create local_load for new kOff>0 and nOff=0.
            FailureOr<Value> awtA = getAsyncWaitTokenForLocalLoad(
              dot2aVals[dot].back().getDefiningOp(), false, builder, &mapping);
            aSlice = generateLocalLoadSlice(
                mapping.lookup(dot2aLoopArg[dot]), 0, false, dotEncoding, builder,
                failed(awtA) ? std::nullopt : std::optional<Value>(*awtA),
                mOff, prefetchWidthM, std::nullopt, std::nullopt, kOff, prefetchWidthK);
            cloneElementwiseOps(aSlice, dot2aVals[dot], builder);
            aSlices[mOff] = aSlice;
            //LDBG("aSliceNew: " << aSlice);
          } else {
            // Need to reuse the data from nOff=0.
            aSlice = aSlices[mOff];
            //LDBG("aSliceReused: " << aSlice);
          }
        }

        Value bSlice;
        if (kOff == 0 && nOff == 0) {
          // Prefetched in prior iter.
          if (Value b = operand2headPrefetch.lookup(dot.getB()))
            bSlice = newForOp.getTiedLoopRegionIterArg(&*b.use_begin());
          else
            bSlice = mapping.lookup(dot.getB());
          bSlices[nOff] = bSlice;
          //LDBG("bSlice0: " << bSlice);
        } else {          
          if (mOff == 0) {
            // Create local_load for new kOff>0 and mOff=0.
          FailureOr<Value> awtB = getAsyncWaitTokenForLocalLoad(
              dot2bVals[dot].back().getDefiningOp(), false, builder, &mapping);
          bSlice = generateLocalLoadSlice(
              mapping.lookup(dot2bLoopArg[dot]), 1, false, dotEncoding, builder,
              failed(awtB) ? std::nullopt : std::optional<Value>(*awtB),
              std::nullopt, std::nullopt, nOff, prefetchWidthN, kOff, prefetchWidthK);
          cloneElementwiseOps(bSlice, dot2bVals[dot], builder);
          bSlices[nOff] = bSlice;
          //LDBG("bSliceNew: " << bSlice);
          } else {
            // Need to reuse the data from mOff=0.
            bSlice = bSlices[nOff];
            //LDBG("bSliceReused: " << bSlice);
          }
        }
        if (lastDotOp)
          builder.setInsertionPointAfter(lastDotOp);
        if (insertSchedBarriers) {
          insertDotLocalLoadSchedBarrier(builder, dot.getLoc());
        }
        // Get the accumulator for this (M,N) tile
        Value cSlice = mnToDot[{mOff, nOff}];
        auto dType = cast<RankedTensorType>(cSlice.getType());
        //LDBG("cSlice[" << mOff << "," << nOff << "]: " << cSlice);
        Operation *newDot = DotOp::create(builder,
            dot.getLoc(), dType,
            ValueRange{aSlice, bSlice, cSlice},
            dotAttrs);
        //LDBG("newDot[" << mOff << "," << nOff << "]: " << *newDot);
        //printContiguity(cast<RankedTensorType>(aSlice.getType()), cast<RankedTensorType>(bSlice.getType()));
        // Update the accumulator for this (M,N) tile
        mnToDot[{mOff, nOff}] = newDot->getResult(0);
        lastDotOp = newDot;
      }
    }
  }

  // Concatenate all M×N tiles back into a single tensor with original shape
  // Join N.
  SmallVector<Value> mJoins;
  for (int32_t mOff = 0; mOff < totalM; mOff += prefetchWidthM) {
    SmallVector<Value> mnSlices;
    for (int32_t nOff = 0; nOff < totalN; nOff += prefetchWidthN) {
      mnSlices.push_back(mnToDot[{mOff, nOff}]);
    }
    Value mJoin = joinValuesAlongAxis(mnSlices, nAxis, dot.getLoc(), builder);
    mJoins.push_back(mJoin);
  }
  // Join M.
  Value result = joinValuesAlongAxis(mJoins, mAxis, dot.getLoc(), builder);
  // Set insertion point to before the last dot for the prefetched local loads
  Operation *newOp = result.getDefiningOp();
  builder.setInsertionPoint(lastDotOp);
  return newOp;
}

// Generates the last local_loads which are local_load[k=0]'
void Prefetcher::generatePrefetchingLocalLoads(triton::DotOp dot, OpBuilder &builder,
                                        IRMapping &mapping,
                                        SmallVector<Value> &yieldValues) {
  Attribute dotEncoding = dot.getType().getEncoding();
  // Get async wait tokens from async_wait at end of prior iteration.
  FailureOr<Value> awtA = getAsyncWaitTokenForLocalLoad(
      dot2aVals[dot].back().getDefiningOp(), true, builder, &mapping);
  FailureOr<Value> awtB = getAsyncWaitTokenForLocalLoad(
      dot2bVals[dot].back().getDefiningOp(), true, builder, &mapping);
  Value aToYield = generateLocalLoadSlice(
      mapping.lookup(dot2aYield[dot]), 0, true, dotEncoding, builder,
      failed(awtA) ? std::nullopt : std::optional<Value>(*awtA));
  cloneElementwiseOps(aToYield, dot2aVals[dot], builder);
  yieldValues.push_back(aToYield);
  Value bToYield = generateLocalLoadSlice(
      mapping.lookup(dot2bYield[dot]), 1, true, dotEncoding, builder,
      failed(awtB) ? std::nullopt : std::optional<Value>(*awtB));
  cloneElementwiseOps(bToYield, dot2bVals[dot], builder);
  yieldValues.push_back(bToYield);
}

// Get async wait token (awt), if any, for new LocalLoad in newForOp
// based on old LocalLoad; args determine 3 cases where to
// get/create awt.
//
// Args
// - fromPriorIter, used for prefetching slice[0], means track the awt
//   through block args, yield and find it in the previous loop iteration.
// - mapping maps original forOp to newForOp, and is not used with
//   not in for loop, e.g. for emitPrologue.
//
// Case 0 - Prologue. awt is loop arg; returns init value before loop.
//  - fromPriorIter=false
//  - mapping=nullptr
// Case 1 - Slice[1,N-1]. awt is loop arg; returns same arg but mapped to
// newForLoop.
//  - fromPriorIter=false
//  - mapping=valid
// Case 2 - Slice[0] prefetched. awt comes from end of prior loop iteration.
//  - fromPriorIter=true
//  - mapping=valid
//
//  NOTE: fromPriorIter=true & mapping=nullptr is invalid combination.
FailureOr<Value> Prefetcher::getAsyncWaitTokenForLocalLoad(Operation *cvt,
                                                           bool fromPriorIter,
                                                           OpBuilder &builder,
                                                           IRMapping *mapping) {
  auto llOp = dyn_cast<triton::gpu::LocalLoadOp>(cvt);
  if (!llOp)
    return failure();
  if (llOp->getNumOperands() != 2)
    return failure();
  Value awt = llOp->getOperand(1);
  if (!isa<AsyncTokenType>(awt.getType()))
    return failure();

  if (!fromPriorIter) {
    if (!mapping) {
      // Case 0: return async wait token in prologue.
      if (mlir::BlockArgument loopArg =
              dyn_cast<mlir::BlockArgument>(awt)) {
        unsigned argIdx =
            loopArg.getArgNumber() - forOp.getNumInductionVars();
        Value initAwt = forOp.getInitArgs()[argIdx];
        return initAwt;
      } else {
        assert(false || "Expected async wait token to be loop arg.");
        return failure();
      }
      return awt;
    } else {
      // Case 1: return new async wait token from for(args) for
      // LocalLoad[1, N-1].
      return mapping->lookup(awt);
    }
  }
  assert(mapping);
  assert(fromPriorIter);


  mlir::BlockArgument loopArg = dyn_cast<mlir::BlockArgument>(awt);
  if (!loopArg) {
    assert(
      false ||
      "fromPriorIter specified but awt isn't a loop arg.");
    return failure();
  }

  // Case 2: return new async wait token from end of prior iteration,
  // this occurs for the prefetching LocalLoads at the end of the loop;
  // which may or may not have been created yet i.e. is in mapping.
  // Note: awt may already be in mapping for two reasons,
  // (a) it is a duplicate of async_wait created below,
  // (b) associated async_wait was already created previously in new loop
  // even though want prior iter of it. Now we want to wrap around the loop
  // body and find this token in the previous iteration because it was
  // prefetched.
  unsigned argIdx = loopArg.getArgNumber() - forOp.getNumInductionVars();
  Value initAwt = forOp.getInitArgs()[argIdx];
  Value yieldedAwt = yieldOp.getOperand(argIdx);
  if (mapping->contains(yieldedAwt))
    return mapping->lookup(yieldedAwt);

  // Want awt fromPriorIter, but it isn't in map yet because the async_wait op
  // hasn't been visited yet, so create and place in mapping.
  LDBG("Case 2 yieldedAwt not yet in map");
  auto awOp = yieldedAwt.getDefiningOp();
  // Create new async_wait op in new loop
  Operation *newAwOp = builder.clone(*awOp, *mapping);
  for (unsigned dstIdx : llvm::seq(unsigned(0), awOp->getNumResults()))
    mapping->map(awOp->getResult(dstIdx), newAwOp->getResult(dstIdx));
  return newAwOp->getResult(0);
}

Value Prefetcher::generateLocalLoadSlice(Value v, unsigned opIdx, bool isPrologue,
                                   Attribute dotEncoding, OpBuilder &builder,
                                   std::optional<Value> asyncWaitToken,
                                   std::optional<int64_t> offsetM,
                                   std::optional<int64_t> shapeM,
                                   std::optional<int64_t> offsetN,
                                   std::optional<int64_t> shapeN,
                                   std::optional<int64_t> offsetK,
                                   std::optional<int64_t> shapeK) {
  // opIdx: 0 => a, 1 => b
  auto type = cast<triton::gpu::MemDescType>(v.getType());
  SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
  auto rank = shape.size();
  SmallVector<int32_t> offset(rank, 0);
  Type elementType = type.getElementType();

  // For operand A (opIdx=0): shape is [M, K], so mIdx=0, kIdx=1
  // For operand B (opIdx=1): shape is [K, N], so kIdx=0, nIdx=1
  int64_t mIdx = 0;  // M dimension index (only for operand A)
  int64_t nIdx = 1;  // N dimension index (only for operand B)
  int64_t kIdx = opIdx == 0 ? rank - 1 : rank - 2;

  // Set default M dimension handling (operand A only)
  if (opIdx == 0) {
    offset[mIdx] = isPrologue ? 0 : prefetchWidthM;
    shape[mIdx] = isPrologue ? prefetchWidthM : (shape[mIdx] - prefetchWidthM);
    if (shapeM)
      shape[mIdx] = *shapeM;
    if (offsetM)
      offset[mIdx] = *offsetM;
  }

  // Set default N dimension handling (operand B only)
  if (opIdx == 1) {
    offset[nIdx] = isPrologue ? 0 : prefetchWidthN;
    shape[nIdx] = isPrologue ? prefetchWidthN : (shape[nIdx] - prefetchWidthN);
    if (shapeN)
      shape[nIdx] = *shapeN;
    if (offsetN)
      offset[nIdx] = *offsetN;
  }

  // Set K dimension handling (both operands)
  offset[kIdx] = isPrologue ? 0 : prefetchWidthK;
  shape[kIdx] = isPrologue ? prefetchWidthK : (shape[kIdx] - prefetchWidthK);

  if (shapeK)
    shape[kIdx] = *shapeK;
  if (offsetK)
    offset[kIdx] = *offsetK;

  Value newSmem = triton::gpu::MemDescSubsliceOp::create(
      builder, v.getLoc(),
      triton::gpu::MemDescType::get(
          shape, elementType, type.getEncoding(), type.getMemorySpace(),
          type.getMutableMemory(), type.getAllocShape()),
      v, offset);
  auto dotOperandEnc = triton::gpu::DotOperandEncodingAttr::get(
      builder.getContext(), opIdx, dotEncoding, kWidth);
  Value prefetchSlice;
  if (asyncWaitToken) {
    prefetchSlice = triton::gpu::LocalLoadOp::create(
        builder, v.getLoc(),
        RankedTensorType::get(shape, elementType, dotOperandEnc), newSmem,
        *asyncWaitToken);
  } else {
    prefetchSlice = triton::gpu::LocalLoadOp::create(
        builder, v.getLoc(),
        RankedTensorType::get(shape, elementType, dotOperandEnc), newSmem);
  }

  return prefetchSlice;
}

LogicalResult Prefetcher::initialize() {
  Block *loop = forOp.getBody();
  
  auto getEncoding = [](Value v) {
    return cast<TensorOrMemDesc>(v.getType()).getEncoding();
  };

  SmallVector<triton::DotOp> dotsInFor;
  for (Operation &op : *loop)
    if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
      // Only accepts dotOps encoded as Nvidia MMA v2 or AMD MFMA
      auto dstMmaEnc =
          dyn_cast<NvidiaMmaEncodingAttr>(getEncoding(dotOp.getResult()));
      auto dstMfmaEnc =
          dyn_cast<AMDMfmaEncodingAttr>(getEncoding(dotOp.getResult()));
      auto dstWmmaEnc =
          dyn_cast<AMDWmmaEncodingAttr>(getEncoding(dotOp.getResult()));
      if (!dstMfmaEnc && (!dstMmaEnc || dstMmaEnc.getVersionMajor() != 2) &&
          !dstWmmaEnc)
        // Don't rewrite if any other type is found.
        return failure();
      dotsInFor.push_back(dotOp);
    }

  if (dotsInFor.empty())
    return failure();

  // TODO: segfault (original for still has uses)
  // when used in flash attention that has 2 dots in the loop
  if (dotsInFor.size() > 1)
    return failure();

  // returns source of cvt
  auto getPrefetchSrc = [](Value v) -> SmallVector<Value> {
    // walk back to conversion
    Operation *op = v.getDefiningOp();
    bool foundConvertFromShared = false;
    SmallVector<Value> rets;
    rets.push_back(op->getResult(0));
    // LDBG("Prefetch src: " << *op);
    while (op) {
      if (!op->getResult(0).hasOneUse())
        break;
      rets.push_back(op->getOperand(0));
      if (auto cvt = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
        // NYI for other encodings, for example if we have transpose
        // in the chain
        if (isa<DotOperandEncodingAttr>(cvt.getType().getEncoding()))
          foundConvertFromShared = true;
        break;
      }
      op = op->getOperand(0).getDefiningOp();
      if (op)
        LDBG("op: " << *op);
    }
    std::reverse(rets.begin(), rets.end());

    if (foundConvertFromShared)
      return rets;
    return {};
  };

  auto getIncomingOp = [this](Value v) -> Value {
    if (auto arg = mlir::dyn_cast<BlockArgument>(v))
      if (arg.getOwner()->getParentOp() == forOp.getOperation())
        return forOp.getTiedLoopInit(arg)->get();
    return Value();
  };

  auto getYieldOperand = [this](Value v) -> Value {
    auto arg = mlir::cast<BlockArgument>(v);
    unsigned yieldIdx = arg.getArgNumber() - forOp.getNumInductionVars();
    return yieldOp.getOperand(yieldIdx);
  };

  for (triton::DotOp dot : dotsInFor) {
    auto aOpd = dot.getA();
    auto bOpd = dot.getB();
    auto aType = aOpd.getType();
    auto bType = bOpd.getType();
    auto dType = cast<RankedTensorType>(dot.getResult().getType());
    auto aEnc =
        mlir::cast<triton::gpu::DotOperandEncodingAttr>(aType.getEncoding());
    auto bEnc =
        mlir::cast<triton::gpu::DotOperandEncodingAttr>(bType.getEncoding());
    assert(aEnc.getKWidth() == bEnc.getKWidth());
    kWidth = aEnc.getKWidth();
    LDBG("kWidth: " << kWidth);

    auto transOp = [&](Operation *op, int opdIdx) -> bool {
      if (auto localLoad = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
        auto srcType = localLoad.getSrc().getType();
        auto order = getOrder(srcType);
        return (order[0] == opdIdx);
      }
      return true;
    };
    bool transA = transOp(aOpd.getDefiningOp(), 0);
    bool transB = transOp(bOpd.getDefiningOp(), 1);
    LDBG("transA: " << transA);
    LDBG("transB: " << transB);

    // Calculate prefetch widths
    unsigned elementWidthA = aType.getElementTypeBitWidth();
    unsigned elementWidthB = bType.getElementTypeBitWidth();

    // Get sizes for all three dimensions
    unsigned mSize = aType.getShape()[0];  // M dimension from operand A
    unsigned nSize = bType.getShape().back();  // N dimension from operand B
    unsigned kSize = aType.getShape().back();  // K dimension
    LDBG("size: " << mSize << "x" << nSize << "x" << kSize);

    //auto shapePerCta = getShapePerCTA(dType);
    //LDBG("shapePerCta: " << shapePerCta[0] << "x" << shapePerCta[1]);

    // Larger prefetch widths means more mfmas in a dot-tile, more prefetching and fewer slices.
    // This determines how to slice a dot into sub-dots. We want to prefetch the least ammount possible,
    // therefore we calculate the smallest tile possible.
    // Transposing A or B means not wanting to slice the respective dimension, so we only slice the other.
    // If we want to bring in more logic, then depending on aspect ratio of warp tile, then that will
    // change whether we want to slice axis 0 or 1 first, and what we want this rectangularity to look like.
    auto prefetchWidthAMD = [&](ArrayRef<unsigned> instrShape, ArrayRef<unsigned> warpsPerCta, unsigned numInsts) -> std::tuple<unsigned, unsigned, unsigned> {
      LDBG("instrShape: " << instrShape[0] << "x" << instrShape[1] << "x" << instrShape[2]);
      LDBG("warpsPerCta: " << warpsPerCta[0] << "x" << warpsPerCta[1]);
      LDBG("TotalInsts: " << mSize / (instrShape[0]*warpsPerCta[0])
          << "x" << nSize / (instrShape[1]*warpsPerCta[1])
          << "x" << kSize / instrShape[2] << " (" << numInsts << ")");

      // TODO(dtanner) need better logic here because the directory doesn't use target info.
      // Get Transpose shape from target info.
      //auto arch = getAMDArch(forOp->getParentOfType<ModuleOp>());
      //triton::AMD::TargetInfo targetInfo(arch ? arch->str() : "");
      //const auto isaFamily = targetInfo.getISAFamily();
      //unsigned minTransposeWidth =
      //  isaFamily == AMD::ISAFamily::GFX1250 ? 128 :
      //  isaFamily == AMD::ISAFamily::CDNA4 ? 64 :
      //  16;
      unsigned maxM = mSize / (instrShape[0] * warpsPerCta[0]);
      unsigned maxN = nSize / (instrShape[1] * warpsPerCta[1]);
      unsigned maxK = kSize / (instrShape[2]);
      unsigned minTransposeWidth = 64;
      unsigned m = 1, n = 1, k = 1;
      if (transA) {
        // M, K must be at least minTransposeWidth for lowerDsReadTr
        m = std::max<unsigned>(1, minTransposeWidth / instrShape[0]);
        k = std::max<unsigned>(1, minTransposeWidth / instrShape[2]);
      }
      if (transB) {
        // N, K must be at least minTransposeWidth for lowerDsReadTr
        n = std::max<unsigned>(1, minTransposeWidth / instrShape[1]);
        k = std::max<unsigned>(1, minTransposeWidth / instrShape[2]);
      }
      numInsts /= (m*n*k);
      // At this point, we've made the tiles the minimum size needed for transposing to work well,
      // so we can keep increasing their size to reach the desired number of instructions.
      LDBG("instr tile m: " << m << ", n: " << n << ", k: " << k);

      // Square-ish tile of num mma instructions; want m >= n because splitting m first (fat rows).
      while (numInsts > 1) {
        bool preferSquare = false;
        if ((m <= n || !preferSquare) && m < maxM && !transA) {
          m *= 2;
        } else if (n < maxN) {
          n *= 2;
        } else if (k < maxK) {
          k *= 2;
        } else {
          // Want to prefetch more but tile already same size as dot.
          break;
        }
        numInsts /= 2;
      }
      // By this point our tile is at lease the size needed for transposing,
      // and has at least the number of instructions.
      LDBG("instr tile m: " << m << ", n: " << n << ", k: " << k);
      // override for testing via environment variables
      unsigned mOverride = tools::getIntEnv("TRITON_PREFETCH_M_OVERRIDE");
      unsigned nOverride = tools::getIntEnv("TRITON_PREFETCH_N_OVERRIDE");
      unsigned kOverride = tools::getIntEnv("TRITON_PREFETCH_K_OVERRIDE");
      if (mOverride > 0) {
        m = mOverride;
      }
      if (nOverride > 0) {
        n = nOverride;
      }
      if (kOverride > 0) {
        k = kOverride;
      }
      LDBG("instr tile m: " << m << ", n: " << n << ", k: " << k);
      m *= instrShape[0]*warpsPerCta[0];
      n *= instrShape[1]*warpsPerCta[1];
      k *= instrShape[2];
      return {m, n, k};
    };

      
    // Query contiguity properties from encodings
    //printContiguity(aType, bType);
    

    // Get the dot result encoding to determine instruction dimensions
    Attribute dotEncoding = dot.getType().getEncoding();
    if (auto mfmaEnc = dyn_cast<AMDMfmaEncodingAttr>(dotEncoding)) {
      unsigned numInsts = 4;
      auto [m, n, k] = prefetchWidthAMD(mfmaEnc.getInstrShape(), mfmaEnc.getWarpsPerCTA(), numInsts);
      prefetchWidthM = m;
      prefetchWidthN = n;
      prefetchWidthK = k;
    } else if (auto wmmaEnc = dyn_cast<AMDWmmaEncodingAttr>(dotEncoding)) {
      unsigned numInsts = 8;
      auto warpsPerCTA = getWarpsPerCTA(wmmaEnc, dType.getShape());
      auto [m, n, k] = prefetchWidthAMD(wmmaEnc.getInstrShape(), warpsPerCTA, numInsts);
      prefetchWidthM = m;
      prefetchWidthN = n;
      prefetchWidthK = k;
    } else if (auto mmaEnc = dyn_cast<NvidiaMmaEncodingAttr>(dotEncoding)) {
      // For NVIDIA MMA, instruction shape depends on version
      // MMAv2: typically 16x8 or similar
      auto instrShape = mmaEnc.getInstrShape();
      auto instrM = instrShape[0];
      auto instrN = instrShape[1];
      // K dimension for MMA is determined by kWidth
      auto instrK = kWidth > 0 ? kWidth : 16;
      // K dimension width: Use 8x instruction K width for better tensor core utilization
      if (kWidth == 0)
        prefetchWidthK = 256 / elementWidthA;
      else
        prefetchWidthK = 8 * kWidth;
      // Prefetch whole MxN tile
      prefetchWidthM = mSize;
      prefetchWidthN = nSize;
      if (kSize < prefetchWidthK)
        continue;
    }

    // Can't prefetch MORE than the tile size
    prefetchWidthM = std::min<unsigned>(prefetchWidthM, mSize);
    prefetchWidthN = std::min<unsigned>(prefetchWidthN, nSize);
    prefetchWidthK = std::min<unsigned>(prefetchWidthK, kSize);
    LDBG("prefetchWidths: " << prefetchWidthM << "x" << prefetchWidthN << "x" << prefetchWidthK);
    LDBG("NumSlices: " << mSize / prefetchWidthM << "x" << nSize / prefetchWidthN << "x" << kSize / prefetchWidthK);
    auto aVals = getPrefetchSrc(dot.getA());
    auto bVals = getPrefetchSrc(dot.getB());

    if (aVals.size() && bVals.size()) {
      Value aSmem = aVals.front();
      Value bSmem = bVals.front();
      Value aHeaderDef = getIncomingOp(aSmem);
      Value bHeaderDef = getIncomingOp(bSmem);
      LDBG("aHeaderDef: " << aHeaderDef);
      LDBG("bHeaderDef: " << bHeaderDef);
      // Only prefetch loop arg
      if (aHeaderDef && bHeaderDef) {
        dots.insert(dot);
        dot2aVals[dot] = aVals;
        dot2bVals[dot] = bVals;
        dot2aHeaderDef[dot] = aHeaderDef;
        dot2bHeaderDef[dot] = bHeaderDef;
        dot2aLoopArg[dot] = aSmem;
        dot2bLoopArg[dot] = bSmem;
        dot2aYield[dot] = getYieldOperand(aSmem);
        dot2bYield[dot] = getYieldOperand(bSmem);
      }
    }
  }

  return success();
}

void Prefetcher::emitPrologue() {
  OpBuilder builder(forOp);

  for (triton::DotOp dot : dots) {
    FailureOr<Value> awtA = getAsyncWaitTokenForLocalLoad(
        dot2aVals[dot].back().getDefiningOp(), false, builder);
    FailureOr<Value> awtB = getAsyncWaitTokenForLocalLoad(
        dot2bVals[dot].back().getDefiningOp(), false, builder);
    Attribute dotEncoding = dot.getType().getEncoding();
    Value aPrefetched = generateLocalLoadSlice(
        dot2aHeaderDef[dot], 0, true, dotEncoding, builder,
        failed(awtA) ? std::nullopt : std::optional<Value>(*awtA));
    cloneElementwiseOps(aPrefetched, dot2aVals[dot], builder);
    Value bPrefetched = generateLocalLoadSlice(
        dot2bHeaderDef[dot], 1, true, dotEncoding, builder,
        failed(awtB) ? std::nullopt : std::optional<Value>(*awtB));
    cloneElementwiseOps(bPrefetched, dot2bVals[dot], builder);

    operand2headPrefetch[dot.getA()] = aPrefetched;
    operand2headPrefetch[dot.getB()] = bPrefetched;
  }
}

scf::ForOp Prefetcher::createNewForOp() {
  OpBuilder builder(forOp);

  SmallVector<Value> loopArgs;
  for (auto v : forOp.getInitArgs())
    loopArgs.push_back(v);
  for (triton::DotOp dot : dots) {
    loopArgs.push_back(operand2headPrefetch[dot.getA()]);
    loopArgs.push_back(operand2headPrefetch[dot.getB()]);
  }

  auto newForOp =
      scf::ForOp::create(builder, forOp.getLoc(), forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), loopArgs);

  builder.setInsertionPointToStart(newForOp.getBody());
  IRMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  // The insertion point should be placed before the yield op
  auto setInsertionPointBeforeYield = [](OpBuilder &builder,
                                         scf::ForOp newForOp) {
    if (newForOp.getBody()->mightHaveTerminator()) {
      builder.setInsertionPoint(newForOp.getBody()->getTerminator());
    } else {
      builder.setInsertionPointToEnd(newForOp.getBody());
    }
  };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    // If we're currently trying to sink a prefetched dot, we need to stop
    // sinking it (by resetting the insertion point to the end) if we find
    // control flow, or anything that depends on the dot op.
    if (op.getNumRegions() > 0) {
      setInsertionPointBeforeYield(builder, newForOp);
    }
    for (auto operand : op.getOperands()) {
      if (auto def = operand.getDefiningOp()) {
        auto dot = dyn_cast<triton::DotOp>(def);
        if (dot && dots.contains(dot)) {
          setInsertionPointBeforeYield(builder, newForOp);
        }
      }
    }
    Operation *newOp = builder.clone(op, mapping);
    auto dot = dyn_cast<triton::DotOp>(&op);
    if (dot && dots.contains(dot)) {
      Attribute dotEncoding = dot.getType().getEncoding();
      newOp = generateDotsAndNonPrefetchingLocalLoads(dot, dotEncoding, builder, mapping,
                                         newForOp);
    }
    // update mapping of results
    for (unsigned dstIdx : llvm::seq(unsigned(0), op.getNumResults()))
      mapping.map(op.getResult(dstIdx), newOp->getResult(dstIdx));
  }

  // prefetch next iteration
  SmallVector<Value> yieldValues;
  for (Value v : forOp.getBody()->getTerminator()->getOperands())
    yieldValues.push_back(mapping.lookupOrDefault(v));
  for (triton::DotOp dot : dots) {
    generatePrefetchingLocalLoads(dot, builder, mapping, yieldValues);
  }
  // Update ops of yield
  builder.setInsertionPointToEnd(newForOp.getBody());
  if (!yieldValues.empty())
    scf::YieldOp::create(builder, yieldOp.getLoc(), yieldValues);
  return newForOp;
}

} // anonymous namespace

struct PrefetchPass : public impl::TritonGPUPrefetchBase<PrefetchPass> {
  void runOnOperation() override {
    LDBG("PrefetchPass");
    // Canonicalize convert ops to make the pattern matching easier.
    RewritePatternSet cleanUpPatterns(&getContext());
    triton::gpu::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                              &getContext());
    if (mlir::applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }
    getOperation()->walk([&](scf::ForOp forOp) {
      Prefetcher prefetcher(forOp);

      if (prefetcher.initialize().failed())
        return;

      prefetcher.emitPrologue();

      scf::ForOp newForOp = prefetcher.createNewForOp();
      //LDBG("newForOp: " << newForOp);

      // replace the original loop
      for (unsigned i = 0; i < forOp->getNumResults(); ++i)
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      forOp->erase();
      LDBG("PrefetchPass - Succeeded");
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
