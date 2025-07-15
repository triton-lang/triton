#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

namespace {

using namespace mlir;
using namespace mlir::triton::gpu;

constexpr int kPtrBitWidth = 64;
struct ConvertLayoutOpUsingLinearLayoutsConversion
    : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
  const TargetInfoBase &targetInfo;

  // Set benefit to 2 so that this pattern applies before other convert-layout
  // conversions.  TODO(jlebar): Eventually we want this to be the only pattern.
  explicit ConvertLayoutOpUsingLinearLayoutsConversion(
      LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();

    const auto &shape = op.getType().getShape();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
    LinearLayout srcLayout = toLinearLayout(srcTy);
    LinearLayout dstLayout = toLinearLayout(dstTy);

    StringAttr kBlock = str_attr("block");
    StringAttr kWarp = str_attr("warp");
    StringAttr kLane = str_attr("lane");
    StringAttr kRegister = str_attr("register");

    assert(to_vector(conversion.getInDimNames()) ==
           to_vector(conversion.getOutDimNames()));
    auto dims = conversion.getInDimNames();
    if (llvm::is_contained(dims, kBlock)) {
      // Case 1: Transfer between values in different CTAs.
      //          This requires moving values through distributed shared memory.
      return rewriter.notifyMatchFailure(
          op, "NYI: Transfer between different CTAs");
    } else if (llvm::is_contained(dims, kWarp)) {
      // Case 2: Transfer between values in the same CTA, in which case we move
      //         values through shared memory.
      return transferWithinBlock(op, srcLayout, dstLayout, adaptor, rewriter);
    } else if (llvm::is_contained(dims, kLane)) {
      // Case 3. Transfer between values in the same warp, in which case we try
      //         to move values using warp shuffles, though if the pattern is
      //         complicated enough we may fall back to using shared memory
      if (auto decomposedCvt =
              getWarpLayoutConvertDecomposition(srcTy, dstTy)) {
        transferWithinWarp(op, *decomposedCvt, adaptor, rewriter);
        return success();
      }
      // TODO: Since data is only transferred within a warp over shared memory,
      // we should use `bar.warp.sync` instead of `barrier`, which will improve
      // latency when warps issue barriers on different cycles.
      return transferWithinBlock(op, srcLayout, dstLayout, adaptor, rewriter);
    } else if (llvm::is_contained(dims, kRegister)) {
      // Case 4. Transfer between values in the same thread, in which case we
      //         simply reorder the elements of adaptor.getSrc().
      return transferWithinThread(op, conversion, adaptor, rewriter);
    } else {
      // Cast 5. The two layouts are equivalent. We should probably remove
      // these in RemoveLayoutConversion.
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
  }

  LogicalResult
  transferWithinThread(ConvertLayoutOp op, const LinearLayout &conversion,
                       OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    StringAttr kRegister = str_attr("register");
    assert(!cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> outVals(conversion.getInDimSize(kRegister));
    for (int i = 0; i < outVals.size(); i++) {
      auto srcIdx = conversion.apply({{kRegister, i}}).begin()->second;
      outVals[i] = inVals[srcIdx];
    }
    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

  SmallVector<Value> transferWithinBlockSwizzlingImpl(
      Location loc, ConversionPatternRewriter &rewriter,
      const LinearLayout &srcLayout, const LinearLayout &dstLayout,
      ArrayRef<Value> inVals, Type llvmElemTy, Value smemBase) const {
    auto *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // We handle transformations recursively as they all need a preprocessing
    // and a postprocessing step.

    // Handle pointer types as 64-bit integers
    if (isa<LLVM::LLVMPointerType>(llvmElemTy)) {
      auto llvmElemTyPtr = i64_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(inVals, [&](Value v) {
        return b.ptrtoint(llvmElemTyPtr, v).getResult();
      }));
      auto outVals =
          transferWithinBlockSwizzlingImpl(loc, rewriter, srcLayout, dstLayout,
                                           newInVals, llvmElemTyPtr, smemBase);
      for (auto &v : outVals) {
        v = b.inttoptr(llvmElemTy, v);
      }
      return outVals;
    }

    // Handle sub-byte elements like i1
    if (llvmElemTy.getIntOrFloatBitWidth() < 8) {
      // Upcast to i8
      auto i8ElemTy = i8_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(
          inVals, [&](Value v) { return b.zext(i8ElemTy, v).getResult(); }));
      auto outVals = transferWithinBlockSwizzlingImpl(
          loc, rewriter, srcLayout, dstLayout, newInVals, i8ElemTy, smemBase);
      for (auto &v : outVals) {
        v = b.trunc(llvmElemTy, v);
      }
      return outVals;
    }

    // Remove broadcasting in src
    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    if (!removeBroadcastSrc.isIdentity()) {
      auto prmtSrc = removeBroadcastSrc.apply(srcLayout);
      auto newInVals = removeBroadcastSrc.apply(inVals);
      return transferWithinBlockSwizzlingImpl(loc, rewriter, prmtSrc, dstLayout,
                                              newInVals, llvmElemTy, smemBase);
    }

    // Remove broadcasting in dst
    auto removeBroadcastDst = actionRemoveBroadcastedRegs(dstLayout);
    if (!removeBroadcastDst.isIdentity()) {
      auto prmtDst = removeBroadcastDst.apply(dstLayout);
      auto outVals = transferWithinBlockSwizzlingImpl(
          loc, rewriter, srcLayout, prmtDst, inVals, llvmElemTy, smemBase);
      return broadcastAs(outVals, dstLayout);
    }

    // At this point we have a type that's at least 8-bit
    // and we don't have broadcasting in the registers
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto smem = optimalSwizzling(srcLayout, dstLayout, bitwidth);

    // Extract reps from smem
    auto kReg = str_attr("register");
    auto kReps = str_attr("reps");
    auto nReps = smem.getInDimSize(kReps);
    auto reps = LinearLayout::identity1D(nReps, kReg, kReps);

    auto totalStoreCvt = srcLayout.invertAndCompose(smem);
    auto totalLoadCvt = dstLayout.invertAndCompose(smem);

    // The permutation exists by construction of the reps dimension in
    // optimalSwizzling
    auto permStore =
        regPermForDivide(totalStoreCvt, reps, /*left=*/false).value();
    totalStoreCvt = permStore.apply(totalStoreCvt);
    auto permutedInVals = permStore.apply(inVals);
    auto permLoad =
        regPermForDivide(totalLoadCvt, reps, /*left=*/false).value();
    totalLoadCvt = permLoad.apply(totalLoadCvt);

    // Remove the reps and flatten into offset
    auto storeCvt = *divideRight(totalStoreCvt, reps);
    auto loadCvt = *divideRight(totalLoadCvt, reps);
    auto kOffset = str_attr("offset");
    storeCvt = storeCvt.reshapeOuts({{kOffset, storeCvt.getTotalOutDimSize()}});
    loadCvt = loadCvt.reshapeOuts({{kOffset, loadCvt.getTotalOutDimSize()}});

    auto tileSize = storeCvt.getInDimSize(kReg);

    assert(permutedInVals.size() == tileSize * nReps);
    SmallVector<Value> outVals;
    for (int i = 0; i < nReps; ++i) {
      if (i > 0)
        b.barrier();

      auto tileInVals =
          ArrayRef<Value>(permutedInVals).slice(i * tileSize, tileSize);
      // Store
      lowerLdStShared(loc, ctx, storeCvt, tileInVals, llvmElemTy, smemBase,
                      rewriter, targetInfo);
      b.barrier();
      // Load
      SmallVector<Value> tileOutVals = lowerLdStShared(
          loc, ctx, loadCvt, {}, llvmElemTy, smemBase, rewriter, targetInfo);
      llvm::append_range(outVals, tileOutVals);
    }

    // Undo the permLoad used to divideRight
    outVals = permLoad.inverse().apply(outVals);
    return outVals;
  }

  LogicalResult
  transferWithinBlockSwizzling(ConvertLayoutOp op, Value src,
                               ConversionPatternRewriter &rewriter) const {
    // Fallback for now to standard lowering if it can use stmatrix
    auto scratchConfig =
        getScratchConfigForCvt(op.getSrc().getType(), op.getType());
    bool isStMatrix = targetInfo.canUseStMatrix(
        op.getSrc().getType(), scratchConfig.repShape,
        scratchConfig.paddedRepShape, scratchConfig.order,
        /*swizzleByteSize=*/0);
    if (isStMatrix) {
      return failure();
    }

    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    // Remove the kBlock dimension from the layout as it's the identity in the
    // cvt
    auto srcLayout = toLinearLayout(srcTy);
    auto dstLayout = toLinearLayout(dstTy);
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    srcLayout = srcLayout.sublayout({kReg, kLane, kWarp},
                                    to_vector(srcLayout.getOutDimNames()));
    dstLayout = dstLayout.sublayout({kReg, kLane, kWarp},
                                    to_vector(dstLayout.getOutDimNames()));

    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto inVals = unpackLLElements(loc, src, rewriter);
    auto outVals = transferWithinBlockSwizzlingImpl(
        loc, rewriter, srcLayout, dstLayout, inVals, llvmElemTy, smemBase);

    Value result =
        packLLElements(loc, getTypeConverter(), outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult transferWithinBlock(ConvertLayoutOp op,
                                    const LinearLayout &srcLayout,
                                    const LinearLayout &dstLayout,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    assert(cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));

    // Try to use swizzling to implement the conversion
    // HACK Remove once AMD tests pass for the swizzling path
    if (targetInfo.isCuda() && succeeded(transferWithinBlockSwizzling(
                                   op, adaptor.getSrc(), rewriter))) {
      return success();
    }

    Value result = transferWithinBlockPadding(op, adaptor.getSrc(), targetInfo,
                                              getTypeConverter(), rewriter);

    rewriter.replaceOp(op, result);
    return success();
  }

  // Use warp shuffles to implement a layout conversion where data only needs to
  // be moved within warps.
  void transferWithinWarp(ConvertLayoutOp op,
                          DecomposedWarpConversion decomposed,
                          OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const;
};

} // namespace

void ConvertLayoutOpUsingLinearLayoutsConversion::transferWithinWarp(
    ConvertLayoutOp op, DecomposedWarpConversion decomposed, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  assert(!cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));
  auto [P1, Cp, P2inv, reducedP1, reducedP2] = std::move(decomposed);

  // Grab the source elements and prepare the outputs of just the shuffles.
  SmallVector<Value> srcValues =
      unpackLLElements(loc, adaptor.getSrc(), rewriter);
  SmallVector<Value> shflOuts(Cp.getInDimSize(kRegister));

  Value laneId = getLaneId(rewriter, loc);

  // Emit one shuffle per destination register.
  for (int i : llvm::seq(shflOuts.size())) {
    // 'Cp' maps a (dst_lane, dst_reg) -> (src_lane, src_reg), and we know that
    // for a register, it does not map to different registers in the same lane.
    // At the same time, for each register, P1 returns the source value index
    // to provide as the shuffle value.
    auto out = applyLinearLayout(loc, rewriter, P1,
                                 {{kRegister, b.i32_val(i)}, {kLane, laneId}});
    assert(out.size() == 1);
    Value srcRegIdx = out.front().second;
    // The size of the input lane dimension is the number of selects to emit.
    // TODO(jeff): For dtypes smaller than i32, we can use byte permutes and
    // shuffle multiple values at a time.
    Value shflSrc = b.undef(srcValues.front().getType());
    for (int j : llvm::seq(reducedP1.getInDimSize(kLane))) {
      int32_t check =
          reducedP1.apply({{kLane, j}, {kRegister, i}}).front().second;
      shflSrc = b.select(b.icmp_eq(srcRegIdx, b.i32_val(check)),
                         srcValues[check], shflSrc);
    }

    out = applyLinearLayout(loc, rewriter, Cp,
                            {{kRegister, b.i32_val(i)}, {kLane, laneId}});
    assert(out.size() == 1);
    Value shflIdx = out.front().second;
    shflOuts[i] = targetInfo.shuffleIdx(rewriter, loc, shflSrc, shflIdx);
  }

  // Finally, we just need to apply P2 to the shflOuts to permute the registers
  // into their final form. Use the same trick to reduce the number of emitted
  // selects.
  SmallVector<Value> results(shflOuts.size());
  for (int i : llvm::seq(results.size())) {
    Value result = b.undef(srcValues.front().getType());

    auto out = applyLinearLayout(loc, rewriter, P2inv,
                                 {{kRegister, b.i32_val(i)}, {kLane, laneId}});
    Value resultIdx = out.front().second;
    for (int j : llvm::seq(reducedP2.getInDimSize(kLane))) {
      int32_t check =
          reducedP2.apply({{kLane, j}, {kRegister, i}}).front().second;
      result = b.select(b.icmp_eq(resultIdx, b.i32_val(check)), shflOuts[check],
                        result);
    }
    results[i] = result;
  }

  Value result =
      packLLElements(loc, getTypeConverter(), results, rewriter, op.getType());
  rewriter.replaceOp(op, result);
}

void mlir::triton::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpUsingLinearLayoutsConversion>(
      typeConverter, targetInfo, benefit);
}
