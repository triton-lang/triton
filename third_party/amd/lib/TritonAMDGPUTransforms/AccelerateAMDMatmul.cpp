#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace {
using triton::AMD::ISAFamily;

int getMfmaVersion(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::CDNA1:
    return 1;
  case ISAFamily::CDNA2:
    return 2;
  case ISAFamily::CDNA3:
    return 3;
  default:
    break;
  }
  return 0;
}

int getWmmaVersion(StringRef archGen) {
  if (archGen.contains("gfx11"))
    return 1;
  if (archGen.contains("gfx12"))
    return 2;
  return 0;
}

SmallVector<unsigned, 2> warpsPerTile(tt::DotOp dotOp,
                                      const ArrayRef<int64_t> shape,
                                      int numWarps,
                                      SmallVector<int64_t, 2> shapePerWarp) {
  auto rank = shape.size();
  // Early exit for batched matmul
  if (rank == 3)
    return {(unsigned)numWarps, 1, 1};

  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  ForwardSliceOptions fwdOpt;
  fwdOpt.filter = filter;
  BackwardSliceOptions bwdOpt;
  bwdOpt.omitBlockArguments = true;
  bwdOpt.filter = filter;
  auto slices = getSlice(dotOp, bwdOpt, fwdOpt);
  for (Operation *op : slices)
    if (isa<tt::DotOp>(op) && (op != dotOp))
      return {(unsigned)numWarps, 1};

  SmallVector<int64_t, 2> tensorShape = {shape[0], shape[1]};
  SmallVector<unsigned, 2> ret = {1, 1};
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (tensorShape[0] / (shapePerWarp[0] * 2) / ret[0] >=
        tensorShape[1] / shapePerWarp[1] / ret[1]) {
      if (ret[0] < tensorShape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);

  if (ret[1] * shapePerWarp[1] > tensorShape[1]) {
    return {ret[1], ret[0]};
  }

  return ret;
}

SmallVector<unsigned, 2>
warpsPerTileMFMA(tt::DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps,
                 SmallVector<int64_t, 2> shapePerWarp) {
  return warpsPerTile(dotOp, shape, numWarps, shapePerWarp);
}

SmallVector<unsigned, 2>
warpsPerTileWMMA(tt::DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps) {
  return warpsPerTile(dotOp, shape, numWarps,
                      {ttg::AMDWmmaEncodingAttr::getMNKDimPerInstr()[0],
                       ttg::AMDWmmaEncodingAttr::getMNKDimPerInstr()[1]});
}

using OperandTypesVector = SmallVector<Type, 4>;
OperandTypesVector
selectMatrixCoreOperandTypes(tt::DotOp dot,
                             ArrayRef<OperandTypesVector> applicableTypes) {
  SmallVector<Value> dotOperands = {dot.getA(), dot.getB(), dot.getC(),
                                    dot.getD()};
  OperandTypesVector initElemTypes;
  llvm::transform(dotOperands, std::back_inserter(initElemTypes), [](Value v) {
    return cast<RankedTensorType>(v.getType()).getElementType();
  });

  // Use simple costmodel to define optimal set of the dot operands.
  // Most expensive - accuracy loss conversions:
  //   - any larger type -> any smaller type;
  //   - float -> int;
  //   - int -> float (not supported for now);
  //   - signed int -> unsigned int;
  //   - unsigned int -> signed int with same or less size.
  // They are never performed, better to use FMA.
  // Supported conversion for now costs `1`, no conversion costs `0`.
  // The model could be improved in the future. For example taken into account
  // chain dot could be detected and result conversion score is decreased.
  int maxConvertCost =
      std::numeric_limits<int32_t>::max() / applicableTypes.front().size();
  auto calcConvertCost = [&](Type fromTy, Type toTy) -> int32_t {
    if (fromTy == toTy)
      return 0;

    // Skip conversion between int and float. Int16/int32 cases are lowered to
    // FMA.
    if (fromTy.isIntOrIndex() != toTy.isIntOrIndex())
      return maxConvertCost;

    if (fromTy.isIntOrIndex() && toTy.isIntOrIndex() &&
        fromTy.isUnsignedInteger() != toTy.isUnsignedInteger())
      return fromTy.isUnsignedInteger() && fromTy.getIntOrFloatBitWidth() <
                                               toTy.getIntOrFloatBitWidth()
                 ? 1
                 : maxConvertCost;

    return fromTy.getIntOrFloatBitWidth() <= toTy.getIntOrFloatBitWidth()
               ? 1
               : maxConvertCost;
  };
  auto minCost = maxConvertCost;
  auto optTypes = OperandTypesVector();
  for (auto types : applicableTypes) {
    assert(types.size() == initElemTypes.size());
    int accumulatedConvertCost = 0;
    for (int i = 0; i < initElemTypes.size(); ++i) {
      accumulatedConvertCost += calcConvertCost(initElemTypes[i], types[i]);
    }
    if (accumulatedConvertCost < minCost) {
      minCost = accumulatedConvertCost;
      optTypes = types;
    }
  }
  return optTypes;
}

OperandTypesVector getOperandTypesForWmmaOp(PatternRewriter &rewriter,
                                            tt::DotOp dot, int version) {
  Type f16 = rewriter.getF16Type();
  Type f32 = rewriter.getF32Type();
  Type bf16 = rewriter.getBF16Type();
  Type i8 = rewriter.getIntegerType(8);
  Type i32 = rewriter.getIntegerType(32);
  SmallVector<OperandTypesVector> applicableTypes = {
      // clang-format off
      {f16, f16, f32, f32},
      {f16, f16, f16, f16},
      {bf16, bf16, f32, f32},
      {bf16, bf16, bf16, bf16},
      {i8, i8, i32, i32},
      // i4, i4, i32, i32 - is supported configuration
      // by WMMA instruction, but not supported by triton
      // clang-format on
  };
  // TODO: support fp8 configurations for WMMAv2. The code should be as
  // following:
  // if (version == 2) {
  //   Type fp8 = rewriter.getFp8Type();
  //   Type bf8 = rewriter.getBF8Type();
  //   applicableTypes.append({
  //       // clang-format off
  //       {fp8, fp8, f32, f32},
  //       {fp8, bf8, f32, f32},
  //       {bf8, fp8, f32, f32},
  //       {bf8, bf8, f32, f32},
  //       // clang-format on
  //   });
  // }
  return selectMatrixCoreOperandTypes(dot, applicableTypes);
}

/**
 * @brief Convert layout and cast element type of a given tensor
 *
 * If old element type is different from new element type, this function
 * creates two new operations:
 * 1. %converted_value = layout_convert %value, newEncoding
 * 2. %casted_value = cast(fext, ftrunc, etc.) %value, newElemType
 *
 * If old element type is same as new element type, this function creates only
 * one operation: %converted_value = layout_convert %value, newEncoding
 *
 * @param rewriter
 * @param value original tensor value, which we need to convert and cast
 * @param newEncoding new encoding for the tenosr
 * @param newElemType new element type for the tensor
 * @return converted and optionaly casted tensor value
 */
Value convertAndCastTensor(PatternRewriter &rewriter, Value value,
                           Attribute newEncoding, Type newElemType) {
  assert(newElemType.isIntOrFloat());

  auto loc = value.getLoc();
  auto oldType = cast<RankedTensorType>(value.getType());
  auto oldElemType = oldType.getElementType();

  assert(oldElemType.isIntOrFloat());
  assert(oldElemType.isIntOrIndex() == newElemType.isIntOrIndex());

  auto convertedType =
      RankedTensorType::get(oldType.getShape(), oldElemType, newEncoding);

  Value convertedTensor =
      rewriter.create<ttg::ConvertLayoutOp>(loc, convertedType, value);

  if (newElemType == oldElemType)
    return convertedTensor;

  Type castedType = convertedType.cloneWith(std::nullopt, newElemType);

  Value castedTensor;

  if (newElemType.isIntOrIndex()) {
    unsigned oldWidth = oldElemType.getIntOrFloatBitWidth();
    unsigned newWidth = newElemType.getIntOrFloatBitWidth();
    if (oldWidth == newWidth)
      castedTensor = rewriter.create<arith::BitcastOp>(loc, convertedType,
                                                       convertedTensor);
    else if (oldWidth > newWidth)
      castedTensor =
          rewriter.create<arith::TruncIOp>(loc, castedType, convertedTensor);
    else if (oldElemType.isSignedInteger())
      castedTensor =
          rewriter.create<arith::ExtSIOp>(loc, castedType, convertedTensor);
    else
      castedTensor =
          rewriter.create<arith::ExtUIOp>(loc, castedType, convertedTensor);
  } else {
    if (oldElemType.isF16() && newElemType.isF32())
      castedTensor =
          rewriter.create<arith::ExtFOp>(loc, castedType, convertedTensor);
    else if (oldElemType.isF32() && newElemType.isF16())
      castedTensor =
          rewriter.create<arith::TruncFOp>(loc, castedType, convertedTensor);
    else
      castedTensor =
          rewriter.create<tt::FpToFpOp>(loc, castedType, convertedTensor);
  }
  return castedTensor;
}

class BlockedToMFMA : public RewritePattern {
  int mfmaVersion;
  int enforcedNonKDim;
  int kPack;

public:
  BlockedToMFMA(MLIRContext *context, int mfmaVersion, int nonKDim, int kPack)
      : RewritePattern(tt::DotOp::getOperationName(), 2, context),
        mfmaVersion(mfmaVersion), enforcedNonKDim(nonKDim), kPack(kPack) {}

  bool isSecondDot(tt::DotOp &dotOp) const {
    auto filter = [&dotOp](Operation *op) {
      return op->getParentRegion() == dotOp->getParentRegion();
    };
    BackwardSliceOptions bwdOpt;
    bwdOpt.omitBlockArguments = true;
    bwdOpt.filter = filter;
    SetVector<Operation *> slices;
    getBackwardSlice(dotOp.getResult(), &slices, bwdOpt);
    if (llvm::find_if(slices, [](Operation *op) {
          return isa<tt::DotOp>(op);
        }) != slices.end())
      return true;
    return false;
  }

  /// @brief Choose MFMA instruction parameters
  /// @param dot target dot operation
  /// @return MfmaInsn or failure
  FailureOr<MfmaInsn> chooseMfmaInstruction(tt::DotOp dot) const {
    // number of matrix elements along k dim per one MFMA intruction
    unsigned kDim = 0;
    auto opType = cast<RankedTensorType>(dot.getA().getType());
    auto dataTypeA = opType.getElementType();
    auto dataTypeB =
        cast<RankedTensorType>(dot.getB().getType()).getElementType();

    auto resType = cast<RankedTensorType>(dot.getD().getType());
    auto resShape = resType.getShape();
    auto rank = resShape.size();
    auto M = resShape[rank - 2];
    auto N = resShape[rank - 1];

    unsigned mDim = 0;
    unsigned nDim = 0;
    if (enforcedNonKDim != 0) {
      mDim = enforcedNonKDim;
      nDim = enforcedNonKDim;
    } else {
      int minSize = std::min(M, N);
      if (minSize >= 32) {
        mDim = 32;
        nDim = 32;
      }
      if (minSize >= 16 && minSize < 32) {
        mDim = 16;
        nDim = 16;
      }
      if (minSize < 16) {
        if (M < 16 && N >= 64) {
          mDim = 4;
          nDim = 64;
        } else if (M >= 64 && N < 16) {
          mDim = 64;
          nDim = 4;
        } else {
          assert(opType.getShape()[rank - 1] >= 64 &&
                 "k should be at least 64 to use this layout");
          mDim = 4;
          nDim = 4;
        }
      }
    }
    assert(mDim != 0 && nDim != 0);

    auto maybeMfmaInsn =
        MfmaInsn::selectMfma(mDim, nDim, dataTypeA, dataTypeB, mfmaVersion);
    if (failed(maybeMfmaInsn))
      llvm::report_fatal_error("No match found in MFMA database\n");

    kDim = maybeMfmaInsn->getKDim();
    assert(kDim != 0);
    assert(M % mDim == 0 && N % nDim == 0);
    assert(opType.getShape()[rank - 1] % kDim == 0);
    return maybeMfmaInsn;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto dotOp = cast<tt::DotOp>(op);

    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        !isa<ttg::BlockedEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    if (!supportMFMA(dotOp))
      return failure();

    auto CTALayout = ttg::getCTALayout(oldRetType.getEncoding());

    // get MFMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());
    auto ctx = oldAType.getContext();

    ttg::AMDMfmaEncodingAttr mfmaEnc;

    auto mfmaInstr = chooseMfmaInstruction(dotOp);
    auto mDim = mfmaInstr.value().getMDim();
    auto nDim = mfmaInstr.value().getNDim();
    auto kDim = mfmaInstr.value().getKDim();
    auto kBase = mfmaInstr.value().getKBase();

    auto warpsPerTile =
        warpsPerTileMFMA(dotOp, retShape, numWarps, {mDim, nDim});

    mfmaEnc = ttg::AMDMfmaEncodingAttr::get(
        oldRetType.getContext(),
        /*versionMajor*/ mfmaVersion, /*versionMinor*/ 0, warpsPerTile,
        /*instrShape*/ mDim, nDim, /*isTransposed*/ true, CTALayout);

    Type mfmaAccType;
    if (oldRetType.getElementType().isIntOrIndex())
      mfmaAccType = rewriter.getIntegerType(32);
    else
      mfmaAccType = rewriter.getF32Type();

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = convertAndCastTensor(rewriter, oldAcc, mfmaEnc, mfmaAccType);

    // Here is a brief explanation of kWidth, kBase, and kDim
    // 1. kWidth: the number of elements each thread loads from shared memory in
    //    preparation for mfma instructions. In theory each thread can issue one
    //    or more load instructions to load a total of kWidth elements, since
    //    those elements are not required to be in contiguous addresses in
    //    shared memory. But in practice, we make sure the kWidth elements can
    //    be loaded from shared memory by a single ds_read instruction by
    //    setting vecSize of the sharedLayout to be kWidth.
    // 2. kDim: the k dimension size of the mfma instruction. E.g. instruction
    //    mfma_32x32x16 has kDim = 16, meaning this mfma instruction can compute
    //    a matmul of operands with shape 32x16 and 16x32.
    // 3. kBase: the number of elements each thread holds for a single mfma
    //    instruction.
    // 4. relation between kBase and kDim:
    //    4.1 For mfma_32, kBase = kDim / 2
    //    4.2 For mfma_16, kBase = kDim / 4
    //    4.3 For mfma_4, it depends on how mfma_4 is used. We'll extend to
    //        mfma_4 later.
    // 5. relation between kWidth and kBase: For now it supports two cases
    //    5.1 kWidth = kBase, i.e. kPack = 1. In this case, each load from
    //        shared memory results in one mfma instruction.
    //    5.2 kWidth = 2 * kBase, i.e. kPack = 2. In this case, each load from
    //        shared memory results in two mfma instructions, since one mfma
    //        can only consume kBase elements from each thread.
    //    Note that we cannot have larger kPack since kPack = 2 means
    //    ds_read_b128, which is the largest vector size for shared memory load.
    auto kWidth = kBase;
    // in mfma 4x4 case argument matrix groups in 16 groups
    if (mDim == 4 && nDim == 4)
      kWidth = kDim / 16;
    if ((mDim == 4 && nDim == 64) || (mDim == 64 && nDim == 4))
      kWidth = kDim;

    // We want to extend kWidth by kPack (kPack=1 means no extension)
    // to increase ds_read vector size
    // However, in FA, the second dot can only use kWidth = kBase since it's
    // limited by the result of the first dot, which is of mfmaLayout.
    if (!isSecondDot(dotOp))
      kWidth *= kPack;

    auto newAEncoding =
        ttg::DotOperandEncodingAttr::get(ctx, 0, mfmaEnc, kWidth);
    auto newBEncoding =
        ttg::DotOperandEncodingAttr::get(ctx, 1, mfmaEnc, kWidth);
    a = convertAndCastTensor(rewriter, a, newAEncoding,
                             mfmaInstr.value().getElementTypeA());
    b = convertAndCastTensor(rewriter, b, newBEncoding,
                             mfmaInstr.value().getElementTypeB());
    auto newDot = rewriter.create<tt::DotOp>(
        dotOp.getLoc(), newAcc.getType(), a, b, newAcc,
        dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());

    Value dotOutput =
        convertAndCastTensor(rewriter, newDot, oldRetType.getEncoding(),
                             oldRetType.getElementType());

    rewriter.replaceOp(op, dotOutput);

    return success();
  }
};
static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  Type tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                .cloneWith(std::nullopt, promotedType);
  return builder.create<triton::FpToFpOp>(loc, tensorPromotedType, operand);
}

// promote operands of dot op if the existing combination is not natively
// supported.
static void decomposeMixedModeDotOp(ModuleOp mod) {
  mod.walk([](triton::DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    Type promoteType;
    if (isa<ttg::AMDMfmaEncodingAttr>(D.getType().getEncoding())) {
      Type BElType = dotOp.getB().getType().getElementType();

      auto maxBitWidth = std::max(AElType.getIntOrFloatBitWidth(),
                                  BElType.getIntOrFloatBitWidth());

      // TODO check mfma tensor core version compatibility
      if (maxBitWidth == 8)
        return;

      if (AElType == BElType)
        return;

      if (maxBitWidth < 16)
        promoteType = builder.getF16Type();
      else if (maxBitWidth <= 32)
        promoteType = builder.getF32Type();
    } else if (isa<ttg::AMDWmmaEncodingAttr>(D.getType().getEncoding())) {
      Type BElType = dotOp.getB().getType().getElementType();

      if (AElType == BElType)
        return;

      // Other cases must be filtered earlier
      promoteType =
          AElType.getIntOrFloatBitWidth() > BElType.getIntOrFloatBitWidth()
              ? AElType
              : BElType;
    } else {
      // FMA case.
      Type AElType = dotOp.getA().getType().getElementType();
      Type DElType = D.getType().getElementType();

      // Convert int operands to FP32 to apply FMA case
      // Do it here instead of introducing new pattern because the pass is more
      // about MMA dots.
      // TODO: Introduce new pass for FMA dots legalization.
      if (AElType.isIntOrIndex()) {
        assert(dotOp.getB().getType().getElementType().isIntOrIndex() &&
               dotOp.getC().getType().getElementType().isIntOrIndex() &&
               DElType.isIntOrIndex());
        auto convertTensorIToFP = [&](Value v) -> Value {
          RankedTensorType vTy = cast<RankedTensorType>(v.getType());
          Type dstType = vTy.cloneWith(std::nullopt, builder.getF32Type());
          Type srcElType = vTy.getElementType();
          return !srcElType.isUnsignedInteger()
                     ? builder
                           .create<arith::SIToFPOp>(dotOp.getLoc(), dstType, v)
                           .getResult()
                     : builder
                           .create<arith::UIToFPOp>(dotOp.getLoc(), dstType, v)
                           .getResult();
        };
        auto convertTensorFPToI = [&](Type dstElType, Value v) -> Value {
          RankedTensorType vTy = cast<RankedTensorType>(v.getType());
          Type dstType = vTy.cloneWith(std::nullopt, dstElType);
          return !dstElType.isUnsignedInteger()
                     ? builder
                           .create<arith::FPToSIOp>(dotOp.getLoc(), dstType, v)
                           .getResult()
                     : builder
                           .create<arith::FPToUIOp>(dotOp.getLoc(), dstType, v)
                           .getResult();
        };

        auto newAOperand = convertTensorIToFP(dotOp.getA());
        auto newBOperand = convertTensorIToFP(dotOp.getB());
        auto newCOperand = convertTensorIToFP(dotOp.getC());
        auto newDot = builder.create<tt::DotOp>(
            dotOp.getLoc(), newCOperand.getType(), newAOperand, newBOperand,
            newCOperand, dotOp.getInputPrecision(),
            dotOp.getMaxNumImpreciseAcc());
        auto newD = convertTensorFPToI(DElType, newDot.getResult());
        D.replaceAllUsesWith(newD);
        dotOp.erase();
        return;
      }

      if (AElType == DElType)
        return;
      promoteType = DElType;
    }
    Location loc = dotOp.getLoc();
    Value promotedA = promoteOperand(builder, loc, dotOp.getA(), promoteType);
    Value promotedB = promoteOperand(builder, loc, dotOp.getB(), promoteType);
    dotOp.setOperand(0, promotedA);
    dotOp.setOperand(1, promotedB);
  });
}

class BlockedToWMMA : public RewritePattern {
  int wmmaVersion;

public:
  BlockedToWMMA(MLIRContext *context, int wmmaVersion)
      : RewritePattern(tt::DotOp::getOperationName(), 2, context),
        wmmaVersion(wmmaVersion) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctx = op->getContext();
    auto dotOp = cast<tt::DotOp>(op);

    Value a = dotOp.getA();
    Value b = dotOp.getB();

    auto oldRetType = cast<RankedTensorType>(dotOp.getResult().getType());
    auto oldRetEncoding = oldRetType.getEncoding();
    if (!oldRetEncoding || !isa<ttg::BlockedEncodingAttr>(oldRetEncoding))
      return failure();

    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());
    auto retShape = oldRetType.getShape();
    auto aShape = oldAType.getShape();
    auto bShape = oldBType.getShape();

    // check shape
    auto mnkDim = ttg::AMDWmmaEncodingAttr::getMNKDimPerInstr();
    auto rank = aShape.size();
    if (aShape[rank - 2] % mnkDim[0] != 0 || // m
        bShape[rank - 1] % mnkDim[1] != 0 || // n
        aShape[rank - 1] % mnkDim[2] != 0)   // k
      return failure();

    if (wmmaVersion == 2 && llvm::isa<FloatType>(oldAType) &&
        oldAType.getIntOrFloatBitWidth() == 8) {
      return rewriter.notifyMatchFailure(op, "not supported yet");
    }

    // get operand types
    auto operandTypes = getOperandTypesForWmmaOp(rewriter, dotOp, wmmaVersion);
    if (operandTypes.empty())
      return failure();

    // get WMMA encoding for the given number of warps
    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);

    ttg::AMDWmmaEncodingAttr wmmaEnc;

    auto warpsPerTile = warpsPerTileWMMA(dotOp, retShape, numWarps);

    auto CTALayout = ttg::getCTALayout(oldRetEncoding);
    wmmaEnc = ttg::AMDWmmaEncodingAttr::get(ctx, wmmaVersion, warpsPerTile,
                                            CTALayout);

    auto newRetType = RankedTensorType::get(retShape, operandTypes[3], wmmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc =
        convertAndCastTensor(rewriter, oldAcc, wmmaEnc, operandTypes[2]);

    auto newAType = RankedTensorType::get(
        aShape, operandTypes[0],
        ttg::DotOperandEncodingAttr::get(
            ctx, 0, wmmaEnc, wmmaEnc.getSizePerThreadForOperands(0)[rank - 1]));
    auto newBType = RankedTensorType::get(
        bShape, operandTypes[1],
        ttg::DotOperandEncodingAttr::get(
            ctx, 1, wmmaEnc, wmmaEnc.getSizePerThreadForOperands(1)[rank - 2]));

    Value castedA = convertAndCastTensor(rewriter, a, newAType.getEncoding(),
                                         operandTypes[0]);
    Value castedB = convertAndCastTensor(rewriter, b, newBType.getEncoding(),
                                         operandTypes[1]);
    auto newDot = rewriter.create<tt::DotOp>(
        dotOp.getLoc(), newRetType, castedA, castedB, newAcc,
        dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());

    Value dotOutput = convertAndCastTensor(rewriter, newDot, oldRetEncoding,
                                           oldRetType.getElementType());
    rewriter.replaceOp(op, dotOutput);
    return success();
  }
};
} // namespace

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

class TritonAMDGPUAccelerateMatmulPass
    : public TritonAMDGPUAccelerateMatmulBase<
          TritonAMDGPUAccelerateMatmulPass> {
public:
  TritonAMDGPUAccelerateMatmulPass() = default;
  TritonAMDGPUAccelerateMatmulPass(StringRef archGen, int matrixInstructionSize,
                                   int kPack) {
    this->archGenerationName = archGen.data();
    this->matrixInstructionSize = matrixInstructionSize;
    this->kPack = kPack;
  }
  void runOnOperation() override {

    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    RewritePatternSet patterns(context);
    switch (auto isaFamily = triton::AMD::deduceISAFamily(archGenerationName)) {
    case ISAFamily::CDNA1:
    case ISAFamily::CDNA2:
    case ISAFamily::CDNA3:
      patterns.add<::BlockedToMFMA>(context, getMfmaVersion(isaFamily),
                                    matrixInstructionSize, kPack);
      break;
    case ISAFamily::RDNA3:
      patterns.add<::BlockedToWMMA>(context,
                                    getWmmaVersion(archGenerationName));
      break;
    default:
      break;
    }
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
    decomposeMixedModeDotOp(m);
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUAccelerateMatmulPass(
    std::string archGen, int matrixInstructionSize, int kPack) {
  return std::make_unique<TritonAMDGPUAccelerateMatmulPass>(
      archGen, matrixInstructionSize, kPack);
}
