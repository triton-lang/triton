#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.h"
#include "triton/Tools/LayoutUtils.h"
#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using ::mlir::LLVM::AMD::scaleDotElemTypeToMLIRType;

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
  case ISAFamily::CDNA4:
    return 4;
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

SmallVector<unsigned, 3>
warpsPerTile(Operation *dotOp, ArrayRef<int64_t> shape, int numWarps,
             std::pair<int64_t, int64_t> shapePerWarp) {
  auto rank = shape.size();
  // Early exit for batched matmul
  if (rank == 3)
    return {(unsigned)numWarps, 1, 1};

  auto filter = [dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  ForwardSliceOptions fwdOpt;
  fwdOpt.filter = filter;
  BackwardSliceOptions bwdOpt;
  bwdOpt.omitBlockArguments = true;
  bwdOpt.filter = filter;
  auto slices = getSlice(dotOp, bwdOpt, fwdOpt);
  for (Operation *op : slices) {
    if (isa<mlir::triton::DotOpInterface>(op) && (op != dotOp))
      return {(unsigned)numWarps, 1};
  }

  SmallVector<int64_t, 2> tensorShape = {shape[0], shape[1]};
  SmallVector<unsigned, 3> ret = {1, 1};
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (tensorShape[0] / (shapePerWarp.first * 2) / ret[0] >=
        tensorShape[1] / shapePerWarp.second / ret[1]) {
      if (ret[0] < tensorShape[0] / shapePerWarp.first) {
        ret[0] *= 2;
      } else {
        ret[1] *= 2;
      }
    } else {
      ret[1] *= 2;
    }
  } while (true);

  if (ret[1] * shapePerWarp.second > tensorShape[1]) {
    return {ret[1], ret[0]};
  }

  return ret;
}

SmallVector<unsigned, 3>
warpsPerTileMFMA(Operation *dotOp, ArrayRef<int64_t> shape, int numWarps,
                 std::pair<int64_t, int64_t> shapePerWarp) {
  return warpsPerTile(dotOp, shape, numWarps, shapePerWarp);
}

SmallVector<unsigned, 3>
warpsPerTileWMMA(Operation *dotOp, ArrayRef<int64_t> shape, int numWarps) {
  auto mnk = ttg::AMDWmmaEncodingAttr::getMNKDimPerInstr();
  return warpsPerTile(dotOp, shape, numWarps, {mnk[0], mnk[1]});
}

// Chooses a proper MFMA instruction that can used to compute the given dot op.
// If enforcedNonKDim is not zero, it will be used to overwrite the default
// logic to chose a MFMA with matching M/N dim.
FailureOr<MfmaInsn> chooseMfmaInstruction(RankedTensorType cType,
                                          Type aElemType, Type bElemType,
                                          int inputKSize, int mfmaVersion,
                                          bool allowXF32, int enforcedNonKDim) {
  // number of matrix elements along k dim per one MFMA intruction
  unsigned kDim = 0;

  auto resShape = cType.getShape();
  auto rank = resShape.size();
  auto M = resShape[rank - 2];
  auto N = resShape[rank - 1];

  unsigned mDim = 0;
  unsigned nDim = 0;
  if (enforcedNonKDim != 0) {
    mDim = nDim = enforcedNonKDim;
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
  }
  if (mDim == 0 || nDim == 0)
    return failure();

  auto maybeMfmaInsn = MfmaInsn::selectMfma(mDim, nDim, aElemType, bElemType,
                                            mfmaVersion, allowXF32);
  if (failed(maybeMfmaInsn))
    llvm::report_fatal_error("No match found in MFMA database\n");

  kDim = maybeMfmaInsn->getKDim();
  assert(kDim != 0);
  assert(enforcedNonKDim != 0 || (M % mDim == 0 && N % nDim == 0));
  // if inputKSize % kDim != 0 this layout will introduce data duplication,
  // consider FMA dot is prefered, except cases MFMA layout is enforced.
  if (enforcedNonKDim == 0 && inputKSize % kDim != 0)
    return failure();
  return maybeMfmaInsn;
}

FailureOr<MfmaInsn> chooseMfmaInstruction(tt::DotOp dot, int mfmaVersion,
                                          int nonKDim) {
  RankedTensorType aType = dot.getA().getType();
  bool allowXF32 =
      dot.getInputPrecision() == InputPrecision::TF32 && mfmaVersion == 3;
  return chooseMfmaInstruction(dot.getC().getType(), aType.getElementType(),
                               dot.getB().getType().getElementType(),
                               aType.getShape().back(), mfmaVersion, allowXF32,
                               nonKDim);
}

FailureOr<MfmaInsn> chooseMfmaInstruction(tt::DotScaledOp dot, int mfmaVersion,
                                          int nonKDim) {
  auto ctx = dot.getContext();
  int64_t inputKDim = dot.getLhs().getType().getShape().back();
  if (dot.getLhsType() == ScaleDotElemType::E2M1) {
    // Since two fp4 are packed into int8, to get the correct K dim size, we
    // need to multiply it by 2.
    inputKDim *= 2;
  }
  return chooseMfmaInstruction(
      dot.getC().getType(), scaleDotElemTypeToMLIRType(ctx, dot.getLhsType()),
      scaleDotElemTypeToMLIRType(ctx, dot.getRhsType()), inputKDim, mfmaVersion,
      /*allowXF32=*/false, nonKDim);
}

FailureOr<MfmaInsn> chooseMfmaInstruction(tt::DotScaledOp dot, int mfmaVersion,
                                          int nonKDim, bool useFp16) {
  // For scaled dot, we handle it with fp16 or bf16 emulation for now.
  Builder b(dot.getContext());
  Type elemType = useFp16 ? b.getF16Type() : b.getBF16Type();
  return chooseMfmaInstruction(dot.getC().getType(), /*aElemType=*/elemType,
                               /*bElemType=*/elemType,
                               dot.getLhs().getType().getShape().back(),
                               mfmaVersion, /*allowXF32=*/false, nonKDim);
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

//===---------------------------------------------------------------------===//
// @brief Convert layout and cast element type of a given tensor
//
// If old element type is different from new element type, this function
// creates two new operations:
// 1. %converted_value = layout_convert %value, newEncoding
// 2. %casted_value = cast(fext, ftrunc, etc.) %value, newElemType
//
// If old element type is same as new element type, this function creates only
// one operation: %converted_value = layout_convert %value, newEncoding
//
// @param rewriter
// @param value original tensor value, which we need to convert and cast
// @param newEncoding new encoding for the tensor
// @param newElemType new element type for the tensor
// @return converted and optionally casted tensor value
//===---------------------------------------------------------------------===//
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

class ScaledBlockedToScaledMFMAF8F6F4 final
    : public OpRewritePattern<triton::DotScaledOp> {
  int mfmaVersion;
  int nonKDim;

public:
  ScaledBlockedToScaledMFMAF8F6F4(MLIRContext *context, int mfmaVersion,
                                  int nonKDim, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), mfmaVersion(mfmaVersion),
        nonKDim(nonKDim) {}

  LogicalResult matchAndRewrite(triton::DotScaledOp dotOp,
                                PatternRewriter &rewriter) const override {
    using TensorValue = TypedValue<RankedTensorType>;

    if (mfmaVersion != 4) {
      return rewriter.notifyMatchFailure(
          dotOp, "F8F6F4 scaled dot is only natively supported on gfx950");
    }

    RankedTensorType oldRetType = dotOp.getType();
    if (!isa_and_nonnull<BlockedEncodingAttr>(oldRetType.getEncoding()))
      return rewriter.notifyMatchFailure(
          dotOp, "expected blocked encoding result tensor");

    unsigned rank = oldRetType.getRank();
    if (rank == 3)
      return rewriter.notifyMatchFailure(dotOp, "NYI: 3d case");

    TensorValue a = dotOp.getLhs();
    TensorValue b = dotOp.getRhs();
    TensorValue aScale = dotOp.getLhsScale();
    TensorValue bScale = dotOp.getRhsScale();
    auto oldShape = oldRetType.getShape();

    if (!aScale || !bScale)
      return rewriter.notifyMatchFailure(dotOp,
                                         "expect scales for both A and B");

    ScaleDotElemType aElemType = dotOp.getLhsType();
    ScaleDotElemType bElemType = dotOp.getRhsType();
    auto supportsTypes = [](ScaleDotElemType elemType) {
      return elemType == ScaleDotElemType::E2M1;
    };

    if (!supportsTypes(aElemType) || !supportsTypes(bElemType))
      return rewriter.notifyMatchFailure(dotOp, "NYI: mxfp6, mxfp8");

    MLIRContext *ctx = dotOp.getContext();
    auto moduleOp = dotOp->getParentOfType<ModuleOp>();

    ttg::CTALayoutAttr ctaLayout = ttg::getCTALayout(oldRetType.getEncoding());
    unsigned numWarps = ttg::TritonGPUDialect::getNumWarps(moduleOp);
    if (numWarps == 1)
      return rewriter.notifyMatchFailure(dotOp,
                                         "num_warps==1 is not supported");

    // Choose a suitable Scaled MFMA instruction for this scaled dot op.
    FailureOr<MfmaInsn> mfmaInstr =
        chooseMfmaInstruction(dotOp, mfmaVersion, nonKDim);
    if (failed(mfmaInstr))
      return rewriter.notifyMatchFailure(dotOp,
                                         "cannot choose scaled mfma intrinsic");

    unsigned mDim = mfmaInstr.value().getMDim();
    unsigned nDim = mfmaInstr.value().getNDim();
    unsigned kDim = mfmaInstr.value().getKDim();
    unsigned kBase = mfmaInstr.value().getKBase();
    assert(mDim == nDim);

    auto warpsPerTile =
        warpsPerTileMFMA(dotOp, oldShape, numWarps, {mDim, nDim});

    // Always use transposed mfma layout. This enables larger vectorization
    // for global store instructions.
    auto mfmaEnc = ttg::AMDMfmaEncodingAttr::get(
        ctx, /*versionMajor=*/mfmaVersion, /*versionMinor=*/0, warpsPerTile,
        /*instrShape=*/mDim, nDim, /*isTransposed=*/true, ctaLayout);

    auto newRetType =
        RankedTensorType::get(oldShape, oldRetType.getElementType(), mfmaEnc);

    auto newAcc = rewriter.create<ttg::ConvertLayoutOp>(
        dotOp.getC().getLoc(), newRetType, dotOp.getC());

    // For the mfma_scale_f32_*_f8f6f4 instructions, each thread consumes 32
    // elements. But since two fp4 elements are packed into one int8, the
    // kWidth is 16 for fp4.
    unsigned kWidth = kBase;
    auto newAEncoding = DotOperandEncodingAttr::get(
        ctx, /*idx=*/0, newRetType.getEncoding(),
        aElemType == ScaleDotElemType::E2M1 ? kWidth / 2 : kWidth);

    auto newBEncoding = DotOperandEncodingAttr::get(
        ctx, /*idx=*/1, newRetType.getEncoding(),
        bElemType == ScaleDotElemType::E2M1 ? kWidth / 2 : kWidth);

    auto convertInputLayout = [&](TensorValue v,
                                  DotOperandEncodingAttr enc) -> TensorValue {
      auto vType = v.getType();

      auto newVType =
          RankedTensorType::get(vType.getShape(), vType.getElementType(), enc);
      return rewriter.create<ttg::ConvertLayoutOp>(v.getLoc(), newVType, v);
    };
    a = convertInputLayout(a, newAEncoding);
    b = convertInputLayout(b, newBEncoding);

    StringAttr kRegister = StringAttr::get(ctx, "register");
    StringAttr kLane = StringAttr::get(ctx, "lane");
    StringAttr kWarp = StringAttr::get(ctx, "warp");
    StringAttr kBlock = StringAttr::get(ctx, "block");

    auto order = ttg::getMatrixOrder(rank, /*rowMajor=*/true);

    // Init register layout. Will be adjusted later
    auto regs = tt::identityStandardND(kRegister, {1, 1}, order);

    auto aEncLL = newAEncoding.toLinearLayout(a.getType().getShape());
    auto standardOutDims = llvm::to_vector(aEncLL.getOutDimNames());

    using basisT = std::vector<std::vector<int32_t>>;
    auto createLinearLayout = [&](int idx, const basisT &warpBasis) {
      LinearLayout lanes = LinearLayout::empty();
      // In scaled dot, the shapes of operands(without batch dimension) are,
      // respectively:
      // - A: [M, K]
      // - B: [K, N]
      // - aScale: [M, K / 32]
      // - bScale: [N, K / 32]
      //
      // To correctly feed A/B and its scale into instruction, we need to
      // distribute aScale/bScale among warps in the same way as A/B. But bScale
      // is not transposed like B. So we need to transpose the warp layout of
      // bScale.
      //
      // The tricky part is, our desired outputs are [dim0, dim1], but
      // at this position, the layouts are transposed to [dim1, dim0]. So
      // instead of reverse bScale's layout, we need to reverse aScale's. There
      // will be a transpose in the end to correct everything.
      basisT warps = warpBasis;
      if (idx == 0) {
        for (auto &basis : warps) {
          std::reverse(basis.begin(), basis.end());
        }
      }
      // In general, for both 32x32 and 16x16 scaled mfma, and no matter what
      // data type the A/B operand is, each lane takes 32 elements from A/B
      // alone K dim, and 1 or 2 elements from scale accordingly. The number of
      // scale's elements in a lane varies because the 32 elements from A/B may
      // not be consecutive.
      //
      // For mxfp4, these 32 elements are consecutive, so only 1 scale element
      // is required. But for mxfp6/mxfp8, there are 2 16-consecutive elements
      // blocks, so 2 scale elements are required.
      if (mDim == 32) {
        // For ROCDL::mfma_scale_f32_32x32x64_f8f6f4 with fp4 input, each lane
        // takes 32 consecutive elements from A alone K dimension. The first
        // 32 lanes collectively handle A[0:32][0:32], and the other 32 lanes
        // collectively handle A[0:32][32:64]. Each lane take 1 scale element
        // accordingly. Similar to B and bScale.
        lanes = LinearLayout(
            {{kLane, {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {1, 0}}},
             {kWarp, warps},
             {kBlock, {}}},
            {standardOutDims[order[0]], standardOutDims[order[1]]});
      } else {
        assert(mDim == 16);
        // For ROCDL::mfma_scale_f32_16x16x128_f8f6f4 with fp4 input, each lane
        // takes 32 consecutive elements from A alone K dimension. The first
        // 16 lanes collectively handle A[0:16][0:32], and another 16 lanes
        // collectively handle A[0:16][32:64] and so on. Each lane take 1 scale
        // element accordingly. Similar to B and bScale.
        lanes = LinearLayout(
            {{kLane, {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}, {2, 0}}},
             {kWarp, warps},
             {kBlock, {}}},
            {standardOutDims[order[0]], standardOutDims[order[1]]});
      }
      return regs * lanes;
    };

    auto convertScaleLayout = [&](TensorValue val, TensorValue scale,
                                  DotOperandEncodingAttr enc,
                                  int idx) -> TensorValue {
      auto dotLL = enc.toLinearLayout(val.getType().getShape());
      LinearLayout::BasesT scaleBases = dotLL.getBases();
      auto &warpBases = scaleBases[kWarp];

      LinearLayout newLL = createLinearLayout(idx, warpBases);

      auto shape = scale.getType().getShape();

      // Adjust register-level layout to fill the shape, at this level, both
      // aScale and bScale should align with A operand.
      for (auto d : newAEncoding.getRepOrder()) {
        auto outDim = standardOutDims[d];
        auto dimSize = newLL.getOutDimSize(outDim);
        newLL *=
            LinearLayout::identity1D(shape[d] / dimSize, kRegister, outDim);
      }
      newLL = newLL.transposeOuts(standardOutDims);
      Attribute newScaleEncoding = ttg::LinearEncodingAttr::get(ctx, newLL);

      auto newScaleType = RankedTensorType::get(
          shape, scale.getType().getElementType(), newScaleEncoding);
      return rewriter.create<ttg::ConvertLayoutOp>(scale.getLoc(), newScaleType,
                                                   scale);
    };
    aScale = convertScaleLayout(a, aScale, newAEncoding, 0);
    bScale = convertScaleLayout(b, bScale, newBEncoding, 1);

    auto newDot = rewriter.create<triton::DotScaledOp>(
        dotOp.getLoc(), newRetType, a, b, newAcc, aScale, bScale, aElemType,
        bElemType, dotOp.getFastMath());

    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(dotOp, oldRetType,
                                                      newDot);
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
      // FMA case is processed in AccelerateBlocked
      return;
    }
    Location loc = dotOp.getLoc();
    Value promotedA = promoteOperand(builder, loc, dotOp.getA(), promoteType);
    Value promotedB = promoteOperand(builder, loc, dotOp.getB(), promoteType);
    dotOp.setOperand(0, promotedA);
    dotOp.setOperand(1, promotedB);
  });
}

class BlockedToWMMA : public OpRewritePattern<tt::DotOp> {
  int wmmaVersion;

public:
  BlockedToWMMA(MLIRContext *context, int wmmaVersion,
                PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), wmmaVersion(wmmaVersion) {}

  LogicalResult matchAndRewrite(tt::DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto ctx = dotOp->getContext();

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
      return rewriter.notifyMatchFailure(dotOp, "not supported yet");
    }

    // get operand types
    auto operandTypes = getOperandTypesForWmmaOp(rewriter, dotOp, wmmaVersion);
    if (operandTypes.empty())
      return failure();

    // get WMMA encoding for the given number of warps
    auto mod = dotOp->getParentOfType<ModuleOp>();
    int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);

    ttg::AMDWmmaEncodingAttr wmmaEnc;

    auto warpsPerTile = warpsPerTileWMMA(dotOp, retShape, numWarps);

    auto CTALayout = ttg::getCTALayout(oldRetEncoding);

    // TODO implement heuristic/option for this parameter
    bool isTransposed = false;
    wmmaEnc = ttg::AMDWmmaEncodingAttr::get(ctx, wmmaVersion, isTransposed,
                                            warpsPerTile, CTALayout);

    auto newRetType = RankedTensorType::get(retShape, operandTypes[3], wmmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getC();
    auto newAcc =
        convertAndCastTensor(rewriter, oldAcc, wmmaEnc, operandTypes[2]);

    auto newAType =
        RankedTensorType::get(aShape, operandTypes[0],
                              ttg::DotOperandEncodingAttr::get(
                                  ctx, 0, wmmaEnc,
                                  wmmaEnc.getSizePerThreadForOperand(
                                      /*kWidth=*/0, /*opIdx=*/0)[rank - 1]));
    auto newBType =
        RankedTensorType::get(bShape, operandTypes[1],
                              ttg::DotOperandEncodingAttr::get(
                                  ctx, 1, wmmaEnc,
                                  wmmaEnc.getSizePerThreadForOperand(
                                      /*kWidth=*/0, /*opIdx=*/1)[rank - 2]));

    Value castedA = convertAndCastTensor(rewriter, a, newAType.getEncoding(),
                                         operandTypes[0]);
    Value castedB = convertAndCastTensor(rewriter, b, newBType.getEncoding(),
                                         operandTypes[1]);
    auto newDot = rewriter.create<tt::DotOp>(
        dotOp.getLoc(), newRetType, castedA, castedB, newAcc,
        dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());

    Value dotOutput = convertAndCastTensor(rewriter, newDot, oldRetEncoding,
                                           oldRetType.getElementType());
    rewriter.replaceOp(dotOp, dotOutput);
    return success();
  }
};

class AccelerateBlocked : public OpRewritePattern<DotOp> {
  StringRef arch;

public:
  AccelerateBlocked(MLIRContext *context, StringRef arch,
                    PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), arch(arch) {}

  bool isFloat(Type t) const { return t.isIntOrFloat() && !t.isIntOrIndex(); }

  Value castToElTy(PatternRewriter &rewriter, Value v, Type elTy) const {
    Location loc = v.getLoc();
    auto srcTy = cast<RankedTensorType>(v.getType());
    auto dstTy = srcTy.cloneWith(std::nullopt, elTy);
    if (srcTy == dstTy)
      return v;
    auto srcElTy = srcTy.getElementType();
    auto dstElTy = dstTy.getElementType();
    if (isFloat(srcElTy) && isFloat(dstElTy)) {
      auto rmode =
          RoundingModeAttr::get(rewriter.getContext(), RoundingMode::RTNE);
      return rewriter.create<FpToFpOp>(loc, dstTy, v, rmode);
    }
    if (!isFloat(srcElTy) && isFloat(dstElTy))
      return rewriter.create<arith::SIToFPOp>(loc, dstTy, v);
    if (isFloat(srcElTy) && !isFloat(dstElTy))
      return rewriter.create<arith::FPToSIOp>(loc, dstTy, v);
    assert(false && "int -> int cast is unexpected in FMA legalization");
    return Value();
  }

  struct DotElTypes {
    Type a, b, c, d;
  };

  bool isLegalFMAForm(DotOp dotOp, const DotElTypes &dotTypes) const {
    if (AMD::supportsVDot(arch)) {
      auto aOpType = dotOp.getA().getType();
      int rank = aOpType.getRank();
      int k = aOpType.getShape()[rank - 1];
      // Try Fp16 x Fp16 -> Fp32 v_dot
      // if k % 2 != 0: can not use fp V_DOT instruction
      if (dotTypes.a.isF16() && dotTypes.b.isF16() && dotTypes.c.isF32() &&
          dotTypes.d.isF32() && k % 2 == 0) {
        return true;
      }

      // TODO: enable this condition, when fp32 -> fp16 cast works correctly
      // Consider this case as non legal, despite this case is covered by fp16
      // FMA. Because v_dot expected to give both better performance and
      // computational precision.
      if (false && dotTypes.a.isF16() && dotTypes.b.isF16() &&
          dotTypes.c.isF16() && dotTypes.d.isF16() && k % 2 == 0) {
        return false;
      }

      // Try I8 x I8 -> I32 v_dot
      // if k % 4 != 0: can not use integer V_DOT instruction
      if (dotTypes.a.isInteger(8) && dotTypes.b.isInteger(8) &&
          dotTypes.c.isInteger(32) && dotTypes.d.isInteger(32) && k % 4 == 0) {
        return true;
      }
    }

    auto expectedElTy = dotTypes.a;
    for (auto operand : dotOp.getOperands()) {
      auto opTy = cast<RankedTensorType>(operand.getType());
      auto elTy = opTy.getElementType();
      if (elTy != expectedElTy)
        return false;
      if (!elTy.isF16() && !elTy.isF32())
        return false;
    }
    return true;
  }

  LogicalResult tryAccelerateF16WithVDot(DotOp dotOp, PatternRewriter &rewriter,
                                         const DotElTypes &dotTypes) const {
    if (!AMD::supportsVDot(arch))
      return failure();

    // If this is fp16 x fp16 ->fp16 case prioritize using v_dot.
    auto aOpType = dotOp.getA().getType();
    int rank = aOpType.getRank();
    int k = aOpType.getShape()[rank - 1];
    if (dotTypes.a.isF16() && dotTypes.b.isF16() && dotTypes.c.isF16() &&
        dotTypes.d.isF16() && k % 2 == 0) {
      auto newC = castToElTy(rewriter, dotOp.getC(), f32_ty);
      auto newDot = rewriter.create<DotOp>(
          dotOp.getLoc(), newC.getType(), dotOp.getA(), dotOp.getB(), newC,
          dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());
      auto newD = castToElTy(rewriter, newDot.getResult(), f16_ty);
      rewriter.replaceOp(dotOp, newD);
      return success();
    }
    return failure();
  }

  LogicalResult tryLegalizeFMA(DotOp dotOp, PatternRewriter &rewriter,
                               const DotElTypes &dotTypes) const {
    // Legalize dot for plain FMA case, i.e. same operands and result type.

    // Find common type, larger or equal of all operand types
    SmallVector<Type> opElTy{dotTypes.a, dotTypes.b, dotTypes.c, dotTypes.d};
    unsigned maxBitsize = 8;
    for (auto elTy : opElTy)
      maxBitsize = std::max(maxBitsize, elTy.getIntOrFloatBitWidth());
    assert(maxBitsize <= 32);
    Type commonTy =
        maxBitsize <= 16 ? rewriter.getF16Type() : rewriter.getF32Type();

    // Check that type is compatible with all operands; fallback to fp32 if not.
    if (commonTy.isF16()) {
      for (auto elTy : opElTy) {
        if (elTy.isInteger() && elTy.getIntOrFloatBitWidth() > 8) {
          commonTy = rewriter.getF32Type();
          break;
        }
        if (elTy.isBF16()) {
          commonTy = rewriter.getF32Type();
          break;
        }
      }
    }

    auto newA = castToElTy(rewriter, dotOp.getA(), commonTy);
    auto newB = castToElTy(rewriter, dotOp.getB(), commonTy);
    auto newC = castToElTy(rewriter, dotOp.getC(), commonTy);

    auto newDot = rewriter.create<DotOp>(dotOp.getLoc(), newC.getType(), newA,
                                         newB, newC, dotOp.getInputPrecision(),
                                         dotOp.getMaxNumImpreciseAcc());
    auto newD = castToElTy(rewriter, newDot.getResult(), dotTypes.d);

    rewriter.replaceOp(dotOp, newD);
    return success();
  }

  LogicalResult matchAndRewrite(DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<BlockedEncodingAttr>(dotOp.getD().getType().getEncoding()))
      return failure();

    DotElTypes dotTypes;
    dotTypes.a = dotOp.getA().getType().getElementType();
    dotTypes.b = dotOp.getB().getType().getElementType();
    dotTypes.c = dotOp.getC().getType().getElementType();
    dotTypes.d = dotOp.getD().getType().getElementType();

    // Check that dot is not legalized already
    if (isLegalFMAForm(dotOp, dotTypes)) {
      return failure();
    }

    // TODO: enable this condition, when fp32 -> fp16 cast works correctly
    if (false &&
        tryAccelerateF16WithVDot(dotOp, rewriter, dotTypes).succeeded()) {
      return success();
    }

    return tryLegalizeFMA(dotOp, rewriter, dotTypes);
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
    // TODO Add generic patterns for scaled_dot op

    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    RewritePatternSet patterns(context);
    ttg::populateDecomposeScaledBlockedPatterns(patterns, /*benefit=*/3);
    switch (auto isaFamily = triton::AMD::deduceISAFamily(archGenerationName)) {
    case ISAFamily::CDNA4:
      patterns.add<::ScaledBlockedToScaledMFMAF8F6F4>(
          context, getMfmaVersion(isaFamily), matrixInstructionSize,
          /*benefit=*/10);
      [[fallthrough]];
    case ISAFamily::CDNA1:
    case ISAFamily::CDNA2:
    case ISAFamily::CDNA3:
      patterns.add<::BlockedToMFMA>(context, getMfmaVersion(isaFamily),
                                    matrixInstructionSize, kPack,
                                    /*benefit=*/2);
      break;
    case ISAFamily::RDNA3:
      patterns.add<::BlockedToWMMA>(context, getWmmaVersion(archGenerationName),
                                    /*benefit=*/2);
      break;
    default:
      break;
    }
    patterns.add<AccelerateBlocked>(context, archGenerationName, /*benefit=*/1);
    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
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
