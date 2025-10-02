#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonAMDGPUTransforms/WmmaGroup.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.h"
#include "triton/Dialect/TritonGPU/Transforms/LayoutPropagationUtility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/TypeSwitch.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using ::mlir::LLVM::AMD::isChainDotHead;
using ::mlir::LLVM::AMD::isChainDotTail;

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamd-accelerate-matmul"

namespace mlir {

namespace {
using triton::AMD::ISAFamily;

constexpr char AttrDecomposedDotScaledSource[] =
    "amdgpu.decomposed_dot_scaled_source";

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
  if (archGen.starts_with("gfx11"))
    return 1;
  if (archGen.starts_with("gfx12") && !archGen.ends_with("50"))
    return 2;
  if (archGen == "gfx1250")
    return 3;

  return 0;
}

FailureOr<ScaleDotElemType> mlirTypeToScaledElemType(Type type) {
  return llvm::TypeSwitch<Type, FailureOr<ScaleDotElemType>>(type)
      .Case<Float8E4M3FNType>([](Type) { return ScaleDotElemType::E4M3; })
      .Case<Float8E5M2Type>([](Type) { return ScaleDotElemType::E5M2; })
      .Case<Float6E3M2FNType>([](Type) { return ScaleDotElemType::E3M2; })
      .Case<Float6E2M3FNType>([](Type) { return ScaleDotElemType::E2M3; })
      .Case<Float4E2M1FNType>([](Type) { return ScaleDotElemType::E2M1; })
      .Default([](Type) { return failure(); });
}

// Data types supported by non-native DotScaledOp
bool isF16F8F4(ScaleDotElemType elemType) {
  return elemType == ScaleDotElemType::E2M1 ||
         elemType == ScaleDotElemType::E4M3 ||
         elemType == ScaleDotElemType::E5M2 ||
         elemType == ScaleDotElemType::BF16 ||
         elemType == ScaleDotElemType::FP16;
}

SmallVector<unsigned, 3>
warpsPerTile(Operation *dotOp, ArrayRef<int64_t> shape, int numWarps,
             std::pair<int64_t, int64_t> shapePerWarp) {
  auto rank = shape.size();
  // Case 1: Early exit for batched matmul
  if (rank == 3)
    return {static_cast<unsigned>(numWarps), 1, 1};

  // Case 2: For FA-like pattern, i.e. result of 1st tl.dot is used as the opA
  // of the 2nd dot, we will set warpsPerCTA differently for 1st and 2nd dot
  auto ttDotOp = cast<tt::DotOpInterface>(dotOp);
  bool isHeadDot = isChainDotHead(ttDotOp);
  bool isTailDot = isChainDotTail(ttDotOp);
  // For the 1st dot in chain-dot, we always set warpsPerCTA={numWarps, 1}
  // because this eliminates
  // 1) inter-warp reduction in the softmax step.
  // 2) layout conversion from #mma to #dot_op of the second dot.
  if (isHeadDot)
    return {static_cast<unsigned>(numWarps), 1};
  // For the 2nd dot in chain-dot, we always distribute warp along dim0 first,
  // then dim1. Because
  // 1) This is how we distribute the warps for the 1st dot. Now the
  //    warpsPerCTA for the 1st dot become the warp layout of the dotOperand
  //    layout of the 2nd dot, which must match the warpsPerCTA of the 2nd dot.
  // 2) When shape[0] is small, as in decode kernels, we don't want to
  //    distribute more warps than shape[0] // mDim. If we do so, each warp
  //    needs to hold more elements in the final output, which increases
  //    register pressure, especially for large head dim (e.g. 512) attention
  //    kernels.
  if (isTailDot) {
    SmallVector<unsigned, 3> ret = {1, 1};
    ret[0] = static_cast<unsigned>(std::min(
        static_cast<int64_t>(numWarps),
        static_cast<int64_t>(llvm::divideCeil(shape[0], shapePerWarp.first))));
    ret[1] = numWarps / ret[0];
    return ret;
  }

  // Case 3: Regular cases
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
warpsPerTileWMMA(Operation *dotOp, ArrayRef<int64_t> shape, int numWarps,
                 std::pair<int64_t, int64_t> shapePerWarp) {
  return warpsPerTile(dotOp, shape, numWarps, shapePerWarp);
}

// Chooses a proper MFMA instruction that can used to compute the given dot op.
// If enforcedNonKDim is not zero, it will be used to overwrite the default
// logic to choose a MFMA with matching M/N dim.
FailureOr<MfmaIntrinsic>
chooseMfmaInstruction(Location loc, int mfmaVersion, RankedTensorType cType,
                      Type aElemType, Type bElemType, int inputKSize,
                      int enforcedNonKDim, bool withScale, bool allowXF32) {
  // number of matrix elements along k dim per one MFMA instruction
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
      // On CNDA2-4, if the element type is f64, we use 16x16 intrinsic as
      // there's no 32x32 intrinsic.
      mDim = nDim = 32;
      if (aElemType.isF64() || bElemType.isF64()) {
        mDim = nDim = 16;
      }
    } else if (minSize >= 16) {
      mDim = nDim = 16;
    } else if (minSize >= 4) {
      if (M >= 64) {
        mDim = 64;
        nDim = 4;
      } else if (N >= 64) {
        mDim = 4;
        nDim = 64;
      }
    }
  }

  FailureOr<MfmaIntrinsic> maybeMfmaIntrinsic =
      MfmaIntrinsic::selectFor(loc, mfmaVersion, mDim, nDim, inputKSize,
                               aElemType, bElemType, withScale, allowXF32);

  // Fallback to FMA if the M/N dim is not supported by MFMA.
  if (failed(maybeMfmaIntrinsic))
    return failure();

  kDim = maybeMfmaIntrinsic->kDim;
  assert(kDim != 0);
  assert(enforcedNonKDim != 0 || (M % mDim == 0 && N % nDim == 0));
  // If inputKSize % kDim != 0 (including the case where inputKSize < kDim),
  // this layout will introduce data duplication.
  if (inputKSize % kDim != 0)
    return failure();
  return maybeMfmaIntrinsic;
}

FailureOr<MfmaIntrinsic> chooseMfmaInstruction(tt::DotOp dot, int mfmaVersion,
                                               int nonKDim,
                                               bool withScale = false) {
  RankedTensorType aType = dot.getA().getType();
  bool allowXF32 =
      dot.getInputPrecision() == InputPrecision::TF32 && mfmaVersion == 3;
  return chooseMfmaInstruction(
      dot.getLoc(), mfmaVersion, dot.getC().getType(), aType.getElementType(),
      dot.getB().getType().getElementType(), aType.getShape().back(), nonKDim,
      withScale, allowXF32);
}

FailureOr<MfmaIntrinsic> chooseMfmaInstruction(tt::DotScaledOp dot,
                                               int mfmaVersion, int nonKDim) {
  using ::mlir::LLVM::AMD::scaleDotElemTypeToMLIRType;

  auto ctx = dot.getContext();
  int64_t inputKDim = dot.getA().getType().getShape().back();
  if (dot.getAElemType() == ScaleDotElemType::E2M1 && dot.getLhsKPack()) {
    // Since two fp4 are packed into int8, to get the correct K dim size, we
    // need to multiply it by 2.
    inputKDim *= 2;
  }
  Type aElemType = scaleDotElemTypeToMLIRType(ctx, dot.getAElemType());
  Type bElemType = scaleDotElemTypeToMLIRType(ctx, dot.getBElemType());
  return chooseMfmaInstruction(dot.getLoc(), mfmaVersion, dot.getC().getType(),
                               aElemType, bElemType, inputKDim, nonKDim,
                               /*withScale=*/true, /*allowXF32=*/false);
}

FailureOr<MfmaIntrinsic> chooseMfmaInstruction(tt::DotScaledOp dot,
                                               int mfmaVersion, int nonKDim,
                                               bool useFp16) {
  // For scaled dot, we handle it with fp16 or bf16 emulation for now.
  Builder b(dot.getContext());
  Type elemType = useFp16 ? b.getF16Type() : b.getBF16Type();
  return chooseMfmaInstruction(dot.getLoc(), mfmaVersion, dot.getC().getType(),
                               elemType, elemType,
                               dot.getA().getType().getShape().back(), nonKDim,
                               /*withScale=*/false, /*allowXF32=*/false);
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
      {bf16, bf16, f32, f32},
      {i8, i8, i32, i32},
      // {f16, f16, f16, f16},
      // {bf16, bf16, bf16, bf16},
      // {i4, i4, i32, i32} - are supported configurations
      // by WMMA instruction, but not supported by triton
      // clang-format on
  };
  if (version == 2 || version == 3) {
    Type fp8e4nv = rewriter.getType<Float8E4M3FNType>();
    Type fp8e5 = rewriter.getType<Float8E5M2Type>();
    applicableTypes.append({
        // clang-format off
        {fp8e4nv, fp8e4nv, f32, f32},
        {fp8e4nv, fp8e5, f32, f32},
        {fp8e5, fp8e4nv, f32, f32},
        {fp8e5, fp8e5, f32, f32},
        // clang-format on
    });
  }
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

Value findScaleAsDecompositionSource(Value v) {
  BackwardSliceOptions options;
  options.omitBlockArguments = true;
  SetVector<Operation *> slice;
  (void)getBackwardSlice(v, &slice, options);
  for (auto &op : slice) {
    if (op->hasAttrOfType<BoolAttr>(AttrDecomposedDotScaledSource)) {
      op->removeAttr(AttrDecomposedDotScaledSource);
      return op->getResult(0);
    }
  }

  return {};
}

// Figure out a best tilesPerWarp parameter that gives largest vector size for
// global load for the given |scale| tensor feeding into dot_scaled op. Returns
// the largest vector size and writes the choice to |result|.
int deduceTilesPerWarp(TypedValue<RankedTensorType> scale, unsigned opIdx,
                       unsigned nonKDim, ArrayRef<unsigned> warpsPerCTA,
                       SmallVectorImpl<unsigned> *result) {
  std::array<unsigned, 2> chosen{1, 1};
  int vecSize = 1;
  if (!scale) {
    result->assign(chosen.begin(), chosen.end());
    return vecSize;
  }

  // Source code have flexibility to preshuffle scale tensor to achieve better
  // global load vectorization. That preshuffle scheme is conveyed via some
  // tl.reshape and tl.trans op combinations. Instead of hardcoding one case or
  // pattern match the op chain here, we try certain scale tensor layouts and
  // see which one gives us better vectorization when pushed upwards to the
  // global load.
  //
  // For 16x16x128 scaled MFMA intrinsic, each thread only reads one i8 value.
  // For better vectorization, we prefer to stick 2x2 such intrinsic together so
  // each thread can read 4xi8 values.
  SmallVector<std::array<unsigned, 2>, 2> choices{{2, 2}, {1, 1}};
  for (const auto &choice : choices) {
    LLVM_DEBUG(llvm::dbgs()
               << "choice: [" << choice[0] << ", " << choice[1] << "]\n");
    LinearLayout layout = ttg::chooseScaledMfmaScaleLayout(
        scale.getContext(), opIdx, scale.getType().getShape(), nonKDim, choice,
        warpsPerCTA);
    LLVM_DEBUG(llvm::dbgs() << "trying scale layout: " << layout << "\n");

    // Infer source layout used for global load using the current scale layout.
    auto loadLayoutPair =
        ttg::inferSourceLoadLayout(layout, scale.getDefiningOp());
    if (!loadLayoutPair)
      continue;
    tt::LoadOp loadOp = loadLayoutPair->first;
    const LinearLayout &inferredLayout = loadLayoutPair->second;
    LLVM_DEBUG(llvm::dbgs()
               << "inferred load layout: " << inferredLayout << "\n");

    auto loadType = cast<RankedTensorType>(loadOp.getType());
    auto loadOrder = ttg::getOrder(loadType);
    auto loadCTALayout = ttg::getCTALayout(loadType.getEncoding());

    // Reuse existing shared memory vectorization utilities by constructing a
    // pass through layout that does linear element mapping.
    MLIRContext *context = scale.getContext();
    auto passThruShared = ttg::SwizzledSharedEncodingAttr::get(
        context, 1, 1, 1, loadOrder, loadCTALayout);
    auto sharedLL =
        triton::gpu::toLinearLayout(loadType.getShape(), passThruShared);
    auto composedLL = inferredLayout.invertAndCompose(sharedLL).flattenOuts();
    auto [v, _] =
        largestVectorisation(context, composedLL, /*bitwidth=*/8, std::nullopt);

    if (v > vecSize) {
      LLVM_DEBUG(llvm::dbgs() << "found vector size: " << v << "\n");
      chosen = choice;
      vecSize = v;
      break;
    }
  }
  result->assign(chosen.begin(), chosen.end());
  return vecSize;
}

class BlockedToMFMA : public OpRewritePattern<tt::DotOp> {
  int mfmaVersion;
  int nonKDim;
  int kPack;

public:
  BlockedToMFMA(MLIRContext *context, int mfmaVersion, int nonKDim, int kPack,
                PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), mfmaVersion(mfmaVersion),
        nonKDim(nonKDim), kPack(kPack) {}

  LogicalResult matchAndRewrite(tt::DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    using TensorValue = TypedValue<RankedTensorType>;
    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        !isa<ttg::BlockedEncodingAttr>(oldRetType.getEncoding()))
      return failure();
    if (!isa_and_nonnull<BlockedEncodingAttr>(dotOp.getType().getEncoding()))
      return rewriter.notifyMatchFailure(
          dotOp, "expected blocked encoding result tensor");

    auto CTALayout = ttg::getCTALayout(oldRetType.getEncoding());

    // get MFMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    int numWarps = ttg::lookupNumWarps(dotOp);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());
    auto ctx = oldAType.getContext();

    Type aElemType = oldAType.getElementType();
    Type bElemType = oldBType.getElementType();
    bool withScale =
        mfmaVersion == 4 && isF8F6F4(aElemType) && isF8F6F4(bElemType);

    // If mfmaVersion == 4 and both inputs are of F8F6F4 types, we will try to
    // use the V_MFMA_*_F8F6F4 instructions since it has higher FLOPs per cycle.
    // If we can't find a proper instruction, we will fall back to select from
    // normal mfma instructions.
    FailureOr<MfmaIntrinsic> mfmaInstr =
        chooseMfmaInstruction(dotOp, mfmaVersion, nonKDim, withScale);
    if (failed(mfmaInstr)) {
      if (!withScale) {
        return failure();
      }
      mfmaInstr = chooseMfmaInstruction(dotOp, mfmaVersion, nonKDim, false);
      if (failed(mfmaInstr))
        return failure();

      withScale = false;
    }

    auto mDim = mfmaInstr->mDim;
    auto nDim = mfmaInstr->nDim;
    auto kDim = mfmaInstr->kDim;
    auto kBase = mfmaInstr->kBase;

    auto warpsPerTile =
        warpsPerTileMFMA(dotOp, retShape, numWarps, {mDim, nDim});

    Type mfmaAccType;
    if (oldRetType.getElementType().isIntOrIndex())
      mfmaAccType = rewriter.getIntegerType(32);
    else if (oldRetType.getElementType().isF64())
      mfmaAccType = rewriter.getF64Type();
    else
      mfmaAccType = rewriter.getF32Type();

    // Use transposed mfma layout to enable larger vectorization for global
    // store instructions. We can not support transposed mfma 4x64 as it
    // requires to broadcast the operand A.
    bool isTransposed = !(mDim == 4 && nDim == 64);
    auto aElemTy = mfmaInstr->aElementType;
    auto is16BitElemTy = (aElemTy.isF16() || aElemTy.isBF16());

    unsigned rank = oldRetType.getRank();
    SmallVector<unsigned, 2> tilesPerWarp{1, 1};
    if (mDim == nDim && (mDim == 32 || mDim == 16)) {
      SmallVector<unsigned, 2> tilesA{1, 1}, tilesB{1, 1};
      Value scaleA = findScaleAsDecompositionSource(a);
      Value scaleB = findScaleAsDecompositionSource(b);
      int vecA = deduceTilesPerWarp(dyn_cast_if_present<TensorValue>(scaleA), 0,
                                    mDim, warpsPerTile, &tilesA);
      int vecB = deduceTilesPerWarp(dyn_cast_if_present<TensorValue>(scaleB), 1,
                                    mDim, warpsPerTile, &tilesB);
      tilesPerWarp = vecA > vecB ? tilesA : tilesB;
      LLVM_DEBUG(llvm::dbgs() << "chosen tilesPerWarp: [" << tilesPerWarp[0]
                              << ", " << tilesPerWarp[1] << "]\n");
    }

    bool hasPreShuffledScale = (tilesPerWarp[0] > 1 && tilesPerWarp[1] > 1);

    // Set tilesPerWarp and isTransposed to enable intra warp conversion for
    // the mfma16x16 layout of a dot op, depending on whether
    // its result is used by operand 0 or operand 1 of another dot op.
    if (mfmaVersion == 4 && is16BitElemTy && mDim == 16 && nDim == 16 &&
        rank == 2 && !hasPreShuffledScale) {
      if (isChainDotHead(dotOp, 0u) &&
          retShape.front() >= 16 * 2 * warpsPerTile.front() &&
          retShape.back() == 16 && warpsPerTile.back() == 1) {
        isTransposed = true;
        tilesPerWarp = {2, 1};
      } else if (isChainDotHead(dotOp, 1u) && retShape.front() == 16 &&
                 retShape.back() >= 16 * 2 * warpsPerTile.back() &&
                 warpsPerTile.front() == 1) {
        isTransposed = false;
        tilesPerWarp = {1, 2};
      }
    }

    if (rank == 3) {
      tilesPerWarp.insert(tilesPerWarp.begin(), 1);
    }

    ttg::AMDMfmaEncodingAttr mfmaEnc = ttg::AMDMfmaEncodingAttr::get(
        oldRetType.getContext(), mfmaVersion, warpsPerTile, {mDim, nDim, kDim},
        isTransposed, CTALayout, tilesPerWarp,
        mfmaAccType.getIntOrFloatBitWidth());

    // convert accumulator
    auto oldAcc = dotOp.getC();
    auto newAcc = convertAndCastTensor(rewriter, oldAcc, mfmaEnc, mfmaAccType);

    // Here is a brief explanation of kWidth, kBase, and kDim
    // 1. kWidth: the number of **consecutive** elements each thread loads from
    //    shared memory in preparation for mfma instructions. In theory, each
    //    thread can issue multiple ds_read to load elements from non-contiguous
    //    addresses in shared memory for one mfma instruction, but that won't be
    //    good for performance. So in practice for better vectorization, we
    //    make sure the kWidth elements can be loaded from shared memory by a
    //    single ds_read instruction by setting vecSize of the sharedLayout
    //    to be kWidth.
    // 2. kDim: the k dimension size of the mfma instruction. E.g. instruction
    //    mfma_32x32x16 has kDim = 16, meaning this mfma instruction can compute
    //    a matmul of operands with shape 32x16 and 16x32.
    // 3. kBase: the number of elements each thread holds for a single mfma
    //    instruction.
    // 4. relation between kBase and kDim:
    //    4.1 For mfma_32, kBase = kDim / 2
    //    4.2 For mfma_16, kBase = kDim / 4
    //    4.3 For mfma_4, kBase = kDim / 16
    // 5. relation between kWidth and kBase: For now it supports two cases
    //    5.1 kWidth = kBase, i.e. kPack = 1. In this case, each load from
    //        shared memory results in one mfma instruction.
    //    5.2 kWidth = 2 * kBase, i.e. kPack = 2. In this case, each load from
    //        shared memory results in two mfma instructions, since one mfma
    //        can only consume kBase elements from each thread.
    //    Note that we cannot have larger kPack since kPack = 2 means
    //    ds_read_b128, which is the largest vector size for shared memory load.
    auto kWidth = kBase;

    // We want to extend kWidth by kPack (kPack=1 means no extension)
    // to increase ds_read vector size
    // However, in FA, the second dot can only use kWidth = kBase since it's
    // limited by the result of the first dot, which is of mfmaLayout.
    auto isDotChainTail = isChainDotTail(dotOp);
    if (!isDotChainTail)
      kWidth *= kPack;

    // For FA fwd kernel with f16 elementTy, we limit the 2nd dot to have
    // kWidth = 4 so that the coversion from #mma (result of 1st dot)
    // to #dotOp (operand 0 of 2nd dot) is a no-op.
    // TODO (lixun): relax the condition for 8-bit elementTy.
    if (is16BitElemTy && isDotChainTail) {
      kWidth = 4;
    }

    Value newDot;
    if (withScale) {
      // If a scaled mfma instruction is chosen, we will rewrite the DotOp to a
      // DotScaledOp.
      auto aScaledElemTy = mlirTypeToScaledElemType(aElemType);
      auto bScaledElemTy = mlirTypeToScaledElemType(bElemType);
      if (failed(aScaledElemTy) || failed(bScaledElemTy))
        return failure();

      assert(kWidth == 32);
      auto newAEncoding =
          DotOperandEncodingAttr::get(ctx, 0, mfmaEnc, kWidth / 2);
      auto newBEncoding =
          DotOperandEncodingAttr::get(ctx, 1, mfmaEnc, kWidth / 2);

      a = convertAndCastTensor(rewriter, a, newAEncoding,
                               mfmaInstr->aElementType);
      b = convertAndCastTensor(rewriter, b, newBEncoding,
                               mfmaInstr->bElementType);
      newDot = rewriter.create<triton::DotScaledOp>(
          dotOp.getLoc(), newAcc.getType(), a, b, newAcc, Value(), Value(),
          aScaledElemTy.value(), bScaledElemTy.value(), /*fastMath=*/false);
    } else {
      auto newAEncoding =
          ttg::DotOperandEncodingAttr::get(ctx, 0, mfmaEnc, kWidth);
      auto newBEncoding =
          ttg::DotOperandEncodingAttr::get(ctx, 1, mfmaEnc, kWidth);
      a = convertAndCastTensor(rewriter, a, newAEncoding,
                               mfmaInstr->aElementType);
      b = convertAndCastTensor(rewriter, b, newBEncoding,
                               mfmaInstr->bElementType);
      newDot = rewriter.create<tt::DotOp>(dotOp.getLoc(), newAcc.getType(), a,
                                          b, newAcc, dotOp.getInputPrecision(),
                                          dotOp.getMaxNumImpreciseAcc());
    }

    Value dotOutput =
        convertAndCastTensor(rewriter, newDot, oldRetType.getEncoding(),
                             oldRetType.getElementType());

    rewriter.replaceOp(dotOp, dotOutput);

    return success();
  }
};

class ScaledBlockedToMFMA final : public OpRewritePattern<triton::DotScaledOp> {
  int mfmaVersion;
  int nonKDim;
  int kPack;

public:
  ScaledBlockedToMFMA(MLIRContext *context, int mfmaVersion, int nonKDim,
                      int kPack, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), mfmaVersion(mfmaVersion),
        nonKDim(nonKDim), kPack(kPack) {}

  LogicalResult matchAndRewrite(triton::DotScaledOp dotOp,
                                PatternRewriter &rewriter) const override {
    // TODO: add support for m/n packed formats.
    if (!dotOp.getLhsKPack() || !dotOp.getRhsKPack())
      return failure();
    using TensorValue = TypedValue<RankedTensorType>;

    RankedTensorType oldRetType = dotOp.getType();
    if (!isa_and_nonnull<BlockedEncodingAttr>(oldRetType.getEncoding()))
      return rewriter.notifyMatchFailure(
          dotOp, "expected blocked encoding result tensor");
    unsigned rank = oldRetType.getRank();
    if (rank == 3)
      return rewriter.notifyMatchFailure(dotOp, "NYI: 3d case");

    TensorValue a = dotOp.getA();
    TensorValue b = dotOp.getB();
    TensorValue aScale = dotOp.getAScale();
    TensorValue bScale = dotOp.getBScale();
    if (aScale && bScale)
      return rewriter.notifyMatchFailure(dotOp, "NYI: both LHS and RHS scale");

    ScaleDotElemType aElemType = dotOp.getAElemType();
    ScaleDotElemType bElemType = dotOp.getBElemType();
    if (!isF16F8F4(aElemType) || !isF16F8F4(bElemType))
      return rewriter.notifyMatchFailure(dotOp, "NYI: mxfp6 operand");

    MLIRContext *ctx = dotOp.getContext();
    auto moduleOp = dotOp->getParentOfType<ModuleOp>();
    int numWarps = ttg::lookupNumWarps(dotOp);

    ttg::CTALayoutAttr ctaLayout = ttg::getCTALayout(oldRetType.getEncoding());
    int numThreads = ttg::TritonGPUDialect::getThreadsPerWarp(moduleOp);

    // Choose a suitable MFMA instruction for this scaled dot op.
    bool useFp16 = aElemType == ScaleDotElemType::FP16 ||
                   bElemType == ScaleDotElemType::FP16;
    FailureOr<MfmaIntrinsic> mfmaInstr =
        chooseMfmaInstruction(dotOp, mfmaVersion, nonKDim, useFp16);
    if (failed(mfmaInstr))
      return rewriter.notifyMatchFailure(dotOp, "cannot choose mfma intrinsic");

    if (useFp16) {
      dotOp.emitRemark(
          "Warning: detected one dot_scaled operand is fp16 tensor so "
          "upcasting to fp16 for computation, which impacts precision; "
          "experimental behavior and may change in future");
    }

    unsigned mDim = mfmaInstr->mDim;
    unsigned nDim = mfmaInstr->nDim;
    unsigned kDim = mfmaInstr->kDim;
    unsigned kBase = mfmaInstr->kBase;

    // For mxfp4 A/B tensor, we pack every two values into one int8 value there.
    // For such cases, we have different initial kWidth for LHS and RHS, which
    // will be "fixed" later by using upcast_mxfp to convert LHS to unpacked
    // values. For such packed cases, we cannot support flexible kPack choices
    // from the developer--it just does not apply here. So mandate the choice
    // here.
    bool isAPacked = aElemType == ScaleDotElemType::E2M1;
    bool isBPacked = bElemType == ScaleDotElemType::E2M1;
    bool isPacked = isAPacked || isBPacked;
    unsigned kWidths[] = {isPacked ? (isAPacked ? 4 : 8) : kBase * kPack,
                          isPacked ? (isAPacked ? 8 : 4) : kBase * kPack};

    // For A/B tensor, 32 consecutive elements along K dim share the same scale.
    // We'd like to keep the scale values together with the base values in the
    // same warp to avoid cross-warp data exchange. It means we want warpsPerCTA
    // = 1 along the N/M dimension for the mxfp A/B case. We achieve that by
    // setting the M/N dimension as numWarps.
    SmallVector<unsigned, 2> mfmaWarpsPerCTA(rank, 1);
    mfmaWarpsPerCTA[aScale ? 0 : 1] = numWarps;

    // Always use transposed mfma layout. This enables larger vectorization
    // for global store instructions.
    auto elementBitWidth = oldRetType.getElementType().getIntOrFloatBitWidth();
    auto mfmaEnc = ttg::AMDMfmaEncodingAttr::get(
        ctx, mfmaVersion, mfmaWarpsPerCTA, {mDim, nDim, kDim},
        /*isTransposed=*/true, ctaLayout, {}, elementBitWidth);

    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), mfmaEnc);

    auto newAcc = rewriter.create<ttg::ConvertLayoutOp>(
        dotOp.getC().getLoc(), newRetType, dotOp.getC());

    auto upcastForMMA = [&](TensorValue v, int idx,
                            ScaleDotElemType type) -> TensorValue {
      auto vType = v.getType();
      auto newVEncoding = DotOperandEncodingAttr::get(
          ctx, idx, newRetType.getEncoding(), kWidths[idx]);
      auto newVType = RankedTensorType::get(
          vType.getShape(), vType.getElementType(), newVEncoding);
      v = rewriter.create<ttg::ConvertLayoutOp>(v.getLoc(), newVType, v);
      // Don't need to covert int8 holding mxfp4--the upcast_mxfp op can
      // take int8 tensor as input.
      if (type == ScaleDotElemType::BF16 || type == ScaleDotElemType::FP16 ||
          type == ScaleDotElemType::E2M1)
        return v;

      auto upcastedType = RankedTensorType::get(
          vType.getShape(),
          useFp16 ? rewriter.getF16Type() : rewriter.getBF16Type(),
          newVEncoding);
      return cast<TensorValue>(
          rewriter.create<FpToFpOp>(v.getLoc(), upcastedType, v).getResult());
    };
    a = upcastForMMA(a, 0, aElemType);
    b = upcastForMMA(b, 1, bElemType);

    // We need to have "matching" encoding between the main tensor and scale
    // tensor to make sure the scale values needed is in the same warp. So we
    // adopt the same CTA layout and warps per CTA. The warp dimensions needs to
    // match along M/N dimension too. With in a warp, we have 64 threads. We let
    // each thread read in one scale value. So we need a threadsPerWarp =
    // mDim/nDim along M/N dimension. Note that For MFMA intrinsics, mDim is
    // always the same as nDim. And for scaled dot scale tensor, we always have
    // K as the innermost dimension. So we have the same threadsPerWarp in the
    // below no matter A or B scale. Similarly for warpsPerCTA, the non-K
    // dimension is always at index 0.
    assert(mDim == nDim);
    SmallVector<unsigned, 2> threadsPerWarp = {mDim, numThreads / mDim};
    SmallVector<unsigned, 2> blockWarpsPerCTA(rank, 1);
    blockWarpsPerCTA[0] = numWarps;
    auto newScaleEncoding = triton::gpu::BlockedEncodingAttr::get(
        ctx, {1, 1}, threadsPerWarp, blockWarpsPerCTA, {1, 0}, ctaLayout);

    auto upcastMXFP = [&](TensorValue v, TensorValue scale,
                          ScaleDotElemType elemType, bool fastMath) -> Value {
      if (!scale)
        return v;

      auto newScaleType = RankedTensorType::get(
          scale.getType().getShape(), scale.getType().getElementType(),
          newScaleEncoding);
      auto convOp = rewriter.create<ttg::ConvertLayoutOp>(scale.getLoc(),
                                                          newScaleType, scale);

      Builder b(v.getContext());
      // TODO: Emit device assert to check scale tensor range fitting into fp16?
      Type outputElemType = useFp16 ? b.getF16Type() : b.getBF16Type();
      auto outputType =
          amdgpu::UpcastMXFPOp::deduceOutputType(v, elemType, outputElemType);
      return rewriter.create<amdgpu::UpcastMXFPOp>(
          dotOp.getLoc(), outputType, v, convOp, elemType, fastMath);
    };

    Value scaledA =
        upcastMXFP(a, aScale, dotOp.getAElemType(), dotOp.getFastMath());
    Value scaledB =
        upcastMXFP(b, bScale, dotOp.getBElemType(), dotOp.getFastMath());
    auto newDot = rewriter.create<DotOp>(dotOp.getLoc(), newRetType, scaledA,
                                         scaledB, newAcc);
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(dotOp, oldRetType,
                                                      newDot);
    return success();
  }
};

class DecomposeAMDScaledBlocked final : public ttg::DecomposeScaledBlocked {
public:
  DecomposeAMDScaledBlocked(MLIRContext *context, PatternBenefit benefit = 1)
      : ttg::DecomposeScaledBlocked(context, benefit) {}
  using TensorValue = TypedValue<RankedTensorType>;

  RankedTensorType getScaleType(RankedTensorType vType, int32_t kDim,
                                bool isFp4) const {
    if (!isFp4)
      return vType;

    // We want scale to have the same layout as the operand. But Fp4 operand
    // is packed along kDim. So we need to double the shape to fit scale.
    auto packedShape = llvm::to_vector(vType.getShape());
    packedShape[kDim] *= 2;
    return vType.clone(packedShape);
  }

  TensorValue scaleArg(PatternRewriter &rewriter, triton::DotScaledOp dotOp,
                       int opIdx, FloatType computeType) const override {
    TensorValue v = (opIdx == 0) ? dotOp.getA() : dotOp.getB();
    TensorValue scale = (opIdx == 0) ? dotOp.getAScale() : dotOp.getBScale();
    ScaleDotElemType elemType =
        (opIdx == 0) ? dotOp.getAElemType() : dotOp.getBElemType();

    // 1) If it's fp16/bf16, we don't upcast
    if (elemType == ScaleDotElemType::BF16 ||
        elemType == ScaleDotElemType::FP16)
      return v;

    // 2) If it's non-scaled F8F4, we reuse the common path
    if (!scale) {
      return ttg::DecomposeScaledBlocked::scaleArg(rewriter, dotOp, opIdx,
                                                   computeType);
    }

    RankedTensorType vType = v.getType();
    unsigned rank = vType.getRank();
    int32_t kDim = opIdx == 0 ? rank - 1 : rank - 2;
    auto vType16 = vType.clone(computeType);
    auto loc = dotOp.getLoc();
    bool isFp4 = (elemType == ScaleDotElemType::E2M1);

    RankedTensorType scaleType16 = getScaleType(vType16, kDim, isFp4);

    // Mark scale to simplify pattern matching during deducing TilesPerWarp
    scale.getDefiningOp()->setAttr(AttrDecomposedDotScaledSource,
                                   BoolAttr::get(rewriter.getContext(), true));

    // 3) Cast scale to bf16, broadcast it and convert the layout
    FloatType bf16Type = rewriter.getBF16Type();
    auto reshapeScale = extendAndBroadcastScale(
        rewriter, dotOp, scale, bf16Type, scaleType16.clone(bf16Type), opIdx);

    // 4) Upcast with scale
    TensorValue result;
    if (isFp4) {
      result = rewriter.create<triton::amdgpu::ScaledUpcastFp4Op>(
          loc, scaleType16, v, reshapeScale, kDim);
    } else {
      result = rewriter.create<triton::amdgpu::ScaledUpcastFp8Op>(
          loc, scaleType16, v, reshapeScale);
    }

    // 5) If the scale is NaN, return NaN, else return the scaled value.
    return maskNan(rewriter, dotOp, result, scale, kDim);
  }
};

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

    TensorValue a = dotOp.getA();
    TensorValue b = dotOp.getB();
    TensorValue aScale = dotOp.getAScale();
    TensorValue bScale = dotOp.getBScale();
    auto oldShape = oldRetType.getShape();

    ScaleDotElemType aElemType = dotOp.getAElemType();
    ScaleDotElemType bElemType = dotOp.getBElemType();
    auto supportsTypes = [](ScaleDotElemType elemType) {
      return elemType == ScaleDotElemType::E2M1 ||
             elemType == ScaleDotElemType::E4M3 ||
             elemType == ScaleDotElemType::E5M2;
    };

    if (!supportsTypes(aElemType) || !supportsTypes(bElemType)) {
      return rewriter.notifyMatchFailure(dotOp, "NYI: mxfp6");
    }

    bool bothScalesAbsent = !aScale && !bScale;

    MLIRContext *ctx = dotOp.getContext();

    ttg::CTALayoutAttr ctaLayout = ttg::getCTALayout(oldRetType.getEncoding());
    unsigned numWarps = ttg::lookupNumWarps(dotOp);
    if (numWarps == 1)
      return rewriter.notifyMatchFailure(dotOp,
                                         "num_warps==1 is not supported");

    // Choose a suitable Scaled MFMA instruction for this scaled dot op.
    FailureOr<MfmaIntrinsic> mfmaInstr =
        chooseMfmaInstruction(dotOp, mfmaVersion, nonKDim);
    if (failed(mfmaInstr))
      return rewriter.notifyMatchFailure(dotOp,
                                         "cannot choose scaled mfma intrinsic");

    auto mDim = mfmaInstr->mDim;
    auto nDim = mfmaInstr->nDim;
    auto kDim = mfmaInstr->kDim;
    auto kBase = mfmaInstr->kBase;
    assert(mDim == nDim);

    auto warpsPerTile =
        warpsPerTileMFMA(dotOp, oldShape, numWarps, {mDim, nDim});

    SmallVector<unsigned, 2> tilesA{1, 1}, tilesB{1, 1}, tilesPerWarp;
    int vecA = deduceTilesPerWarp(aScale, 0, mDim, warpsPerTile, &tilesA);
    int vecB = deduceTilesPerWarp(bScale, 1, mDim, warpsPerTile, &tilesB);
    tilesPerWarp = vecA > vecB ? tilesA : tilesB;
    LLVM_DEBUG(llvm::dbgs() << "chosen tilesPerWarp: [" << tilesPerWarp[0]
                            << ", " << tilesPerWarp[1] << "]\n");

    // Always use transposed mfma layout. This enables larger vectorization
    // for global store instructions.
    auto elementBitWidth = oldRetType.getElementType().getIntOrFloatBitWidth();
    mlir::Attribute mfmaEnc = ttg::AMDMfmaEncodingAttr::get(
        ctx, mfmaVersion, warpsPerTile, {mDim, nDim, kDim},
        /*isTransposed=*/true, ctaLayout, tilesPerWarp, elementBitWidth);

    auto newRetType =
        RankedTensorType::get(oldShape, oldRetType.getElementType(), mfmaEnc);

    auto newAcc = rewriter.create<ttg::ConvertLayoutOp>(
        dotOp.getC().getLoc(), newRetType, dotOp.getC());

    auto order = ttg::getMatrixOrder(rank, /*rowMajor=*/true);
    auto standardOutDims = standardOutDimNames(ctx, rank);

    // For the mfma_scale_f32_*_f8f6f4 instructions, each thread consumes 32
    // elements. But since two fp4 elements are packed into one int8, the
    // kWidth is 16 for fp4.
    const unsigned kWidth = kBase;
    assert(kWidth == 32);
    using basisT = std::vector<std::vector<int32_t>>;

    auto aShape = a.getType().getShape();
    auto bShape = b.getType().getShape();
    auto aEncLL = LinearLayout::empty();
    auto bEncLL = LinearLayout::empty();

    auto convertInputLayout = [&](TensorValue v,
                                  unsigned opIdx) -> TensorValue {
      auto vType = v.getType();

      auto newEnc =
          DotOperandEncodingAttr::get(ctx, opIdx, mfmaEnc, kWidth / 2);

      bool kPacked = opIdx == 0 ? dotOp.getLhsKPack() : dotOp.getRhsKPack();
      if (kPacked == false) {
        // This is FP4 with M/N packing. Create local alloc + local load here
        // so we have control of the shared layout
        // A, M packed: tensor<16x64xi8> --> 32x32
        // B, N packed: tensor<64x16xi8> --> 32x32
        SmallVector<int64_t> newShape(vType.getShape());
        newShape[opIdx == 0 ? 0 : 1] = newShape[opIdx == 0 ? 0 : 1] * 2;
        newShape[opIdx == 0 ? 1 : 0] = newShape[opIdx == 0 ? 1 : 0] / 2;
        auto newVType =
            RankedTensorType::get(newShape, vType.getElementType(), newEnc);
        OpBuilder builder(dotOp);
        auto srcEncoding = vType.getEncoding();
        auto originalOrder = triton::gpu::getOrderForMemory(vType);
        SmallVector<unsigned> newOrder = originalOrder;
        if (opIdx == 1) {
          newOrder = {1, 0};
        } else {
          newOrder = {0, 1};
        }
        auto sharedMemorySpace =
            triton::gpu::SharedMemorySpaceAttr::get(vType.getContext());
        auto tmpType = triton::gpu::MemDescType::get(
            vType.getShape(), vType.getElementType(),
            triton::gpu::SwizzledSharedEncodingAttr::get(
                v.getContext(), newEnc, vType.getShape(), newOrder,
                triton::gpu::getCTALayout(srcEncoding), vType.getElementType()),
            sharedMemorySpace);
        auto tmp = builder.create<triton::gpu::LocalAllocOp>(dotOp.getLoc(),
                                                             tmpType, v);
        auto newConvert =
            builder.create<triton::amdgpu::LocalLoadPackedTransposedOp>(
                dotOp.getLoc(), newVType, tmp);
        if (opIdx == 0) {
          aShape = newConvert.getType().getShape();
          aEncLL *= newEnc.toLinearLayout(aShape);
        } else {
          bShape = newConvert.getType().getShape();
          bEncLL *= newEnc.toLinearLayout(bShape);
        }
        return newConvert;
      } else {
        if (opIdx == 0)
          aEncLL *= newEnc.toLinearLayout(aShape);
        else
          bEncLL *= newEnc.toLinearLayout(bShape);
        auto newVType = RankedTensorType::get(vType.getShape(),
                                              vType.getElementType(), newEnc);
        return rewriter.create<ttg::ConvertLayoutOp>(v.getLoc(), newVType, v);
      }
    };
    a = convertInputLayout(a, 0);
    b = convertInputLayout(b, 1);

    StringAttr kWarp = StringAttr::get(ctx, "warp");
    auto convertScaleLayout = [&](TensorValue scale,
                                  llvm::ArrayRef<int64_t> valShape,
                                  LinearLayout dotLL, int idx) -> Value {
      if (bothScalesAbsent)
        return Value();

      SmallVector<int64_t> shape;
      if (!scale) {
        int64_t nonKDim = idx == 0 ? valShape[0] : valShape[1];
        int64_t k = idx == 0 ? valShape[1] : valShape[0];
        ScaleDotElemType &elemType = idx == 0 ? aElemType : bElemType;
        int packSize = elemType == ScaleDotElemType::E2M1 ? 2 : 1;
        shape = {nonKDim, k * packSize / 32};
      } else {
        shape = llvm::to_vector(scale.getType().getShape());
      }

      LinearLayout newLL = ttg::chooseScaledMfmaScaleLayout(
          ctx, idx, shape, mDim, tilesPerWarp, warpsPerTile);

      Attribute newScaleEncoding = ttg::LinearEncodingAttr::get(ctx, newLL);
      // Scale's data type is always i8
      auto newScaleType = RankedTensorType::get(shape, i8_ty, newScaleEncoding);

      if (!scale) {
        // 0x7F is 1.0 in E8M0
        return rewriter.create<arith::ConstantOp>(
            dotOp->getLoc(), newScaleType,
            DenseElementsAttr::get(newScaleType, llvm::APInt(8, 0x7F)));
      } else {
        return rewriter.create<ttg::ConvertLayoutOp>(scale.getLoc(),
                                                     newScaleType, scale);
      }
    };
    auto newAScale =
        convertScaleLayout(aScale, aShape, aEncLL, /*dotOperandIdx=*/0);
    auto newBScale =
        convertScaleLayout(bScale, bShape, bEncLL, /*dotOperandIdx=*/1);

    auto newDot = rewriter.create<triton::DotScaledOp>(
        dotOp.getLoc(), newRetType, a, b, newAcc, newAScale, newBScale,
        aElemType, bElemType, dotOp.getFastMath());

    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(dotOp, oldRetType,
                                                      newDot);

    return success();
  }
};

class ScaledBlockedToScaledWMMAF8F6F4 final
    : public OpRewritePattern<triton::DotScaledOp> {
  int wmmaVersion;

public:
  ScaledBlockedToScaledWMMAF8F6F4(MLIRContext *context, int wmmaVersion,
                                  PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), wmmaVersion(wmmaVersion) {}

  LogicalResult matchAndRewrite(triton::DotScaledOp dotOp,
                                PatternRewriter &rewriter) const override {
    using TensorValue = TypedValue<RankedTensorType>;

    if (wmmaVersion != 3) {
      return rewriter.notifyMatchFailure(
          dotOp, "F8F6F4 scaled dot is only natively supported on gfx1250");
    }

    RankedTensorType oldRetType = dotOp.getType();
    if (!isa_and_nonnull<BlockedEncodingAttr>(oldRetType.getEncoding())) {
      return rewriter.notifyMatchFailure(
          dotOp, "expected blocked encoding result tensor");
    }

    unsigned rank = oldRetType.getRank();
    if (rank == 3)
      return rewriter.notifyMatchFailure(dotOp, "NYI: 3d case");

    TensorValue a = dotOp.getA();
    TensorValue b = dotOp.getB();
    TensorValue aScale = dotOp.getAScale();
    TensorValue bScale = dotOp.getBScale();
    auto oldShape = oldRetType.getShape();

    ScaleDotElemType aElemType = dotOp.getAElemType();
    ScaleDotElemType bElemType = dotOp.getBElemType();
    // TODO: Add more supported types
    auto supportsTypes = [](ScaleDotElemType elemType) {
      return elemType == ScaleDotElemType::E2M1;
    };

    if (!supportsTypes(aElemType) || !supportsTypes(bElemType)) {
      return rewriter.notifyMatchFailure(dotOp, "Not supported yet mxfp type");
    }

    MLIRContext *ctx = dotOp.getContext();

    ttg::CTALayoutAttr ctaLayout = ttg::getCTALayout(oldRetType.getEncoding());
    unsigned numWarps = ttg::lookupNumWarps(dotOp);

    constexpr unsigned mDim = 16;
    constexpr unsigned nDim = 16;
    constexpr unsigned kDim = 128;

    auto warpsPerTile =
        warpsPerTileWMMA(dotOp, oldShape, numWarps, {mDim, nDim});

    auto wmmaEnc = ttg::AMDWmmaEncodingAttr::get(
        ctx, wmmaVersion, true, warpsPerTile, ctaLayout, {mDim, nDim, kDim});
    auto wmmaPackedEnc =
        ttg::AMDWmmaEncodingAttr::get(ctx, wmmaVersion, true, warpsPerTile,
                                      ctaLayout, {mDim, nDim, kDim / 2});

    auto newRetType =
        RankedTensorType::get(oldShape, oldRetType.getElementType(), wmmaEnc);

    auto newAcc = rewriter.create<ttg::ConvertLayoutOp>(
        dotOp.getC().getLoc(), newRetType, dotOp.getC());

    StringAttr kRegister = StringAttr::get(ctx, "register");
    StringAttr kLane = StringAttr::get(ctx, "lane");
    StringAttr kWarp = StringAttr::get(ctx, "warp");
    StringAttr kBlock = StringAttr::get(ctx, "block");

    auto order = ttg::getMatrixOrder(rank, /*rowMajor=*/true);
    auto standardOutDims = standardOutDimNames(ctx, rank);

    using basisT = std::vector<std::vector<int32_t>>;

    auto aShape = a.getType().getShape();
    auto bShape = b.getType().getShape();

    auto aEncLL = LinearLayout::empty();
    auto bEncLL = LinearLayout::empty();

    auto convertInputLayout = [&](TensorValue v, unsigned opIdx,
                                  bool isFp4) -> TensorValue {
      auto parent = isFp4 ? wmmaPackedEnc : wmmaEnc;
      auto vType = v.getType();
      auto newEnc = DotOperandEncodingAttr::get(ctx, opIdx, parent, 16);
      auto newVType = RankedTensorType::get(vType.getShape(),
                                            vType.getElementType(), newEnc);
      (opIdx == 0 ? aEncLL : bEncLL) *=
          newEnc.toLinearLayout(opIdx == 0 ? aShape : bShape);
      return rewriter.create<ttg::ConvertLayoutOp>(v.getLoc(), newVType, v);
    };
    a = convertInputLayout(a, 0, aElemType == ScaleDotElemType::E2M1);
    b = convertInputLayout(b, 1, bElemType == ScaleDotElemType::E2M1);

    auto convertScaleLayout = [&](TensorValue scale,
                                  llvm::ArrayRef<int64_t> valShape,
                                  LinearLayout dotLL, int idx) -> Value {
      LinearLayout::BasesT scaleBases = dotLL.getBases();
      auto &warpBases = scaleBases[kWarp];

      SmallVector<int64_t> shape;
      if (!scale) {
        int64_t nonKDim = idx == 0 ? valShape[0] : valShape[1];
        int64_t k = idx == 0 ? valShape[1] : valShape[0];
        ScaleDotElemType &elemType = idx == 0 ? aElemType : bElemType;
        int packSize = elemType == ScaleDotElemType::E2M1 ? 2 : 1;
        shape = {nonKDim, k * packSize / 32};
      } else {
        shape = llvm::to_vector(scale.getType().getShape());
      }

      LinearLayout newLL =
          ttg::chooseScaledWmmaScaleLayout(ctx, idx, warpBases, shape);
      Attribute newScaleEncoding = ttg::LinearEncodingAttr::get(ctx, newLL);
      // Scale's data type is always i8
      auto newScaleType = RankedTensorType::get(shape, i8_ty, newScaleEncoding);

      if (!scale) {
        // 0x7F is 1.0 in E8M0
        return rewriter.create<arith::ConstantOp>(
            dotOp->getLoc(), newScaleType,
            DenseElementsAttr::get(newScaleType, llvm::APInt(8, 0x7F)));
      } else {
        return rewriter.create<ttg::ConvertLayoutOp>(scale.getLoc(),
                                                     newScaleType, scale);
      }
    };
    auto newAScale =
        convertScaleLayout(aScale, aShape, aEncLL, /*dotOperandIdx=*/0);
    auto newBScale =
        convertScaleLayout(bScale, bShape, bEncLL, /*dotOperandIdx=*/1);

    auto newDot = rewriter.create<triton::DotScaledOp>(
        dotOp.getLoc(), newRetType, a, b, newAcc, newAScale, newBScale,
        aElemType, bElemType, dotOp.getFastMath());

    auto m = dotOp->getParentOfType<ModuleOp>();
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

// Promote operands of dot op if the existing combination is not natively
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

FailureOr<WmmaIntrinsic> chooseWmmaInstruction(Location loc, int wmmaVersion,
                                               RankedTensorType cType,
                                               Type aElemType, Type bElemType,
                                               Type cElemType, int inputKSize,
                                               int enforcedNonKDim) {
  // number of matrix elements along k dim per one WMMA instruction
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
    if (minSize >= 16) {
      mDim = 16;
      nDim = 16;
    }
  }
  if (mDim == 0 || nDim == 0)

    return failure();

  FailureOr<WmmaIntrinsic> maybeWmmaIntrinsic = WmmaIntrinsic::selectFor(
      wmmaVersion, mDim, nDim, inputKSize, aElemType, bElemType, cElemType);
  if (failed(maybeWmmaIntrinsic))
    return emitError(loc, "no matching matrix core intrinsic due to "
                          "unsupported element type: A=")
           << aElemType << " B=" << bElemType << " C=" << cElemType;

  kDim = maybeWmmaIntrinsic->kDim;
  assert(kDim != 0);
  assert(enforcedNonKDim != 0 || (M % mDim == 0 && N % nDim == 0));
  // if inputKSize % kDim != 0 this layout will introduce data duplication,
  // consider FMA dot is preferred, except cases Wmma layout is enforced.
  if (enforcedNonKDim == 0 && inputKSize % kDim != 0)
    return failure();
  return maybeWmmaIntrinsic;
}

FailureOr<WmmaIntrinsic> chooseWmmaInstruction(tt::DotOp dot,
                                               OperandTypesVector operandTypes,
                                               int wmmaVersion, int nonKDim) {

  return chooseWmmaInstruction(dot.getLoc(), wmmaVersion, dot.getC().getType(),
                               operandTypes[0], operandTypes[1],
                               operandTypes[2],
                               dot.getA().getType().getShape().back(), nonKDim);
}

class BlockedToWMMA : public OpRewritePattern<tt::DotOp> {
  int wmmaVersion;
  int nonKDim;

public:
  BlockedToWMMA(MLIRContext *context, int wmmaVersion, int nonKDim,
                PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), wmmaVersion(wmmaVersion),
        nonKDim(nonKDim) {}

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

    // get operand types
    auto operandTypes = getOperandTypesForWmmaOp(rewriter, dotOp, wmmaVersion);
    if (operandTypes.empty())
      return failure();

    // check shape
    FailureOr<WmmaIntrinsic> wmmaInstr =
        chooseWmmaInstruction(dotOp, operandTypes, wmmaVersion, nonKDim);
    if (failed(wmmaInstr)) {
      return failure();
    }

    auto mDim = wmmaInstr->mDim;
    auto nDim = wmmaInstr->nDim;
    auto kDim = wmmaInstr->kDim;
    auto kBase = wmmaInstr->kBase;

    // get WMMA encoding for the given number of warps
    int numWarps = ttg::lookupNumWarps(dotOp);

    ttg::AMDWmmaEncodingAttr wmmaEnc;

    auto warpsPerTile =
        warpsPerTileWMMA(dotOp, retShape, numWarps, {mDim, nDim});

    auto CTALayout = ttg::getCTALayout(oldRetEncoding);

    // Use transposed wmma layout to enable larger vectorization for global
    // store instructions.
    bool isTransposed = true;
    wmmaEnc = ttg::AMDWmmaEncodingAttr::get(ctx, wmmaVersion, isTransposed,
                                            warpsPerTile, CTALayout,
                                            {mDim, nDim, kDim});

    auto newRetType = RankedTensorType::get(retShape, operandTypes[3], wmmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getC();
    auto newAcc =
        convertAndCastTensor(rewriter, oldAcc, wmmaEnc, operandTypes[2]);

    // kBase, kWidth and kDim follow the same logic as in mfma
    // for now kwidth = kbase always
    auto kWidth = kBase;
    auto newAType = RankedTensorType::get(
        aShape, operandTypes[0],
        ttg::DotOperandEncodingAttr::get(ctx, 0, wmmaEnc, kWidth));
    auto newBType = RankedTensorType::get(
        bShape, operandTypes[1],
        ttg::DotOperandEncodingAttr::get(ctx, 1, wmmaEnc, kWidth));

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
      // When converting a floating point number with a smaller precision (such
      // as float16) to one with a larger precision (such as float32), no
      // rounding occurs. There is no need for, nor does it involve, a rounding
      // mode. This kind of conversion is exact and lossless.
      RoundingModeAttr rmode;
      if (srcElTy.getIntOrFloatBitWidth() > dstElTy.getIntOrFloatBitWidth()) {
        rmode =
            RoundingModeAttr::get(rewriter.getContext(), RoundingMode::RTNE);
      }
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

      // CDNA4 has Bf16 v_dot2
      if (AMD::deduceISAFamily(arch) == ISAFamily::CDNA4 &&
          dotTypes.a.isBF16() && dotTypes.b.isBF16() && dotTypes.c.isF32() &&
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
      if (!elTy.isF16() && !elTy.isF32() && !elTy.isF64())
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

#define GEN_PASS_DEF_TRITONAMDGPUACCELERATEMATMUL
#include "TritonAMDGPUTransforms/Passes.h.inc"

struct TritonAMDGPUAccelerateMatmulPass
    : impl::TritonAMDGPUAccelerateMatmulBase<TritonAMDGPUAccelerateMatmulPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    RewritePatternSet mfmaPatterns(context);
    switch (auto isaFamily = triton::AMD::deduceISAFamily(archGenerationName)) {
    case ISAFamily::GFX1250:
      mfmaPatterns.add<ScaledBlockedToScaledWMMAF8F6F4>(
          context, getWmmaVersion(archGenerationName), /*benefit=*/3);
      break;
    case ISAFamily::CDNA4:
      mfmaPatterns.add<::ScaledBlockedToScaledMFMAF8F6F4>(
          context, getMfmaVersion(isaFamily), matrixInstructionSize,
          /*benefit=*/4);
      mfmaPatterns.add<::DecomposeAMDScaledBlocked>(context, /*benefit=*/3);
      [[fallthrough]];
    case ISAFamily::CDNA3:
    case ISAFamily::CDNA2:
    case ISAFamily::CDNA1:
      mfmaPatterns.add<::BlockedToMFMA, ::ScaledBlockedToMFMA>(
          context, getMfmaVersion(isaFamily), matrixInstructionSize, kPack,
          /*benefit=*/2);
      break;
    case ISAFamily::RDNA3:
    case ISAFamily::RDNA4:
      ttg::populateDecomposeScaledBlockedPatterns(mfmaPatterns,
                                                  /*benefit=*/3);
      mfmaPatterns.add<::BlockedToWMMA>(
          context, getWmmaVersion(archGenerationName), matrixInstructionSize,
          /*benefit=*/2);
      break;
    default:
      break;
    }
    if (applyPatternsGreedily(m, std::move(mfmaPatterns)).failed())
      signalPassFailure();

    RewritePatternSet patterns(context);
    patterns.add<AccelerateBlocked>(context, archGenerationName, /*benefit=*/1);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
    decomposeMixedModeDotOp(m);
  }
};

} // namespace mlir
