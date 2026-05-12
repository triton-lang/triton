#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonAMDGPUTransforms/WmmaGroup.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/TargetFeatures.h"
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
using triton::amdgpu::ISAFamily;
using triton::amdgpu::TargetFeatures;

constexpr char AttrDecomposedDotScaledSource[] =
    "amdg.decomposed_dot_scaled_source";

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

int getWmmaVersion(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::RDNA3:
    return 1;
  case ISAFamily::RDNA4:
    return 2;
  case ISAFamily::GFX1250:
    return 3;
  default:
    break;
  }

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

// This function is to lay out numWarps warps into 2d dimension for the given
// dot operation, where
//  - shape : is the shape of the resulting data a single CTA will produce
//  - instrShape: is shape of data a single wmma/mfma hardware instruction
//     will consume
//
// This function takes into account three situations
//  a) 1st dot in a chained dot operations (e.g. in FA)
//  b) 2nd dot in a chained dot operations
//  c) single dot operation
//
// In case a), it will return {numWarp, 1} for the first dot in an attempt to
// reduce subsequent reduction overhead.
//
// TODO: describe b) using terse and intuitive way.
//
// Here is an example for case c). Assume instrShape is 16x16, and the shape is
// 160x320. So, the CTA worth of data is partitioned into 10x20 grid. This
// function is to lay out the numWarps into a mxn dimension, such that
//   - m*n = numWarps, and
//   - 10/m is close to 20/n
//
SmallVector<unsigned, 3> planWarps(Operation *dotOp, ArrayRef<int64_t> shape,
                                   int numWarps,
                                   std::pair<int64_t, int64_t> instrShape) {
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
        static_cast<int64_t>(llvm::divideCeil(shape[0], instrShape.first))));
    ret[1] = numWarps / ret[0];
    return ret;
  }

  // Case 3: Regular cases
  SmallVector<int64_t, 2> tensorShape = {shape[0], shape[1]};
  SmallVector<unsigned, 3> ret = {1, 1};
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (tensorShape[0] / (instrShape.first * 2) / ret[0] >=
        tensorShape[1] / instrShape.second / ret[1]) {
      if (ret[0] < tensorShape[0] / instrShape.first) {
        ret[0] *= 2;
      } else {
        ret[1] *= 2;
      }
    } else {
      ret[1] *= 2;
    }
  } while (true);

  if (ret[1] * instrShape.second > tensorShape[1]) {
    return {ret[1], ret[0]};
  }

  return ret;
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
  if (failed(maybeMfmaIntrinsic)) {
    mlir::emitRemark(loc) << "Unable to select MFMA intrinsic for the request: "
                          << "version=" << mfmaVersion << ", result-shape=("
                          << M << "x" << N << "), selected-tiles=(" << mDim
                          << "x" << nDim << "), inputKSize=" << inputKSize
                          << ", aElemType=" << aElemType
                          << ", bElemType=" << bElemType
                          << ", withScale=" << (withScale ? "true" : "false")
                          << ", allowXF32=" << (allowXF32 ? "true" : "false")
                          << (enforcedNonKDim != 0
                                  ? (llvm::Twine(", enforcedNonKDim=") +
                                     llvm::Twine(enforcedNonKDim))
                                        .str()
                                  : "");
    return failure();
  }

  kDim = maybeMfmaIntrinsic->kDim;
  assert(kDim != 0);
  assert(enforcedNonKDim != 0 || (M % mDim == 0 && N % nDim == 0));
  // If inputKSize % kDim != 0 (including the case where inputKSize < kDim),
  // this layout will introduce data duplication.
  if (inputKSize % kDim != 0) {
    mlir::emitRemark(loc)
        << "Unable to select MFMA intrinsic '" << maybeMfmaIntrinsic->name
        << "' as MFMA intrinsic k-dimension size kDim=" << kDim
        << ", which is not a multiple of tile k-dimension size inputKSize="
        << inputKSize
        << ". Using this intrinsic would introduce data duplication.";
    return failure();
  }
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
  if (version == 3) {
    applicableTypes.append({
        // clang-format off
        {f32, f32, f32, f32},
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
      ttg::ConvertLayoutOp::create(rewriter, loc, convertedType, value);

  if (newElemType == oldElemType)
    return convertedTensor;

  Type castedType = convertedType.cloneWith(std::nullopt, newElemType);

  Value castedTensor;

  if (newElemType.isIntOrIndex()) {
    unsigned oldWidth = oldElemType.getIntOrFloatBitWidth();
    unsigned newWidth = newElemType.getIntOrFloatBitWidth();
    if (oldWidth == newWidth)
      castedTensor = arith::BitcastOp::create(rewriter, loc, convertedType,
                                              convertedTensor);
    else if (oldWidth > newWidth)
      castedTensor =
          arith::TruncIOp::create(rewriter, loc, castedType, convertedTensor);
    else if (oldElemType.isSignedInteger())
      castedTensor =
          arith::ExtSIOp::create(rewriter, loc, castedType, convertedTensor);
    else
      castedTensor =
          arith::ExtUIOp::create(rewriter, loc, castedType, convertedTensor);
  } else {
    if (oldElemType.isF16() && newElemType.isF32())
      castedTensor =
          arith::ExtFOp::create(rewriter, loc, castedType, convertedTensor);
    else if (oldElemType.isF32() && newElemType.isF16())
      castedTensor =
          arith::TruncFOp::create(rewriter, loc, castedType, convertedTensor);
    else {
      RoundingModeAttr rmode;
      if (oldElemType.getIntOrFloatBitWidth() >
          newElemType.getIntOrFloatBitWidth()) {
        rmode =
            RoundingModeAttr::get(rewriter.getContext(), RoundingMode::RTNE);
      }
      castedTensor = tt::FpToFpOp::create(rewriter, loc, castedType,
                                          convertedTensor, rmode);
    }
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

// Figure out the best tilesPerWarp that gives largest vector size for |scale|
// tensors feeding into dot_scaled op.
SmallVector<unsigned, 2> deduceTilesPerWarpForScale(
    TypedValue<RankedTensorType> scaleA, TypedValue<RankedTensorType> scaleB,
    unsigned nonKDim, unsigned m, unsigned n, ArrayRef<unsigned> warpsPerCTA) {
  // Source code have flexibility to preshuffle scale tensor to achieve better
  // global load vectorization. That preshuffle scheme is conveyed via some
  // tl.reshape and tl.trans op combinations. Instead of hardcoding one case or
  // pattern match the op chain here, we try certain scale tensor layouts and
  // see which one gives us better vectorization when pushed upwards to the
  // global load.
  auto inferScaleSrcVecSize =
      [&](unsigned opIdx, TypedValue<RankedTensorType> scale,
          SmallVector<unsigned, 2> tilesPerWarp) -> unsigned {
    if (!scale)
      return 1;

    LinearLayout layout = ttg::chooseScaledMfmaScaleLayout(
        scale.getContext(), opIdx, scale.getType().getShape(), nonKDim,
        tilesPerWarp, warpsPerCTA);
    LLVM_DEBUG(llvm::dbgs() << "trying scale layout: " << layout << "\n");

    auto scaleDef = scale.getDefiningOp();
    // assume vec=4 for constant scale
    if (isa_and_nonnull<arith::ConstantOp, triton::SplatOp>(scaleDef))
      return 4;
    // Infer source layout used for global load using the current scale layout.
    auto loadLayoutPair = ttg::inferSourceLoadLayout(layout, scaleDef);
    if (!loadLayoutPair)
      return 1;
    tt::LoadOp loadOp = loadLayoutPair->first;
    const LinearLayout &inferredLayout = loadLayoutPair->second;
    LLVM_DEBUG(llvm::dbgs()
               << "inferred load layout: " << inferredLayout << "\n");

    auto loadType = cast<RankedTensorType>(loadOp.getType());
    auto loadOrder = ttg::getOrder(loadType);
    auto loadCGALayout = ttg::getCGALayout(loadType.getEncoding());

    // Reuse existing shared memory vectorization utilities by constructing a
    // pass through layout that does linear element mapping.
    MLIRContext *context = scale.getContext();
    auto passThruShared = ttg::SwizzledSharedEncodingAttr::get(
        context, 1, 1, 1, loadOrder, loadCGALayout);
    auto sharedLL =
        triton::gpu::toLinearLayout(loadType.getShape(), passThruShared);
    auto composedLL = inferredLayout.invertAndCompose(sharedLL).flattenOuts();
    LLVM_DEBUG(llvm::dbgs()
               << "inferred composed layout: " << composedLL << "\n");
    auto [v, _] =
        largestVectorisation(context, composedLL, /*bitwidth=*/8, std::nullopt);
    return v;
  };

  unsigned largest = 2;
  SmallVector<unsigned, 2> chosen{1, 1};
  // For scaled MFMA intrinsic, each thread only reads one i8 value.
  // For better vectorization, we prefer to stick tilesPerWarp 2x2 for 16x16x128
  // and 1x1 for 32x32x64 so that each thread can read 4xi8 values.
  // limit tilesPerWarp to block boundary
  for (unsigned mDimTiles = 1; mDimTiles <= std::min(2u, m / nonKDim);
       mDimTiles++) {
    for (unsigned nDimTiles = 1; nDimTiles <= std::min(2u, n / nonKDim);
         nDimTiles++) {
      SmallVector<unsigned, 2> tilesPerWarp{mDimTiles, nDimTiles};
      unsigned vecSizeA = inferScaleSrcVecSize(0, scaleA, tilesPerWarp);
      unsigned vecSizeB = inferScaleSrcVecSize(1, scaleB, tilesPerWarp);
      LLVM_DEBUG(llvm::dbgs() << "when tilesPerWarp: " << tilesPerWarp[0]
                              << ", " << tilesPerWarp[1] << "\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "inferred scaleA vecSize: " << vecSizeA << "\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "inferred scaleB vecSize: " << vecSizeB << "\n");
      unsigned score = vecSizeA + vecSizeB;
      if (score > largest) {
        largest = score;
        chosen = tilesPerWarp;
      }
    }
  }
  assert(largest <= 8 && "at most pack 4 scales for scale a & b respectively");
  // fixup: align with dimension that has scale
  if (!scaleA && scaleB)
    chosen[0] = std::min(ceil<unsigned>(m, nonKDim), chosen[1]);
  if (!scaleB && scaleA)
    chosen[1] = std::min(ceil<unsigned>(n, nonKDim), chosen[0]);
  return chosen;
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
    if (!isa_and_nonnull<BlockedEncodingAttr>(oldRetType.getEncoding()))
      return rewriter.notifyMatchFailure(
          dotOp, "expected blocked encoding result tensor");

    auto CGALayout = ttg::getCGALayout(oldRetType.getEncoding());

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
        return rewriter.notifyMatchFailure(
            dotOp,
            "Unable to choose preferable MFMA intrinsic for dot operation.");
      }
      mfmaInstr = chooseMfmaInstruction(dotOp, mfmaVersion, nonKDim, false);
      if (failed(mfmaInstr)) {
        return rewriter.notifyMatchFailure(
            dotOp, "Unable to choose MFMA intrinsic for dot operation.");
      }

      withScale = false;
    }

    auto mDim = mfmaInstr->mDim;
    auto nDim = mfmaInstr->nDim;
    auto kDim = mfmaInstr->kDim;
    auto kBase = mfmaInstr->kBase;

    auto warpsPerTile = planWarps(dotOp, retShape, numWarps, {mDim, nDim});

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
      // For decomposed scaled dots, pack warps along the free (non-scaled)
      // dimension to keep scale values warp-local. Cap at tile extent to
      // avoid overflow (e.g. 8 warps, 128-wide tile -> [2,4] not [1,8]).
      if ((scaleA || scaleB) && !(scaleA && scaleB)) {
        unsigned freeDim = scaleA ? 0 : 1;
        unsigned scaledDim = 1 - freeDim;
        unsigned maxFree = retShape[freeDim] / mDim;
        warpsPerTile[freeDim] =
            std::min(static_cast<unsigned>(numWarps), maxFree);
        warpsPerTile[scaledDim] = numWarps / warpsPerTile[freeDim];
      }
      tilesPerWarp = deduceTilesPerWarpForScale(
          dyn_cast_if_present<TensorValue>(scaleA),
          dyn_cast_if_present<TensorValue>(scaleB), mDim, retShape[0],
          retShape[1], warpsPerTile);
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
        isTransposed, CGALayout, tilesPerWarp,
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
      if (failed(aScaledElemTy))
        return rewriter.notifyMatchFailure(
            dotOp, "failed to deduce scaled element type fo A");

      auto bScaledElemTy = mlirTypeToScaledElemType(bElemType);
      if (failed(bScaledElemTy))
        return rewriter.notifyMatchFailure(
            dotOp, "failed to deduce scaled element type fo A");

      assert(kWidth == 32);
      auto newAEncoding =
          DotOperandEncodingAttr::get(ctx, 0, mfmaEnc, kWidth / 2);
      auto newBEncoding =
          DotOperandEncodingAttr::get(ctx, 1, mfmaEnc, kWidth / 2);

      a = convertAndCastTensor(rewriter, a, newAEncoding,
                               mfmaInstr->aElementType);
      b = convertAndCastTensor(rewriter, b, newBEncoding,
                               mfmaInstr->bElementType);
      newDot = triton::DotScaledOp::create(
          rewriter, dotOp.getLoc(), newAcc.getType(), a, b, newAcc, Value(),
          Value(), aScaledElemTy.value(), bScaledElemTy.value(),
          /*fastMath=*/false);
    } else {
      auto newAEncoding =
          ttg::DotOperandEncodingAttr::get(ctx, 0, mfmaEnc, kWidth);
      auto newBEncoding =
          ttg::DotOperandEncodingAttr::get(ctx, 1, mfmaEnc, kWidth);
      a = convertAndCastTensor(rewriter, a, newAEncoding,
                               mfmaInstr->aElementType);
      b = convertAndCastTensor(rewriter, b, newBEncoding,
                               mfmaInstr->bElementType);
      newDot = tt::DotOp::create(rewriter, dotOp.getLoc(), newAcc.getType(), a,
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

class DecomposeAMDScaledBlocked final : public ttg::DecomposeScaledBlocked {
public:
  DecomposeAMDScaledBlocked(MLIRContext *context,
                            const TargetFeatures &targetFeatures,
                            PatternBenefit benefit = 1)
      : ttg::DecomposeScaledBlocked(context, benefit),
        targetFeatures(targetFeatures) {}
  using TensorValue = TypedValue<RankedTensorType>;

  LogicalResult matchAndRewrite(tt::DotScaledOp dotOp,
                                PatternRewriter &rewriter) const override {
    dotOp.emitRemark() << "Decomposing scaled dot operation into regular dot "
                          "operation with explicit scaling.";
    return ttg::DecomposeScaledBlocked::matchAndRewrite(dotOp, rewriter);
  }

  RankedTensorType getScaledUpcastResultType(RankedTensorType vType,
                                             FloatType computeType,
                                             int32_t kDim, bool isFp4,
                                             Location loc) const {
    if (!isFp4)
      return vType.clone(computeType);

    auto resultType = mlir::triton::gpu::inferFp4ToFpResultType(
        vType, computeType, kDim, loc);
    assert(succeeded(resultType));
    return *resultType;
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
    auto loc = dotOp.getLoc();
    bool isFp4 = (elemType == ScaleDotElemType::E2M1);

    RankedTensorType resultType =
        getScaledUpcastResultType(vType, computeType, kDim, isFp4, loc);

    // Mark scale to simplify pattern matching during deducing TilesPerWarp
    scale.getDefiningOp()->setAttr(AttrDecomposedDotScaledSource,
                                   BoolAttr::get(rewriter.getContext(), true));

    Value reshapeScale;
    if (targetFeatures.supportsCvtPkScalePk8()) {
      // On architectures with CvtPkScalePk8 (e.g., GFX1250), the scale type
      // is int8, required by hardware instruction so type should not be
      // converted.
      if (opIdx == 1) {
        auto order = getTransposeOrder(rank);
        scale = TransOp::create(rewriter, loc, scale, order);
      }

      reshapeScale = broadcastScale(
          rewriter, dotOp, dotOp->getParentOfType<ModuleOp>(), scale, kDim);

      auto newScaleType = resultType.clone(scale.getType().getElementType());
      reshapeScale = mlir::triton::gpu::ConvertLayoutOp::create(
          rewriter, loc, newScaleType, reshapeScale);
    } else {
      // Cast scale to bf16, broadcast it and convert the layout
      FloatType bf16Type = rewriter.getBF16Type();
      reshapeScale = extendAndBroadcastScale(rewriter, dotOp, scale, bf16Type,
                                             resultType.clone(bf16Type), opIdx);
    }

    // Upcast with scale
    TensorValue result;
    if (isFp4) {
      result = triton::amdgpu::ScaledUpcastFp4Op::create(
          rewriter, loc, resultType, v, reshapeScale, kDim);
    } else {
      result = triton::amdgpu::ScaledUpcastFp8Op::create(
          rewriter, loc, resultType, v, reshapeScale);
    }

    // If the scale is NaN, return NaN, else return the scaled value.
    return maskNan(rewriter, dotOp, result, scale, kDim);
  }

private:
  const TargetFeatures &targetFeatures;
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

    ttg::CGAEncodingAttr cgaLayout =
        ttg::getCGALayout(oldRetType.getEncoding());
    unsigned numWarps = ttg::lookupNumWarps(dotOp);
    if (numWarps == 1)
      return rewriter.notifyMatchFailure(dotOp,
                                         "num_warps==1 is not supported");

    // Choose a suitable Scaled MFMA instruction for this scaled dot op.
    FailureOr<MfmaIntrinsic> mfmaInstr =
        chooseMfmaInstruction(dotOp, mfmaVersion, nonKDim);
    if (failed(mfmaInstr)) {
      return rewriter.notifyMatchFailure(dotOp,
                                         "Unable to choose preferable MFMA "
                                         "intrinsic for scaled dot operation.");
    }

    auto mDim = mfmaInstr->mDim;
    auto nDim = mfmaInstr->nDim;
    auto kDim = mfmaInstr->kDim;
    auto kBase = mfmaInstr->kBase;
    assert(mDim == nDim);

    auto warpsPerTile = planWarps(dotOp, oldShape, numWarps, {mDim, nDim});

    SmallVector<unsigned, 2> tilesPerWarp = deduceTilesPerWarpForScale(
        aScale, bScale, mDim, oldShape[0], oldShape[1], warpsPerTile);
    LLVM_DEBUG(llvm::dbgs() << "chosen tilesPerWarp: [" << tilesPerWarp[0]
                            << ", " << tilesPerWarp[1] << "]\n");

    // Always use transposed mfma layout. This enables larger vectorization
    // for global store instructions.
    auto elementBitWidth = oldRetType.getElementType().getIntOrFloatBitWidth();
    mlir::Attribute mfmaEnc = ttg::AMDMfmaEncodingAttr::get(
        ctx, mfmaVersion, warpsPerTile, {mDim, nDim, kDim},
        /*isTransposed=*/true, cgaLayout, tilesPerWarp, elementBitWidth);

    auto newRetType =
        RankedTensorType::get(oldShape, oldRetType.getElementType(), mfmaEnc);

    auto newAcc = ttg::ConvertLayoutOp::create(rewriter, dotOp.getC().getLoc(),
                                               newRetType, dotOp.getC());

    auto order = ttg::getMatrixOrder(rank, /*rowMajor=*/true);
    auto standardOutDims = standardOutDimNames(ctx, rank);

    // For the mfma_scale_f32_*_f8f6f4 instructions, each thread consumes 32
    // elements. But since two fp4 elements are packed into one int8, the
    // kWidth is 16 for fp4.
    const unsigned kWidth = kBase;
    assert(kWidth == 32);

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
                triton::gpu::getCGALayout(srcEncoding), vType.getElementType()),
            sharedMemorySpace);
        auto tmp = triton::gpu::LocalAllocOp::create(builder, dotOp.getLoc(),
                                                     tmpType, v);
        auto newConvert = triton::amdgpu::LocalLoadPackedTransposedOp::create(
            builder, dotOp.getLoc(), newVType, tmp);
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
        return ttg::ConvertLayoutOp::create(rewriter, v.getLoc(), newVType, v);
      }
    };
    a = convertInputLayout(a, 0);
    b = convertInputLayout(b, 1);

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

      Attribute newScaleEncoding =
          ttg::LinearEncodingAttr::get(ctx, std::move(newLL));
      // Scale's data type is always i8
      auto newScaleType = RankedTensorType::get(shape, i8_ty, newScaleEncoding);

      if (!scale) {
        // 0x7F is 1.0 in E8M0
        return arith::ConstantOp::create(
            rewriter, dotOp->getLoc(), newScaleType,
            DenseElementsAttr::get(newScaleType, llvm::APInt(8, 0x7F)));
      } else {
        return ttg::ConvertLayoutOp::create(rewriter, scale.getLoc(),
                                            newScaleType, scale);
      }
    };
    auto newAScale =
        convertScaleLayout(aScale, aShape, aEncLL, /*dotOperandIdx=*/0);
    auto newBScale =
        convertScaleLayout(bScale, bShape, bEncLL, /*dotOperandIdx=*/1);

    auto newDot = triton::DotScaledOp::create(
        rewriter, dotOp.getLoc(), newRetType, a, b, newAcc, newAScale,
        newBScale, aElemType, bElemType, dotOp.getFastMath());

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
    auto supportsTypes = [](ScaleDotElemType elemType) {
      return elemType == ScaleDotElemType::E2M1 ||
             elemType == ScaleDotElemType::E4M3 ||
             elemType == ScaleDotElemType::E5M2;
    };

    if (!supportsTypes(aElemType) || !supportsTypes(bElemType)) {
      return rewriter.notifyMatchFailure(dotOp, "Not supported yet mxfp type");
    }

    unsigned scaleFactor = dotOp.deduceScaleFactor();

    MLIRContext *ctx = dotOp.getContext();

    ttg::CGAEncodingAttr cgaLayout =
        ttg::getCGALayout(oldRetType.getEncoding());
    unsigned numWarps = ttg::lookupNumWarps(dotOp);
    auto oldShapePerCTA =
        ttg::getShapePerCTA(cgaLayout.getCTASplitNum(), oldShape);

    constexpr unsigned mDim = 16;
    constexpr unsigned nDim = 16;
    constexpr unsigned kDim = 128;

    auto warpsPerTile =
        planWarps(dotOp, oldShapePerCTA, numWarps, {mDim, nDim});
    // TODO: Select tilesPerWarp in Triton
    SmallVector<unsigned> tilesPerWarp(rank, 1u);

    auto ctaLayout =
        ttg::chooseWmmaCTALinearLayout(ctx, rank, warpsPerTile, tilesPerWarp);
    auto wmmaEnc = ttg::AMDWmmaEncodingAttr::get(
        ctx, wmmaVersion, ctaLayout, true, cgaLayout, {mDim, nDim, kDim});
    auto wmmaPackedEnc = ttg::AMDWmmaEncodingAttr::get(
        ctx, wmmaVersion, ctaLayout, true, cgaLayout, {mDim, nDim, kDim / 2});

    auto newRetType =
        RankedTensorType::get(oldShape, oldRetType.getElementType(), wmmaEnc);

    auto newAcc = ttg::ConvertLayoutOp::create(rewriter, dotOp.getC().getLoc(),
                                               newRetType, dotOp.getC());

    auto order = ttg::getMatrixOrder(rank, /*rowMajor=*/true);
    auto standardOutDims = standardOutDimNames(ctx, rank);

    RankedTensorType aType = a.getType();
    RankedTensorType bType = b.getType();
    auto aCgaLayout = ttg::getCGALayout(aType.getEncoding());
    auto bCgaLayout = ttg::getCGALayout(bType.getEncoding());
    auto aShape = aType.getShape();
    auto bShape = bType.getShape();
    auto aShapePerCTA =
        ttg::getShapePerCTA(aCgaLayout.getCTASplitNum(), aShape);
    auto bShapePerCTA =
        ttg::getShapePerCTA(bCgaLayout.getCTASplitNum(), bShape);

    auto aEncLL = LinearLayout::empty();
    auto bEncLL = LinearLayout::empty();

    auto convertInputLayout = [&](TensorValue v, unsigned opIdx,
                                  bool isFp4) -> TensorValue {
      auto parent = isFp4 ? wmmaPackedEnc : wmmaEnc;
      auto vType = v.getType();
      auto newEnc = DotOperandEncodingAttr::get(ctx, opIdx, parent, 16);
      auto newVType = RankedTensorType::get(vType.getShape(),
                                            vType.getElementType(), newEnc);
      if (opIdx == 0)
        aEncLL *= newEnc.toLinearLayout(aShapePerCTA);
      else
        bEncLL *= newEnc.toLinearLayout(bShapePerCTA);

      return ttg::ConvertLayoutOp::create(rewriter, v.getLoc(), newVType, v);
    };
    a = convertInputLayout(a, 0, aElemType == ScaleDotElemType::E2M1);
    b = convertInputLayout(b, 1, bElemType == ScaleDotElemType::E2M1);

    auto getDefaultScaleTypeValue = [&](int idx) -> std::pair<Type, Attribute> {
      // If both scales are absent, use E8M0 for generality
      if (!aScale && !bScale) {
        return {i8_ty, rewriter.getIntegerAttr(i8_ty, 0x7F)};
      }

      if (aElemType == ScaleDotElemType::E2M1 &&
          bElemType == ScaleDotElemType::E2M1) {
        // Fp4 x Fp4 requires to use the same scale dtype for both operands.
        TensorValue otherScale = idx == 0 ? bScale : aScale;
        Type scaleTy = otherScale.getType().getElementType();
        // int8 in this context is actually represents fp8e8m0 dtype,
        // 127 constant is how 1.0 is represented in this data type.
        if (scaleTy.isInteger(8))
          return {scaleTy, rewriter.getIntegerAttr(scaleTy, 127)};
        return {scaleTy, rewriter.getOneAttr(scaleTy)};
      }

      return {i8_ty, rewriter.getIntegerAttr(i8_ty, 0x7F)};
    };

    auto convertScaleLayout = [&](TensorValue scale,
                                  llvm::ArrayRef<int64_t> valShape,
                                  LinearLayout dotLL, int idx,
                                  ttg::CGAEncodingAttr cgaLayout) -> Value {
      SmallVector<int64_t> shape;
      Type scaleType;
      // 0x7F is 1.0 in E8M0
      Attribute scaleValue = rewriter.getIntegerAttr(i8_ty, 0x7F);
      if (!scale) {
        int64_t nonKDim = idx == 0 ? valShape[0] : valShape[1];
        int64_t k = idx == 0 ? valShape[1] : valShape[0];
        ScaleDotElemType &elemType = idx == 0 ? aElemType : bElemType;
        int packSize = elemType == ScaleDotElemType::E2M1 ? 2 : 1;
        shape = {nonKDim, k * packSize / scaleFactor};
        std::tie(scaleType, scaleValue) = getDefaultScaleTypeValue(idx);
      } else {
        scaleType = scale.getType().getElementType();
        shape = llvm::to_vector(scale.getType().getShape());
      }

      LinearLayout newLL = ttg::chooseScaledWmmaScaleLayout(
          ctx, idx, shape, mDim, nDim, wmmaEnc.getIsTransposed(), scaleFactor,
          ctaLayout, cgaLayout);
      Attribute newScaleEncoding = ttg::LinearEncodingAttr::get(ctx, newLL);
      auto newScaleType =
          RankedTensorType::get(shape, scaleType, newScaleEncoding);

      if (!scale) {
        auto denseAttr = DenseElementsAttr::get(newScaleType, scaleValue);
        return arith::ConstantOp::create(rewriter, dotOp->getLoc(),
                                         newScaleType, denseAttr);
      } else {
        return ttg::ConvertLayoutOp::create(rewriter, scale.getLoc(),
                                            newScaleType, scale);
      }
    };

    // Operands A and A-scale are of shape MxK and Mx(K/grp), respectively.
    // They both split along M-dimension in the same way, and hence have the
    // same CGA layout.
    auto aScaleCgaLayout = aCgaLayout;
    assert(!aScale || (aScaleCgaLayout ==
                       ttg::getCGALayout((aScale.getType()).getEncoding())));
    auto newAScale = convertScaleLayout(aScale, aShape, aEncLL,
                                        /*dotOperandIdx=*/0, aScaleCgaLayout);

    auto bScaleCgaLayout = inferBScaleCgaLayout(ctx, cgaLayout);
    assert(!bScale || (ttg::getCGALayout(bScale.getType().getEncoding()) ==
                       bScaleCgaLayout));
    auto newBScale = convertScaleLayout(bScale, bShape, bEncLL,
                                        /*dotOperandIdx=*/1, bScaleCgaLayout);

    auto newDot = triton::DotScaledOp::create(
        rewriter, dotOp.getLoc(), newRetType, a, b, newAcc, newAScale,
        newBScale, aElemType, bElemType, dotOp.getFastMath(),
        dotOp.getLhsKPack(), dotOp.getRhsKPack());

    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(dotOp, oldRetType,
                                                      newDot);

    return success();
  }

  // If Bscale were present, we could directly grab the CGA layout from it.
  // Unfortunately, it is optional; on top of that, sometime we need to know
  // its CGA layout even if it's not present.
  //
  // Note that split only take place along M and N dimension. For A and Ascale,
  // their shapes are MxK and Mx(K/grp), respectively. Hence, A and A-scale can
  // share the CGA layout. For B and Bscale, their shapes are KxN and Nx(K/grp),
  // respectively. Since N shows up on different positions, we cannot "reuse"
  // B's CGA layout for B-scale.
  //
  // We can grab N's split number from D's CGA layout, and construct Bscale's
  // using that info.
  //
  static ttg::CGAEncodingAttr
  inferBScaleCgaLayout(MLIRContext *ctx, ttg::CGAEncodingAttr dCgaLayout) {
    auto resultSplit = dCgaLayout.getCTASplitNum();
    unsigned nSplit = resultSplit[1];
    unsigned numCtas = mlir::product(dCgaLayout.getCTAsPerCGA());

    return ttg::CGAEncodingAttr::fromSplitParams(
        ctx,
        /*CTAsPerCGA=*/{nSplit, numCtas / nSplit},
        /*CTASplitNum=*/{nSplit, 1}, /*CTAOrder*/ {0, 1});
  }
};

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  Type tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                .cloneWith(std::nullopt, promotedType);
  return triton::FpToFpOp::create(builder, loc, tensorPromotedType, operand);
}

// Promote operands of dot op if the existing combination is not natively
// supported.
static void decomposeMixedModeDotOp(ModuleOp mod) {
  mod.walk([](triton::DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    Type BElType = dotOp.getB().getType().getElementType();
    auto maxBitWidth = std::max(AElType.getIntOrFloatBitWidth(),
                                BElType.getIntOrFloatBitWidth());
    Type promoteType;
    if (isa<ttg::AMDMfmaEncodingAttr>(D.getType().getEncoding())) {
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
      if (maxBitWidth == 8)
        return;

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
                                               Type cElemType, int inputKSize) {
  // number of matrix elements along k dim per one WMMA instruction
  unsigned kDim = 0;

  unsigned mDim = 16;
  unsigned nDim = 16;

  FailureOr<WmmaIntrinsic> maybeWmmaIntrinsic = WmmaIntrinsic::selectFor(
      wmmaVersion, mDim, nDim, inputKSize, aElemType, bElemType, cElemType);
  if (failed(maybeWmmaIntrinsic))
    return emitError(loc, "no matching matrix core intrinsic ")
           << "for wmma version " << wmmaVersion << " with instruction shape ["
           << mDim << ", " << nDim << ", " << inputKSize
           << "] and element types A=" << aElemType << ", B=" << bElemType
           << ", C=" << cElemType << ". Check whether the mfma version,"
           << " instruction shape, and data types "
           << "are supported on the current AMD GPU architecture.";

  kDim = maybeWmmaIntrinsic->kDim;
  assert(kDim != 0);
  return maybeWmmaIntrinsic;
}

FailureOr<WmmaIntrinsic> chooseWmmaInstruction(tt::DotOp dot,
                                               OperandTypesVector operandTypes,
                                               int wmmaVersion) {

  return chooseWmmaInstruction(
      dot.getLoc(), wmmaVersion, dot.getC().getType(), operandTypes[0],
      operandTypes[1], operandTypes[2], dot.getA().getType().getShape().back());
}

class BlockedToWMMA : public OpRewritePattern<tt::DotOp> {
  int wmmaVersion;

public:
  BlockedToWMMA(MLIRContext *context, int wmmaVersion, int nonKDim,
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
      return rewriter.notifyMatchFailure(
          dotOp, "expected `BlockedEncodingAttr` for the result type");

    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());
    auto retShape = oldRetType.getShape();
    auto aShape = oldAType.getShape();
    auto bShape = oldBType.getShape();

    // get operand types
    auto operandTypes = getOperandTypesForWmmaOp(rewriter, dotOp, wmmaVersion);
    if (operandTypes.empty())
      return rewriter.notifyMatchFailure(
          dotOp, "failed to get operands types for wmma op");

    auto kDimTensor = aShape.back();
    if (kDimTensor == 1) {
      return rewriter.notifyMatchFailure(dotOp,
                                         "Skipping WMMA for dot op with K=1");
    }
    // check shape
    FailureOr<WmmaIntrinsic> wmmaInstr =
        chooseWmmaInstruction(dotOp, operandTypes, wmmaVersion);
    if (failed(wmmaInstr)) {
      return rewriter.notifyMatchFailure(
          dotOp, "Unable to choose WMMA intrinsic for dot operation.");
    }

    auto mDim = wmmaInstr->mDim;
    auto nDim = wmmaInstr->nDim;
    auto kDim = wmmaInstr->kDim;
    auto kBase = wmmaInstr->kBase;

    // get WMMA encoding for the given number of warps
    int numWarps = ttg::lookupNumWarps(dotOp);

    ttg::AMDWmmaEncodingAttr wmmaEnc;

    auto CGALayout = ttg::getCGALayout(oldRetEncoding);

    // Note: Following few lines of code could be very confusing for some
    // readers. If this is the case, please read AMDWmmaEncodingAttr defined
    // in include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td for the
    // meaning of `ctaLayout`, which depicit how warps and optionally, "tile"
    // are arranged. Some occurrences of "tile" refer to the portion of tensor
    // being processed by all entire CGA as a whole, and some occurrences
    // means small portion of tensor which feed to a *SINGLE* wmma instruction.
    //
    // TODO: clean up and refine terms to be clearer.
    auto retShapePerCTA =
        ttg::getShapePerCTA(CGALayout.getCTASplitNum(), retShape);
    auto warpsPerTile =
        planWarps(dotOp, retShapePerCTA, numWarps, {mDim, nDim});

    // Use transposed wmma layout to enable larger vectorization for global
    // store instructions.
    bool isTransposed = true;
    SmallVector<unsigned> tilesPerWarp(retShape.size(), 1u);
    auto ctaLayout = ttg::chooseWmmaCTALinearLayout(ctx, retShape.size(),
                                                    warpsPerTile, tilesPerWarp);

    wmmaEnc =
        ttg::AMDWmmaEncodingAttr::get(ctx, wmmaVersion, ctaLayout, isTransposed,
                                      CGALayout, {mDim, nDim, kDim});

    auto newRetType = RankedTensorType::get(retShape, operandTypes[3], wmmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getC();
    auto newAcc =
        convertAndCastTensor(rewriter, oldAcc, wmmaEnc, operandTypes[2]);

    // deduce `kWidth` which is the number of consecutive elements along the K
    // dimension for a lane. Derive it from `kBase` which is the number of
    // elements along the K dimension in a WMMA instruction per lane. Note:
    // `kBase` can consist of several separated groups of consecutive elements.
    // This depends on the instruction encoding.

    // kWidth is always equals to kBase for WMMA v1/2
    auto kWidth = kBase;
    if (wmmaVersion == 3) {
      const bool isF32 = oldAType.getElementType().isF32();
      // kBase always consits of several groups of 8 elments except F32 case
      kWidth = isF32 ? 2 : 8;
    }
    assert(kWidth != 0);

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
    auto newDot = tt::DotOp::create(
        rewriter, dotOp.getLoc(), newRetType, castedA, castedB, newAcc,
        dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());

    Value dotOutput = convertAndCastTensor(rewriter, newDot, oldRetEncoding,
                                           oldRetType.getElementType());
    rewriter.replaceOp(dotOp, dotOutput);
    return success();
  }
};

class AccelerateBlocked : public OpRewritePattern<DotOp> {
  TargetFeatures targetFeatures;

public:
  AccelerateBlocked(MLIRContext *context, const TargetFeatures &targetFeatures,
                    PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), targetFeatures(targetFeatures) {}

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
      return FpToFpOp::create(rewriter, loc, dstTy, v, rmode);
    }
    if (!isFloat(srcElTy) && isFloat(dstElTy))
      return arith::SIToFPOp::create(rewriter, loc, dstTy, v);
    if (isFloat(srcElTy) && !isFloat(dstElTy))
      return arith::FPToSIOp::create(rewriter, loc, dstTy, v);
    assert(false && "int -> int cast is unexpected in FMA legalization");
    return Value();
  }

  struct DotElTypes {
    Type a, b, c, d;
  };

  bool isLegalFMAForm(DotOp dotOp, const DotElTypes &dotTypes) const {
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
    if (targetFeatures.isCDNA4() && dotTypes.a.isBF16() &&
        dotTypes.b.isBF16() && dotTypes.c.isF32() && dotTypes.d.isF32() &&
        k % 2 == 0) {
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
    // If this is fp16 x fp16 ->fp16 case prioritize using v_dot.
    auto aOpType = dotOp.getA().getType();
    int rank = aOpType.getRank();
    int k = aOpType.getShape()[rank - 1];
    if (dotTypes.a.isF16() && dotTypes.b.isF16() && dotTypes.c.isF16() &&
        dotTypes.d.isF16() && k % 2 == 0) {
      auto newC = castToElTy(rewriter, dotOp.getC(), f32_ty);
      auto newDot = DotOp::create(
          rewriter, dotOp.getLoc(), newC.getType(), dotOp.getA(), dotOp.getB(),
          newC, dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());
      auto newD = castToElTy(rewriter, newDot.getResult(), f16_ty);
      rewriter.replaceOp(dotOp, newD);
      return success();
    }
    return rewriter.notifyMatchFailure(
        dotOp, "Unable to choose V_DOT instruction for dot operation.");
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

    auto newDot = DotOp::create(rewriter, dotOp.getLoc(), newC.getType(), newA,
                                newB, newC, dotOp.getInputPrecision(),
                                dotOp.getMaxNumImpreciseAcc());
    auto newD = castToElTy(rewriter, newDot.getResult(), dotTypes.d);

    rewriter.replaceOp(dotOp, newD);
    return success();
  }

  LogicalResult matchAndRewrite(DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<BlockedEncodingAttr>(dotOp.getD().getType().getEncoding()))
      return rewriter.notifyMatchFailure(
          dotOp, "expected blocked encoding result tensor");

    dotOp.emitRemark() << "Attempting to map dot operation to FMA intrinsic.";

    DotElTypes dotTypes;
    dotTypes.a = dotOp.getA().getType().getElementType();
    dotTypes.b = dotOp.getB().getType().getElementType();
    dotTypes.c = dotOp.getC().getType().getElementType();
    dotTypes.d = dotOp.getD().getType().getElementType();

    // Check that dot is not legalized already
    if (isLegalFMAForm(dotOp, dotTypes)) {
      return rewriter.notifyMatchFailure(
          dotOp, "Dot operation is already in FMA form.");
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
    TargetFeatures targetFeatures{llvm::StringRef(gfxArch)};
    auto isaFamily = targetFeatures.getISAFamily();
    unsigned wmmaVersion = getWmmaVersion(isaFamily);
    switch (isaFamily) {
    case ISAFamily::GFX1250:
      mfmaPatterns.add<ScaledBlockedToScaledWMMAF8F6F4>(context, wmmaVersion,
                                                        /*benefit=*/4);
      mfmaPatterns.add<::DecomposeAMDScaledBlocked>(context, targetFeatures,
                                                    /*benefit=*/3);
      mfmaPatterns.add<BlockedToWMMA>(context, wmmaVersion, 16, /*benefit=*/2);
      break;
    case ISAFamily::CDNA4:
      mfmaPatterns.add<::ScaledBlockedToScaledMFMAF8F6F4>(
          context, getMfmaVersion(isaFamily), matrixInstructionSize,
          /*benefit=*/4);
      [[fallthrough]];
    case ISAFamily::CDNA3:
    case ISAFamily::CDNA2:
    case ISAFamily::CDNA1:
      mfmaPatterns.add<::DecomposeAMDScaledBlocked>(context, targetFeatures,
                                                    /*benefit=*/3);
      mfmaPatterns.add<::BlockedToMFMA>(context, getMfmaVersion(isaFamily),
                                        matrixInstructionSize, kPack,
                                        /*benefit=*/2);
      break;
    case ISAFamily::RDNA3:
    case ISAFamily::RDNA4:
      ttg::populateDecomposeScaledBlockedPatterns(mfmaPatterns,
                                                  /*benefit=*/3);
      mfmaPatterns.add<::BlockedToWMMA>(context, wmmaVersion,
                                        matrixInstructionSize,
                                        /*benefit=*/2);
      break;
    default:
      break;
    }
    if (applyPatternsGreedily(m, std::move(mfmaPatterns)).failed())
      signalPassFailure();

    RewritePatternSet patterns(context);
    patterns.add<AccelerateBlocked>(context, targetFeatures, /*benefit=*/1);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
    decomposeMixedModeDotOp(m);
  }
};

} // namespace mlir
