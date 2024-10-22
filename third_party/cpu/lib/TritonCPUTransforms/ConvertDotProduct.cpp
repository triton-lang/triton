#include "cpu/include/TritonCPUTransforms/OptCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "include/triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include <iostream>
#include <utility>

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTDOTPRODUCT
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

// TODO: support SVE and different vector width
// We currently only supported Arm Neon (128 bit vector).
// To support scalable vectors in SVE, we need to generate
// vector-length agnostic (VLA) code using vector.vscale.
// To support other platform (AVX512 for X86), we need to
// change the vectorBitWidth and the intrinsics.
constexpr int vectorBitWidth = 128;

// This function is used to identify bf16 dot product (expressed by elementwise
// multiplication follwed by a sum).
// For example, the following pattern: tl.sum(a * x[None, :], axis=1)
// is used to express a dot product.
// Since x is broadcated for the elementwise multiplication. And tl.sum will
// cast its bf16 input to fp32.
// The pattern in MLIR will be:
// BroadcastOp -> MulFOp -> ExtFOp -> MultiDimReductionOp
bool isBf16DotProduct(vector::MultiDimReductionOp op, bool useHorizontalSum,
                      Value &matInput, Value &vecInput,
                      PatternRewriter &rewriter) {
  Value src = op.getSource();
  Value acc = op.getAcc();
  auto srcTy = cast<VectorType>(src.getType());
  auto accTy = cast<VectorType>(acc.getType());
  auto resTy = cast<VectorType>(op.getType());

  auto srcRank = srcTy.getRank();
  auto outNum = srcTy.getDimSize(0);

  if (resTy != accTy || srcRank != 2 || !isFp32(srcTy))
    return false;

  if (op.isReducedDim(0) || !op.isReducedDim(1))
    return false;

  if (op.getKind() != vector::CombiningKind::ADD)
    return false;

  auto extFOp = src.getDefiningOp<arith::ExtFOp>();

  if (!extFOp || !extFOp->hasOneUse())
    return false;

  auto mulFOp = extFOp.getIn().getDefiningOp<arith::MulFOp>();

  if (!mulFOp || !mulFOp->hasOneUse())
    return false;

  Value lhs = mulFOp.getLhs();
  Value rhs = mulFOp.getRhs();

  auto lhsTy = cast<VectorType>(lhs.getType());
  auto rhsTy = cast<VectorType>(rhs.getType());

  if (!isBf16(lhsTy) || !isBf16(rhsTy))
    return false;

  const int lanes =
      vectorBitWidth / lhsTy.getElementType().getIntOrFloatBitWidth();
  const int resultLanes =
      vectorBitWidth / resTy.getElementType().getIntOrFloatBitWidth();
  int64_t kVal = lhsTy.getDimSize(1);

  if (outNum < 1)
    return false;

  if (!useHorizontalSum) {
    // TODO: masking is not currrently supported
    if (outNum % resultLanes != 0)
      return false;
  }

  // TODO: masking is not currrently supported
  if (kVal % lanes != 0)
    return false;

  if (outNum == 1) {
    matInput = lhs;
    vecInput = rhs;
  } else {
    vector::BroadcastOp broadCastOp;
    if (rhs.getDefiningOp<vector::BroadcastOp>()) {
      matInput = lhs;
      broadCastOp = rhs.getDefiningOp<vector::BroadcastOp>();
    } else {
      matInput = rhs;
      broadCastOp = lhs.getDefiningOp<vector::BroadcastOp>();
    }
    if (!broadCastOp || !broadCastOp->hasOneUse())
      return false;
    vecInput = broadCastOp.getSource();
  }

  if (cast<VectorType>(vecInput.getType()).getDimSize(0) != 1 ||
      cast<VectorType>(matInput.getType()).getDimSize(0) != outNum)
    return false;

  return true;
}

struct ConvertMulSumToDotHorizontalSum
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    Value acc = op.getAcc();
    auto resTy = cast<VectorType>(op.getType());

    Value matInput;
    Value vecInput;

    bool isMatch = isBf16DotProduct(op, /*useHorizontalSum=*/true, matInput,
                                    vecInput, rewriter);
    if (!isMatch)
      return failure();

    // Once we get the matrix input (NxK) and vector input (K),
    // where N is the output channel dimension
    // and K is the reduction dimension.
    // We will generate the following code to perform the dot product.
    // For each output channel:
    // we will pull 8 bf16 elements from the vector and matrix each time when
    // we iterate over the K dimension.
    // We will then use bfdot to perform sum-of-products on pairs of
    // bf16 elements, accumulate and get 4 fp32 outputs.
    // After the iteration over the K dimension finishes, we will use a
    // horizontal sum (faddv) to sum the 4 fp32 into a single fp32.
    // We will also share the vector input across the output channels
    // to reduce the number of loads.
    // For example, if we dot product a size 2x16 matrix with a size 16 vector,
    // the pseudo code will be:
    // matrix = shapecast(matrix, 2x2x8)
    // vector = shapecast(vector, 2x8)
    // out    = zeros(2x4, fp32)
    // out[0] = bfdot(out[0], matrix[0][0], vector[0])
    // out[1] = bfdot(out[1], matrix[1][0], vector[0])
    // out[0] = bfdot(out[0], matrix[0][1], vector[1])
    // out[1] = bfdot(out[1], matrix[1][1], vector[1])
    // out_0  = faddv(out[0]) : 4xfp32 -> fp32
    // out_1  = faddv(out[1]) : 4xfp32 -> fp32

    auto matInputTy = cast<VectorType>(matInput.getType());
    auto vecInputTy = cast<VectorType>(vecInput.getType());

    const int lanes =
        vectorBitWidth / matInputTy.getElementType().getIntOrFloatBitWidth();
    const int resultLanes =
        vectorBitWidth / resTy.getElementType().getIntOrFloatBitWidth();
    int64_t kVal = matInputTy.getDimSize(1);

    // numOfOutputChannels is the number of output channels (N)
    const int numOfOutputChannels = matInputTy.getDimSize(0);
    // numOfBfdotOps is the number of bfdots needed for each output channel.
    const int numOfBfdotOps = kVal / lanes;

    matInput = shapeCast(loc, matInput,
                         {numOfOutputChannels, numOfBfdotOps, lanes}, rewriter);
    vecInput = shapeCast(loc, vecInput, {numOfBfdotOps, lanes}, rewriter);

    SmallVector<Value> outRes(numOfOutputChannels);
    SmallVector<Value> mats(numOfOutputChannels);

    Type outResTy = VectorType::get(resultLanes, resTy.getElementType());

    Value zeroRes = rewriter.create<arith::ConstantOp>(
        loc, outResTy, rewriter.getZeroAttr(outResTy));
    for (int64_t outIdx = 0; outIdx < numOfOutputChannels; outIdx += 1) {
      outRes[outIdx] = zeroRes;
      // Intermediate array to store each row of the input matrix.
      mats[outIdx] = rewriter.create<vector::ExtractOp>(loc, matInput, outIdx);
    }

    SmallVector<Type> resultTypes = {outResTy};
    // TODO: this intrinsic is hard-coded for Arm Neon
    auto bfdot = StringAttr::get(ctx, "llvm.aarch64.neon.bfdot.v4f32.v8bf16");
    SmallVector<Value> args;

    for (int64_t idx = 0; idx < numOfBfdotOps; idx += 1) {
      auto subVec = rewriter.create<vector::ExtractOp>(loc, vecInput, idx);
      for (int64_t outIdx = 0; outIdx < numOfOutputChannels; outIdx += 1) {
        auto subMat =
            rewriter.create<vector::ExtractOp>(loc, mats[outIdx], idx);
        args = {outRes[outIdx], subMat, subVec};
        // bfdot instruction:
        // https://developer.arm.com/documentation/ddi0602/2024-06/SIMD-FP-Instructions/BFDOT--vector---BFloat16-floating-point-dot-product--vector--
        // LLVM fast math flags:
        // https://llvm.org/docs/LangRef.html#fast-math-flags
        // This bfdot intrinsic will perform an unfused sum-of-products of each
        // pair of adjacent bf16 elements in the source vectors (8 bf16), and
        // output 4 fp32 elements.
        auto callIntrOp = rewriter.create<LLVM::CallIntrinsicOp>(
            loc, resultTypes, bfdot, args,
            LLVM::FastmathFlagsAttr::get(ctx, LLVM::FastmathFlags::fast));
        outRes[outIdx] = callIntrOp.getResult(0);
      }
    }

    Value res = rewriter.create<arith::ConstantOp>(loc, resTy,
                                                   rewriter.getZeroAttr(resTy));

    resultTypes = {resTy.getElementType()};
    // TODO: this intrinsic is hard-coded for Arm Neon
    auto horzSum = StringAttr::get(ctx, "llvm.aarch64.neon.faddv.f32.v4f32");
    for (int64_t outIdx = 0; outIdx < numOfOutputChannels; outIdx += 1) {
      args = {outRes[outIdx]};
      // This horizontal sum intrinsic will sum all fp32 elements in the source
      // vector into a single fp32 element
      auto callIntrOp = rewriter.create<LLVM::CallIntrinsicOp>(
          loc, resultTypes, horzSum, args,
          LLVM::FastmathFlagsAttr::get(ctx, LLVM::FastmathFlags::fast));
      res = rewriter.create<vector::InsertOp>(loc, callIntrOp.getResult(0), res,
                                              outIdx);
    }

    if (!isZeroConst(acc)) {
      res = rewriter.create<arith::AddFOp>(loc, res, acc);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertMulSumToDotPack
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    Value acc = op.getAcc();
    auto resTy = cast<VectorType>(op.getType());

    Value matInput;
    Value vecInput;

    bool isMatch = isBf16DotProduct(op, /*useHorizontalSum=*/false, matInput,
                                    vecInput, rewriter);
    if (!isMatch)
      return failure();

    // Once we get the matrix input (NxK) and vector input (K),
    // where N is the output channel dimension
    // and K is the reduction dimension.
    // We will generate the following code to perform the dot product.
    // We will first transpose the matrix so that the output channel dimension
    // is continuous, so we can store multiple output channels in one
    // SIMD register.
    // Then we will loop over the K dimension.
    // For each iteration over K, we will pull 2 bf16 from the input vector.
    // Inside the K loop, we will also iterate over the output channels.
    // For each iteration over the output channel, we will pull
    // 4 output channel (each containing 2 bf16).
    // Then we will broadcast the 2 bf16 from the input vector,
    // dot product it with the 4 output channels (each containing 2 bf16),
    // and accumulate it with 4 outputs.
    // We will iterate over N until all output channels are processed.
    // Then we will move to the next 2 bf16 from the input vector (the K loop).
    // We will also share the vector input across the output channels.
    // For example, if we dot product a size 8x8 matrix with a size 8 vector,
    // the generated pseudo code will be:
    // Dimension:
    //            N:  the output channel dimension
    //            n0: the number of SIMD registers needed to store the output
    //                -- N / 4 (2 in this case)
    //            n1: the number of outputs stored per SIMD register
    //                -- 4
    //            K:  the reduction dimension
    //            k0: the number of SIMD registers needed for the input vector
    //                -- K / 8 (1 in this case)
    //            k1: the number of lanes per SIMD register
    //                -- 4
    //            k2: the number of bf16 elements per SIMD lane
    //                -- 2
    // matrix = shapecast(matrix, 8x4x2)
    //          shape: NxK -> Nx(k0xk1)xk2
    // matrix = tranpose(matrix, 1, 0, 2) : 8x4x2xbf16 -> 4x8x2xbf16
    //          shape: Nx(k0xk1)xk2 -> (k0xk1)xNxk2
    // matrix = shapecast(matrix, 1x4x2x4x2xbf16)
    //          shape: (k0xk1)xNxk2 -> k0xk1xn0xn1xk2
    // vector = shapecast(vector, 1x4x2)
    //          shape: K -> k0xk1xk2
    // out    = zeros(2x4, fp32)
    //          shape: n0xn1
    // subvec = broadcast(vector[0][0]) : 2xbf16 -> 4x2xbf16
    //          shape: k2 -> k1xk2
    // out[0] = bfdot(out[0], matrix[0][0][0], subvec)
    //          shape: (n1, n1xk2, k1xk2) -> n1
    // out[1] = bfdot(out[1], matrix[0][0][1], subvec)
    //          shape: (n1, n1xk2, k1xk2) -> n1
    // subvec = broadcast(vector[0][1]) : 2xbf16 -> 4x2xbf16
    //          shape: k2 -> k1xk2
    // out[0] = bfdot(out[0], matrix[0][1][0], subvec)
    //          shape: (n1, n1xk2, k1xk2) -> n1
    // out[1] = bfdot(out[1], matrix[0][1][1], subvec)
    //          shape: (n1, n1xk2, k1xk2) -> n1
    // subvec = broadcast(vector[0][2]) : 2xbf16 -> 4x2xbf16
    //          shape: k2 -> k1xk2
    // out[0] = bfdot(out[0], matrix[0][2][0], subvec)
    //          shape: (n1, n1xk2, k1xk2) -> n1
    // out[1] = bfdot(out[1], matrix[0][2][1], subvec)
    //          shape: (n1, n1xk2, k1xk2) -> n1
    // subvec = broadcast(vector[0][3]) : 2xbf16 -> 4x2xbf16
    //          shape: k2 -> k1xk2
    // out[0] = bfdot(out[0], matrix[0][3][0], subvec)
    //          shape: (n1, n1xk2, k1xk2) -> n1
    // out[1] = bfdot(out[1], matrix[0][3][1], subvec)
    //          shape: (n1, n1xk2, k1xk2) -> n1
    // out    = shapecast(out, 8) : 2x4xfp32 -> 8xfp32
    //          shape: n0xn1 -> N

    auto matInputTy = cast<VectorType>(matInput.getType());
    auto vecInputTy = cast<VectorType>(vecInput.getType());

    const int lanes =
        vectorBitWidth / matInputTy.getElementType().getIntOrFloatBitWidth();
    const int resultLanes =
        vectorBitWidth / resTy.getElementType().getIntOrFloatBitWidth();
    int64_t kVal = matInputTy.getDimSize(1);

    // numOfOutputChannels is the number of output channels (N)
    const int numOfOutputChannels = matInputTy.getDimSize(0);
    // numOfOutputRegs is the number of SIMD registers needed to store the
    // output.
    const int numOfOutputRegs = numOfOutputChannels / resultLanes;
    // numOfVecRegs is the number of SIMD registers needed for the
    // input vector.
    const int numOfVecRegs = kVal / lanes;
    // numOfVecPairs is the number of pairs (pair of bf16 elements) for the
    // input vector.
    const int numOfVecPairs = numOfVecRegs * resultLanes;

    VectorType fullResTy =
        VectorType::get({numOfOutputRegs, resultLanes}, resTy.getElementType());

    VectorType subResTy = VectorType::get(resultLanes, resTy.getElementType());

    acc = shapeCast(loc, acc, fullResTy, rewriter);

    Type inElemTy = matInputTy.getElementType();
    // Integer type for a pair of bf16 elements
    Type pairTy = IntegerType::get(ctx, 32);

    vecInput =
        shapeCast(loc, vecInput, {numOfVecRegs, resultLanes, 2}, rewriter);
    // We bitcast here because we are pulling pairs of bf16 each time.
    vecInput = rewriter.create<vector::BitCastOp>(
        loc, VectorType::get({numOfVecRegs, resultLanes, 1}, pairTy), vecInput);
    vecInput = shapeCast(loc, vecInput, {numOfVecRegs, resultLanes}, rewriter);

    matInput = shapeCast(loc, matInput, {numOfOutputChannels, numOfVecPairs, 2},
                         rewriter);
    // We bitcast here because we are pulling pairs of bf16 each time.
    matInput = rewriter.create<vector::BitCastOp>(
        loc, VectorType::get({numOfOutputChannels, numOfVecPairs, 1}, pairTy),
        matInput);
    matInput = shapeCast(loc, matInput, {numOfOutputChannels, numOfVecPairs},
                         rewriter);
    // Packing/Transposing the weight matrix so that
    // the output channel is continuous
    matInput = rewriter.create<vector::TransposeOp>(
        loc, matInput, SmallVector<int64_t, 2>{1, 0});
    matInput = shapeCast(
        loc, matInput,
        {numOfVecRegs, resultLanes, numOfOutputRegs, resultLanes}, rewriter);

    Value res = rewriter.create<arith::ConstantOp>(
        loc, fullResTy, rewriter.getZeroAttr(fullResTy));
    SmallVector<Type> resultTypes = {subResTy};
    // TODO: this intrinsic is hard-coded for Arm Neon
    auto bfdot = StringAttr::get(ctx, "llvm.aarch64.neon.bfdot.v4f32.v8bf16");
    SmallVector<Value> args;

    SmallVector<Value> subRes(numOfOutputRegs);
    for (int64_t outIdx = 0; outIdx < numOfOutputRegs; outIdx += 1) {
      subRes[outIdx] = rewriter.create<vector::ExtractOp>(loc, acc, outIdx);
    }
    for (int64_t idx = 0; idx < numOfVecRegs; idx += 1) {
      Value fullVec = rewriter.create<vector::ExtractOp>(loc, vecInput, idx);
      for (int64_t vecIdx = 0; vecIdx < resultLanes; vecIdx += 1) {
        // shuffle mask used to broadcast the 'vecIdx'th lane of fullVec
        SmallVector<int64_t> shuffleMask(resultLanes, vecIdx);
        // Broadcasting the 'vecIdx'th lane of fullVec
        Value subVec = rewriter.create<vector::ShuffleOp>(loc, fullVec, fullVec,
                                                          shuffleMask);
        subVec = rewriter.create<vector::BitCastOp>(
            loc, VectorType::get({lanes}, inElemTy), subVec);
        for (int64_t outIdx = 0; outIdx < numOfOutputRegs; outIdx += 1) {
          Value subMat = rewriter.create<vector::ExtractOp>(
              loc, matInput, SmallVector<int64_t, 3>{idx, vecIdx, outIdx});
          subMat = rewriter.create<vector::BitCastOp>(
              loc, VectorType::get({lanes}, inElemTy), subMat);
          args = {subRes[outIdx], subMat, subVec};
          // bfdot instruction:
          // https://developer.arm.com/documentation/ddi0602/2024-06/SIMD-FP-Instructions/BFDOT--vector---BFloat16-floating-point-dot-product--vector--
          // LLVM fast math flags:
          // https://llvm.org/docs/LangRef.html#fast-math-flags
          // This bfdot intrinsic will perform an unfused sum-of-products of
          // each pair of adjacent bf16 elements in the source vectors
          // (8 bf16), and output 4 fp32 elements.
          auto callIntrOp = rewriter.create<LLVM::CallIntrinsicOp>(
              loc, resultTypes, bfdot, args,
              LLVM::FastmathFlagsAttr::get(ctx, LLVM::FastmathFlags::fast));
          subRes[outIdx] = callIntrOp.getResult(0);
        }
      }
    }

    for (int64_t outIdx = 0; outIdx < numOfOutputRegs; outIdx += 1) {
      res = rewriter.create<vector::InsertOp>(loc, subRes[outIdx], res, outIdx);
    }

    res = shapeCast(loc, res, resTy, rewriter);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertDotProduct
    : public triton::cpu::impl::ConvertDotProductBase<ConvertDotProduct> {
  ConvertDotProduct() = default;
  ConvertDotProduct(bool useHorizontalSum) {
    this->useHorizontalSum = useHorizontalSum;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    RewritePatternSet patterns(context);

    if (useHorizontalSum) {
      patterns.add<ConvertMulSumToDotHorizontalSum>(context);
    } else {
      patterns.add<ConvertMulSumToDotPack>(context);
    }

    if (failed(mlir::applyPatternsAndFoldGreedily(mod, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotProduct() {
  return std::make_unique<ConvertDotProduct>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertDotProduct(bool useHorizontalSum) {
  return std::make_unique<ConvertDotProduct>(useHorizontalSum);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
