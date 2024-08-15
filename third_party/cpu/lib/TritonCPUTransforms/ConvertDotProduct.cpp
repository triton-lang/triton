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
bool isBf16DotProduct(vector::MultiDimReductionOp op, Value &matInput,
                      Value &vecInput, PatternRewriter &rewriter) {
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
  int64_t kVal = lhsTy.getDimSize(1);

  if (outNum < 1)
    return false;

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

    bool isMatch = isBf16DotProduct(op, matInput, vecInput, rewriter);
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
    const int resLanes =
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

    Type outResTy = VectorType::get(resLanes, resTy.getElementType());

    Value zeroRes = rewriter.create<arith::ConstantOp>(
        loc, outResTy, rewriter.getZeroAttr(outResTy));
    for (int64_t outIdx = 0; outIdx < numOfOutputChannels; outIdx += 1) {
      outRes[outIdx] = zeroRes;
      // Intermediate array to store each row of the input matrix.
      mats[outIdx] = rewriter.create<vector::ExtractOp>(loc, matInput, outIdx);
    }

    SmallVector<Type> resultTypes = {outResTy};
    // TODO: this intrinsic is hard-coded for Arm Neon
    llvm::StringRef bfdotIntrinsic("llvm.aarch64.neon.bfdot.v4f32.v8bf16");
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
            loc, resultTypes, bfdotIntrinsic, args, LLVM::FastmathFlags::fast);
        outRes[outIdx] = callIntrOp.getResult(0);
      }
    }

    Value res = rewriter.create<arith::ConstantOp>(loc, resTy,
                                                   rewriter.getZeroAttr(resTy));

    resultTypes = {resTy.getElementType()};
    // TODO: this intrinsic is hard-coded for Arm Neon
    llvm::StringRef horizSumIntrinsic("llvm.aarch64.neon.faddv.f32.v4f32");
    for (int64_t outIdx = 0; outIdx < numOfOutputChannels; outIdx += 1) {
      args = {outRes[outIdx]};
      // This horizontal sum intrinsic will sum all fp32 elements in the source
      // vector into a single fp32 element
      auto callIntrOp = rewriter.create<LLVM::CallIntrinsicOp>(
          loc, resultTypes, horizSumIntrinsic, args, LLVM::FastmathFlags::fast);
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

struct ConvertDotProduct
    : public triton::cpu::impl::ConvertDotProductBase<ConvertDotProduct> {
  ConvertDotProduct() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    RewritePatternSet patterns(context);

    patterns.add<ConvertMulSumToDotHorizontalSum>(context);

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

} // namespace cpu
} // namespace triton
} // namespace mlir
