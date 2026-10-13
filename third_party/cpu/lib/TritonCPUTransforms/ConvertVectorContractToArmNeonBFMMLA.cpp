#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <utility>

#define DEBUG_TYPE "triton-cpu-convert-vector-contract-to-arm-neon-bfmmla"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTVECTORCONTRACTTOARMNEONBFMMLA
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {
// Native BFMMLA tile dimensions for (M, N, K).
constexpr int64_t tileSizeM = 2;
constexpr int64_t tileSizeN = 2;
constexpr int64_t tileSizeK = 4;

enum class Form { MM, MMT };

struct Candidate {
  Form form;
  int64_t m;
  int64_t n;
  int64_t k;
};

// Inspect only the contraction structure and masking context. Unsupported
// contractions remain available for existing FP32 legalization.
FailureOr<Candidate> matchCandidate(vector::ContractionOp op) {
  // Normalization must not insert a transpose into a vector.mask region.
  if (op.isMasked()) {
    LDBG("Drop candidate. Masked contractions are not supported.");
    return failure();
  }

  VectorType lhsType = op.getLhsType();
  VectorType rhsType = op.getRhsType();
  auto accType = dyn_cast<VectorType>(op.getAccType());
  auto resultType = dyn_cast<VectorType>(op.getResultType());
  for (VectorType type : {lhsType, rhsType, accType, resultType}) {
    if (!type || type.getRank() != 2 || type.isScalable()) {
      LDBG("Drop candidate. Expected fixed-width rank-2 vectors.");
      return failure();
    }
  }
  if (!lhsType.getElementType().isBF16() ||
      !rhsType.getElementType().isBF16() || !accType.getElementType().isF32() ||
      !resultType.getElementType().isF32()) {
    LDBG("Drop candidate. Expected BF16 inputs and FP32 accumulator/result.");
    return failure();
  }

  // The MM-to-MMT helper rebuilds the contraction with default attributes.
  // Only accept attributes whose effective semantics it preserves.
  if (op.getKind() != vector::CombiningKind::ADD) {
    LDBG("Drop candidate. Expected ADD combining kind.");
    return failure();
  }
  if (op.getFastmath() != arith::FastMathFlags::none) {
    LDBG("Drop candidate. Expected fastmath=none.");
    return failure();
  }
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (name != "indexing_maps" && name != "iterator_types" && name != "kind" &&
        name != "fastmath") {
      LDBG("Drop candidate. Unsupported attribute: " << name);
      return failure();
    }
  }

  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() != 3 || iterators[0] != vector::IteratorType::parallel ||
      iterators[1] != vector::IteratorType::parallel ||
      iterators[2] != vector::IteratorType::reduction) {
    LDBG("Drop candidate. Expected parallel, parallel, reduction iterators.");
    return failure();
  }

  MLIRContext *ctx = op.getContext();
  auto lhsMap = AffineMap::getMultiDimMapWithTargets(3, {0, 2}, ctx);
  // (m, n, k) -> (k, n)
  auto mmRhsMap = AffineMap::getMultiDimMapWithTargets(3, {2, 1}, ctx);
  // (m, n, k) -> (n, k)
  auto mmtRhsMap = AffineMap::getMultiDimMapWithTargets(3, {1, 2}, ctx);
  auto accMap = AffineMap::getMultiDimMapWithTargets(3, {0, 1}, ctx);
  auto maps = op.getIndexingMapsArray();
  if (maps.size() != 3 || maps[0] != lhsMap || maps[2] != accMap ||
      (maps[1] != mmRhsMap && maps[1] != mmtRhsMap)) {
    LDBG("Drop candidate. Expected MM or MMT indexing maps.");
    return failure();
  }
  Form form = maps[1] == mmRhsMap ? Form::MM : Form::MMT;

  int64_t m = lhsType.getDimSize(0);
  int64_t k = lhsType.getDimSize(1);
  int64_t n = rhsType.getDimSize(form == Form::MM ? 1 : 0);
  int64_t rhsK = rhsType.getDimSize(form == Form::MM ? 0 : 1);
  if (m <= 0 || n <= 0 || k <= 0) {
    LDBG("Drop candidate. Expected positive M, N and K dimensions.");
    return failure();
  }
  if (rhsK != k || accType != resultType || accType.getDimSize(0) != m ||
      accType.getDimSize(1) != n) {
    LDBG("Drop candidate. Inconsistent operand or result shapes.");
    return failure();
  }
  if (m % tileSizeM != 0 || n % tileSizeN != 0 || k % tileSizeK != 0) {
    LDBG("Drop candidate. Expected M/N multiples of 2 and K a multiple of 4, "
         "but got M="
         << m << ", N=" << n << ", K=" << k);
    return failure();
  }

  return Candidate{form, m, n, k};
}

// SBGemm means matrix multiplication with BF16 inputs and FP32 outputs.
struct LowerVectorContractSBGemm
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override;

private:
  Value rewrite(vector::ContractionOp op, const Candidate &candidate,
                PatternRewriter &rewriter) const;
};

LogicalResult
LowerVectorContractSBGemm::matchAndRewrite(vector::ContractionOp op,
                                           PatternRewriter &rewriter) const {
  auto candidate = matchCandidate(op);
  if (failed(candidate))
    return failure();
  if (candidate->form != Form::MMT)
    return rewriter.notifyMatchFailure(op, "expected MMT contraction");

  rewriter.replaceOp(op, rewrite(op, *candidate, rewriter));
  return success();
}

Value LowerVectorContractSBGemm::rewrite(vector::ContractionOp op,
                                         const Candidate &candidate,
                                         PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value result = op.getAcc();
  auto tileType =
      VectorType::get({tileSizeM, tileSizeN}, rewriter.getF32Type());

  // Unroll the native 2x2x4 tiles at compile time; do not emit runtime loops.
  for (int64_t m = 0; m < candidate.m; m += tileSizeM) {
    // Extract the original LHS and accumulator row pairs once per M tile.
    // Each LHS row is vector<Kxbf16>; the K loop packs a 2x4 input tile.
    Value lhsRow0 = vector::ExtractOp::create(rewriter, loc, op.getLhs(), m);
    Value lhsRow1 =
        vector::ExtractOp::create(rewriter, loc, op.getLhs(), m + 1);
    // Each accumulator row is vector<Nxf32>; the N loop packs a 2x2 tile.
    Value accRow0 = vector::ExtractOp::create(rewriter, loc, op.getAcc(), m);
    Value accRow1 =
        vector::ExtractOp::create(rewriter, loc, op.getAcc(), m + 1);

    for (int64_t n = 0; n < candidate.n; n += tileSizeN) {
      // MMT stores each output column as a complete K-element RHS row.
      Value rhsRow0 = vector::ExtractOp::create(rewriter, loc, op.getRhs(), n);
      Value rhsRow1 =
          vector::ExtractOp::create(rewriter, loc, op.getRhs(), n + 1);

      // Pack the original 2x2 FP32 tile into vector<4xf32> in row-major order.
      // The second row starts at N in the concatenated accumulator rows.
      const int64_t accMask[] = {n, n + 1, candidate.n + n,
                                 candidate.n + n + 1};
      Value packedAcc =
          vector::ShuffleOp::create(rewriter, loc, accRow0, accRow1, accMask);

      for (int64_t k = 0; k < candidate.k; k += tileSizeK) {
        // Four elements from each K-element row form a vector<8xbf16>.
        // LHS packs a row-major 2x4 tile; the two MMT RHS rows pack the
        // corresponding 4x2 RHS tile in column-major order.
        const int64_t inputMask[] = {k,
                                     k + 1,
                                     k + 2,
                                     k + 3,
                                     candidate.k + k,
                                     candidate.k + k + 1,
                                     candidate.k + k + 2,
                                     candidate.k + k + 3};
        Value packedLhs = vector::ShuffleOp::create(rewriter, loc, lhsRow0,
                                                    lhsRow1, inputMask);
        Value packedRhs = vector::ShuffleOp::create(rewriter, loc, rhsRow0,
                                                    rhsRow1, inputMask);

        // Keep the 2x2 accumulator in vector<4xf32> across all K steps.
        packedAcc =
            arm_neon::BfmmlaOp::create(rewriter, loc, packedAcc.getType(),
                                       packedAcc, packedLhs, packedRhs);
      }

      // Assemble each completed vector<2x2xf32> output tile exactly once.
      Value storeTile =
          rewriter.createOrFold<vector::ShapeCastOp>(loc, tileType, packedAcc);
      const int64_t offsets[] = {m, n};
      const int64_t strides[] = {1, 1};
      result = rewriter.createOrFold<vector::InsertStridedSliceOp>(
          loc, storeTile, result, offsets, strides);
    }
  }

  return result;
}

struct ConvertVectorContractToArmNeonBFMMLA
    : public triton::cpu::impl::ConvertVectorContractToArmNeonBFMMLABase<
          ConvertVectorContractToArmNeonBFMMLA> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    vector::populateVectorContractCanonicalizeMatmulToMMT(
        patterns, [](vector::ContractionOp op) -> LogicalResult {
          auto candidate = matchCandidate(op);
          if (failed(candidate) || candidate->form != Form::MM)
            return failure();
          return success();
        });
    patterns.add<LowerVectorContractSBGemm>(ctx);

    // Normalize eligible MM contracts, then lower their MMT replacements.
    // Rejected contracts remain available for existing FP32 legalization.
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertVectorContractToArmNeonBFMMLA() {
  return std::make_unique<ConvertVectorContractToArmNeonBFMMLA>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
