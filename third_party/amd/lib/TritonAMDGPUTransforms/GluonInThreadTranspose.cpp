#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

// GluonInThreadTranspose pass optimizes ttg.local_alloc and ttg.local_store
// by inserting in-thread transpose operations if shared memory order is
// different from in-thread order of src tensor.
//
// Example of performed transformation:
//
//     %data = tt.load : #blocked<{sizePerThread = [4, 4], order = [0, 1]}>
//     %buf = local_alloc %data : #ttg.swizzled_shared<{order = [1, 0]}>
//
//   Transforms to
//
//      register bases = [[1, 0], [2, 0], [0, 1], [0, 2]]
//        |
//        V
//     %data = tt.load : #blocked<{sizePerThread = [4, 4], order = [0, 1]}>>
//      register bases = [[0, 1], [0, 2], [1, 0], [2, 0]]
//          |
//          V
//     %transposed = amdg.in_thread_transpose %data : #linear
//     %buf = local_alloc %transposed : #ttg.swizzled_shared<{order = [1, 0]}>
//
// This is a simplified version of the InThreadTranspose pass designed
// specifically for the Gluon frontend. Optimization assumes kernel author take
// care about transposable source layout, i.e. shapePerThread is not a vector
// ([4, 16], not [1, 16]). It also does not depend on or change operations,
// except optimized local_alloc or local_store.

#define DEBUG_TYPE "tritonamdgpu-gluon-in-thread-transpose"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttag = mlir::triton::amdgpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUGLUONINTHREADTRANSPOSE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

Type replaceEncoding(Type type, Attribute encoding) {
  RankedTensorType tensorType = cast<RankedTensorType>(type);
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType(), encoding);
}

bool isTransposableEncoding(RankedTensorType srcType,
                            ttg::MemDescType dstType) {
  auto srcEnc = srcType.getEncoding();
  auto dstEnc = dstType.getEncoding();
  auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcEnc);
  if (!blockedEnc)
    return false;

  auto srcOrder = ttg::getOrder(srcType);
  auto dstOrder = ttg::getOrder(dstType);
  return srcOrder[0] != dstOrder[0];
}

unsigned getTransposableSizePerThread(RankedTensorType srcType) {
  auto blockedEnc = cast<ttg::BlockedEncodingAttr>(srcType.getEncoding());
  auto inThreadShape = blockedEnc.getSizePerThread();
  assert(inThreadShape.size() == 2);
  return std::min(inThreadShape[0], inThreadShape[1]);
}

struct LocalAllocOpTraits {
  using OpT = ttg::LocalAllocOp;

  static Value getMemDescValue(OpT op) { return op.getResult(); }
};

struct LocalStoreOpTraits {
  using OpT = ttg::LocalStoreOp;

  static Value getMemDescValue(OpT op) { return op.getDst(); }
};

/// Checks if local memory store could be optimized by in-thread transpose.
/// Works with both LocalAllocOp and LocalStoreOp.
template <typename TraitsT>
static std::optional<Value>
matchTransposableLocalMemPattern(typename TraitsT::OpT localMemOp) {
  // Check if local memory op has a source operand.
  // This check could fail for local_alloc.
  Value src = localMemOp.getSrc();
  if (src == nullptr) {
    LDBG("local memory op has no source operand");
    return std::nullopt;
  }

  Value memDescVal = TraitsT::getMemDescValue(localMemOp);
  auto srcType = cast<RankedTensorType>(src.getType());
  auto dstType = cast<ttg::MemDescType>(memDescVal.getType());

  // Only support 2D tensors for now
  if (srcType.getRank() != 2) {
    LDBG("only 2D tensors are supported");
    return std::nullopt;
  }

  // Check if the encoding is transposable
  if (!isTransposableEncoding(srcType, dstType)) {
    LDBG("src encoding is not suitable for transpose");
    return std::nullopt;
  }

  // Check that there's room for transpose
  auto kDimMaxSizePerThread = getTransposableSizePerThread(srcType);
  if (kDimMaxSizePerThread < 2) {
    LDBG("kDim size per thread < 2, cannot transpose");
    return std::nullopt;
  }

  return src;
}

/// Template pattern that works with both LocalAllocOp and LocalStoreOp.
/// TraitsT provides operation-specific accessors.
template <typename TraitsT>
class GluonInThreadTransposePattern
    : public OpRewritePattern<typename TraitsT::OpT> {
public:
  using OpT = typename TraitsT::OpT;

  GluonInThreadTransposePattern(MLIRContext *context,
                                PatternBenefit benefit = 1)
      : OpRewritePattern<OpT>(context, benefit) {}

  LogicalResult matchAndRewrite(OpT localMemOp,
                                PatternRewriter &rewriter) const override {
    LDBG("Consider " << localMemOp);

    auto matchResult = matchTransposableLocalMemPattern<TraitsT>(localMemOp);
    if (!matchResult) {
      LDBG("Failed to match a transposable local memory operation");
      return failure();
    }

    auto srcValue = *matchResult;

    auto srcType = cast<RankedTensorType>(srcValue.getType());
    auto srcShape = srcType.getShape();
    auto srcEncoding = cast<ttg::BlockedEncodingAttr>(srcType.getEncoding());

    // Create the transposed layout
    auto transposedLayout =
        ttag::InThreadTransposeOp::deduceOutputLayout(srcShape, srcEncoding);
    auto transposedEncoding = ttg::LinearEncodingAttr::get(
        localMemOp->getContext(), std::move(transposedLayout));

    auto loc = localMemOp->getLoc();

    // Create the in-thread transpose operation
    rewriter.setInsertionPoint(localMemOp);
    auto transposedType = replaceEncoding(srcType, transposedEncoding);
    Value inThreadTransposed = ttag::InThreadTransposeOp::create(
        rewriter, loc, transposedType, srcValue);

    // Update the local memory op to use the transposed value
    rewriter.startOpModification(localMemOp);
    localMemOp.getSrcMutable().assign(inThreadTransposed);
    rewriter.finalizeOpModification(localMemOp);

    LDBG("Successfully inserted in-thread transpose");
    return success();
  }
};

class TritonAMDGPUGluonInThreadTransposePass
    : public impl::TritonAMDGPUGluonInThreadTransposeBase<
          TritonAMDGPUGluonInThreadTransposePass> {

public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    auto ctx = m.getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<GluonInThreadTransposePattern<LocalAllocOpTraits>>(
        ctx, /*benefit=*/1);
    patterns.add<GluonInThreadTransposePattern<LocalStoreOpTraits>>(
        ctx, /*benefit=*/1);
    walkAndApplyPatterns(m, std::move(patterns));
  }
};

} // anonymous namespace

} // namespace mlir
