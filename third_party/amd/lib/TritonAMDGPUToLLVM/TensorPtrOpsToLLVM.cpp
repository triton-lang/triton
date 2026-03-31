#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TDMUtility.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
// Validates that the tensor descriptor's strides and shared layout are
// compatible with TDM. Requirements:
//  - The shared order must be [rank-1, rank-2, ..., 0].
//  - All stride-1 dimensions must be consecutive trailing dims.
// Additionally, a single stride-1 dimension may appear at the rank-2
// position (col-major) if the shared order has rank-2 and rank-1 swapped.
LogicalResult validateStridesAndSharedOrder(triton::MakeTensorDescOp op,
                                            Attribute sharedEnc,
                                            ArrayRef<int64_t> shape,
                                            ValueRange strides) {
  int rank = shape.size();
  auto sharedOrder = triton::gpu::getOrder(
      cast<triton::gpu::SharedEncodingTrait>(sharedEnc), shape);

  SmallVector<unsigned> strideOneDims;
  for (auto [dim, strideVal] : llvm::enumerate(strides)) {
    if (getConstantIntValue(getAsOpFoldResult(strideVal)).value_or(0) == 1)
      strideOneDims.push_back(dim);
  }

  if (strideOneDims.empty())
    return op.emitError() << "requires at least one dimension to have stride 1";

  // If the only stride-1 dim is the second-to-last dimension (col-major) we can
  // safely reorder the dimensions during lowering.
  bool isColMajor =
      strideOneDims.size() == 1 && strideOneDims.front() == rank - 2;

  SmallVector<unsigned> expectedOrder(llvm::reverse(llvm::seq<unsigned>(rank)));
  if (isColMajor)
    std::swap(expectedOrder[0], expectedOrder[1]);

  if (sharedOrder != ArrayRef(expectedOrder)) {
    if (isColMajor)
      return op.emitError()
             << "requires shared order [rank-2, rank-1, rank-3, "
                "rank-4, ..., 0] because dim[rank-2] has stride 1";
    return op.emitError() << "requires shared order [rank-1, rank-2, ..., 0]";
  }

  if (strideOneDims.size() > 1) {
    unsigned k = strideOneDims.size();
    unsigned numStride1Dims = strideOneDims.size();
    for (unsigned i = 0; i < numStride1Dims; ++i) {
      if (strideOneDims[i] != rank - numStride1Dims + i)
        return op.emitError() << "requires all stride 1 dimensions to be "
                                 "consecutive starting from the last dimension";
    }
  }

  return success();
}

// Collects all users of the value beyond the basic block boundaries
// defining a given value.
void collectUsers(Value value, llvm::SetVector<Operation *> &users) {
  for (OpOperand &use : value.getUses()) {
    Operation *userOp = use.getOwner();
    if (users.contains(userOp)) {
      // stop recursion; avoid loops
      return;
    }
    users.insert(userOp);
    const unsigned argIdx = use.getOperandNumber();

    if (auto unrealCast = dyn_cast<mlir::UnrealizedConversionCastOp>(userOp)) {
      collectUsers(unrealCast->getResult(argIdx), users);
    }

    if (auto branch = dyn_cast<mlir::BranchOpInterface>(userOp)) {
      auto successors = branch->getSuccessors();
      for (auto [idx, successor] : llvm::enumerate(successors)) {
        auto operands = branch.getSuccessorOperands(idx);
        if (argIdx < operands.size()) {
          collectUsers(successor->getArgument(argIdx), users);
        }
      }
    }
  }
}

struct MakeTensorDescOpConversion
    : public ConvertOpToLLVMPattern<triton::MakeTensorDescOp> {
  using ConvertOpToLLVMPattern<
      triton::MakeTensorDescOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::MakeTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto basePtr = adaptor.getBase();
    auto tensorShape = llvm::to_vector(adaptor.getShape());
    auto tensorStride = llvm::to_vector(adaptor.getStrides());
    auto result = op.getResult();

    auto tensorDescTy = result.getType();
    auto blockTy = tensorDescTy.getBlockType();
    auto sharedEnc = blockTy.getEncoding();
    if (!sharedEnc) {
      if (!sharedEnc)
        return rewriter.notifyMatchFailure(
            op, "Descriptor has no shared memory layout assigned.");
    }
    unsigned padInterval = 0;
    unsigned padAmount = 0;
    if (auto padEnc = getPaddedEncoding(sharedEnc)) {
      if (padEnc.getIntervals().size() != 1 || padEnc.getPaddings().size() != 1)
        return rewriter.notifyMatchFailure(
            op, "NYI: Multiple interval-padding pairs in TDM.");
      padInterval = padEnc.getIntervals()[0];
      padAmount = padEnc.getPaddings()[0];
    }

    Type elementType =
        getTypeConverter()->convertType(blockTy.getElementType());
    SmallVector<int64_t> blockShape = to_vector(blockTy.getShape());
    int numWarps = lookupNumWarps(op);
    auto shapePerCTA = triton::gpu::getShapePerCTA(sharedEnc, blockShape);

    if (failed(validateStridesAndSharedOrder(op, sharedEnc, shapePerCTA,
                                             tensorStride))) {
      return failure();
    }
    auto sharedOrder = triton::gpu::getOrder(
        cast<triton::gpu::SharedEncodingTrait>(sharedEnc), shapePerCTA);
    bool isRowMajor = sharedOrder[0] == (sharedOrder.size() - 1);
    // Create TDM descriptor for 2D-5D tensors
    auto tdmDesc = LLVM::AMD::createTDMDescriptor(
        rewriter, loc, getTypeConverter(), elementType, shapePerCTA, numWarps,
        padInterval, padAmount, tensorShape, tensorStride, basePtr, isRowMajor,
        sharedEnc);

    SmallVector<Value> groups = tdmDesc.getAllGroups();

    auto desc =
        packLLElements(loc, getTypeConverter(), groups, rewriter, tensorDescTy);

    rewriter.replaceOp(op, desc);
    return success();
  }
};
} // namespace

void mlir::triton::AMD::populateTensorPtrOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<MakeTensorDescOpConversion>(typeConverter, benefit);
  return;
}
