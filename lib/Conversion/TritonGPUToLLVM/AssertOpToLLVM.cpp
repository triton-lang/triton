#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace {

using namespace mlir;

struct AssertOpConversion : public ConvertOpToLLVMPattern<triton::AssertOp> {
  explicit AssertOpConversion(LLVMTypeConverter &typeConverter,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AssertOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto elems =
        unpackUniqueTensorElements(loc, adaptor.getCondition(), rewriter);
    auto elemTy = elems[0].getType();
    Value condition = b.int_val(elemTy.getIntOrFloatBitWidth(), 0);
    for (auto elem : elems) {
      if (elemTy.isSignedInteger() || elemTy.isSignlessInteger()) {
        condition = b.or_(condition,
                          b.icmp_eq(elem, LLVM::ConstantOp::create(
                                              rewriter, loc, elemTy,
                                              rewriter.getZeroAttr(elemTy))));
      } else {
        return op->emitError("Unsupported type for assert");
      }
    }
    llAssert(op, condition, adaptor.getMessage(), rewriter);
    if (isa<RankedTensorType>(op.getCondition().getType())) {
      // Add a barrier to avoid a race condition in case an assert is followed
      // by an op that may trap if the assert condition is true. Since the
      // tensor in those two operations may have different layout we need to
      // make sure all the threads are done executing the assert before going to
      // the next op.
      b.barrier(triton::gpu::AddrSpace::None);
    }
    rewriter.eraseOp(op);
    return success();
  }
  // op: the op at which the assert is inserted. Unlike printf, we need to
  // know about the op to split the block.
  void llAssert(AssertOp op, Value condition, StringRef message,
                ConversionPatternRewriter &rewriter) const {

    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    StringRef file = "unknown";
    StringRef func = "unknown";
    int line = 0;

    while (auto callLoc = dyn_cast<CallSiteLoc>(loc))
      loc = callLoc.getCallee();

    while (auto nameLoc = dyn_cast<NameLoc>(loc))
      loc = nameLoc.getChildLoc();

    if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(loc)) {
      file = fileLineColLoc.getFilename();
      line = fileLineColLoc.getLine();
    }

    if (targetInfo.isCuda() &&
        op->getParentOfType<triton::gpu::WarpSpecializeOp>()) {
      Block *previousBlock = rewriter.getInsertionBlock();
      Region *region = previousBlock->getParent();
      std::string diagnostic = message.str();
      diagnostic.push_back('\0');
      diagnostic.append(file);
      diagnostic.push_back('\0');
      diagnostic.append(func);
      diagnostic.push_back('\0');
      diagnostic.append(std::to_string(line));

      // Keep failure paths inside their owning warp-specialization region.
      // CUDA's assertion function is noreturn, so identical diagnostics within
      // the same region can share one terminal failure block.
      Block *&failureBlock = failureBlocks[region][diagnostic];
      if (!failureBlock) {
        OpBuilder::InsertionGuard guard(rewriter);
        failureBlock = rewriter.createBlock(region);
        targetInfo.assertFail(rewriter, loc, message, file, func, line);
        LLVM::UnreachableOp::create(rewriter, loc);
      }

      Block *continueBlock =
          rewriter.splitBlock(previousBlock, rewriter.getInsertionPoint());
      rewriter.setInsertionPointToEnd(previousBlock);
      LLVM::CondBrOp::create(rewriter, loc, condition, failureBlock,
                             continueBlock);
      rewriter.setInsertionPointToStart(continueBlock);
      return;
    }

    auto [prevBlock, ifBlock, thenBlock] =
        createIfBlock(rewriter, loc, condition);

    rewriter.setInsertionPointToStart(ifBlock);
    targetInfo.assertFail(rewriter, loc, message, file, func, line);

    // Split a block after the call.
    rewriter.setInsertionPointToStart(thenBlock);
  }

protected:
  const TargetInfoBase &targetInfo;
  mutable llvm::DenseMap<Region *, llvm::StringMap<Block *>> failureBlocks;
};

} // namespace

void mlir::triton::populateAssertOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<AssertOpConversion>(typeConverter, targetInfo, benefit);
}
