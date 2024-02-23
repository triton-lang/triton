#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
namespace {
using namespace mlir;
using namespace mlir::triton;
struct AssertOpConversion : public ConvertOpToLLVMPattern<triton::AssertOp> {
  using ConvertOpToLLVMPattern<triton::AssertOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();
    auto elems = unpackLLElements(loc, adaptor.getCondition(), rewriter);
    auto elemTy = elems[0].getType();
    Value condition = int_val(elemTy.getIntOrFloatBitWidth(), 0);
    for (auto elem : elems) {
      if (elemTy.isSignedInteger() || elemTy.isSignlessInteger()) {
        condition =
            or_(condition,
                icmp_eq(elem, rewriter.create<LLVM::ConstantOp>(
                                  loc, elemTy, rewriter.getZeroAttr(elemTy))));
      } else {
        assert(false && "Unsupported type for assert");
        return failure();
      }
    }
    llAssert(op, condition, adaptor.getMessage(), adaptor.getFile(),
             adaptor.getFunc(), adaptor.getLine(), rewriter);
    rewriter.eraseOp(op);
    return success();
  }
  // op: the op at which the assert is inserted. Unlike printf, we need to
  // know about the op to split the block.
  static void llAssert(Operation *op, Value condition, StringRef message,
                       StringRef file, StringRef func, int line,
                       ConversionPatternRewriter &rewriter) {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    auto ctx = rewriter.getContext();
    auto loc = op->getLoc();
    // #block1
    // if (condition) {
    //   #block2
    //   __assertfail(message);
    // }
    // #block3
    Block *prevBlock = op->getBlock();
    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);
    auto funcOp = getAssertfailDeclaration(rewriter);
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    Value messageString =
        LLVM::addStringToModule(loc, rewriter, "assertMessage_", message);
    Value fileString =
        LLVM::addStringToModule(loc, rewriter, "assertFile_", file);
    Value funcString =
        LLVM::addStringToModule(loc, rewriter, "assertFunc_", func);
    Value lineNumber = i32_val(line);
    Value charSize = int_val(sizeof(size_t) * 8, sizeof(char));
    SmallVector<Value> operands = {messageString, fileString, lineNumber,
                                   funcString, charSize};
    auto ret = call(funcOp, operands);
    // Split a block after the call.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<cf::BranchOp>(loc, thenBlock);
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<cf::CondBranchOp>(loc, condition, ifBlock, thenBlock);
  }
  static LLVM::LLVMFuncOp
  getAssertfailDeclaration(ConversionPatternRewriter &rewriter) {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    StringRef funcName("__assertfail");
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);
    // void __assert_fail(const char * assertion, const char * file, unsigned
    // int line, const char * function);
    auto *ctx = rewriter.getContext();
    SmallVector<Type> argsType{ptr_ty(ctx), ptr_ty(ctx), i32_ty, ptr_ty(ctx),
                               rewriter.getIntegerType(sizeof(size_t) * 8)};
    auto funcType = LLVM::LLVMFunctionType::get(void_ty(ctx), argsType);
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                             funcType);
  }
};

} // namespace

void mlir::triton::populateAssertOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<AssertOpConversion>(typeConverter, benefit);
}
