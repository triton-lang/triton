#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/Passes.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "third_party/proton/dialect/include/Dialect/ProtonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton::proton {
#define GEN_PASS_DEF_ADDPROTONKERNELARG
#include "proton/dialect/include/Conversion/ProtonGPUToLLVM/Passes.h.inc"
} // namespace triton::proton
} // namespace mlir

namespace {

static void filterFuncAttributes(triton::FuncOp op, bool filterArgAttrs,
                                 SmallVectorImpl<NamedAttribute> &result) {

  for (const auto &attr : op->getAttrs()) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == op.getFunctionTypeAttrName() ||
        attr.getName() == "std.varargs" ||
        (filterArgAttrs && attr.getName() == op.getArgAttrsAttrName()))
      continue;
    result.push_back(attr);
  }
}

triton::FuncOp insertAmendedFuncOp(triton::FuncOp funcOp,
                                   IRRewriter &rewriter) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  Location loc = moduleOp->getLoc();
  auto ctx = funcOp->getContext();
  auto funcTy = funcOp.getFunctionType();
  SmallVector<Type> amendedInputTy(funcTy.getInputs());
  Type globalPtrTy = triton::PointerType::get(rewriter.getI8Type(), 1);

  amendedInputTy.push_back(globalPtrTy);
  auto amendedFuncTy =
      FunctionType::get(ctx, amendedInputTy, funcTy.getResults());
  SmallVector<NamedAttribute> amendedAttrs;
  filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, amendedAttrs);
  if (auto argAttrs = funcOp.getAllArgAttrs()) {
    llvm::SmallVector<mlir::Attribute> amendedArgAttrs(argAttrs.begin(),
                                                       argAttrs.end());
    while (amendedArgAttrs.size() < amendedInputTy.size()) {
      amendedArgAttrs.emplace_back(DictionaryAttr::get(ctx));
    }
    amendedAttrs.push_back(rewriter.getNamedAttr(
        funcOp.getArgAttrsAttrName(), rewriter.getArrayAttr(amendedArgAttrs)));
  }
  auto amendedFuncOp = rewriter.create<triton::FuncOp>(
      funcOp.getLoc(), funcOp.getName(), amendedFuncTy, amendedAttrs);
  auto &region = funcOp.getBody();
  region.addArgument(globalPtrTy, loc);
  rewriter.inlineRegionBefore(region, amendedFuncOp.getBody(),
                              amendedFuncOp.end());
  return amendedFuncOp;
}

struct AddProtonKernelArg
    : public mlir::triton::proton::impl::AddProtonKernelArgBase<
          AddProtonKernelArg> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    bool hasProtonGlobalScratchAllocOp = false;
    moduleOp.walk([&](triton::FuncOp funcOp) {
      funcOp.walk([&](proton::gpu::GlobalScratchAllocOp op) {
        hasProtonGlobalScratchAllocOp = true;
      });
    });

    assert(hasProtonGlobalScratchAllocOp &&
           "Proton scratch alloc op not found");
    // TODO(crobeck): add support for multiple kernels
    assert(llvm::range_size(moduleOp.getOps<triton::FuncOp>()) == 1);
    triton::FuncOp funcOp = *moduleOp.getOps<triton::FuncOp>().begin();

    IRRewriter rewriter(ctx);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto amendedFuncOp = insertAmendedFuncOp(funcOp, rewriter);
    rewriter.eraseOp(funcOp);
    return;
  }
};

} // namespace

namespace mlir {

namespace triton::proton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>> createAddProtonKernelArgPass() {
  return std::make_unique<AddProtonKernelArg>();
}

} // namespace gpu

} // namespace triton::proton

} // namespace mlir
