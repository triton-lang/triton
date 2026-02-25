#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>

namespace mlir {
namespace triton {
namespace instrument {

#define GEN_PASS_DEF_TRITONINSTRUMENTGLOBALSANITIZER
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h.inc"

namespace {

static constexpr const char kGSanGlobalStateArgAttr[] = "tti.gsan_global_state";

bool isPointerLikeValue(Type type) {
  if (isa<triton::PointerType>(type))
    return true;
  auto tensorTy = dyn_cast<RankedTensorType>(type);
  return tensorTy && isa<triton::PointerType>(tensorTy.getElementType());
}

int32_t getBytesPerElem(Type ptrType) {
  unsigned bitWidth = triton::getPointeeBitWidth(ptrType);
  return static_cast<int32_t>(std::max<unsigned>(1, (bitWidth + 7) / 8));
}

class GlobalSanitizerPass
    : public impl::TritonInstrumentGlobalSanitizerBase<GlobalSanitizerPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module);
    Type gsanStatePtrTy = triton::PointerType::get(builder.getI8Type(), 1);
    DenseSet<StringRef> calledFuncs;
    module.walk(
        [&](triton::CallOp callOp) { calledFuncs.insert(callOp.getCallee()); });

    SmallVector<triton::FuncOp> funcs;
    module.walk([&](triton::FuncOp func) { funcs.push_back(func); });
    for (triton::FuncOp func : funcs) {
      auto funcTy = func.getFunctionType();
      SmallVector<Type> inputTys(funcTy.getInputs().begin(),
                                 funcTy.getInputs().end());
      inputTys.push_back(gsanStatePtrTy);
      func.setType(FunctionType::get(module.getContext(), inputTys,
                                     funcTy.getResults()));

      func.getBody().addArgument(gsanStatePtrTy, func.getLoc());
      SmallVector<Attribute> newArgAttrs;
      if (auto argAttrs = func.getAllArgAttrs())
        newArgAttrs.append(argAttrs.begin(), argAttrs.end());
      while (newArgAttrs.size() < func.getNumArguments()) {
        newArgAttrs.push_back(DictionaryAttr::get(module.getContext()));
      }
      if (!newArgAttrs.empty())
        func.setAllArgAttrs(newArgAttrs);
      func.setArgAttr(func.getNumArguments() - 1, kGSanGlobalStateArgAttr,
                      builder.getUnitAttr());

      bool isEntry = !calledFuncs.contains(func.getSymName());
      if (isEntry) {
        OpBuilder b(&func.front(), func.front().begin());
        ExperimentalGSanInitOp::create(b, func.getLoc());
      }
    }

    SmallVector<triton::CallOp> callOps;
    module.walk([&](triton::CallOp op) { callOps.push_back(op); });
    for (triton::CallOp callOp : callOps) {
      auto caller = callOp->getParentOfType<triton::FuncOp>();
      assert(caller && caller.getNumArguments() > 0 &&
             "expected triton.call to be nested under a Triton function");

      SmallVector<Value> operands(callOp.getOperands().begin(),
                                  callOp.getOperands().end());
      operands.push_back(caller.getArgument(caller.getNumArguments() - 1));

      OpBuilder b(callOp);
      auto newCallOp =
          triton::CallOp::create(b, callOp.getLoc(), callOp.getCallee(),
                                 callOp.getResultTypes(), operands);
      newCallOp->setAttrs(callOp->getAttrs());
      callOp->replaceAllUsesWith(newCallOp->getResults());
      callOp.erase();
    }

    module.walk([&](triton::LoadOp op) {
      Type ptrType = op.getPtr().getType();
      OpBuilder b(op);
      ExperimentalGSanTensorAccessOp::create(
          b, op.getLoc(), op.getPtr(), op.getMask(), getBytesPerElem(ptrType),
          /*isStore=*/false);
    });
    module.walk([&](triton::StoreOp op) {
      Type ptrType = op.getPtr().getType();
      OpBuilder b(op);
      ExperimentalGSanTensorAccessOp::create(
          b, op.getLoc(), op.getPtr(), op.getMask(), getBytesPerElem(ptrType),
          /*isStore=*/true);
    });
  }
};

} // namespace

} // namespace instrument
} // namespace triton
} // namespace mlir
