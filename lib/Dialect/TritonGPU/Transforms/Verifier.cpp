#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include <memory>

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUVerifier : public TritonGPUVerifierBase<TritonGPUVerifier> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // The idea is similar to mlir/lib/IR/Verifier.cpp
    verifyImpl(m.getOperation());
  }

private:
  LogicalResult verifySingleOp(Operation *op) {
    if (auto dotOp = llvm::dyn_cast<triton::DotOp>(op)) {
      Type aType = dotOp.a().getType();
      Type bType = dotOp.b().getType();
      Type cType = dotOp.c().getType();
      Type dType = dotOp.d().getType();
      for (auto it : llvm::zip(llvm::SmallVector<Type>{aType, bType},
                               llvm::SmallVector<char>{'a', 'b'})) {
        Type type = std::get<0>(it);
        char name = std::get<1>(it);
        if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
          Attribute encoding = tensorType.getEncoding();
          if (!encoding)
            return dotOp.emitError() << name << " should have encoding";
          if (!encoding.isa<triton::gpu::SharedEncodingAttr>())
            return dotOp.emitError() << name << " should be of shared layout";
        } else
          return dotOp.emitError()
                 << name << "'s type should be of RankedTensorType";
      }

      Attribute cLayout;
      for (auto it : llvm::zip(llvm::SmallVector<Type>{cType, dType},
                               llvm::SmallVector<char>{'c', 'd'})) {
        Type type = std::get<0>(it);
        char name = std::get<1>(it);
        if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
          Attribute encoding = tensorType.getEncoding();
          if (!encoding)
            return dotOp.emitError() << name << " should have encoding";
          if (!encoding.isa<triton::gpu::MmaEncodingAttr>() &&
              !encoding.isa<triton::gpu::BlockedEncodingAttr>())
            return dotOp.emitError()
                   << name << " should be of distributed layout";
          if (name == 'c')
            cLayout = encoding;
          else if (encoding != cLayout)
            return dotOp.emitError() << "d & c should have the same layout";
        } else
          return dotOp.emitError()
                 << name << "'s type should be of RankedTensorType";
      }

      // signalPassFailure();
    }
    if (auto loadOp = llvm::dyn_cast<triton::LoadOp>(op)) {
      // TODO: fill this
    }
    if (auto storeOp = llvm::dyn_cast<triton::StoreOp>(op)) {
      // TODO: fill this
    }
    if (auto addptrOp = llvm::dyn_cast<triton::AddPtrOp>(op)) {
      // TODO: fill this
    }
    // Triton builtin Ops
    if (llvm::isa<triton::GetProgramIdOp, triton::GetNumProgramsOp,
                  triton::MakeRangeOp>(op)) {
      // TODO: fill this
    }
    if (auto atomicRmw = llvm::dyn_cast<triton::AtomicRMWOp>(op)) {
      // TODO: fill this
    }
    if (auto atomicCas = llvm::dyn_cast<triton::AtomicCASOp>(op)) {
      // TODO: fill this
    }

    // TODO: Arithmetic, SCF, TritonGPU ops
    return success();
  }

  void verifyImpl(Operation *op) {
    if (verifySingleOp(op).failed())
      signalPassFailure();

    // verify that all child regions are ok
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &childOp : block)
          verifyImpl(&childOp);
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUVerifier() {
  return std::make_unique<TritonGPUVerifier>();
}
