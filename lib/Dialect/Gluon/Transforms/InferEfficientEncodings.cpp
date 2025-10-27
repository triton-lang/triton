#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gluon {

#define GEN_PASS_DEF_GLUONINFEREFFICIENTENCODINGSPASS
#include "triton/Dialect/Gluon/Transforms/Passes.h.inc"

#define DEBUG_TYPE "gluon-infer-efficient-encodings"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

class GluonInferEfficientEncodingsPass : public impl::GluonInferEfficientEncodingsPassBase<GluonInferEfficientEncodingsPass> {
    void runOnOperation() override {
    getOperation()->walk([](triton::LoadOp load) {
        auto ptrTy = dyn_cast<RankedTensorType>(load.getPtr().getType());
        if (!ptrTy || !ptrTy.getEncoding())
        return;
        llvm::outs() << "tt.load @" << load.getLoc() << " uses encoding ";
        ptrTy.getEncoding().print(llvm::outs());
        llvm::outs() << "\n";
    });
    }
};
//

} // namespace mlir::triton::gluon