#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonamdgpu-in-thread-transpose"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

static Type getNewType(Type type, Attribute encoding) {
  RankedTensorType tensorType = dyn_cast<RankedTensorType>(type);
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType(), encoding);
}

// This function is mostly copied over from coalesce.cpp since it uses almost
// the same functionality.
void convertLayout(Attribute encoding, Operation *op) {
  OpBuilder builder(op);
  SmallVector<Value, 4> newArgs;
  for (auto operand : op->getOperands()) {
    auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
    if (tensorType) {
      Type newType = getNewType(tensorType, encoding);
      newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), newType, operand));
    } else {
      newArgs.push_back(operand);
    }
  }

  // Convert output types
  SmallVector<Type, 4> newTypes;
  for (auto t : op->getResultTypes()) {
    newTypes.push_back(getNewType(t, encoding));
  }

  // Construct new op with the new encoding
  Operation *newOp = builder.create(op->getLoc(), op->getName().getIdentifier(),
                                    newArgs, newTypes, op->getAttrs());

  // Cast the results back to the original layout
  for (size_t i = 0; i < op->getNumResults(); i++) {
    Value newResult = newOp->getResult(i);
    if (newTypes[i] != op->getResultTypes()[i]) {
      newResult = builder.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), op->getResult(i).getType(), newResult);
    }
    op->getResult(i).replaceAllUsesWith(newResult);
  }
  op->erase();
}

SmallVector<Operation *> getLoadInsts(Operation *op) {
  SmallVector<Operation *> ret;
  auto v = op->getOperand(0);
  auto prevOp = v.getDefiningOp();
  if (isa<RegionBranchOpInterface>(prevOp)) {
    // Deal with the case that convert_layout intakes from scf.if, etc.
    LDBG("Dealing with scf blocks");
    auto idx = cast<OpResult>(v).getResultNumber();
    llvm::SmallVector<scf::YieldOp> yieldOps;
    prevOp->walk([&](Operation *op) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        yieldOps.push_back(yieldOp);
      }
    });

    for (auto yieldOp : yieldOps) {
      auto maybeLoadOp = yieldOp.getOperand(idx).getDefiningOp();
      if (isa<tt::LoadOp>(maybeLoadOp))
        ret.push_back(maybeLoadOp);
    }
  } else if (isa<tt::LoadOp>(prevOp)) {
    // regular case
    LDBG("Regular cases");
    ret.push_back(prevOp);
  } else {
    // can't find any loadOp
    LDBG("we assume load->convert_layout->dot chain but we cannot find it.");
  }
  return ret;
}

bool needCvtToThreadRaked(Value operand) {
  auto opTensorTy = cast<RankedTensorType>(operand.getType());
  auto opEnc = opTensorTy.getEncoding();
  auto opDotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(opEnc);
  // dotOperand has to have dotOp and MFMA encoding
  if (!opDotOpEnc)
    return false;
  if (!isa<ttg::AMDMfmaEncodingAttr>(opDotOpEnc.getParent())) {
    LDBG("Operand's parent encoding is not MFMA");
    return false;
  }
  auto cvtOp = operand.getDefiningOp();
  // make sure the previous op is convert_layout
  if (!cvtOp || !isa<ttg::ConvertLayoutOp>(cvtOp))
    return false;
  auto cvtOperand = cvtOp->getOperand(0);
  auto cvtOperandEnc =
      cast<RankedTensorType>(cvtOperand.getType()).getEncoding();
  auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(cvtOperandEnc);
  // make sure it is converted from blocked layout
  if (!blockedEnc)
    return false;
  // check whether it's contiguous on K dimension
  int kDimNum = opDotOpEnc.getOpIdx() == 0 ? 1 : 0;
  auto order = blockedEnc.getOrder();
  if (order[0] != kDimNum) {
    return true;
  }

  return false;
}

ttg::BlockedEncodingAttr getThreadRakedBlockedEnc(Value operand,
                                                  ModuleOp &mod) {
  // get the K dim according to dotOp operand's index
  auto tensorTy = cast<RankedTensorType>(operand.getType());
  auto shape = tensorTy.getShape();
  auto opEnc = tensorTy.getEncoding();
  auto opDotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(opEnc);
  int kDimNum = opDotOpEnc.getOpIdx() == 0 ? 1 : 0;
  // get the current blocked encoding
  auto cvtOperand = operand.getDefiningOp()->getOperand(0);
  auto cvtOperandEnc =
      cast<RankedTensorType>(cvtOperand.getType()).getEncoding();
  auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(cvtOperandEnc);
  // compute the sizePerThread for the new encoding
  auto sizePerThread = blockedEnc.getSizePerThread();
  auto elemsPerIter = product(sizePerThread);
  auto elemsTotal = blockedEnc.getTotalElemsPerThread(shape, tensorTy);
  // we need to know how many iteration each thread will load
  LDBG("elemsPerIter = " << elemsPerIter << "; elemsTotal = " << elemsTotal);
  auto numMaxIters = elemsTotal / elemsPerIter;
  auto bitwidth = tensorTy.getElementType().getIntOrFloatBitWidth();
  // Current the widest is set to ds_write_b64
  auto newKOuterDim = std::min(numMaxIters, 64 / bitwidth);
  LDBG("Choose the minimum of numIters: " << numMaxIters << " and numDtype: "
                                          << 64 / bitwidth);
  SmallVector<unsigned> newSizePerThread(sizePerThread);
  newSizePerThread[kDimNum] = newKOuterDim;

  // return the new blocked encoding
  auto order = blockedEnc.getOrder();
  int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
  return ttg::BlockedEncodingAttr::get(mod.getContext(), shape,
                                       newSizePerThread, order, numWarps,
                                       threadsPerWarp, numCTAs);
}

class TritonAMDGPUInThreadTransposePass
    : public TritonAMDGPUInThreadTransposeBase<
          TritonAMDGPUInThreadTransposePass> {

public:
  TritonAMDGPUInThreadTransposePass() = default;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    m.walk([&](Operation *op) {
      auto dotOp = dyn_cast<tt::DotOp>(op);
      if (!dotOp)
        return;

      LDBG("DotOp under inspection: " << dotOp);
      auto mod = op->getParentOfType<ModuleOp>();

      // helper function
      auto cvtNonKContigDotOperand = [&](Value op) {
        if (needCvtToThreadRaked(op)) {
          auto loadOps = getLoadInsts(op.getDefiningOp());
          // when we cannot find the associated loadOp
          if (!loadOps.size())
            return;
          auto newBlockedEnc = getThreadRakedBlockedEnc(op, mod);
          LDBG("newBlockedEnc = " << newBlockedEnc);
          for (auto loadOp : loadOps)
            convertLayout(newBlockedEnc, (Operation *)loadOp);
        }
      };

      cvtNonKContigDotOperand(dotOp.getA());
      cvtNonKContigDotOperand(dotOp.getB());
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUInThreadTransposePass() {
  return std::make_unique<TritonAMDGPUInThreadTransposePass>();
}
