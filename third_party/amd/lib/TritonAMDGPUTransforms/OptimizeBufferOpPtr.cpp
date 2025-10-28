#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-optimize-buffer-op-ptr"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using ::mlir::LLVM::AMD::getVectorSize;
using mlir::triton::AMD::ISAFamily;

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;
namespace amdttg = mlir::triton::amdgpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUOPTIMIZEBUFFEROPPTR
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {
// /*-----------------Base pointer increment optimization-------------------*/

// Optimization tries to transfer increments from offsets to base pointer in
// buffer loads:
//
// for ... (offsets = offsets_init):
//   val = buffer_load basePtr [ offsets ]
//   offsets_new = offsets + cst
//   yield offsets_new
// }
//
// transforms to:
//
// for ... (basePtr = basePtr_init):
//   val = buffer_load basePtr [ offsets ]
//   basePtr_new = basePtr + cst
//   yield basePtr_new
// }
//
// This lowers register consumption and reduces time spend for address
// computation.
struct AdvanceBasePointer : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  // Check if same value is stored in every element of tensor val
  // Return true if all elelemts are equal, false if not or it was not able for
  // proove it.
  static bool isScalarizableValue(mlir::Value val) {
    if (isa<IntegerType>(val.getType()))
      return true;
    auto defOp = val.getDefiningOp();
    if (!defOp)
      return false;
    if (auto constantOp = dyn_cast<mlir::arith::ConstantOp>(defOp)) {
      auto denseAttr =
          dyn_cast<DenseIntElementsAttr>(constantOp.getValueAttr());
      if (!denseAttr)
        return false;
      if (!denseAttr.isSplat())
        return false;
      return true;
    } else if (auto splat = dyn_cast<triton::SplatOp>(defOp)) {
      return isScalarizableValue(splat.getSrc());
    }
    return false;
  }

  // Generate a scalar value that is stored in provided RankedTensor val
  static mlir::Value scalarizeValue(PatternRewriter &rewriter,
                                    mlir::Value val) {
    if (isa<mlir::IntegerType>(val.getType()))
      return val;
    auto defOp = val.getDefiningOp();
    assert(defOp);
    if (auto constantOp = dyn_cast<mlir::arith::ConstantOp>(defOp)) {
      auto denseAttr =
          dyn_cast<DenseIntElementsAttr>(constantOp.getValueAttr());
      assert(denseAttr);
      auto splatVal = denseAttr.getSplatValue<llvm::APInt>();
      rewriter.setInsertionPoint(constantOp);
      return rewriter.create<mlir::arith::ConstantIntOp>(
          constantOp.getLoc(), denseAttr.getElementType(), splatVal);
    } else if (auto splat = dyn_cast<triton::SplatOp>(defOp)) {
      return scalarizeValue(rewriter, splat.getSrc());
    }
    return Value();
  }

  // load is a target buffer load
  // offsetIncrement is a tensor added to offsets on each iteration
  // baseIncrement is a scalar which will be added to base pointer after
  // optimization offsetInitialized is a value of offset on first loop iteration
  // incrementOp is an operation that advances offset tensor
  struct LoadData {
    Operation *load;
    Value offsetIncrement;
    Value baseIncrement;
    Value offsetInitializer;
    Operation *incrementOp;
  };

  static Value getOffset(Operation *load) {
    if (auto specific = dyn_cast<amdttg::BufferLoadOp>(load))
      return specific.getOffsets();
    if (auto specific = dyn_cast<amdttg::BufferLoadToLocalOp>(load))
      return specific.getOffsets();
    assert(false && "unsupported operation type");
  }

  static Value getBasePtr(Operation *load) {
    if (auto specific = dyn_cast<amdttg::BufferLoadOp>(load))
      return specific.getPtr();
    if (auto specific = dyn_cast<amdttg::BufferLoadToLocalOp>(load))
      return specific.getPtr();
    assert(false && "unsupported operation type");
  }

  static void setOffset(Operation *load, Value newOffset) {
    assert((isa<amdttg::BufferLoadOp, amdttg::BufferLoadToLocalOp>(load)));
    const int offsetIdx = isa<amdttg::BufferLoadOp>(load) ? 1 : 2;
    load->setOperand(offsetIdx, newOffset);
  }

  static void setBasePtr(Operation *load, Value newBasePtr) {
    assert((isa<amdttg::BufferLoadOp, amdttg::BufferLoadToLocalOp>(load)));
    const int ptrIdx = isa<amdttg::BufferLoadOp>(load) ? 0 : 1;
    load->setOperand(ptrIdx, newBasePtr);
  }

  // Perform series of checks to decide if given operation could be optimized.
  // If optimization is possible, return filled LoadData
  static std::optional<LoadData> analyzeLoad(Operation *loadOp,
                                             scf::ForOp targetFor) {
    LDBG("Analyzing: " << *loadOp);
    Value maybeOffsetsBlockArg = getOffset(loadOp);
    auto maybeOffsetDefOp = maybeOffsetsBlockArg.getDefiningOp();
    if (maybeOffsetDefOp && isa<arith::AddIOp>(maybeOffsetDefOp)) {
      for (auto &use : maybeOffsetDefOp->getUses()) {
        auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner());
        if (!yieldOp || yieldOp->getParentOp() != targetFor) {
          continue;
        }
        auto loopBody = targetFor.getBody();

        int blockOpNo =
            use.getOperandNumber() + targetFor.getNumInductionVars();
        maybeOffsetsBlockArg = loopBody->getArgument(blockOpNo);
        break;
      }
    }
    if (!isa<BlockArgument>(maybeOffsetsBlockArg)) {
      LDBG("Rejected: expect load offset to be a loop argument");
      return {};
    }
    auto blockArg = dyn_cast<BlockArgument>(maybeOffsetsBlockArg);
    auto loopBlock = blockArg.getOwner();
    auto forOp = dyn_cast<scf::ForOp>(loopBlock->getParentOp());
    if (!forOp || forOp != targetFor) {
      LDBG("Rejected: expect load offset to be a target loop argument");
      return {};
    }
    auto basePtr = getBasePtr(loadOp);
    auto defOpBlock = basePtr.getParentBlock();
    if (!defOpBlock->getParentOp()->isProperAncestor(targetFor)) {
      LDBG("Rejected: expect load base Ptr to be invariant to the loop");
      return {};
    }
    auto yield = dyn_cast<scf::YieldOp>(loopBlock->getTerminator());
    int offsetOperandNo = blockArg.getArgNumber() - forOp.getNumInductionVars();
    auto offsetYieldOperand = yield.getOperand(offsetOperandNo);
    auto incrementOp = offsetYieldOperand.getDefiningOp();
    if (!isa<arith::AddIOp>(incrementOp)) {
      LDBG("Rejected: expect arith::addi used for pointer advanceent");
      return {};
    }
    Value advanceStep;
    if (incrementOp->getOperand(0) == blockArg &&
        incrementOp->getOperand(1) != blockArg) {
      advanceStep = incrementOp->getOperand(1);
    }
    if (incrementOp->getOperand(1) == blockArg &&
        incrementOp->getOperand(0) != blockArg) {
      advanceStep = incrementOp->getOperand(0);
    }
    if (!advanceStep) {
      LDBG("Rejected: expect arith::addi to advance same block argument as "
           "used in load");
      return {};
    }
    if (!isScalarizableValue(advanceStep)) {
      LDBG("Rejected: ptr increment step is not supported");
      return {};
    }
    Value offsetInitializer = forOp.getInitArgs()[offsetOperandNo];
    LoadData data = {loadOp, advanceStep, Value(), offsetInitializer,
                     incrementOp};
    LDBG("Load is suitable for offset pointer optimization");
    return data;
  }

  // Create scalar values which will increment load base ptr
  // Fills appropriate fields in given LoadData structures
  static void createScalarIncrements(PatternRewriter &rewriter,
                                     SmallVector<LoadData> &loads) {
    for (auto &loadData : loads) {
      auto scalarStep = scalarizeValue(rewriter, loadData.offsetIncrement);
      loadData.baseIncrement = scalarStep;
    }
  }

  static bool isAddFirst(LoadData &ld) {
    return getOffset(ld.load).getDefiningOp() == ld.incrementOp;
  }

  static scf::ForOp
  cloneLoopWithBasePtrIncrements(PatternRewriter &rewriter, scf::ForOp forOp,
                                 SmallVector<LoadData> &loads) {
    // Create new loop with additional arguments
    llvm::SmallVector<Value> newLoopArgs(forOp.getInitArgs());
    for (auto loadData : loads) {
      newLoopArgs.push_back(getBasePtr(loadData.load));
    }
    rewriter.setInsertionPoint(forOp);
    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newLoopArgs);
    // Clone old loop body without terminator
    IRMapping mapping;
    auto oldBlock = forOp.getBody();
    auto newBlock = newForOp.getBody();
    for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
      mapping.map(oldBlock->getArgument(i), newBlock->getArgument(i));
    }
    rewriter.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->end());
    for (auto &op : oldBlock->without_terminator()) {
      rewriter.clone(op, mapping);
    }
    // Create base pointer increment operations
    auto basePtrs = newBlock->getArguments().take_back(loads.size());
    llvm::SmallVector<Value> nextIterBasePtrs;
    for (auto [loadData, basePtr] : llvm::zip(loads, basePtrs)) {
      if (isAddFirst(loadData)) {
        rewriter.setInsertionPoint(newBlock, newBlock->begin());
      } else {
        rewriter.setInsertionPoint(newBlock, newBlock->end());
      }
      Value step = loadData.baseIncrement;
      if (mapping.contains(loadData.baseIncrement)) {
        step = mapping.lookup(loadData.baseIncrement);
      }
      auto loc = loadData.incrementOp->getLoc();
      auto ptrType = basePtr.getType();
      auto nextIterBasePtr =
          rewriter.create<triton::AddPtrOp>(loc, ptrType, basePtr, step);
      nextIterBasePtrs.push_back(nextIterBasePtr);
    }
    // Create yield operation
    llvm::SmallVector<Value> newYieldOperands;
    auto oldTerminator = forOp.getBody()->getTerminator();
    for (auto operand : oldTerminator->getOperands()) {
      newYieldOperands.push_back(mapping.lookup(operand));
    }
    newYieldOperands.append(nextIterBasePtrs);

    rewriter.setInsertionPoint(newBlock, newBlock->end());
    rewriter.create<scf::YieldOp>(oldBlock->getTerminator()->getLoc(),
                                  newYieldOperands);
    // Replace dynamic load offsets with invariant value
    // Replace base ptr with incrementing value
    for (auto [loadData, basePtr, nextBasePtr] :
         llvm::zip(loads, basePtrs, nextIterBasePtrs)) {
      auto newLoad = mapping.lookup<Operation *>(loadData.load);
      setOffset(newLoad, loadData.offsetInitializer);
      // two cases:
      // 1. first advance pointer, then load
      // 2. load uses pointers from loop arguments, advanced pointer used on
      // next iteration
      Value advancingBasePtr = isAddFirst(loadData) ? nextBasePtr : basePtr;
      setBasePtr(newLoad, advancingBasePtr);
    }
    return newForOp;
  }

  template <typename OpType>
  static void collectLoads(SmallVector<LoadData> &loads, scf::ForOp forOp) {
    forOp.walk([&loads, forOp](OpType loadOp) {
      auto loadData = analyzeLoad(loadOp, forOp);
      if (loadData.has_value()) {
        loads.push_back(loadData.value());
      }
    });
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    LDBG("Analyzing ForOp for for offset pointer optimization: " << forOp);
    // Gather buffer loads which could be optimized
    SmallVector<LoadData> loads;
    collectLoads<triton::amdgpu::BufferLoadOp>(loads, forOp);
    collectLoads<triton::amdgpu::BufferLoadToLocalOp>(loads, forOp);

    if (loads.empty())
      return rewriter.notifyMatchFailure(forOp, "no suitable buffer loads");

    // Perform IR transformation
    createScalarIncrements(rewriter, loads);
    auto newForOp = cloneLoopWithBasePtrIncrements(rewriter, forOp, loads);
    rewriter.replaceAllUsesWith(forOp.getResults(),
                                newForOp.getResults().drop_back(loads.size()));
    rewriter.eraseOp(forOp);
    return success();
  }
};

} // anonymous namespace

struct TritonAMDGPUOptimizeBufferOpPtrPass
    : impl::TritonAMDGPUOptimizeBufferOpPtrBase<
          TritonAMDGPUOptimizeBufferOpPtrPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    FuncOp func = getOperation();

    patterns.add<AdvanceBasePointer>(context, /*benefit=*/1);

    if (applyPatternsGreedily(func, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
