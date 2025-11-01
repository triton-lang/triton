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
// buffer operations:
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

  // Description of struct fields:
  // - op is a target BufferOp, for example BufferLoadOp or BufferLoadToLocalOp
  // - offsetIncrement is a tensor added to offsets on each iteration
  // - baseIncrement is a scalar which will be added to base pointer
  // - offsetInitializer is a value of offset on first loop iteration
  // - incrementOp is an operation that advances offset tensor
  struct BufferOpInfo {
    amdttg::BufferOpAddressinInterface op;
    Value offsetIncrement;
    Value baseIncrement;
    Value offsetInitializer;
    Operation *incrementOp;
  };

  // Perform series of checks to decide if given operation could be optimized.
  // If optimization is possible, return filled BufferOpInfo
  static std::optional<BufferOpInfo>
  analyzeBufferOp(amdttg::BufferOpAddressinInterface op, scf::ForOp targetFor) {
    LDBG("Analyzing: " << *op);
    Value maybeOffsetsBlockArg = op.getOffsets();
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
      LDBG("Rejected: expect buffer op offset to be a loop argument");
      return {};
    }
    auto blockArg = dyn_cast<BlockArgument>(maybeOffsetsBlockArg);
    auto loopBlock = blockArg.getOwner();
    auto forOp = dyn_cast<scf::ForOp>(loopBlock->getParentOp());
    if (!forOp || forOp != targetFor) {
      LDBG("Rejected: expect buffer op offset to be a target loop argument");
      return {};
    }
    auto basePtr = op.getPtr();
    auto defOpBlock = basePtr.getParentBlock();
    if (!defOpBlock->getParentOp()->isProperAncestor(targetFor)) {
      LDBG("Rejected: expect buffer op base Ptr to be invariant to the loop");
      return {};
    }
    auto yield = dyn_cast<scf::YieldOp>(loopBlock->getTerminator());
    int offsetOperandNo = blockArg.getArgNumber() - forOp.getNumInductionVars();
    auto offsetYieldOperand = yield.getOperand(offsetOperandNo);
    auto incrementOp = offsetYieldOperand.getDefiningOp();
    if (!incrementOp || !isa<arith::AddIOp>(incrementOp)) {
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
           "used in buffer op");
      return {};
    }
    if (!isScalarizableValue(advanceStep)) {
      LDBG("Rejected: ptr increment step is not supported");
      return {};
    }
    Value offsetInitializer = forOp.getInitArgs()[offsetOperandNo];
    BufferOpInfo data = {op, advanceStep, Value(), offsetInitializer,
                         incrementOp};
    LDBG("Buffer op is suitable for offset pointer optimization");
    return data;
  }

  // Create scalar values which will increment buffer op base ptr
  // Fills appropriate fields in given BufferOpInfo structures
  static void createScalarIncrements(PatternRewriter &rewriter,
                                     SmallVector<BufferOpInfo> &infoList) {
    for (auto &BufferOpInfo : infoList) {
      auto scalarStep = scalarizeValue(rewriter, BufferOpInfo.offsetIncrement);
      BufferOpInfo.baseIncrement = scalarStep;
    }
  }

  static bool isAddFirst(BufferOpInfo &info) {
    return info.op.getOffsets().getDefiningOp() == info.incrementOp;
  }

  static scf::ForOp
  cloneLoopWithBasePtrIncrements(PatternRewriter &rewriter, scf::ForOp forOp,
                                 SmallVector<BufferOpInfo> &infoList) {
    // Create new loop with additional arguments
    llvm::SmallVector<Value> newLoopArgs(forOp.getInitArgs());
    for (auto info : infoList) {
      newLoopArgs.push_back(info.op.getPtr());
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
    auto basePtrs = newBlock->getArguments().take_back(infoList.size());
    llvm::SmallVector<Value> nextIterBasePtrs;
    for (auto [info, basePtr] : llvm::zip(infoList, basePtrs)) {
      if (isAddFirst(info)) {
        rewriter.setInsertionPoint(newBlock, newBlock->begin());
      } else {
        rewriter.setInsertionPoint(newBlock, newBlock->end());
      }
      Value step = info.baseIncrement;
      if (mapping.contains(info.baseIncrement)) {
        step = mapping.lookup(info.baseIncrement);
      }
      auto loc = info.incrementOp->getLoc();
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
    // Replace dynamic buffer op offsets with invariant value
    // Replace base ptr with incrementing value
    for (auto [info, basePtr, nextBasePtr] :
         llvm::zip(infoList, basePtrs, nextIterBasePtrs)) {
      auto newBufferOp = cast<amdttg::BufferOpAddressinInterface>(
          mapping.lookup<Operation *>(info.op.getOperation()));
      newBufferOp.getOffsetsMutable().assign(info.offsetInitializer);
      // two cases:
      // 1. buffer op uses pointer after increment
      // 2. buffer op uses pointers from loop arguments,
      //    incremented pointer is used on next iteration
      Value advancingBasePtr = isAddFirst(info) ? nextBasePtr : basePtr;
      newBufferOp.getPtrMutable().assign(advancingBasePtr);
    }
    return newForOp;
  }

  static SmallVector<BufferOpInfo> collectBufferOps(scf::ForOp forOp) {
    SmallVector<BufferOpInfo> list;
    forOp.walk([&list, forOp](amdttg::BufferOpAddressinInterface op) {
      auto info = analyzeBufferOp(op, forOp);
      if (info.has_value()) {
        list.push_back(info.value());
      }
    });
    return list;
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    LDBG("Analyzing ForOp for offset pointer optimization: " << forOp);
    // Gather buffer buffer operations which could be optimized
    SmallVector<BufferOpInfo> infoList = collectBufferOps(forOp);

    if (infoList.empty())
      return rewriter.notifyMatchFailure(forOp,
                                         "no suitable buffer operations");

    // Perform IR transformation
    createScalarIncrements(rewriter, infoList);
    auto newForOp = cloneLoopWithBasePtrIncrements(rewriter, forOp, infoList);
    rewriter.replaceAllUsesWith(
        forOp.getResults(), newForOp.getResults().drop_back(infoList.size()));
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
