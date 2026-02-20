#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/RangeAnalysis.h"
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
// Goal of this transformation is to decrease amount of work done by vector
// instructions(decrease number of v_add). This could save cycles in a loop, and
// give more parallelism on architectures where MFMA and vector instrcutions
// executed on the same hardware module.
struct AdvanceBasePointer : public OpRewritePattern<scf::ForOp> {

  AdvanceBasePointer(MLIRContext *context, DataFlowSolver *solver,
                     PatternBenefit benefit = 1)
      : OpRewritePattern<scf::ForOp>(context, benefit), solver(solver) {}

  // Check if same value is stored in every element of tensor val
  // Return true if all elelemts are equal, false if not or it was not able for
  // proove it.
  static bool isScalarizableValue(mlir::Value val) {
    if (isa<IntegerType>(val.getType()))
      return true;
    auto defOp = val.getDefiningOp();
    if (!defOp)
      return false;
    APInt constant;
    if (matchPattern(val, m_ConstantInt(&constant))) {
      return true;
    } else if (auto splat = dyn_cast<triton::SplatOp>(defOp)) {
      return isScalarizableValue(splat.getSrc());
    }
    return false;
  }

  // Generate a scalar value that is stored in provided RankedTensor val
  static mlir::Value scalarizeValue(PatternRewriter &rewriter,
                                    mlir::Value val) {
    if (isa<IntegerType>(val.getType()))
      return val;
    auto defOp = val.getDefiningOp();
    assert(defOp);
    APInt constant;
    if (matchPattern(val, m_ConstantInt(&constant))) {
      rewriter.setInsertionPoint(defOp);
      auto intType =
          IntegerType::get(rewriter.getContext(), constant.getBitWidth());
      return arith::ConstantIntOp::create(rewriter, defOp->getLoc(), intType,
                                          constant);
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
    amdttg::BufferOpInterface op;
    Value offsetIncrement;
    Value baseIncrement;
    Value offsetInitializer;
    Operation *incrementOp;
  };

  static std::optional<ConstantIntRanges>
  getValueRange(Value value, DataFlowSolver *solver) {
    auto stepType = cast<RankedTensorType>(value.getType());
    assert(stepType.getElementType().getIntOrFloatBitWidth() < 64);

    const auto *stepRange =
        solver->lookupState<dataflow::IntegerValueRangeLattice>(value);
    if (stepRange->getValue().isUninitialized()) {
      LDBG("Rejected: value range is unintialized");
      return {};
    }
    return stepRange->getValue().getValue();
  }

  static bool isStrictlyNegative(Value advanceStep, DataFlowSolver *solver) {
    auto stepRangeValue = getValueRange(advanceStep, solver);
    if (!stepRangeValue.has_value())
      return false;

    return stepRangeValue.value().smax().isNegative();
  }

  static bool isNonNegative(Value advanceStep, DataFlowSolver *solver) {
    auto stepRangeValue = getValueRange(advanceStep, solver);
    if (!stepRangeValue.has_value())
      return false;

    return stepRangeValue.value().smax().isNonNegative();
  }

  // Estimates range of offset used in buffer op.
  // offset is 32 bit unsigned integer, equal to blockArg + advancedStep.
  // blockArg and advancedStep unit is a potentially multibyte element, offset
  // is measured in bytes.
  static std::optional<std::pair<int64_t, int64_t>>
  estimateRangeOfOffsetWithCarryValues(int elemByteWidth, Value blockArg,
                                       Value advanceStep,
                                       DataFlowSolver *solver) {
    auto stepType = cast<RankedTensorType>(advanceStep.getType());
    assert(stepType.getElementType().getIntOrFloatBitWidth() < 64);

    const auto *blockArgRange =
        solver->lookupState<dataflow::IntegerValueRangeLattice>(blockArg);
    if (blockArgRange->getValue().isUninitialized()) {
      LDBG("Rejected: blockArg range is unintialized");
      return {};
    }

    const auto *stepRange =
        solver->lookupState<dataflow::IntegerValueRangeLattice>(advanceStep);
    if (stepRange->getValue().isUninitialized()) {
      LDBG("Rejected: step range is unintialized");
      return {};
    }

    auto blockArgRangeValue = blockArgRange->getValue().getValue();
    auto stepRangeValue = stepRange->getValue().getValue();

    // Use limit to crop MSB from negative indexing
    constexpr uint64_t maxOffsetValue = 0xff'ff'ff'ff;

    // Range analysys for block argument and step should not return values
    // larger than maximum uint32_t
    assert(blockArgRangeValue.umax().getLimitedValue() <= maxOffsetValue &&
           "expect block argument to be a 32 bit value");
    assert(stepRangeValue.umax().getLimitedValue() <= maxOffsetValue &&
           "expect step value to be a 32 bit value");

    // Offset value for current iteration is a sum of block argument (offset
    // from previous iteration) and step (increment of offset).
    // These values are measured in number of elements.
    // Offset in buffer instruction is measured in bytes instead of elements.
    // In order to convert offset in elements to offset in
    // bytest compiler left shifts offset. This shift is emulated by
    // multiplication with "elemByteWidth". Potentially this could lead to
    // overflow of 32 bit value, but it should happen only if step or offset is
    // a negative number, i.e. we can discard most significant bits.
    auto elemsToBytes = [maxOffsetValue, elemByteWidth](const llvm::APInt &x) {
      return (x * elemByteWidth).getLimitedValue(maxOffsetValue);
    };
    uint32_t blockArgInBytesMax = elemsToBytes(blockArgRangeValue.umax());
    uint32_t blockArgInBytesMin = elemsToBytes(blockArgRangeValue.umin());
    uint32_t stepArgInBytesMax = elemsToBytes(stepRangeValue.umax());
    uint32_t stepArgInBytesMin = elemsToBytes(stepRangeValue.umin());

    int64_t operationUncappedResultMax =
        (int64_t)blockArgInBytesMax + stepArgInBytesMax;
    int64_t operationUncappedResultMin =
        (int64_t)blockArgInBytesMin + stepArgInBytesMin;
    return {{operationUncappedResultMin, operationUncappedResultMax}};
  }

  static bool unsignedOverflowImpossible(int elemByteWidth, Value blockArg,
                                         Value advanceStep,
                                         DataFlowSolver *solver) {
    auto maybeOffsetRange = estimateRangeOfOffsetWithCarryValues(
        elemByteWidth, blockArg, advanceStep, solver);
    if (!maybeOffsetRange.has_value())
      return false;
    auto offsetWithCarryRange = maybeOffsetRange.value();
    assert(offsetWithCarryRange.first >= 0 && "offset is unsigned");
    assert(offsetWithCarryRange.first <= offsetWithCarryRange.second);

    return offsetWithCarryRange.second <= 0xff'ff'ff'ff;
  }

  static bool unsignedOverflowInevitable(int elemByteWidth, Value blockArg,
                                         Value advanceStep,
                                         DataFlowSolver *solver) {
    auto maybeOffsetRange = estimateRangeOfOffsetWithCarryValues(
        elemByteWidth, blockArg, advanceStep, solver);
    if (!maybeOffsetRange.has_value())
      return false;
    auto offsetWithCarryRange = maybeOffsetRange.value();
    assert(offsetWithCarryRange.first <= offsetWithCarryRange.second);

    return offsetWithCarryRange.first > 0xff'ff'ff'ff;
  }

  static bool isTransformationEquivalent(int elemByteWidth, Value blockArg,
                                         Value advanceStep,
                                         DataFlowSolver *solver) {
    if (isNonNegative(advanceStep, solver) &&
        unsignedOverflowImpossible(elemByteWidth, blockArg, advanceStep,
                                   solver))
      return true;
    if (isStrictlyNegative(advanceStep, solver) &&
        unsignedOverflowInevitable(elemByteWidth, blockArg, advanceStep,
                                   solver))
      return true;
    return false;
  }

  static BlockArgument getLoopBlockArgument(Value val, scf::ForOp loop) {
    auto blockArg = dyn_cast<BlockArgument>(val);
    if (!blockArg || blockArg.getOwner()->getParentOp() != loop)
      return nullptr;
    return blockArg;
  }

  static BlockArgument getLoopArgInAdd(arith::AddIOp addOp, scf::ForOp forOp) {
    for (auto operand : addOp->getOperands())
      if (auto loopArg = getLoopBlockArgument(operand, forOp))
        return loopArg;
    return nullptr;
  }

  // Assume that offset step is not a loop argument.
  // This optimization supports only cases when step could be analyzed without
  // traversing control flow.
  static Value getStepOperandInAdd(arith::AddIOp addOp, scf::ForOp forOp) {
    for (auto operand : addOp->getOperands())
      if (!getLoopBlockArgument(operand, forOp))
        return operand;
    return nullptr;
  }

  static bool isAddFirst(amdttg::BufferOpInterface bufferOp) {
    return isa_and_nonnull<arith::AddIOp>(
        bufferOp.getOffsets().getDefiningOp());
  }

  static BlockArgument findOffsetLoopArgumentCandidate(Value offsetValue,
                                                       scf::ForOp targetFor) {
    BlockArgument offsetLoopArgument;
    if (auto offsetAdd =
            dyn_cast_or_null<arith::AddIOp>(offsetValue.getDefiningOp())) {
      offsetLoopArgument = getLoopArgInAdd(offsetAdd, targetFor);
    } else {
      offsetLoopArgument = getLoopBlockArgument(offsetValue, targetFor);
    }
    return offsetLoopArgument;
  }

  static Value findIncrementedOffsetCandidate(BlockArgument offsetLoopArgument,
                                              scf::ForOp targetFor) {
    int yieldArgNumber =
        offsetLoopArgument.getArgNumber() - targetFor.getNumInductionVars();
    auto yield = cast<scf::YieldOp>(targetFor.getBody()->getTerminator());
    return yield.getOperand(yieldArgNumber);
  }

  static arith::AddIOp findOffsetAddCandidate(Value incrementedOffset) {
    auto addOp =
        dyn_cast_or_null<arith::AddIOp>(incrementedOffset.getDefiningOp());
    return addOp;
  }

  // Checks that bufferOp uses given OffsetAddOp as an offset
  // and offsetAddOp takes one of it's operands from a given BlockArgument.
  static bool incrementChainConsistent(amdttg::BufferOpInterface bufferOp,
                                       BlockArgument offset,
                                       arith::AddIOp offsetAddOp) {
    auto targetFor = cast<scf::ForOp>(offset.getOwner()->getParentOp());
    auto maybeOffsetLoopArgument = getLoopArgInAdd(offsetAddOp, targetFor);
    if (offset != maybeOffsetLoopArgument) {
      return false;
    }
    if (isAddFirst(bufferOp) &&
        bufferOp.getOffsets().getDefiningOp() != offsetAddOp) {
      return false;
    }
    return true;
  }

  static bool isInvariantForLoop(Value val, scf::ForOp loop) {
    return val.getParentBlock()->getParentOp()->isProperAncestor(loop);
  }

  // Perform series of checks to decide if given operation could be optimized.
  // If optimization is possible, return filled BufferOpInfo.
  static std::optional<BufferOpInfo>
  analyzeBufferOp(amdttg::BufferOpInterface op, scf::ForOp targetFor,
                  DataFlowSolver *solver) {
    auto offsetValue = op.getOffsets();

    // Try to match components of two following patterns:
    //
    // Case 1(add first):
    //   for (%offsetLoopArgument):
    //     %incrementedOffset = arith.addi %offsetLoopArgument, %advanceStep
    //     bufferOp %base[%incrementedOffset]
    //     yield %incrementedOffset
    //
    // Case 2(buffer op first)
    //   for (%offsetLoopArgument):
    //     bufferOp %base[%offsetLoopArgument]
    //     %incrementedOffset = arith.addi %offsetLoopArgument, %advanceStep
    //     yield %incrementedOffset
    //
    // These patterns are similar. but buffer operation uses different values:
    // loop argument or loop argument after increment.
    //
    // Offset incrementing data flow graph is same for both of them:
    //
    //           advanceStep
    //                      \
    // offsetLoopArgument -addi-> incrementedOffset -yield->
    auto offsetLoopArgument =
        findOffsetLoopArgumentCandidate(offsetValue, targetFor);
    if (!offsetLoopArgument) {
      LDBG("Rejected: Could not find a candidate for loop_arg in "
           "loop_arg->addi->yield chain");
      return {};
    }

    auto incrementedOffset =
        findIncrementedOffsetCandidate(offsetLoopArgument, targetFor);
    if (!incrementedOffset) {
      LDBG("Rejected: Could not find a candidate for yield operand");
      return {};
    }

    auto addOp = findOffsetAddCandidate(incrementedOffset);
    if (!addOp) {
      LDBG("Rejected: Could not find addi in loop_arg->addi->yield chain");
      return {};
    }

    Value advanceStep = getStepOperandInAdd(addOp, targetFor);
    if (!advanceStep) {
      LDBG("Rejected: Could not find an offset step");
      return {};
    }

    // End of pattern component search.
    // Following core verifies that found pattern is optimizable and consistent.

    if (!incrementChainConsistent(op, offsetLoopArgument, addOp)) {
      LDBG("Rejected: Found offset increment chain is not consistent");
      return {};
    }

    if (!isInvariantForLoop(op.getPtr(), targetFor)) {
      LDBG("Rejected: Buffer op base Ptr is not an invariant in the loop");
      return {};
    }

    if (!isScalarizableValue(advanceStep)) {
      LDBG("Rejected: ptr increment step is not uniform or not supported by "
           "scalarizer yet");
      return {};
    }

    auto ptrType = cast<PointerType>(op.getPtr().getType());
    auto elemBitwidth = ptrType.getPointeeType().getIntOrFloatBitWidth();
    auto dtypeByteWidth = elemBitwidth / 8;
    assert(dtypeByteWidth > 0);
    if (!isTransformationEquivalent(dtypeByteWidth, offsetLoopArgument,
                                    advanceStep, solver)) {
      LDBG("Rejected: it is arithmetically unsafe to split offset computation");
      return {};
    }

    LDBG("Buffer op is suitable for offset pointer optimization");

    int offsetInitNo =
        offsetLoopArgument.getArgNumber() - targetFor.getNumInductionVars();
    auto offsetInitializer = targetFor.getInitArgs()[offsetInitNo];
    BufferOpInfo info{op, advanceStep, nullptr, offsetInitializer, addOp};

    return info;
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

  static scf::ForOp
  cloneLoopWithBasePtrIncrements(PatternRewriter &rewriter, scf::ForOp forOp,
                                 SmallVector<BufferOpInfo> &infoList) {
    // Create new loop with additional arguments
    llvm::SmallVector<Value> newLoopArgs(forOp.getInitArgs());
    for (auto info : infoList) {
      newLoopArgs.push_back(info.op.getPtr());
    }
    rewriter.setInsertionPoint(forOp);
    auto newForOp =
        scf::ForOp::create(rewriter, forOp.getLoc(), forOp.getLowerBound(),
                           forOp.getUpperBound(), forOp.getStep(), newLoopArgs);
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
      if (isAddFirst(info.op)) {
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
          triton::AddPtrOp::create(rewriter, loc, ptrType, basePtr, step);
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
    scf::YieldOp::create(rewriter, oldBlock->getTerminator()->getLoc(),
                         newYieldOperands);
    // Replace dynamic buffer op offsets with invariant value
    // Replace base ptr with incrementing value
    for (auto [info, basePtr, nextBasePtr] :
         llvm::zip(infoList, basePtrs, nextIterBasePtrs)) {
      auto newBufferOp = cast<amdttg::BufferOpInterface>(
          mapping.lookup<Operation *>(info.op.getOperation()));
      newBufferOp.getOffsetsMutable().assign(info.offsetInitializer);
      // two cases:
      // 1. buffer op uses pointer after increment
      // 2. buffer op uses pointers from loop arguments,
      //    incremented pointer is used on next iteration
      Value advancingBasePtr = isAddFirst(info.op) ? nextBasePtr : basePtr;
      newBufferOp.getPtrMutable().assign(advancingBasePtr);
    }
    return newForOp;
  }

  SmallVector<BufferOpInfo> collectBufferOps(scf::ForOp forOp) const {
    SmallVector<BufferOpInfo> list;
    forOp.walk([&list, this, forOp](amdttg::BufferOpInterface op) {
      auto info = analyzeBufferOp(op, forOp, solver);
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

  DataFlowSolver *solver;
};

} // anonymous namespace

struct TritonAMDGPUOptimizeBufferOpPtrPass
    : impl::TritonAMDGPUOptimizeBufferOpPtrBase<
          TritonAMDGPUOptimizeBufferOpPtrPass> {
  using Base::Base;

  void runOnOperation() override {
    FuncOp funcOp = getOperation();

    DenseMap<Value, SetVector<Operation *>> assumptions =
        AMD::TritonIntegerRangeAnalysis::collectAssumptions(funcOp);
    std::shared_ptr<DataFlowSolver> solver = createDataFlowSolver();
    DominanceInfo *dominanceInfo = &getAnalysis<DominanceInfo>();
    AMD::TritonIntegerRangeAnalysis *rangeAnalysis =
        solver->load<AMD::TritonIntegerRangeAnalysis>(assumptions,
                                                      dominanceInfo);
    AMD::initializeFuncOps(funcOp, rangeAnalysis);
    if (failed(solver->initializeAndRun(funcOp))) {
      LDBG("Integer range analysis failed to initialize, exiting");
      return;
    }

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<AdvanceBasePointer>(context, solver.get(), /*benefit=*/1);
    walkAndApplyPatterns(funcOp, std::move(patterns));
  }
};

} // namespace mlir
