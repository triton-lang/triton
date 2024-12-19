// Reporting error messages for scheduling errors.

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

#include "triton/Dialect/TritonGPU/Transforms/PipelineErrorReporter.h"

#define DEBUG_TYPE "triton-pipeline-error-reporter"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::scf;

void PipelineErrorReporter::findRootDefiningOp(Operation *op,
                                               unsigned int resultNumber) {
  LDBG("findRootDefiningOp: " << *op << "\n from its " << resultNumber
                              << "th result\n");

  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    // then branch.
    {
      auto operandFromThenBranch = ifOp.thenYield()->getOperand(resultNumber);
      if (auto opResult = dyn_cast<OpResult>(operandFromThenBranch)) {
        findRootDefiningOp(operandFromThenBranch.getDefiningOp(),
                           opResult.getResultNumber());
      } else if (!dyn_cast<BlockArgument>(operandFromThenBranch)) {
        rootDefiningOps.insert(operandFromThenBranch.getDefiningOp());
      }
    }
    // else branch.
    {
      auto operandFromElseBranch = ifOp.thenYield()->getOperand(resultNumber);
      if (auto opResult = dyn_cast<OpResult>(operandFromElseBranch)) {
        findRootDefiningOp(opResult.getDefiningOp(),
                           opResult.getResultNumber());
      } else if (!dyn_cast<BlockArgument>(operandFromElseBranch)) {
        rootDefiningOps.insert(operandFromElseBranch.getDefiningOp());
      }
    }
  } else {
    rootDefiningOps.insert(op);
  }
}

std::optional<Value>
PipelineErrorReporter::getBlockArgYieldValueFromForLoop(BlockArgument arg) {
  if (arg.getOwner() != forOp.getBody())
    return std::nullopt;
  // Ignore induction variable.
  if (arg.getArgNumber() == 0)
    return std::nullopt;
  return forOp.getBody()->getTerminator()->getOperand(arg.getArgNumber() - 1);
}

void PipelineErrorReporter::findRootSchedulingErrorLoopCarryDep(
    Operation *consumer, Operation *producer, Value operand) {
  DenseSet<Operation *> rootDefiningOps;
  LDBG("findRootSchedulingErrorLoopCarryDep: this operand is not ready at "
       "the consumer: "
       << operand << "\n");
  if (auto arg = dyn_cast<BlockArgument>(operand)) {
    LDBG("operand is a block arg. Arg number: " << arg.getArgNumber() << "\n");
    // This is a loop-carried dependency. Find which value yields the arg.
    auto yieldValue = getBlockArgYieldValueFromForLoop(arg);
    if (!yieldValue) {
      LDBG("no yield value for arg " << arg << " -> BAIL");
      return;
    }

    assert(producer == yieldValue->getDefiningOp() &&
           "producer should be the def of the yield value of operand");
    // We repeat the process of computing the producer, because we need to
    // know the result number of the producer, which is only available in the
    // yield value.
    LDBG("yield value (loop-carry): " << *yieldValue << "\n");
    if (auto opResult = dyn_cast<OpResult>(*yieldValue)) {
      findRootDefiningOp(producer, opResult.getResultNumber());
    } else
      rootDefiningOps.insert(producer);
  }
}

void PipelineErrorReporter::printSchedulingError(int64_t distance,
                                                 Operation *consumer,
                                                 Operation *producer,
                                                 Value operand) {

  std::string errorMessage = "operation scheduled before its operands.";
  std::string likelyBuggyMessage = "This is likely to be a bug. Please "
                                   "report it.";
  // We only find the root defining ops for loop-carried dependencies.
  // When distance is 0, we let the set of root defining ops to be empty.
  if (distance > 0) {
    // TODO: I only find the root defining ops of the producer. We should also
    // find the root user ops of the consumer.
    findRootSchedulingErrorLoopCarryDep(consumer, producer, operand);
  }
  if (rootDefiningOps.empty()) {
    // We failed to find the root defining ops. Whether the disntance is 0 or
    // not, an empty set means we have some bugs in the pipeline expander. We
    // should let the user help report the bug.
    consumer->emitError() << errorMessage << " " << likelyBuggyMessage;
  } else {
    consumer->emitError() << errorMessage;
    for (auto op : rootDefiningOps) {
      op->emitError() << "This line likely causes scheduling conflict. "
                         "Consider moving it "
                         "to an earlier position within the loop body.";
    }
  }
}
