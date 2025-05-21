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
      LDBG("operandFromThenBranch: " << operandFromThenBranch);
      if (auto opResult = dyn_cast<OpResult>(operandFromThenBranch)) {
        findRootDefiningOp(opResult.getDefiningOp(),
                           opResult.getResultNumber());
      } else if (!dyn_cast<BlockArgument>(operandFromThenBranch)) {
        rootDefiningOps.insert(operandFromThenBranch.getDefiningOp());
      }
    }
    // else branch.
    {
      auto operandFromElseBranch = ifOp.elseYield()->getOperand(resultNumber);
      LDBG("operandFromElseBranch: " << operandFromElseBranch);
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

DenseSet<Operation *> findUsersInBlockHierarchy(BlockArgument arg,
                                                Operation *consumerOp) {

  DenseSet<Operation *> usersInBlockHierarchy;

  for (Operation *user : arg.getUsers()) {
    Operation *currentOp = user;
    while (currentOp) {
      if (currentOp == consumerOp) {
        usersInBlockHierarchy.insert(user);
        break;
      }
      currentOp = currentOp->getParentOp();
    }
  }

  return usersInBlockHierarchy;
}

void PipelineErrorReporter::findRootSchedulingErrorLoopCarryDep(
    Operation *consumer, Operation *producer, Value operand) {
  LDBG("findRootSchedulingErrorLoopCarryDep: this operand is not ready at "
       "the consumer: "
       << operand << "\n");

  if (auto arg = dyn_cast<BlockArgument>(operand)) {

    // This is a loop-carried dependency. Find which value yields the arg.
    auto yieldValue = getBlockArgYieldValueFromForLoop(arg);
    if (!yieldValue) {
      LDBG("no yield value for arg " << arg << " -> BAIL");
      return;
    }

    // first find the root consumer.
    rootUserOps = std::move(findUsersInBlockHierarchy(arg, consumer));

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

std::optional<unsigned int> PipelineErrorReporter::findStage(Operation *op) {
  auto it = stages.find(op);
  if (it != stages.end()) {
    return it->second;
  }
  return std::nullopt;
}

void printImplicitUse(Operation *op, InFlightDiagnostic &mainError) {
  auto parentOpLoc = op->getParentOp()->getLoc();
  if (isa<IfOp>(op->getParentOp())) {
    // When an if branch yields a value, the original value is used implicitly
    // when the condition is false. In this case, we don't have the source
    // location of the implicit use. We can only attach a note to the if
    // operation.
    // TODO: we can report the location of the yield value in the if branch.
    mainError.attachNote(parentOpLoc)
        << "Value is implicitly used here when the condition is false in "
           "TTIR, because the variable is updated when the condition is "
           "true.";
  } else {
    mainError.attachNote(parentOpLoc) << "Value is implicitly used here. ";
  }
}

// Comparator for sorting FileLineColLoc operations
bool compareFileLineColLoc(Operation *a, Operation *b) {
  auto locA = a->getLoc();
  auto locB = b->getLoc();
  if (!isa<FileLineColLoc>(locA))
    return false;
  if (!isa<FileLineColLoc>(locB))
    return true;
  auto fileLineLocA = dyn_cast<FileLineColLoc>(a->getLoc());
  auto fileLineLocB = dyn_cast<FileLineColLoc>(b->getLoc());
  if (!fileLineLocA || !fileLineLocB)
    return false; // Should not happen if used correctly
  if (fileLineLocA.getLine() != fileLineLocB.getLine())
    return fileLineLocA.getLine() < fileLineLocB.getLine();
  return fileLineLocA.getColumn() < fileLineLocB.getColumn();
}

void PipelineErrorReporter::printSchedulingError(int64_t distance,
                                                 Operation *consumer,
                                                 Operation *producer,
                                                 Value operand) {
  LDBG("printSchedulingError: distance: " << distance << "\n");
  const char *errorMessage =
      "The software pipeliner failed due to a dependency conflict, resulting "
      "in suboptimal loop performance.";
  const char *likelyBuggyMessage = "This is likely to be a bug. Please "
                                   "report it.";
  // We only find the root defining ops for loop-carried dependencies.
  // When distance is 0, we let the set of root defining ops to be empty.
  if (distance > 0) {
    findRootSchedulingErrorLoopCarryDep(consumer, producer, operand);
  }
  if (rootDefiningOps.empty()) {
    // We failed to find the root defining ops. Whether the distance is 0 or
    // not, an empty set means we have some bugs in the pipeline expander. We
    // should let the user help report the bug.
    consumer->emitWarning() << errorMessage << " " << likelyBuggyMessage;
    return;
  }
  // find the stage of the consumer and the producer.
  auto consumerStage = findStage(consumer);
  auto producerStage = findStage(producer);
  if (!consumerStage || !producerStage) {
    // We failed to find the stage of the consumer or the producer. This is
    // likely to be a bug. We should let the user help report the bug.
    consumer->emitWarning() << errorMessage << " " << likelyBuggyMessage;
    return;
  }
  InFlightDiagnostic mainError = forOp->emitWarning() << errorMessage;
  mainError.attachNote()
      << "The loop body is divided into " << numStages
      << " stages to optimize GPU I/O and computation resources. Different "
         "parts of the loop body are computed in different iterations. "
      << "In iteration i, the update of the following variable is rescheduled "
         "to execute in iteration i + "
      << *producerStage - *consumerStage
      << ". However, it must be updated before its use in iteration i.";
  auto firstRootDefiningOp = *rootDefiningOps.begin();
  auto firstRootDefiningOpName = firstRootDefiningOp->getName().getStringRef();

  for (auto op : rootDefiningOps) {
    mainError.attachNote(op->getLoc()) << "The variable is updated here:";
  }

  auto firstRootUserOp = *rootUserOps.begin();
  auto firstRootUserOpName = firstRootUserOp->getName().getStringRef();

  // Sort rootUserOps using the custom comparator
  std::vector<Operation *> sortedRootUserOps(rootUserOps.begin(),
                                             rootUserOps.end());
  std::sort(sortedRootUserOps.begin(), sortedRootUserOps.end(),
            compareFileLineColLoc);
  // Print sorted operations
  for (auto op : sortedRootUserOps) {
    auto loc = op->getLoc();
    if (isa<UnknownLoc>(loc)) {
      printImplicitUse(op, mainError);
    } else {
      // TODO: we can't find the variable name in the source code. Once we use
      // FileLineColRange instead of FileLineColLoc, we can find the variable
      // name and provide more detailed debug information.
      mainError.attachNote(op->getLoc()) << "The variable is used here:";
    }
  }
  // TODO: we can also add more detailed debug information for more advanced
  // users.
}
