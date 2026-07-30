#ifndef TRITON_ANALYSIS_FUNCTION_H
#define TRITON_ANALYSIS_FUNCTION_H

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <deque>
#include <iterator>
#include <utility>

namespace mlir::triton {

/// A forward dataflow analysis over the CFG and nested region control flow of
/// one function. StateT must be default constructible and provide join() and
/// equality. FuncMapT retains the joined exit state for each analyzed function
/// and makes postorder callee summaries available to the transfer function.
template <typename StateT> class PostOrderFunctionAnalysis {
  using VirtualBlock = std::pair<Block *, Block::iterator>;

public:
  using FuncMapT = DenseMap<FunctionOpInterface, StateT>;

  virtual ~PostOrderFunctionAnalysis() = default;

  void run(FunctionOpInterface function, FuncMapT &funcMap) {
    OpBuilder builder(function.getContext());
    resolve(function, &funcMap, &builder);
  }

protected:
  virtual void update(Operation *operation, StateT *state, FuncMapT *funcMap,
                      OpBuilder *builder) = 0;

private:
  void resolve(FunctionOpInterface function, FuncMapT *funcMap,
               OpBuilder *builder) {
    // A virtual block is a straight-line segment ending at a region branch or
    // basic-block terminator. Splitting basic blocks this way lets one
    // worklist handle both block successors (BranchOpInterface) and nested
    // region control flow (RegionBranchOpInterface). The iterator points to
    // the operation before the segment; an invalid iterator denotes the
    // beginning of the block.
    DenseMap<VirtualBlock, StateT> inputs;
    DenseMap<VirtualBlock, StateT> outputs;
    std::deque<VirtualBlock> worklist;
    worklist.emplace_back(&function.getBlocks().front(), Block::iterator());

    while (!worklist.empty()) {
      VirtualBlock block = worklist.front();
      worklist.pop_front();
      StateT state = inputs[block];
      SmallVector<VirtualBlock> successors;
      Block::iterator begin = block.second.isValid() ? std::next(block.second)
                                                     : block.first->begin();
      for (Operation &operation : llvm::make_range(begin, block.first->end())) {
        // Branch-like operations can have their own effects, so apply the
        // transfer function before following their successors.
        update(&operation, &state, funcMap, builder);
        if (operation.hasTrait<OpTrait::IsTerminator>() ||
            isa<RegionBranchOpInterface>(operation)) {
          visitTerminator(&operation, successors);
          break;
        }
      }

      auto output = outputs.find(block);
      if (output != outputs.end() && state == output->second)
        continue;
      // A transfer may clear state at a synchronization point, so replace the
      // previous output instead of joining with it.
      outputs[block] = state;
      for (VirtualBlock successor : successors) {
        inputs[successor].join(state);
        worklist.push_back(successor);
      }
    }

    StateT &summary = (*funcMap)[function];
    for (Block &exit : function.getBlocks()) {
      if (!exit.getTerminator()->hasTrait<OpTrait::ReturnLike>())
        continue;
      // A basic block may contain several virtual blocks. Its return is in the
      // final segment, whose starting iterator follows all the others.
      SmallVector<std::pair<VirtualBlock, StateT>> exitBlocks;
      for (auto &[block, state] : outputs)
        if (block.first == &exit)
          exitBlocks.emplace_back(block, state);
      auto last = llvm::max_element(exitBlocks, [](auto &lhs, auto &rhs) {
        Block::iterator lhsIt = lhs.first.second, rhsIt = rhs.first.second;
        return !lhsIt.isValid() ||
               (rhsIt.isValid() && lhsIt->isBeforeInBlock(&*rhsIt));
      });
      summary.join(last->second);
    }
  }

  static void visitTerminator(Operation *operation,
                              SmallVector<VirtualBlock> &successors) {
    if (isa<BranchOpInterface>(operation)) {
      for (Block *successor : operation->getSuccessors())
        successors.emplace_back(successor, Block::iterator());
      return;
    }

    if (auto branch = dyn_cast<RegionBranchOpInterface>(operation)) {
      // A region branch either enters one of its regions or continues after
      // the operation in its parent block.
      SmallVector<RegionSuccessor> regions;
      branch.getSuccessorRegions(RegionBranchPoint::parent(), regions);
      for (RegionSuccessor &region : regions) {
        if (region.isOperation())
          successors.emplace_back(branch->getBlock(), branch->getIterator());
        else
          successors.emplace_back(&region.getSuccessor()->front(),
                                  Block::iterator());
      }
      return;
    }

    // ReturnLike unexpectedly implements RegionBranchTerminatorOpInterface,
    // so only query region successors when the parent is a region branch.
    auto branch = dyn_cast<RegionBranchTerminatorOpInterface>(operation);
    if (branch && isa<RegionBranchOpInterface>(branch->getParentOp())) {
      // A region terminator either enters another region of its parent or
      // continues after the parent operation.
      SmallVector<Attribute> operands(branch->getNumOperands());
      SmallVector<RegionSuccessor> regions;
      branch.getSuccessorRegions(operands, regions);
      for (RegionSuccessor &region : regions) {
        if (region.isOperation()) {
          Operation *parent = branch->getParentOp();
          successors.emplace_back(parent->getBlock(), parent->getIterator());
        } else {
          successors.emplace_back(&region.getSuccessor()->front(),
                                  Block::iterator());
        }
      }
      return;
    }

    if (operation->hasTrait<OpTrait::ReturnLike>())
      return;
    llvm_unreachable("unknown terminator in function analysis");
  }
};

} // namespace mlir::triton

#endif // TRITON_ANALYSIS_FUNCTION_H
