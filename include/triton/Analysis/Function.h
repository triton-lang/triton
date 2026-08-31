#ifndef TRITON_ANALYSIS_FUNCTION_H
#define TRITON_ANALYSIS_FUNCTION_H

#include "triton/Analysis/CallGraph.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <deque>
#include <iterator>
#include <optional>
#include <utility>

namespace mlir::triton {

/// Distinguishes an uninitialized join destination from an initialized state.
template <typename StateT> class DataflowState {
public:
  StateT &get() {
    if (!state)
      state.emplace();
    return *state;
  }

  void join(const StateT &other) {
    if (state)
      state->join(other);
    else
      state = other;
  }

private:
  std::optional<StateT> state;
};

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

  template <typename Factory>
  static void runModule(ModuleOp module, Factory makeAnalysis) {
    CallGraph<StateT> callGraph(module);
    FuncMapT summaries;
    callGraph.template walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        [](CallOpInterface, FunctionOpInterface) {},
        [&](FunctionOpInterface function) {
          if (summaries.try_emplace(function).second)
            makeAnalysis(function).run(function, summaries);
        });
  }

protected:
  /// Copy the callee summary before mapping its accesses into the caller.
  static std::optional<StateT> getCallSummary(
      CallOpInterface call, const FuncMapT &summaries,
      llvm::function_ref<void(StateT &, FunctionOpInterface)> translate = {}) {
    auto callee = dyn_cast_or_null<FunctionOpInterface>(call.resolveCallable());
    auto it = summaries.find(callee);
    if (it == summaries.end())
      return std::nullopt;
    StateT summary = it->second;
    if (translate)
      translate(summary, callee);
    return summary;
  }

  virtual void update(Operation *operation, StateT *state, FuncMapT *funcMap,
                      OpBuilder *builder) = 0;

  /// Called after `state` is joined into a CFG or region successor.
  virtual void updateSuccessor(Operation *, Block *, StateT *) {}

  /// Called on a copy of each return block's state before it is summarized.
  virtual void updateExitState(StateT *) {}

private:
  void resolve(FunctionOpInterface function, FuncMapT *funcMap,
               OpBuilder *builder) {
    // Initialize the blockList. Operations are organized into "virtual blocks",
    // which represent segments of straight-line code analyzed by each iteration
    // of the dataflow analysis. Virtual blocks abstract over both control flow
    // represented by basic blocks and block successors (i.e.
    // `BranchOpInterface`) and control flow represented by regions (i.e.
    // `RegionBranchOpInterface`).
    //
    // A virtual block consists of a parent block and a starting iterator, where
    // the virtual block starts on the operation *after* the starting iterator.
    // A null iterator is used to represent the beginning of the block. The
    // virtual block ends at any region branch operation or the basic block
    // terminator. Thus, basic blocks are broken up into multiple virtual blocks
    // at each region operation.
    //
    // Entry virtual blocks are represented by a null iterator. Populate the
    // blockList with the entry virtual blocks in the function. Then, each
    // iteration scans until a terminator or region branch operation is found.
    DenseMap<VirtualBlock, DataflowState<StateT>> inputs;
    DenseMap<VirtualBlock, StateT> outputs;
    std::deque<VirtualBlock> worklist;
    // Start the analysis from the entry block of the function.
    worklist.emplace_back(&function.getBlocks().front(), Block::iterator());

    // A fixed point algorithm
    while (!worklist.empty()) {
      VirtualBlock block = worklist.front();
      worklist.pop_front();
      // Make a copy of the inputblockInfo but not update
      StateT state = inputs[block].get();
      SmallVector<VirtualBlock> successors;
      Operation *terminator = nullptr;
      Block::iterator begin = block.second.isValid() ? std::next(block.second)
                                                     : block.first->begin();
      for (Operation &operation : llvm::make_range(begin, block.first->end())) {
        // Update inputBlockInfo based on the current operation. Note that we do
        // this before we process terminators and branch-like ops, because some
        // of them (e.g. WarpSpecializePartitionsOp) may have synchronizing
        // effects.
        update(&operation, &state, funcMap, builder);
        if (operation.hasTrait<OpTrait::IsTerminator>() ||
            isa<RegionBranchOpInterface>(operation)) {
          visitTerminator(&operation, successors);
          terminator = &operation;
          break;
        }
      }

      auto output = outputs.find(block);
      // If we have seen the block before and the inputBlockInfo is the same as
      // the outputBlockInfo, we skip the successors
      if (output != outputs.end() && state == output->second)
        continue;
      // Update the current block. The block transfer function is not monotonic,
      // so overwrite the output state entirely.
      outputs[block] = state;
      for (VirtualBlock successor : successors) {
        inputs[successor].join(state);
        updateSuccessor(terminator, successor.first, &inputs[successor].get());
        worklist.push_back(successor);
      }
    }

    // Update the final dangling buffers that haven't been synced
    DataflowState<StateT> summary;
    for (Block &exit : function.getBlocks()) {
      if (!exit.getTerminator()->hasTrait<OpTrait::ReturnLike>())
        continue;
      // A basic block can be broken into several virtual blocks. Find all
      // virtual blocks that belong to the basic block containing the return.
      SmallVector<std::pair<VirtualBlock, StateT>> exitBlocks;
      for (auto &[block, state] : outputs)
        if (block.first == &exit)
          exitBlocks.emplace_back(block, state);
      // The return is a terminator, so the virtual block that contains this
      // return starts after all other ones. Find it by comparing the start
      // iterators of the virtual blocks.
      auto last = llvm::max_element(exitBlocks, [](auto &lhs, auto &rhs) {
        Block::iterator lhsIt = lhs.first.second, rhsIt = rhs.first.second;
        return !lhsIt.isValid() ||
               (rhsIt.isValid() && lhsIt->isBeforeInBlock(&*rhsIt));
      });
      StateT exitState = last->second;
      updateExitState(&exitState);
      summary.join(exitState);
    }
    (*funcMap)[function] = std::move(summary.get());
  }

  static void visitTerminator(Operation *operation,
                              SmallVector<VirtualBlock> &successors) {
    // The parent continuation must retain worker memory effects even though
    // the region value-flow interface has no successor for worker returns.
    if (isa<gpu::WarpReturnOp>(operation)) {
      auto ws = operation->getParentOfType<gpu::WarpSpecializeOp>();
      successors.emplace_back(ws->getBlock(), ws->getIterator());
      return;
    }

    if (isa<BranchOpInterface>(operation)) {
      // Collect the block successors of the branch.
      for (Block *successor : operation->getSuccessors())
        successors.emplace_back(successor, Block::iterator());
      return;
    }

    if (auto branch = dyn_cast<RegionBranchOpInterface>(operation)) {
      // The successors of an operation with regions can be queried via an
      // interface. The operation branches to the entry blocks of its region
      // successors. It can also branch to after itself.
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

    // FIXME: `ReturnLike` adds `RegionBranchTerminatorOpInterface` for some
    // reason. Check that the parent is actually a `RegionBranchOpInterface`.
    auto branch = dyn_cast<RegionBranchTerminatorOpInterface>(operation);
    if (branch && isa<RegionBranchOpInterface>(branch->getParentOp())) {
      // Check the successors of a region branch terminator. It can branch to
      // another region of its parent operation or to after the parent op.
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

    // Otherwise, it could be a return op
    if (operation->hasTrait<OpTrait::ReturnLike>())
      return;
    llvm_unreachable("unknown terminator in function analysis");
  }
};

} // namespace mlir::triton

#endif // TRITON_ANALYSIS_FUNCTION_H
