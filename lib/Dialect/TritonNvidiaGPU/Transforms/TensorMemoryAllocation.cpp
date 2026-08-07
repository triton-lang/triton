#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Traits.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

namespace ttg = triton::gpu;

#define GEN_PASS_DEF_TRITONTENSORMEMORYALLOCATIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// Granularity of row allocations.
static constexpr int allocGranularity = 64;
// Number of allocGranularity-sized row groups in tensor memory.
static constexpr int kNumRows = 2;
struct TMemChunk {
  int startRow;
  int startCol;
  int numCols;
  int numRows;
};

// Use a simple bitmap to track memory usage. This is a slow but it allows us to
// handle 2D memory without extra algorithmic complexity. The number of
// allocations is expected to be small so the compile time is unlikely to be a
// problem.
struct MemoryBitMap {
  MemoryBitMap() : elements(512 * kNumRows, false) {}
  void free(const TMemChunk &chunk) {
    for (int i = 0; i < chunk.numCols; i++) {
      for (int j = 0; j < chunk.numRows; j++) {
        setUsed(chunk.startRow + j, chunk.startCol + i, false);
      }
    }
  }
  void alloc(const TMemChunk &chunk) {
    // Ensure the underlying data fits the allocation.
    while ((chunk.startCol + chunk.numCols) * kNumRows >= elements.size())
      elements.resize(2 * elements.size(), false);

    for (int i = 0; i < chunk.numCols; i++) {
      for (int j = 0; j < chunk.numRows; j++) {
        setUsed(chunk.startRow + j, chunk.startCol + i, true);
      }
    }
  }

  TMemChunk findFirstFit(TMemAllocation allocSize,
                         std::optional<int> rowIdConstraint,
                         int columnAlignment) const {
    int numRows = allocSize.numRows / allocGranularity;
    assert(kNumRows - numRows >= 0);
    assert(allocSize.numRows % allocGranularity == 0);
    int startCol = 0;
    while (1) {
      // Skip to the next aligned address.
      if (startCol % columnAlignment != 0) {
        startCol = (startCol / columnAlignment + 1) * columnAlignment;
      }
      // Iterate over possible starting rows
      for (int startRow = 0; startRow <= kNumRows - numRows; ++startRow) {
        if (rowIdConstraint && *rowIdConstraint != startRow)
          continue;
        bool fits = true;

        // Check if the block starting at (startRow, startCol) is free
        for (int i = 0; i < allocSize.numCols && fits; ++i) {
          for (int j = 0; j < numRows; ++j) {
            if (isUsed(startRow + j, startCol + i)) {
              fits = false;
              break;
            }
          }
        }

        // If a suitable block is found, return it
        if (fits) {
          TMemChunk chunk;
          chunk.startRow = startRow;
          chunk.startCol = startCol;
          chunk.numRows = numRows;
          chunk.numCols = allocSize.numCols;
          return chunk;
        }
      }
      startCol++;
    }
    return TMemChunk();
  }

private:
  bool isUsed(int row, int col) const {
    if (row + col * kNumRows >= elements.size())
      return false;
    return elements[row + col * kNumRows];
  }
  void setUsed(int row, int col, bool used) {
    assert(row + col * kNumRows < elements.size());
    elements[row + col * kNumRows] = used;
  }

  std::vector<bool> elements;
};

static Interval<int> getLiveIntervals(Value value, Liveness &liveness,
                                      DenseMap<Operation *, int> &operationId) {
  auto liveOperations = liveness.resolveLiveness(value);
  // Merge the alloc liverange with the liverange of any subview of the
  // allocation.
  SmallVector<Operation *> users(value.getUsers());
  while (!users.empty()) {
    Operation *user = users.pop_back_val();
    if (!user->hasTrait<OpTrait::MemDescViewTrait>())
      continue;
    auto usersLivness = liveness.resolveLiveness(user->getResult(0));
    liveOperations.insert(liveOperations.end(), usersLivness.begin(),
                          usersLivness.end());
    users.append(user->getResult(0).getUsers().begin(),
                 user->getResult(0).getUsers().end());
  }
  auto minId = std::numeric_limits<int>::max();
  auto maxId = std::numeric_limits<int>::min();
  std::for_each(liveOperations.begin(), liveOperations.end(),
                [&](Operation *liveOp) {
                  if (operationId[liveOp] < minId) {
                    minId = operationId[liveOp];
                  }
                  if ((operationId[liveOp] + 1) > maxId) {
                    maxId = operationId[liveOp] + 1;
                  }
                });
  return Interval(minId, maxId);
}

static void updateMap(MemoryBitMap &memoryMap, Interval<int> liveInterval,
                      std::multimap<int, TMemChunk> &intervalLiverangeEnd) {
  int start = liveInterval.start();
  // Add any dead liverange to the list of free intervals.
  for (auto it = intervalLiverangeEnd.begin();
       it != intervalLiverangeEnd.end();) {
    if (it->first > start)
      break;
    memoryMap.free(it->second);
    it = intervalLiverangeEnd.erase(it);
  }
}

static TMemChunk allocFirstFit(MemoryBitMap &memoryMap,
                               TMemAllocation allocSize,
                               std::optional<int> rowIdConstraint,
                               ArrayRef<TMemChunk> coexistingChunks,
                               int columnAlignment) {
  // `coexistingChunks` are all the allocations that might need to be live at
  // the same time as the current allocation plus what is known to be currently
  // live. Union those allocations with a copy of the current memory map and use
  // that to find the actual offsets.
  MemoryBitMap mapForAlloc = memoryMap;
  for (const TMemChunk &chunk : coexistingChunks)
    mapForAlloc.alloc(chunk);
  TMemChunk chunk =
      mapForAlloc.findFirstFit(allocSize, rowIdConstraint, columnAlignment);

  // Mark this chunk as allocated in the actual memory map.
  memoryMap.alloc(chunk);
  return chunk;
}

static SmallVector<Operation *> getAlloc(Value value) {
  SmallVector<Operation *> allocs;
  DenseSet<Value> seen;
  SmallVector<Value> worklist{value};

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second)
      continue;

    // Handle block arguments.
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      Block *block = arg.getOwner();
      Operation *parentOp = block->getParentOp();

      // Handle block with predecessors.
      if (!block->isEntryBlock()) {
        for (Block *pred : block->getPredecessors()) {
          Operation *predOp = pred->getTerminator();
          auto br = dyn_cast<BranchOpInterface>(predOp);
          if (!br) {
            llvm::report_fatal_error("unhandled branch op: " +
                                     predOp->getName().getStringRef());
          }
          SmallVector<Attribute> operands(br->getNumOperands());
          auto it = llvm::find(br->getSuccessors(), block);
          unsigned idx = std::distance(br->getSuccessors().begin(), it);
          SuccessorOperands args = br.getSuccessorOperands(idx);
          Value operand =
              args.getForwardedOperands()[arg.getArgNumber() -
                                          args.getProducedOperandCount()];
          worklist.push_back(operand);
        }
        continue;
      }

      // Handle region entry arguments.
      if (auto wsOp = dyn_cast<ttg::WarpSpecializePartitionsOp>(parentOp)) {
        worklist.push_back(wsOp.getExplicitCaptures()[arg.getArgNumber()]);
      } else if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        unsigned idx = arg.getArgNumber() - 1;
        worklist.push_back(forOp.getYieldedValues()[idx]);
        worklist.push_back(forOp.getInits()[idx]);
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
        unsigned idx = arg.getArgNumber();
        if (arg.getParentRegion() == &whileOp.getAfter()) {
          worklist.push_back(whileOp.getConditionOp().getArgs()[idx]);
        } else {
          worklist.push_back(whileOp.getYieldedValues()[idx]);
          worklist.push_back(whileOp.getInits()[idx]);
        }
      } else {
        llvm::report_fatal_error(
            "unhandled parent op when looking for TMEM alloc: " +
            parentOp->getName().getStringRef());
      }
      continue;
    }

    Operation *defOp = v.getDefiningOp();
    unsigned idx = cast<OpResult>(v).getResultNumber();
    if (isa<TMEMAllocOp>(defOp)) {
      allocs.push_back(defOp);
    } else if (defOp->hasTrait<OpTrait::MemDescViewTrait>()) {
      worklist.push_back(defOp->getOperand(0));
    } else if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
      worklist.push_back(selectOp.getTrueValue());
      worklist.push_back(selectOp.getFalseValue());
    } else if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      worklist.push_back(ifOp.thenYield().getOperand(idx));
      worklist.push_back(ifOp.elseYield().getOperand(idx));
    } else if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      worklist.push_back(forOp.getYieldedValues()[idx]);
      worklist.push_back(forOp.getInits()[idx]);
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(defOp)) {
      worklist.push_back(whileOp.getConditionOp().getArgs()[idx]);
    } else {
      llvm::report_fatal_error("unhandled op when looking for TMEM alloc: " +
                               defOp->getName().getStringRef());
    }
  }

  return allocs;
}

class RowIdConstraints {
  llvm::EquivalenceClasses<Operation *> dependentAllocs;
  llvm::SmallDenseMap<Operation *, int> rowIndex;

public:
  void joinOps(Operation *op1, Operation *op2) {
    dependentAllocs.unionSets(op1, op2);
  }

  std::optional<int> getRowIdConstraint(Operation *op) {
    auto it = dependentAllocs.findLeader(op);
    if (it == dependentAllocs.member_end())
      return std::nullopt;
    auto rowIt = rowIndex.find(*it);
    if (rowIt == rowIndex.end())
      return std::nullopt;
    return rowIt->second;
  }

  void addConstraints(Operation *op, int rowId) {
    auto it = dependentAllocs.findLeader(op);
    if (it == dependentAllocs.member_end())
      return;
    rowIndex[*it] = rowId;
  }
};

static FailureOr<int>
allocateTMem(FunctionOpInterface func,
             const DenseMap<FunctionOpInterface, int> &funcTMemCols,
             SymbolTableCollection &symbolTable) {
  SmallVector<triton::nvidia_gpu::TMEMAllocOp> allocs;
  // Direct calls to functions that (transitively) use tensor memory, together
  // with the callee's column footprint.
  SmallVector<std::pair<Operation *, int>> tmemCalls;
  DenseMap<Operation *, int> operationId;
  RowIdConstraints rowIdConstraints;
  func->walk<WalkOrder::PostOrder>([&](Operation *op) {
    operationId[op] = operationId.size();
    if (auto alloc = dyn_cast<triton::nvidia_gpu::TMEMAllocOp>(op)) {
      allocs.push_back(alloc);
    }
    if (auto callOp = dyn_cast<CallOpInterface>(op)) {
      auto callee = dyn_cast_or_null<FunctionOpInterface>(
          callOp.resolveCallableInTable(&symbolTable));
      if (callee) {
        int calleeCols = funcTMemCols.lookup(callee);
        if (calleeCols > 0)
          tmemCalls.push_back({op, calleeCols});
      }
    }
    if (auto mmaOp = dyn_cast<MMAv5OpInterface>(op)) {
      if (isa<TensorMemoryEncodingAttr>(mmaOp.getA().getType().getEncoding())) {
        TMemAllocation allocSize = getTmemAllocSizes(mmaOp.getA().getType());
        if (allocSize.numRows == 64) {
          // HW restriction, the A alloc and accumulator needs to be in the same
          // rows.
          SmallVector<Operation *> lhsAllocs = getAlloc(mmaOp.getA());
          SmallVector<Operation *> accAllocs = getAlloc(mmaOp.getAccumulator());
          for (Operation *lhsAlloc : lhsAllocs)
            for (Operation *accAlloc : accAllocs)
              rowIdConstraints.joinOps(lhsAlloc, accAlloc);
        } else {
          // TODO: we need to handle cases where the format is blockM and we
          // have multiple blocks.
          assert((cast<TensorMemoryEncodingAttr>(
                      mmaOp.getA().getType().getEncoding())
                          .getBlockM() != 64 &&
                  cast<TensorMemoryEncodingAttr>(
                      mmaOp.getAccumulator().getType().getEncoding())
                          .getBlockM() != 64) &&
                 "interleaved layout with TMEM operand is not supported yet.");
        }
      }
    }
  });
  // A callee's allocations keep their absolute column offsets in every caller
  // because the lowering passes the kernel's tensor memory base unchanged
  // through direct calls. Model each call as an immovable full-height
  // reservation of the callee's footprint at column 0. Since the reservation
  // cannot be relocated, concurrent execution of tensor-memory-using calls is
  // not supported.
  for (auto &[callOp, calleeCols] : tmemCalls) {
    if (callOp->getParentOfType<triton::gpu::WarpSpecializeOp>() ||
        callOp->getParentOfType<ttg::WarpSpecializePartitionsOp>())
      return callOp->emitError(
          "calls to functions that use tensor memory are not supported inside "
          "warp specialize regions");
  }

  int totalMemorySize = 0;
  // The callee's footprint is occupied while a call executes, even if the
  // caller has no allocations of its own.
  for (auto &[callOp, calleeCols] : tmemCalls)
    totalMemorySize = std::max(totalMemorySize, calleeCols);

  MemoryBitMap memoryMap;
  Liveness liveness(func.getOperation());
  std::multimap<int, TMemChunk> intervalLiverangeEnd;
  DenseMap<TMEMAllocOp, TMemChunk> allocChunks;
  // Implement a linear scan first fit algorithm. We expect that fragmentation
  // won't be a problem, if it is this should be revisited.
  for (auto it = allocs.begin(), e = allocs.end(); it != e; ++it) {
    TMEMAllocOp alloc = *it;
    Interval<int> liveInterval = getLiveIntervals(alloc, liveness, operationId);

    // Find all allocations in code that may execute at the same time. Only look
    // at processed allocations.
    SmallVector<TMemChunk> coexistingChunks;
    if (auto ws = alloc->getParentOfType<triton::gpu::WarpSpecializeOp>()) {
      for (auto prevIt = allocs.begin(); prevIt != it; ++prevIt) {
        TMEMAllocOp prevAlloc = *prevIt;
        auto prevWs =
            prevAlloc->getParentOfType<triton::gpu::WarpSpecializeOp>();
        if (prevWs && prevWs == ws &&
            alloc->getParentRegion() != prevAlloc->getParentRegion())
          coexistingChunks.push_back(allocChunks.at(prevAlloc));
      }
    }

    // Allocations live across a call must avoid the columns reserved for the
    // callee.
    for (auto &[callOp, calleeCols] : tmemCalls) {
      if (liveInterval.contains(operationId.lookup(callOp))) {
        TMemChunk calleeChunk;
        calleeChunk.startRow = 0;
        calleeChunk.startCol = 0;
        calleeChunk.numCols = calleeCols;
        calleeChunk.numRows = kNumRows;
        coexistingChunks.push_back(calleeChunk);
      }
    }

    auto memDescType = alloc.getType();
    TMemAllocation allocSize = getTmemAllocSizes(memDescType);
    updateMap(memoryMap, liveInterval, intervalLiverangeEnd);

    std::optional<int> rowIdConstraint =
        rowIdConstraints.getRowIdConstraint(alloc);
    // TODO: clarify the alignment requirements for different allocations. For
    // now enforce an alignment of 4 columns.
    const int columnAlignment = 4;
    TMemChunk chunkAllocated =
        allocFirstFit(memoryMap, allocSize, rowIdConstraint, coexistingChunks,
                      columnAlignment);
    allocChunks.insert({alloc, chunkAllocated});
    // currently naively constraint allocs based on the first one we find.
    rowIdConstraints.addConstraints(alloc, chunkAllocated.startRow);
    intervalLiverangeEnd.insert({liveInterval.end(), chunkAllocated});
    int colOffset = chunkAllocated.startCol;
    int rowOffset = chunkAllocated.startRow * 16;

    alloc->setAttr(
        "tensor_memory_col_offset",
        IntegerAttr::get(IntegerType::get(func->getContext(), 32), colOffset));
    alloc->setAttr(
        "tensor_memory_row_offset",
        IntegerAttr::get(IntegerType::get(func->getContext(), 32), rowOffset));
    totalMemorySize = std::max(totalMemorySize, colOffset + allocSize.numCols);
  }
  return totalMemorySize;
}

// Allocate tensor memory for `func` after all of its callees so that call
// sites know the callee's footprint. `callStack` detects cycles.
static LogicalResult allocateFunctionAndCallees(
    FunctionOpInterface func, DenseMap<FunctionOpInterface, int> &funcTMemCols,
    SymbolTableCollection &symbolTable, SetVector<Operation *> &callStack) {
  if (funcTMemCols.contains(func))
    return success();
  if (!callStack.insert(func.getOperation()))
    return func->emitError(
        "cannot allocate tensor memory for recursive function calls");
  WalkResult result = func->walk([&](CallOpInterface callOp) {
    auto callee = dyn_cast_or_null<FunctionOpInterface>(
        callOp.resolveCallableInTable(&symbolTable));
    if (callee && failed(allocateFunctionAndCallees(callee, funcTMemCols,
                                                    symbolTable, callStack)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  callStack.remove(func.getOperation());
  if (result.wasInterrupted())
    return failure();
  FailureOr<int> totalMemorySize =
      allocateTMem(func, funcTMemCols, symbolTable);
  if (failed(totalMemorySize))
    return failure();
  funcTMemCols[func] = *totalMemorySize;
  return success();
}

} // anonymous namespace

class TritonTensorMemoryAllocationPass
    : public impl::TritonTensorMemoryAllocationPassBase<
          TritonTensorMemoryAllocationPass> {
public:
  IntegerAttr getI32Attr(int32_t value) {
    return Builder(&getContext()).getI32IntegerAttr(value);
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Allocate functions in the call graph bottom-up so that call sites can
    // reserve the tensor memory used by their callee.
    SymbolTableCollection symbolTable;
    DenseMap<FunctionOpInterface, int> funcTMemCols;
    SetVector<Operation *> callStack;
    for (auto func : mod.getOps<FunctionOpInterface>()) {
      if (failed(allocateFunctionAndCallees(func, funcTMemCols, symbolTable,
                                            callStack)))
        return signalPassFailure();
    }

    // Only kernels allocate tensor memory at runtime; device functions are
    // accounted for through the reservations made by their callers.
    int totalMemorySize = 0;
    for (auto func : mod.getOps<FunctionOpInterface>()) {
      if (triton::isKernel(func))
        totalMemorySize = std::max(totalMemorySize, funcTMemCols.lookup(func));
    }

    std::vector<int> possibleAllocations = {0, 32, 64, 128, 256, 512, 576};
    // NOTE: if totalMemorySize > the maximum available for the target (512
    // for Blackwell and 576 for Rubin), we exceeded the maximum amount of
    // tensor memory, but we let the compilation finish so that we can raise an
    // exception in python for the auto-tuner.
    if (totalMemorySize <= possibleAllocations.back()) {
      for (int size : possibleAllocations) {
        if (totalMemorySize <= size) {
          totalMemorySize = size;
          break;
        }
      }
    }
    if (totalMemorySize > 0) {
      // We use a small smem allocation to get the tensor memory base address
      // from tcgen05.alloc, ensure the block has at least 4 bytes of smem
      int shared = 0;
      if (auto sharedAttr = mod->getAttr("ttg.shared")) {
        shared = cast<IntegerAttr>(sharedAttr).getInt();
      }
      if (shared < 4) {
        mod->setAttr("ttg.shared", getI32Attr(4));
      }
    }
    mod->setAttr("ttg.tensor_memory_size", getI32Attr(totalMemorySize));
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
