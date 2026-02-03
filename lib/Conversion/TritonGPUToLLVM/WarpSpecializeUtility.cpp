#include "triton/Conversion/TritonGPUToLLVM/WarpSpecializeUtility.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OperationSupport.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// lowerWarpSpecializeBarriers
//===----------------------------------------------------------------------===//

LogicalResult WarpSpecializeBarrierHelper::createBarrier(
    TritonLLVMIRRewriter &b, unsigned numWarps,
    std::optional<unsigned> partitionIdx) {
  FailureOr<Value> handle = getBarrierHandle(b, partitionIdx);
  if (failed(handle))
    return failure();
  createBarrier(b, numWarps, *handle);
  return success();
}

static std::string mangleFunctionName(StringRef name) {
  return (name + "_ws").str();
}

static LogicalResult lowerBarrier(Operation *op, unsigned numWarps,
                                  std::optional<unsigned> partitionIdx,
                                  WarpSpecializeBarrierHelper &barrierHelper) {
  TritonLLVMIRRewriter b(op->getLoc(), op);
  if (failed(barrierHelper.createBarrier(b, numWarps, partitionIdx)))
    return failure();
  op->erase();
  return success();
}

static LogicalResult lowerCallOp(LLVM::CallOp callOp, unsigned numWarps,
                                 std::optional<unsigned> partitionIdx,
                                 WarpSpecializeBarrierHelper &barrierHelper) {
  TritonLLVMIRRewriter b(callOp->getLoc(), callOp);
  FailureOr<Value> handle = barrierHelper.getBarrierHandle(b, partitionIdx);
  if (failed(handle))
    return failure();
  // Forward the barrier handle.
  callOp.setCallee(mangleFunctionName(*callOp.getCallee()));
  callOp.getCalleeOperandsMutable().append(*handle);
  return success();
}

static LogicalResult
lowerKernelBarriers(LLVM::LLVMFuncOp func,
                    const DenseSet<StringAttr> &innerFunctions,
                    WarpSpecializeBarrierHelper &barrierHelper) {
  unsigned defaultNumWarps = lookupNumWarps(func);

  // Turn all barrier ops into warp group barriers.
  // HACK: Right now, higher-level passes generate all barriers that we
  // interpret as warp group barriers, but they generate explicit warp group
  // barriers.
  SmallVector<WarpSpecializeOp> wsOps;
  WalkResult result = func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Walk into default regions but not partition regions.
    if (auto wsOp = dyn_cast<WarpSpecializePartitionsOp>(op)) {
      wsOps.push_back(wsOp.getParentOp());
      return WalkResult::skip();
    }
    if (barrierHelper.isBarrierOp(op)) {
      return WalkResult(lowerBarrier(op, defaultNumWarps, /*partitionIdx=*/{},
                                     barrierHelper));
    }
    if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
      if (!innerFunctions.contains(callOp.getCalleeAttr().getAttr()))
        return WalkResult::advance();
      return WalkResult(lowerCallOp(callOp, defaultNumWarps,
                                    /*partitionIdx=*/{}, barrierHelper));
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  // Each partition executes simultaneously, so each will get a different
  // barrier ID, but note this means there is a maximum of 16 barriers.
  for (WarpSpecializeOp op : wsOps) {
    for (auto [idx, partition] : llvm::enumerate(op.getPartitionRegions())) {
      unsigned numWarps = op.getPartitionNumWarps()[idx];
      WalkResult result = partition->walk([&, idx = idx](Operation *op) {
        if (barrierHelper.isBarrierOp(op)) {
          return WalkResult(lowerBarrier(op, numWarps, idx, barrierHelper));
        }
        if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
          if (!innerFunctions.contains(callOp.getCalleeAttr().getAttr()))
            return WalkResult::advance();
          return WalkResult(lowerCallOp(callOp, numWarps, idx, barrierHelper));
        }
        return WalkResult::advance();
      });
      if (result.wasInterrupted())
        return failure();
    }
  }

  return success();
}

static LogicalResult
lowerInnerFunctionBarriers(LLVM::LLVMFuncOp func,
                           const DenseSet<StringAttr> &innerFunctions,
                           WarpSpecializeBarrierHelper &barrierHelper) {
  // Append a barrier handle argument.
  LLVM::LLVMFunctionType type = func.getFunctionType();
  SmallVector<Type> newArgTypes = to_vector(type.getParams());
  newArgTypes.push_back(barrierHelper.getBarrierHandleType(type.getContext()));
  func.setFunctionType(LLVM::LLVMFunctionType::get(
      type.getReturnType(), newArgTypes, type.isVarArg()));
  Value handle = func.getBody().addArgument(newArgTypes.back(), func.getLoc());

  // Mangle the function to distinguish it from non-warp-specialized versions.
  func.setSymName(mangleFunctionName(func.getSymName()));
  if (ArrayAttr argAttrs = func.getArgAttrsAttr()) {
    SmallVector<Attribute> newArgAttrs = to_vector(argAttrs.getValue());
    newArgAttrs.push_back(DictionaryAttr::get(func.getContext(), {}));
    func.setArgAttrsAttr(ArrayAttr::get(func.getContext(), newArgAttrs));
  }

  // Lower barrier ops.
  auto numWarpsAttr = func->getAttrOfType<IntegerAttr>("ws_num_warps");
  if (!numWarpsAttr) {
    return func.emitError("function missing '") << "ws_num_warps"
                                                << "' attribute";
  }
  unsigned numWarps = numWarpsAttr.getInt();

  func.walk([&](Operation *op) {
    if (barrierHelper.isBarrierOp(op)) {
      TritonLLVMIRRewriter b(op->getLoc(), op);
      barrierHelper.createBarrier(b, numWarps, handle);
      op->erase();
    } else if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
      if (!innerFunctions.contains(callOp.getCalleeAttr().getAttr()))
        return;
      callOp.setCallee(mangleFunctionName(*callOp.getCallee()));
      callOp.getCalleeOperandsMutable().append(handle);
    }
  });

  return success();
}

LogicalResult mlir::triton::lowerWarpSpecializeBarriers(
    ModuleOp module, WarpSpecializeBarrierHelper &barrierHelper) {
  SmallVector<LLVM::LLVMFuncOp> wsKernels;
  // Find all kernels and the warp specialize ops in them.
  for (LLVM::LLVMFuncOp func : module.getOps<LLVM::LLVMFuncOp>()) {
    WalkResult result =
        func.walk([&](WarpSpecializeOp op) { return WalkResult::interrupt(); });
    // Nothing to do. This kernel is not warp specialized.
    if (!result.wasInterrupted())
      continue;
    if (func.getLinkage() != LLVM::Linkage::External) {
      return func.emitError(
          "only top-level kernel functions can be warp-specialized");
    }
    wsKernels.push_back(func);
  }
  // No warp specialization found.
  if (wsKernels.empty())
    return success();

  DenseSet<StringAttr> innerFunctions;
  for (LLVM::LLVMFuncOp func : module.getOps<LLVM::LLVMFuncOp>()) {
    if (func.getLinkage() != LLVM::Linkage::External)
      innerFunctions.insert(func.getSymNameAttr());
  }

  for (LLVM::LLVMFuncOp func : wsKernels) {
    if (failed(lowerKernelBarriers(func, innerFunctions, barrierHelper)))
      return failure();
  }

  for (LLVM::LLVMFuncOp func : module.getOps<LLVM::LLVMFuncOp>()) {
    if (func.getLinkage() == LLVM::Linkage::External)
      continue;
    if (failed(lowerInnerFunctionBarriers(func, innerFunctions, barrierHelper)))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// convertOpTypes
//===----------------------------------------------------------------------===//

void mlir::triton::convertOpTypes(Operation *op,
                                  const TypeConverter &typeConverter) {
  ImplicitLocOpBuilder b(op->getLoc(), op);
  // WarpSpecializePartitionsOp exists in a region that must only contain a
  // single op. This also means that we know that its operands always dominate
  // the enclosing WarpSpecializeOp, so we can insert the casts there instead.
  if (isa<WarpSpecializePartitionsOp>(op))
    b.setInsertionPoint(op->getParentOp());
  SmallVector<Value> operands = llvm::to_vector(op->getOperands());
  for (Value &operand : operands) {
    Type type = typeConverter.convertType(operand.getType());
    if (type != operand.getType()) {
      operand =
          UnrealizedConversionCastOp::create(b, type, operand).getResult(0);
    }
  }
  op->setOperands(operands);

  for (Region &region : op->getRegions()) {
    b.setInsertionPointToStart(&region.front());
    for (BlockArgument arg : llvm::to_vector(region.getArguments())) {
      Type type = typeConverter.convertType(arg.getType());
      BlockArgument newArg = region.addArgument(type, arg.getLoc());
      auto cast = UnrealizedConversionCastOp::create(b, arg.getType(), newArg);
      arg.replaceAllUsesWith(cast.getResult(0));
      region.eraseArgument(0);
    }
  }

  SmallVector<Type> resultTypes;
  (void)typeConverter.convertTypes(op->getResultTypes(), resultTypes);
  if (TypeRange(resultTypes) == op->getResultTypes())
    return;
  OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                       resultTypes, op->getAttrs());
  for (Region &region : op->getRegions())
    state.addRegion()->takeBody(region);
  b.setInsertionPoint(op);
  Operation *newOp = b.create(state);

  SmallVector<Value> results;
  for (auto [i, result, type] :
       llvm::enumerate(newOp->getResults(), op->getResultTypes())) {
    auto cast = UnrealizedConversionCastOp::create(b, type, result);
    op->getResult(i).replaceAllUsesWith(cast.getResult(0));
  }
  op->erase();
}

//===----------------------------------------------------------------------===//
// elideTrivialCaptures
//===----------------------------------------------------------------------===//

static LogicalResult findTrivialSubcomputation(LLVM::LLVMFuncOp func,
                                               Value capture,
                                               SetVector<Operation *> &ops) {
  SetVector<Value> worklist;
  worklist.insert(capture);
  for (unsigned i = 0; i != worklist.size(); ++i) {
    Value capture = worklist[i];
    // Check for a kernel argument.
    if (auto arg = dyn_cast<BlockArgument>(capture)) {
      if (arg.getOwner() == &func.getBody().front())
        continue;
      // Otherwise, this is some other block argument that cannot be elided.
      return failure();
    }

    Operation *op = capture.getDefiningOp();
    // Check if the defining op can be rematerialized. At the LLVM level,
    // checking for pure is probably a good enough heuristic.
    if (isPure(op)) {
      ops.insert(op);
      worklist.insert(op->operand_begin(), op->operand_end());
      continue;
    }
    // The op cannot be rematerialized.
    return failure();
  }

  // Cap the number of ops that can be rematerialized.
  // FIXME: This is arbitrary.
  return success(ops.size() <= 16);
}

void mlir::triton::elideTrivialCaptures(LLVM::LLVMFuncOp func,
                                        ArrayRef<WarpSpecializeOp> wsOps) {
  // The goal is to completely eliminate captures by hoisting or rematerializing
  // computations. We could minimize captures by rematerializing
  // subcomputations, but that is much more complicated. Prefer rematerializing
  // because that reduces liveranges. If subgraphs are duplicated more than
  // once, we will rely on CSE to clean them up.
  SetVector<Operation *> subgraph;
  for (WarpSpecializeOp wsOp : wsOps) {
    auto partOp = wsOp.getPartitionOp();
    llvm::BitVector toErase(partOp.getNumOperands());
    for (auto [i, capture] : llvm::enumerate(partOp.getExplicitCaptures())) {
      subgraph.clear();
      if (failed(findTrivialSubcomputation(func, capture, subgraph)))
        continue;
      toErase.set(i);
      subgraph = topologicalSort(subgraph);

      for (Region *region : wsOp.getPartitionRegions()) {
        OpBuilder b(region);
        IRMapping mapping;
        for (Operation *op : subgraph) {
          b.clone(*op, mapping);
        }
        Value remat = capture;
        if (!subgraph.empty()) {
          unsigned resultIdx = cast<OpResult>(capture).getResultNumber();
          remat = mapping.lookup(subgraph.back())->getResult(resultIdx);
        }
        region->getArgument(i).replaceAllUsesWith(remat);
      }
    }

    partOp->eraseOperands(toErase);
    for (Region *region : wsOp.getPartitionRegions()) {
      region->front().eraseArguments(toErase);
    }
  }
}

/// Disable LICM (Loop Invariant Code Motion) for a loop. This prevents LLVM
/// from hoisting code out of the switch loop generated by the
/// `ttg.warp_specialize` lowering, which could result in long liveranges and
/// cause register spilling in partition regions.
static void disableLICM(LLVM::BrOp latchBr) {
  Builder b(latchBr.getContext());
  MLIRContext *ctx = b.getContext();
  auto licmMD = LLVM::LoopLICMAttr::get(ctx, b.getBoolAttr(true), {});
  auto loopMD =
      LLVM::LoopAnnotationAttr::get(b.getContext(), {}, {}, {}, {}, {}, licmMD,
                                    {}, {}, {}, {}, {}, {}, {}, {}, {});
  latchBr.setLoopAnnotationAttr(loopMD);
}

//===----------------------------------------------------------------------===//
// lowerWarpSpecializeCommon
//===----------------------------------------------------------------------===//

static void rewritePartitionRegions(WarpSpecializeOp ws, Block *switchLoop,
                                    const TargetInfoBase &targetInfo,
                                    const WarpSpecializeCallbacks &callbacks,
                                    unsigned switchLoopBarrierIdx) {
  TritonLLVMIRRewriter b(ws.getLoc(), ws.getContext());
  for (Region *partition : ws.getPartitionRegions()) {
    // Load the explicit captures from shared memory and replace the block args
    // if there are any.
    b.setInsertionPointToStart(&partition->front());

    callbacks.reallocRegisters(b, ws,
                               RegisterReallocPhase::WorkerPartitionStart,
                               partition->getRegionNumber());

    if (partition->getNumArguments()) {
      auto captureType = LLVM::LLVMStructType::getLiteral(
          b.getContext(), llvm::to_vector(partition->getArgumentTypes()),
          /*isPacked=*/true);
      Value capturePtr =
          LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, ws);
      LLVM::LLVMPointerType ptrTy = ptr_ty(b.getContext(), 3);
      for (auto [i, arg] :
           llvm::zip(llvm::seq<int32_t>(partition->getNumArguments()),
                     partition->getArguments())) {
        Value ptr =
            b.gep(ptrTy, captureType, capturePtr, ArrayRef<LLVM::GEPArg>{0, i});
        // Each thread in the warp group needs a copy of the value.
        Value value = b.load(arg.getType(), ptr, /*align=*/1);
        arg.replaceAllUsesWith(value);
      }
      partition->front().eraseArguments([](auto) { return true; });
    }

    // The shared memory is only live for the entry into the region, so put
    // another barrier here.
    callbacks.createAllBarrier(b, switchLoopBarrierIdx);

    // Rewrite all warp returns.
    partition->walk([&](WarpReturnOp op) {
      TritonLLVMIRRewriter b(op.getLoc(), op);
      callbacks.createAllBarrier(b, switchLoopBarrierIdx);
      callbacks.reallocRegisters(b, ws,
                                 RegisterReallocPhase::WorkerPartitionEnd,
                                 partition->getRegionNumber());
      b.replaceOpWithNewOp<LLVM::BrOp>(op, switchLoop);
    });
  }
}

LogicalResult mlir::triton::lowerWarpSpecializeCommon(
    LLVM::LLVMFuncOp func, ArrayRef<WarpSpecializeOp> wsOps, Block *entry,
    Block *header, Block *switchLoop, Value wid, MLIRContext *ctx,
    unsigned defaultNumWarps, unsigned totalNumWarps,
    const TargetInfoBase &targetInfo, const WarpSpecializeCallbacks &callbacks,
    unsigned switchLoopBarrierIdx) {

  TritonLLVMIRRewriter b(func.getLoc(), ctx);
  Type int8Type = b.getIntegerType(8);
  LLVM::LLVMPointerType ptrTy = ptr_ty(ctx, 3);

  b.setInsertionPointToStart(switchLoop);
  callbacks.reallocRegisters(b, wsOps[0], RegisterReallocPhase::SwitchLoopStart,
                             0);
  callbacks.createAllBarrier(b, switchLoopBarrierIdx);
  Value statePtr = LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, func);
  Value relWid = b.sub(wid, b.i32_val(defaultNumWarps));

  // The default warp group will populate the state pointer with the state ID
  // for all warps.
  // %warp_state_ptr = getelementptr ptr %state_tr[%rel_wid]
  // %warp_state = load i8 %warp_state_ptr
  Value warpStatePtr = b.gep(ptrTy, int8Type, statePtr, relWid);
  // All threads in a warp reading from the same smem address will not create
  // bank conflicts and is better than predicated load.
  Value warpState = b.load(int8Type, warpStatePtr);

  // Pull the partition regions out. Switch based on the state ID to the right
  // partition.
  SmallVector<Block *> partitionBlocks;
  SmallVector<int32_t> partitionStates;
  int32_t partitionStateCounter = 0;
  // This represents the data that the default warp group will fill into the
  // state pointer before entering each `warp_specialize` region, which maps
  // a warp ID to a state ID in the switch.
  int32_t maxNumWarps = totalNumWarps - defaultNumWarps;
  SmallVector<SmallVector<int32_t>> warpToState(
      wsOps.size(), SmallVector<int32_t>(maxNumWarps, -1));

  for (size_t i = 0; i < wsOps.size(); ++i) {
    WarpSpecializeOp op = wsOps[i];
    auto &stateMap = warpToState[i];
    rewritePartitionRegions(op, switchLoop, targetInfo, callbacks,
                            switchLoopBarrierIdx);
    for (auto [partition, partitionNumWarps, startId] :
         llvm::zip(op.getPartitionRegions(), op.getPartitionNumWarps(),
                   *op.getWarpGroupStartIds())) {
      partitionStates.push_back(partitionStateCounter++);
      partitionBlocks.push_back(&partition->front());
      for (int32_t &stateId : MutableArrayRef(stateMap).slice(
               startId - defaultNumWarps, partitionNumWarps))
        stateId = partitionStates.back();
    }
  }

  if (partitionStateCounter > std::numeric_limits<uint8_t>::max()) {
    return mlir::emitError(func.getLoc(),
                           "FIXME: too many warp group partitions");
  }

  // Splice them in reverse order so the IR is easier to read.
  Region::BlockListType &funcBlocks = func.getBody().getBlocks();
  for (Block *block : llvm::reverse(partitionBlocks)) {
    Region *region = block->getParent();
    funcBlocks.splice(std::next(switchLoop->getIterator()),
                      region->getBlocks());
  }

  // Default destination.
  Block *defaultBlock = new Block;
  funcBlocks.insert(std::next(switchLoop->getIterator()), defaultBlock);
  b.setInsertionPointToStart(defaultBlock);
  callbacks.createAllBarrier(b, switchLoopBarrierIdx);
  callbacks.createAllBarrier(b, switchLoopBarrierIdx);
  auto latchBr = LLVM::BrOp::create(b, b.getLoc(), switchLoop);
  disableLICM(latchBr);

  // Exit state.
  Block *switchExit = new Block;
  funcBlocks.insert(std::next(defaultBlock->getIterator()), switchExit);
  partitionBlocks.push_back(switchExit);
  partitionStates.push_back(partitionStateCounter);

  // Create the switch.
  b.setInsertionPointToEnd(switchLoop);
  SmallVector<APInt> caseValues;
  for (int32_t state : partitionStates)
    caseValues.push_back(APInt(8, state));
  LLVM::SwitchOp::create(b, b.getLoc(), warpState, defaultBlock, ValueRange(),
                         caseValues, partitionBlocks,
                         SmallVector<ValueRange>(partitionBlocks.size()));

  // Now add synchronization around the default regions.
  for (size_t i = 0; i < wsOps.size(); ++i) {
    WarpSpecializeOp ws = wsOps[i];
    auto &stateMap = warpToState[i];
    Block *before = ws->getBlock();
    Block *after = b.splitBlock(before, ws->getIterator());
    TritonLLVMIRRewriter b(ws.getLoc(), OpBuilder::atBlockEnd(before));
    Type int8Type = b.getIntegerType(8);
    Value statePtrWs =
        LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, func);
    for (auto [j, state] : llvm::enumerate(stateMap)) {
      Value stateVal = b.i8_val(state);
      b.store(stateVal, b.gep(ptrTy, int8Type, statePtrWs, LLVM::GEPArg(j)));
    }

    // Store the captures if there are any.
    auto partOp = ws.getPartitionOp();
    if (partOp.getNumOperands()) {
      auto captureType = LLVM::LLVMStructType::getLiteral(
          b.getContext(), llvm::to_vector(partOp.getOperandTypes()),
          /*isPacked=*/true);
      Value capturePtr =
          LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, ws);
      for (auto [j, arg] :
           llvm::zip(llvm::seq<int32_t>(partOp.getNumOperands()),
                     partOp.getOperands())) {
        Value ptr =
            b.gep(ptrTy, captureType, capturePtr, ArrayRef<LLVM::GEPArg>{0, j});
        b.store(arg, ptr, /*align=*/1);
      }
    }

    // First barrier releases the waiting warpgroups. The second barrier ensures
    // they have read the captures before the memory is released upon entry.
    callbacks.createAllBarrier(b, switchLoopBarrierIdx);
    callbacks.reallocRegisters(b, ws,
                               RegisterReallocPhase::DefaultPartitionStart, 0);
    callbacks.createAllBarrier(b, switchLoopBarrierIdx);
    LLVM::BrOp::create(b, b.getLoc(), &ws.getDefaultRegion().front());

    ws.getDefaultRegion().walk([&, ws = ws](WarpYieldOp op) mutable {
      TritonLLVMIRRewriter b(op.getLoc(), op);
      callbacks.createAllBarrier(b, switchLoopBarrierIdx);
      callbacks.reallocRegisters(b, ws,
                                 RegisterReallocPhase::DefaultPartitionEnd, 0);
      b.replaceOpWithNewOp<LLVM::BrOp>(op, op.getOperands(), after);
    });
    after->getParent()->getBlocks().splice(after->getIterator(),
                                           ws.getDefaultRegion().getBlocks());

    // Replace the results.
    auto outputs = after->addArguments(
        ws.getResultTypes(),
        SmallVector<Location>(ws.getNumResults(), ws.getLoc()));
    ws.replaceAllUsesWith(outputs);
    ws.erase();
  }

  // Signal all warp groups to exit.
  func.walk([&](LLVM::ReturnOp op) {
    TritonLLVMIRRewriter b(op.getLoc(), op);
    Type int8Type = b.getIntegerType(8);
    Value statePtrExit =
        LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, func);
    Value cst = b.i8_val(partitionStateCounter);
    for (int32_t i : llvm::seq(maxNumWarps))
      b.store(cst, b.gep(ptrTy, int8Type, statePtrExit, LLVM::GEPArg(i)));
    callbacks.createAllBarrier(b, switchLoopBarrierIdx);
  });
  b.setInsertionPointToStart(switchExit);
  LLVM::ReturnOp::create(b, b.getLoc(), ValueRange());

  return success();
}
