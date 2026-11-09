#include "triton/Conversion/TritonGPUToLLVM/WarpSpecializeUtility.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OperationSupport.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/MathExtras.h"

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

static LogicalResult
lowerBarrierHandle(Operation *op, std::optional<unsigned> partitionIdx,
                   WarpSpecializeBarrierHelper &barrierHelper) {
  TritonLLVMIRRewriter b(op->getLoc(), op);
  FailureOr<Value> handle = barrierHelper.getBarrierHandle(b, partitionIdx);
  if (failed(handle))
    return failure();
  assert(op->getNumResults() == 1 && "barrier handle op must have one result");
  op->getResult(0).replaceAllUsesWith(*handle);
  op->erase();
  return success();
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
    if (barrierHelper.isBarrierHandleOp(op)) {
      auto result = lowerBarrierHandle(op, /*partitionIdx=*/{}, barrierHelper);
      return succeeded(result) ? WalkResult::skip() : WalkResult::interrupt();
    }
    if (barrierHelper.isBarrierOp(op)) {
      auto barRes =
          lowerBarrier(op, defaultNumWarps, /*partitionIdx=*/{}, barrierHelper);
      // Since this is a PreOrder walk, we have to return skip() on success so
      // we do not attempt to visit the regions in the now deleted op.
      return barRes.succeeded() ? WalkResult::skip() : WalkResult::interrupt();
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
        if (barrierHelper.isBarrierHandleOp(op))
          return WalkResult(lowerBarrierHandle(op, idx, barrierHelper));
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
    if (barrierHelper.isBarrierHandleOp(op)) {
      assert(op->getNumResults() == 1 &&
             "barrier handle op must have one result");
      op->getResult(0).replaceAllUsesWith(handle);
      op->erase();
    } else if (barrierHelper.isBarrierOp(op)) {
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
  if (wsKernels.empty()) {
    WalkResult result = module.walk([&](Operation *op) {
      if (!barrierHelper.isBarrierHandleOp(op))
        return WalkResult::advance();
      return WalkResult(
          lowerBarrierHandle(op, /*partitionIdx=*/{}, barrierHelper));
    });
    if (result.wasInterrupted())
      return failure();
    return success();
  }

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

static void forEachCapture(TritonLLVMIRRewriter &b, WarpSpecializeOp ws,
                           ValueRange captures,
                           const TargetInfoBase &targetInfo,
                           llvm::function_ref<void(Value, Value)> access) {
  if (captures.empty())
    return;
  auto type = LLVM::LLVMStructType::getLiteral(
      b.getContext(), llvm::to_vector(captures.getTypes()), /*isPacked=*/true);
  Value base = LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, ws);
  auto ptrTy = ptr_ty(b.getContext(), targetInfo.getSharedAddressSpace());
  for (auto [i, value] : llvm::enumerate(captures))
    access(value, b.gep(ptrTy, type, base, ArrayRef<LLVM::GEPArg>{0, i}));
}

static void rewritePartitionRegion(WarpSpecializeOp ws, Region &partition,
                                   Block *resume,
                                   const TargetInfoBase &targetInfo,
                                   const WarpSpecializeCallbacks &callbacks,
                                   Block *workerLoop) {
  TritonLLVMIRRewriter b(ws.getLoc(), &partition.front().front());
  forEachCapture(
      b, ws, partition.getArguments(), targetInfo, [&](Value arg, Value ptr) {
        b.replaceAllUsesWith(arg, b.load(arg.getType(), ptr, /*align=*/1));
      });
  partition.front().eraseArguments(0, partition.getNumArguments());

  // Capture storage can be reused as soon as all workers have read it.
  if (workerLoop || ws.getPartitionOp().getNumOperands())
    callbacks.createAllBarrier(b);

  for (Block &block : partition) {
    auto ret = dyn_cast<WarpReturnOp>(block.getTerminator());
    if (!ret)
      continue;
    b.setInsertionPoint(ret);
    auto br = b.replaceOpWithNewOp<LLVM::BrOp>(ret, resume);
    if (resume == workerLoop) {
      b.setInsertionPoint(br);
      callbacks.createAllBarrier(b);
      disableLICM(br);
    }
  }
}

static void rewriteDefaultRegion(WarpSpecializeOp ws,
                                 const TargetInfoBase &targetInfo,
                                 const WarpSpecializeCallbacks &callbacks,
                                 bool direct) {
  Block *before = ws->getBlock();
  TritonLLVMIRRewriter b(ws.getLoc(), ws);
  Block *after = b.splitBlock(before, ws->getIterator());
  b.setInsertionPointToEnd(before);
  auto partOp = ws.getPartitionOp();
  forEachCapture(
      b, ws, partOp.getOperands(), targetInfo,
      [&](Value value, Value ptr) { b.store(value, ptr, /*align=*/1); });
  callbacks.createAllBarrier(b);
  callbacks.reallocDefaultRegisters(b, ws, /*acquire=*/true);
  if (!direct || partOp.getNumOperands())
    callbacks.createAllBarrier(b);
  for (Type type : ws.getResultTypes())
    after->addArgument(type, ws.getLoc());
  for (Block &block : ws.getDefaultRegion())
    if (auto yield = dyn_cast<WarpYieldOp>(block.getTerminator())) {
      b.setInsertionPoint(yield);
      b.replaceOpWithNewOp<LLVM::BrOp>(yield, yield.getOperands(), after);
    }
  b.mergeBlocks(&ws.getDefaultRegion().front(), before);
  b.inlineRegionBefore(ws.getDefaultRegion(), after);
  b.setInsertionPointToStart(after);
  callbacks.createAllBarrier(b);
  callbacks.reallocDefaultRegisters(b, ws, /*acquire=*/false);
  b.replaceOp(ws, after->getArguments());
}

bool mlir::triton::canDirectDispatch(ArrayRef<WarpSpecializeOp> sites) {
  if (sites.size() != 1)
    return false;
  Block *site = sites.front()->getBlock();
  DominanceInfo dom;
  return !site->isReachable(site) &&
         llvm::all_of(site->getParent()->getOps<LLVM::ReturnOp>(),
                      [&](LLVM::ReturnOp ret) {
                        return dom.dominates(site, ret->getBlock());
                      });
}

static SmallVector<unsigned> getPartitionOrder(ArrayRef<int32_t> starts,
                                               bool byWarpId) {
  auto order = llvm::to_vector(llvm::seq<unsigned>(starts.size()));
  if (byWarpId)
    llvm::sort(order,
               [&](unsigned a, unsigned b) { return starts[a] < starts[b]; });
  return order;
}

static void
dispatchByWarpId(TritonLLVMIRRewriter &b, WarpSpecializeOp ws, Value wid,
                 ArrayRef<unsigned> partitions,
                 llvm::function_ref<Block *(unsigned)> destination) {
  auto starts = *ws.getWarpGroupStartIds();
  auto sizes = ws.getPartitionNumWarps();
  Block *otherwise = destination(partitions.front());
  SmallVector<int32_t> values;
  SmallVector<Block *> destinations;
  for (unsigned p : partitions.drop_front()) {
    Block *dest = destination(p);
    if (dest == otherwise)
      continue;
    llvm::append_range(values, llvm::seq(starts[p], starts[p] + sizes[p]));
    destinations.append(sizes[p], dest);
  }
  if (values.empty())
    LLVM::BrOp::create(b, b.getLoc(), otherwise);
  else
    LLVM::SwitchOp::create(b, b.getLoc(), wid, otherwise, ValueRange(), values,
                           destinations,
                           SmallVector<ValueRange>(values.size()));
}

// A shared entry acquires its register cap before dispatching by warp ID.
static Block *lowerWorkerGroup(WarpSpecializeOp ws,
                               ArrayRef<unsigned> partitions, int32_t cap,
                               Value wid, Block *before, Block *workerLoop,
                               const TargetInfoBase &targetInfo,
                               const WarpSpecializeCallbacks &callbacks,
                               bool commonCap = false) {
  TritonLLVMIRRewriter b(ws.getLoc(), ws);
  auto regions = ws.getPartitionRegions();
  auto body = [&](unsigned p) { return &regions[p]->front(); };
  bool commandToBody = workerLoop && partitions.size() == 1;
  Block *start =
      commandToBody ? body(partitions.front()) : b.createBlock(before);
  Block *finish = commandToBody && cap < 0 ? workerLoop : b.createBlock(before);
  for (unsigned p : partitions)
    rewritePartitionRegion(ws, *regions[p], finish, targetInfo, callbacks,
                           workerLoop);
  if (finish == workerLoop)
    return start;
  b.setInsertionPointToStart(start);
  if (!commonCap && cap >= 0)
    callbacks.reallocWorkerRegisters(b, cap, /*acquire=*/true);
  if (!commandToBody)
    dispatchByWarpId(b, ws, wid, partitions, body);
  b.setInsertionPointToStart(finish);
  callbacks.createAllBarrier(b);
  if (cap >= 0)
    callbacks.reallocWorkerRegisters(b, cap, /*acquire=*/false);
  auto br = LLVM::BrOp::create(b, b.getLoc(), workerLoop ? workerLoop : before);
  if (workerLoop)
    disableLICM(br);
  return start;
}

static void
lowerDirectWarpSpecialize(WarpSpecializeOp ws, Block *workers, Value wid,
                          unsigned numWorkers, const TargetInfoBase &targetInfo,
                          const WarpSpecializeCallbacks &callbacks) {
  TritonLLVMIRRewriter b(ws.getLoc(), ws.getContext());
  Block *exit =
      b.createBlock(workers->getParent(), std::next(workers->getIterator()));

  auto order = getPartitionOrder(*ws.getWarpGroupStartIds(), /*byWarpId=*/true);
  // Explicit layouts with gaps route unassigned warps to partition zero.
  if (ws.getTotalPartitionWarps() < numWorkers) {
    auto zero = llvm::find(order, 0u);
    std::rotate(order.begin(), zero, std::next(zero));
  }

  // Keep different register limits on separate CFG paths through the payload.
  // Subpartitions sharing a hardware warpgroup adjust at the same instruction.
  llvm::MapVector<int32_t, SmallVector<unsigned>> groups;
  for (unsigned p : order)
    groups[callbacks.getWorkerRegisterCap(ws, p)].push_back(p);

  SmallVector<Block *> entries(order.size());
  for (const auto &[cap, partitions] : groups) {
    Block *entry =
        lowerWorkerGroup(ws, partitions, cap, wid, exit,
                         /*workerLoop=*/nullptr, targetInfo, callbacks);
    for (unsigned p : partitions)
      entries[p] = entry;
  }

  b.setInsertionPointToEnd(workers);
  callbacks.createAllBarrier(b);
  dispatchByWarpId(b, ws, wid, order, [&](unsigned p) { return entries[p]; });

  for (Region *region : ws.getPartitionRegions())
    b.inlineRegionBefore(*region, exit);

  rewriteDefaultRegion(ws, targetInfo, callbacks, /*direct=*/true);
  b.setInsertionPointToStart(exit);
  LLVM::ReturnOp::create(b, b.getLoc(), ValueRange());
}

// A partition crossing a warpgroup boundary carries its command into the next
// warpgroup. Per-warp command IDs follow partition order.
static SmallVector<SmallVector<unsigned>>
groupWorkerPartitions(ArrayRef<int32_t> starts, unsigned defaultNumWarps,
                      unsigned warpsPerSlot) {
  SmallVector<SmallVector<unsigned>> groups;
  for (unsigned p : getPartitionOrder(starts, warpsPerSlot > 1)) {
    if (groups.empty() || (starts[p] - defaultNumWarps) % warpsPerSlot == 0)
      groups.emplace_back();
    groups.back().push_back(p);
  }
  return groups;
}

LogicalResult mlir::triton::lowerWarpSpecializeCommon(
    LLVM::LLVMFuncOp func, ArrayRef<WarpSpecializeOp> wsOps, bool direct,
    Block *switchLoop, Value wid, unsigned defaultNumWarps,
    unsigned totalNumWarps, const TargetInfoBase &targetInfo,
    const WarpSpecializeCallbacks &callbacks) {

  int32_t numWorkers = totalNumWarps - defaultNumWarps;
  if (direct) {
    lowerDirectWarpSpecialize(wsOps.front(), switchLoop, wid, numWorkers,
                              targetInfo, callbacks);
    return success();
  }
  TritonLLVMIRRewriter b(func.getLoc(), func.getContext());
  Type int8Type = b.getIntegerType(8);
  auto ptrTy = ptr_ty(b.getContext(), targetInfo.getSharedAddressSpace());

  b.setInsertionPointToStart(switchLoop);
  callbacks.createAllBarrier(b);
  Value statePtr = LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, func);
  unsigned warpsPerSlot = callbacks.warpsPerCommand;
  unsigned numSlots = llvm::divideCeil(numWorkers, warpsPerSlot);
  Value warpStatePtr = statePtr;
  if (numSlots > 1 || warpsPerSlot == 1) {
    Value group =
        warpsPerSlot == 1 ? wid : b.udiv(wid, b.i32_val(warpsPerSlot));
    Value slot = b.sub(group, b.i32_val(defaultNumWarps / warpsPerSlot));
    warpStatePtr = b.gep(ptrTy, int8Type, statePtr, slot);
  }
  // All threads in a warp reading from the same smem address will not create
  // bank conflicts and is better than predicated load.
  Value warpState = b.load(int8Type, warpStatePtr);

  SmallVector<Block *> partitionBlocks;
  SmallVector<SmallVector<int32_t>> slotToState(wsOps.size(),
                                                SmallVector<int32_t>(numSlots));
  Block *before = switchLoop->getNextNode();
  for (size_t i = 0; i < wsOps.size(); ++i) {
    WarpSpecializeOp op = wsOps[i];
    auto &stateMap = slotToState[i];
    TritonLLVMIRRewriter b(op.getLoc(), op);
    auto starts = *op.getWarpGroupStartIds();
    auto sizes = op.getPartitionNumWarps();
    auto groups = groupWorkerPartitions(starts, defaultNumWarps, warpsPerSlot);
    unsigned firstState = partitionBlocks.size();
    if (firstState + groups.size() > std::numeric_limits<uint8_t>::max())
      return op.emitError("too many warp specialization dispatcher states");

    auto regions = op.getPartitionRegions();
    SmallVector<int32_t> caps;
    for (unsigned p : llvm::seq<unsigned>(regions.size()))
      caps.push_back(callbacks.getWorkerRegisterCap(op, p));
    // Acquire a site-wide cap before the command selects its body.
    Block *commonEntry = nullptr;
    if (groups.size() > 1 && caps.front() >= 0 && llvm::all_equal(caps))
      commonEntry = b.createBlock(before);
    for (const auto &partitions : groups) {
      int32_t state = partitionBlocks.size();
      unsigned nextSlot = 0;
      for (unsigned p : partitions) {
        unsigned firstWarp = starts[p] - defaultNumWarps;
        unsigned endSlot = llvm::divideCeil(firstWarp + sizes[p], warpsPerSlot);
        nextSlot = std::max(nextSlot, firstWarp / warpsPerSlot);
        for (; nextSlot < endSlot; ++nextSlot)
          stateMap[nextSlot] = state;
      }
      partitionBlocks.push_back(lowerWorkerGroup(
          op, partitions, caps[partitions.front()], wid, before, switchLoop,
          targetInfo, callbacks, commonEntry != nullptr));
    }
    if (commonEntry) {
      b.setInsertionPointToStart(commonEntry);
      callbacks.reallocWorkerRegisters(b, caps.front(), /*acquire=*/true);
      auto cases = MutableArrayRef(partitionBlocks).drop_front(firstState);
      SmallVector<APInt> values;
      for (unsigned state :
           llvm::seq<unsigned>(firstState, partitionBlocks.size() - 1))
        values.emplace_back(8, state);
      LLVM::SwitchOp::create(b, b.getLoc(), warpState, cases.back(),
                             ValueRange(), values, cases.drop_back(),
                             SmallVector<ValueRange>(values.size()));
      llvm::fill(cases, commonEntry);
    }
    for (Region *region : regions)
      b.inlineRegionBefore(*region, before);
  }

  int32_t exitState = partitionBlocks.size();
  // Every worker has a case at each site. The only other command is exit.
  Block *switchExit =
      b.createBlock(&func.getBody(), std::next(switchLoop->getIterator()));

  // Create the switch.
  b.setInsertionPointToEnd(switchLoop);
  SmallVector<APInt> caseValues;
  for (int32_t state : llvm::seq(partitionBlocks.size()))
    caseValues.push_back(APInt(8, state));
  LLVM::SwitchOp::create(b, b.getLoc(), warpState, switchExit, ValueRange(),
                         caseValues, partitionBlocks,
                         SmallVector<ValueRange>(partitionBlocks.size()));

  // Now add synchronization around the default regions.
  for (size_t i = 0; i < wsOps.size(); ++i) {
    WarpSpecializeOp ws = wsOps[i];
    auto &stateMap = slotToState[i];
    TritonLLVMIRRewriter b(ws.getLoc(), ws);
    Type int8Type = b.getIntegerType(8);
    Value statePtrWs =
        LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, func);
    for (auto [j, state] : llvm::enumerate(stateMap)) {
      Value stateVal = b.i8_val(state);
      b.store(stateVal, b.gep(ptrTy, int8Type, statePtrWs, LLVM::GEPArg(j)));
    }

    rewriteDefaultRegion(ws, targetInfo, callbacks, /*direct=*/false);
  }

  // Signal all warp groups to exit.
  func.walk([&](LLVM::ReturnOp op) {
    TritonLLVMIRRewriter b(op.getLoc(), op);
    Type int8Type = b.getIntegerType(8);
    Value statePtrExit =
        LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, func);
    Value cst = b.i8_val(exitState);
    for (int32_t i : llvm::seq(numSlots))
      b.store(cst, b.gep(ptrTy, int8Type, statePtrExit, LLVM::GEPArg(i)));
    callbacks.createAllBarrier(b);
  });
  b.setInsertionPointToStart(switchExit);
  LLVM::ReturnOp::create(b, b.getLoc(), ValueRange());

  return success();
}
