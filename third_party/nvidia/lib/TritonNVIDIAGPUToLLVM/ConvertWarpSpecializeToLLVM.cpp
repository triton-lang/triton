#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTWARPSPECIALIZETOLLVM
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Reserve one barrier for the default warp group, one for the start barrier,
// and one for the end barrier.
enum BarrierIndex {
  kDefaultWarpGroupBarrierIdx,
  kSwitchLoopBarrierIdx,

  kNumReservedBarriers,
  kNumBarriers = 16
};

static void createBarrier(TritonLLVMOpBuilder2 &b, unsigned barIdx,
                          std::optional<unsigned> numThreads, bool aligned) {
  assert(barIdx < 16 && "not enough barriers");

  PTXBuilder ptxBuilder;
  std::string ptxString;
  llvm::raw_string_ostream os(ptxString);
  os << "barrier.sync";
  if (aligned)
    os << ".aligned";
  os << ' ' << barIdx;
  if (numThreads)
    os << ", " << *numThreads;

  (*ptxBuilder.create<>(ptxString))();
  ptxBuilder.launch(b, b.getLoc(), void_ty(b.getContext()));
}

//===----------------------------------------------------------------------===//
// lowerWarpSpecialize
//===----------------------------------------------------------------------===//

// Assign hardware barriers to each warp group and rewrite warp group barriers
// into `barrier.sync` instructions. There is a maximum number of barriers.
static LogicalResult rewriteWarpGroupBarriers(LLVM::LLVMFuncOp func,
                                              ArrayRef<WarpSpecializeOp> wsOps,
                                              unsigned threadsPerWarp,
                                              unsigned defaultWarpGroupSize) {
  // HACK: Turn all `nvvm.barrier0` ops into warp group barriers.
  func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Walk into default regions but not partition regions.
    if (isa<WarpSpecializePartitionsOp>(op))
      return WalkResult::skip();

    if (auto bar = dyn_cast<NVVM::Barrier0Op>(op)) {
      TritonLLVMOpBuilder2 b(bar.getLoc(), bar);
      createBarrier(b, /*barIdx=*/0, defaultWarpGroupSize, /*aligned=*/true);
      bar.erase();
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });

  // Each partition executes simultaneously, so each will get a different
  // barrier ID, but note this means there is a maximum of 16 barriers.
  for (WarpSpecializeOp op : wsOps) {
    for (auto [idx, partition] : llvm::enumerate(op.getPartitionRegions())) {
      unsigned barIdx = idx + kNumReservedBarriers;
      if (barIdx >= kNumBarriers) {
        return func.emitError("cannot support more than ")
               << (kNumBarriers - kNumReservedBarriers)
               << " warp group partitions";
      }
      unsigned warpGroupSize = threadsPerWarp * op.getPartitionNumWarps()[idx];
      partition->walk([&](NVVM::Barrier0Op bar) {
        TritonLLVMOpBuilder2 b(bar.getLoc(), bar);
        createBarrier(b, barIdx, warpGroupSize, /*aligned=*/true);
        bar.erase();
      });
    }
  }

  return success();
}

static void rewritePartitionRegions(WarpSpecializeOp ws, Block *switchLoop,
                                    const NVIDIA::TargetInfo &targetInfo) {
  TritonLLVMOpBuilder2 b(ws.getLoc(), ws.getContext());

  for (Region *partition : ws.getPartitionRegions()) {
    // Load the explicit captures from shared memory and replace the block args
    // if there are any.
    b.setInsertionPointToStart(&partition->front());
    if (partition->getNumArguments()) {
      auto captureType = LLVM::LLVMStructType::getLiteral(
          b.getContext(), llvm::to_vector(partition->getArgumentTypes()));
      Value capturePtr =
          LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, ws);
      LLVM::LLVMPointerType ptrTy = ptr_ty(b.getContext(), 3);
      for (auto [i, arg] :
           llvm::zip(llvm::seq<int32_t>(partition->getNumArguments()),
                     partition->getArguments())) {
        Value ptr =
            b.gep(ptrTy, captureType, capturePtr, ArrayRef<LLVM::GEPArg>{0, i});
        Value value = b.load(arg.getType(), ptr);
        arg.replaceAllUsesWith(value);
      }
      partition->front().eraseArguments([](auto) { return true; });
    }

    // The shared memory is only live for the entry into the region, so put
    // another barrier here.
    createBarrier(b, kSwitchLoopBarrierIdx, /*numThreads=*/std::nullopt,
                  /*aligned=*/false);

    // Rewrite all warp returns.
    partition->walk([&](WarpReturnOp op) {
      b.setInsertionPoint(op);
      createBarrier(b, kSwitchLoopBarrierIdx, /*numThreads=*/std::nullopt,
                    /*aligned=*/false);
      b.replaceOpWithNewOp<LLVM::BrOp>(op, switchLoop);
    });
  }
}

static LogicalResult lowerWarpSpecialize(LLVM::LLVMFuncOp func,
                                         const NVIDIA::TargetInfo &targetInfo) {
  SmallVector<WarpSpecializeOp> wsOps;
  func.walk([&](WarpSpecializeOp op) { wsOps.push_back(op); });
  // Nothing to do. This kernel is not warp specialized.
  if (wsOps.empty())
    return success();

  // Before lowering away `ttg.warp_specialize`, lower warp group barriers.
  auto module = cast<ModuleOp>(func->getParentOp());
  unsigned threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(module);
  unsigned defaultNumWarps = lookupNumWarps(func);
  unsigned defaultWarpGroupSize = threadsPerWarp * defaultNumWarps;
  if (failed(rewriteWarpGroupBarriers(func, wsOps, threadsPerWarp,
                                      defaultWarpGroupSize)))
    return failure();

  MLIRContext *ctx = func.getContext();
  TritonLLVMOpBuilder2 b(func.getLoc(), ctx);
  Builder rewriter(ctx);

  // Generate the function header.
  Block *entry = &func.getBody().front();
  SmallVector<Location> argLocs = llvm::to_vector(llvm::map_range(
      func.getArguments(), [](BlockArgument arg) { return arg.getLoc(); }));
  Block *header = b.createBlock(entry, func.getArgumentTypes(), argLocs);
  Block *switchLoop = b.createBlock(entry);
  b.setInsertionPointToStart(header);

  // This is the absolute thread ID.
  Value tid = b.create<NVVM::ThreadIdXOp>(i32_ty);
  Value isDefault = b.icmp_ult(tid, b.i32_val(defaultWarpGroupSize));
  b.create<LLVM::CondBrOp>(isDefault, entry, switchLoop);

  // Forward arguments from the header into the old entry block.
  for (auto [arg, oldArg] :
       llvm::zip(header->getArguments(), entry->getArguments()))
    oldArg.replaceAllUsesWith(arg);
  entry->eraseArguments([](auto) { return true; });

  // Generate the switch loop.
  auto totalNumWarpsAttr =
      module->getAttrOfType<IntegerAttr>("ttg.total-num-warps");
  if (!totalNumWarpsAttr) {
    return mlir::emitError(module.getLoc(),
                           "module missing 'ttg.total-num-warps' attribute");
  }
  unsigned totalNumThreads = totalNumWarpsAttr.getInt() * threadsPerWarp;

  // ^switchLoop:
  //   barrier.sync 1
  //   %state_ptr = getelementptr (ptr @shared), <offset>
  //   %rel_tid = sub %tid, <default_warp_group_size>
  //   %rel_wid = udiv %rel_tid, 32
  b.setInsertionPointToStart(switchLoop);
  createBarrier(b, kSwitchLoopBarrierIdx, /*numThreads=*/std::nullopt,
                /*aligned=*/false);
  Value statePtr = LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, func);
  Value relTid = b.sub(tid, b.i32_val(defaultWarpGroupSize));
  Value relWid = b.udiv(relTid, b.i32_val(threadsPerWarp));
  // Tell PTXAS this value is warp-uniform.
  relWid = targetInfo.shuffleIdx(b, b.getLoc(), relWid, 0);

  // The default warp group will populate the state pointer with the state ID
  // for all warps.
  // %warp_state_ptr = getelementptr ptr %state_tr[%rel_wid]
  // %warp_state = load i8 %warp_state_ptr
  LLVM::LLVMPointerType ptrTy = ptr_ty(ctx, 3);
  Value warpStatePtr = b.gep(ptrTy, i8_ty, statePtr, relWid);
  Value warpState = b.load(i8_ty, warpStatePtr);

  // Pull the partition regions out. Switch based on the state ID to the right
  // partition.
  SmallVector<Block *> partitionBlocks;
  SmallVector<int32_t> partitionStates;
  int32_t partitionStateCounter = 0;
  // This represents the data that the default warp group will fill into the
  // state pointer before entering each `warp_specialize` region, which maps
  // a warp ID to a state ID in the switch.
  int32_t maxNumWarps = totalNumWarpsAttr.getInt() - defaultNumWarps;
  SmallVector<SmallVector<int32_t>> warpToState(
      wsOps.size(), SmallVector<int32_t>(maxNumWarps, -1));
  for (auto [op, stateMap] : llvm::zip(wsOps, warpToState)) {
    rewritePartitionRegions(op, switchLoop, targetInfo);
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
  createBarrier(b, kSwitchLoopBarrierIdx, /*numThreads=*/std::nullopt,
                /*aligned=*/false);
  createBarrier(b, kSwitchLoopBarrierIdx, /*numThreads=*/std::nullopt,
                /*aligned=*/false);
  b.create<LLVM::BrOp>(switchLoop);

  // Create the switch.
  b.setInsertionPointToEnd(switchLoop);
  SmallVector<APInt> caseValues;
  for (int32_t state : partitionStates)
    caseValues.push_back(APInt(8, state));
  b.create<LLVM::SwitchOp>(warpState, defaultBlock, ValueRange(), caseValues,
                           partitionBlocks,
                           SmallVector<ValueRange>(partitionBlocks.size()));

  // Now add synchronization around the default regions.
  for (auto [ws, stateMap] : llvm::zip(wsOps, warpToState)) {
    Block *before = ws->getBlock();
    Block *after = b.splitBlock(before, ws->getIterator());
    b.setInsertionPointToEnd(before);
    Value statePtr = LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, func);
    for (auto [i, state] : llvm::enumerate(stateMap)) {
      b.store(b.i8_val(state), b.gep(ptrTy, i8_ty, statePtr, LLVM::GEPArg(i)));
    }

    // Store the captures if there are any.
    if (ws.getNumOperands()) {
      auto captureType = LLVM::LLVMStructType::getLiteral(
          b.getContext(), llvm::to_vector(ws.getOperandTypes()));
      Value capturePtr =
          LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, ws);
      for (auto [i, arg] : llvm::zip(llvm::seq<int32_t>(ws.getNumOperands()),
                                     ws.getOperands())) {
        Value ptr =
            b.gep(ptrTy, captureType, capturePtr, ArrayRef<LLVM::GEPArg>{0, i});
        b.store(arg, ptr);
      }
    }

    // First barrier releases the waiting warpgroups. The second barrier ensures
    // they have read the captures before the memory is released upon entry.
    createBarrier(b, kSwitchLoopBarrierIdx, /*numThreads=*/std::nullopt,
                  /*aligned=*/false);
    createBarrier(b, kSwitchLoopBarrierIdx, /*numThreads=*/std::nullopt,
                  /*aligned=*/false);
    b.create<LLVM::BrOp>(&ws.getDefaultRegion().front());

    ws.getDefaultRegion().walk([&](WarpYieldOp op) {
      b.setInsertionPoint(op);
      createBarrier(b, kSwitchLoopBarrierIdx, /*numThreads=*/std::nullopt,
                    /*aligned=*/false);
      b.replaceOpWithNewOp<LLVM::BrOp>(op, after);
    });
    after->getParent()->getBlocks().splice(after->getIterator(),
                                           ws.getDefaultRegion().getBlocks());
    ws.erase();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertWarpSpecializeToLLVM
    : public mlir::triton::impl::ConvertWarpSpecializeToLLVMBase<
          ConvertWarpSpecializeToLLVM> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // FIXME: Assume warp specialization only happens on Blackwell.
    NVIDIA::TargetInfo targetInfo(/*computeCapability=*/100,
                                  /*ptxVersion=*/100);

    SmallVector<LLVM::LLVMFuncOp> kernels;
    for (auto func : mod.getOps<LLVM::LLVMFuncOp>()) {
      if (func.isPublic())
        kernels.push_back(func);
    }
    for (LLVM::LLVMFuncOp kernel : kernels)
      if (failed(lowerWarpSpecialize(kernel, targetInfo)))
        return signalPassFailure();
  }
};
} // namespace
