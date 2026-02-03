#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/WarpSpecializeUtility.h"
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

class NVIDIAWarpSpecializeBarrierHelper : public WarpSpecializeBarrierHelper {
public:
  NVIDIAWarpSpecializeBarrierHelper(unsigned numThreadsPerWarp)
      : numThreadsPerWarp(numThreadsPerWarp) {}

  bool isBarrierOp(Operation *op) const override {
    return isa<NVVM::Barrier0Op>(op);
  }

  Type getBarrierHandleType(MLIRContext *ctx) const override {
    return IntegerType::get(ctx, 32);
  }

  FailureOr<Value>
  getBarrierHandle(TritonLLVMIRRewriter &b,
                   std::optional<unsigned> partitionIdx) override {
    unsigned barIdx;
    if (!partitionIdx) {
      barIdx = kDefaultWarpGroupBarrierIdx;
    } else {
      barIdx = *partitionIdx + kNumReservedBarriers;
      if (barIdx >= kNumBarriers) {
        return mlir::emitError(b.getLoc(), "cannot support more than ")
               << (kNumBarriers - kNumReservedBarriers)
               << " warp group partitions";
      }
    }
    return b.i32_val(barIdx);
  }

  void createBarrier(TritonLLVMIRRewriter &b, unsigned numWarps,
                     Value handle) override {
    unsigned numThreads = numWarps * numThreadsPerWarp;
    // If a partition has only 1 warp, use `bar.warp.sync`.
    if (numThreads == 32) {
      LLVM::NVIDIA::createSyncWarp(b.getLoc(), b);
    } else {
      NVVM::BarrierOp::create(b, b.getLoc(), TypeRange{}, handle,
                              b.i32_val(numThreads), {}, Value{});
    }
  }

private:
  unsigned numThreadsPerWarp;
};

//===----------------------------------------------------------------------===//
// lowerWarpSpecialize
//===----------------------------------------------------------------------===//

static void createRegRealloc(TritonLLVMIRRewriter &b, int curRegs,
                             int adjRegs) {
  curRegs = std::min(256, curRegs);
  adjRegs = std::min(256, adjRegs);
  auto action = adjRegs < curRegs ? NVVM::SetMaxRegisterAction::decrease
                                  : NVVM::SetMaxRegisterAction::increase;
  NVVM::SetMaxRegisterOp::create(b, b.getLoc(), adjRegs, action);
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

  auto totalNumWarpsAttr =
      module->getAttrOfType<IntegerAttr>("ttg.total-num-warps");
  if (!totalNumWarpsAttr) {
    return mlir::emitError(module.getLoc(),
                           "module missing 'ttg.total-num-warps' attribute");
  }
  unsigned totalNumThreads = totalNumWarpsAttr.getInt() * threadsPerWarp;

  // Determine how many registers the worker warps can surrender before they
  // begin execution.
  auto maxnreg = func->getParentOfType<ModuleOp>()->getAttrOfType<IntegerAttr>(
      AttrMaxRegistersName);
  int lowRegs = -1;
  int defRegs = -1;
  if (maxnreg) {
    int numWorkerWarps = totalNumWarpsAttr.getInt() - defaultNumWarps;
    int startRegs = maxnreg.getInt();

    // First determine how many extra registers the default warp group can get
    // if the workers surrender the maximum number of registers.
    lowRegs = 24;
    int extraRegs = (startRegs - lowRegs) * numWorkerWarps / defaultNumWarps;
    defRegs = (startRegs + extraRegs) / 8 * 8;

    // If the default warp group goes over 256 registers, the workers don't need
    // to give up this much.
    if (defRegs > 256) {
      defRegs = 256;
      int giveRegs = (defRegs - startRegs) * defaultNumWarps / numWorkerWarps;
      lowRegs = (startRegs - giveRegs) / 8 * 8;
    }
  }

  // Attempt to elide captures of trivial computations by hoisting them into the
  // header or rematerializing them into each partition.
  elideTrivialCaptures(func, wsOps);

  MLIRContext *ctx = func.getContext();
  TritonLLVMIRRewriter b(func.getLoc(), ctx);
  Builder rewriter(ctx);

  // Generate the function header.
  Block *entry = &func.getBody().front();
  SmallVector<Location> argLocs = llvm::to_vector(llvm::map_range(
      func.getArguments(), [](BlockArgument arg) { return arg.getLoc(); }));
  Block *header = b.createBlock(entry, func.getArgumentTypes(), argLocs);
  Block *switchLoop = b.createBlock(entry);
  b.setInsertionPointToStart(header);

  // This is the absolute thread ID.
  Value tid = NVVM::ThreadIdXOp::create(b, b.getLoc(), i32_ty);
  Value wid = b.udiv(tid, b.i32_val(threadsPerWarp));
  // Tell PTXAS this value is warp-uniform.
  wid = targetInfo.shuffleIdx(b, b.getLoc(), wid, 0);
  Value isDefault = b.icmp_ult(wid, b.i32_val(defaultNumWarps));
  LLVM::CondBrOp::create(b, b.getLoc(), isDefault, entry, switchLoop);

  // Forward arguments from the header into the old entry block.
  for (auto [arg, oldArg] :
       llvm::zip(header->getArguments(), entry->getArguments()))
    oldArg.replaceAllUsesWith(arg);
  entry->eraseArguments([](auto) { return true; });
  b.setInsertionPointToStart(entry);
  if (maxnreg)
    createRegRealloc(b, maxnreg.getInt(), defRegs);

  WarpSpecializeCallbacks callbacks;
  callbacks.createAllBarrier = [](TritonLLVMIRRewriter &b, unsigned barIdx) {
    assert(barIdx < kNumBarriers && "not enough barriers");
    LLVM::createLLVMIntrinsicCallOp(
        b, b.getLoc(), "llvm.nvvm.barrier.cta.sync.all", {}, b.i32_val(barIdx));
  };

  callbacks.reallocRegisters = [&](TritonLLVMIRRewriter &b, WarpSpecializeOp ws,
                                   RegisterReallocPhase phase,
                                   unsigned regionNumber) {
    if (phase == RegisterReallocPhase::SwitchLoopStart) {
      if (maxnreg)
        createRegRealloc(b, maxnreg.getInt(), lowRegs);
      return;
    }

    if (auto actRegs = ws.getActualRegisters()) {
      switch (phase) {
      case RegisterReallocPhase::WorkerPartitionStart:
        createRegRealloc(b, lowRegs, (*actRegs)[regionNumber + 1]);
        break;
      case RegisterReallocPhase::WorkerPartitionEnd:
        createRegRealloc(b, (*actRegs)[regionNumber + 1], lowRegs);
        break;
      case RegisterReallocPhase::DefaultPartitionStart:
        createRegRealloc(b, defRegs, actRegs->front());
        break;
      case RegisterReallocPhase::DefaultPartitionEnd:
        createRegRealloc(b, actRegs->front(), defRegs);
        break;
      default:
        break;
      }
    }
  };

  // ^switchLoop:
  //   barrier.sync 1
  //   %state_ptr = getelementptr (ptr @shared), <offset>
  //   %rel_tid = sub %tid, <default_warp_group_size>
  //   %rel_wid = udiv %rel_tid, 32

  return lowerWarpSpecializeCommon(
      func, wsOps, entry, header, switchLoop, wid, ctx, defaultNumWarps,
      totalNumWarpsAttr.getInt(), targetInfo, callbacks, kSwitchLoopBarrierIdx);
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
    NVIDIA::TargetInfo targetInfo(/*computeCapability=*/100, /*ptxVersion=*/87);

    // Convert types and cleanup unrealized conversions.
    mlir::LowerToLLVMOptions option(&getContext());
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(&getContext(), option,
                                               targetInfo);
    mod.walk([&](Operation *op) {
      if (isa<WarpSpecializeOp, WarpSpecializePartitionsOp, WarpYieldOp>(op))
        convertOpTypes(op, typeConverter);
    });
    OpPassManager pm;
    pm.addPass(createReconcileUnrealizedCastsPass());
    if (failed(runPipeline(pm, mod)))
      return signalPassFailure();

    unsigned threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    NVIDIAWarpSpecializeBarrierHelper barrierHelper(threadsPerWarp);
    if (failed(lowerWarpSpecializeBarriers(mod, barrierHelper)))
      return signalPassFailure();

    SmallVector<LLVM::LLVMFuncOp> kernels;
    for (auto func : mod.getOps<LLVM::LLVMFuncOp>()) {
      if (func.getLinkage() == LLVM::Linkage::External)
        kernels.push_back(func);
    }
    for (LLVM::LLVMFuncOp kernel : kernels)
      if (failed(lowerWarpSpecialize(kernel, targetInfo)))
        return signalPassFailure();
  }
};
} // namespace
