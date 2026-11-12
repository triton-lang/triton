#include "Dialect/NVGPU/IR/Dialect.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
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

static constexpr const char kDisableSetMaxRegisterAttr[] =
    "tti.disable_setmaxregister";

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Reserve distinct barrier IDs for the default warp group and the whole CTA.
enum BarrierIndex {
  kDefaultWarpGroupBarrierIdx,
  kNumReservedBarriers = LLVM::NVIDIA::kWarpSpecializeCTABarrierId + 1,
  kNumBarriers = 16
};

class NVIDIAWarpSpecializeBarrierHelper : public WarpSpecializeBarrierHelper {
public:
  NVIDIAWarpSpecializeBarrierHelper(unsigned numThreadsPerWarp)
      : numThreadsPerWarp(numThreadsPerWarp) {}

  bool isBarrierOp(Operation *op) const override {
    return isa<NVVM::BarrierOp>(op);
  }

  bool isBarrierHandleOp(Operation *op) const override {
    return isa<mlir::triton::nvgpu::WarpGroupBarrierIdOp>(op);
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
      // Only this warp partition participates, so the barrier is not
      // CTA-aligned.
      NVVM::BarrierOp::create(b, b.getLoc(), handle, b.i32_val(numThreads),
                              /*aligned=*/false);
    }
  }

private:
  unsigned numThreadsPerWarp;
};

static void rewriteWarpSpecializeWarpIdsOnce(ModuleOp mod) {
  SmallVector<mlir::triton::gpu::WarpIdOp> wsWarpIds;
  mod.walk([&](mlir::triton::gpu::WarpIdOp op) {
    if (getWarpGroupStartWarpId(op->getBlock()))
      wsWarpIds.push_back(op);
  });

  for (mlir::triton::gpu::WarpIdOp op : wsWarpIds) {
    std::optional<int> startWarpId = getWarpGroupStartWarpId(op->getBlock());
    assert(startWarpId &&
           "expected warp-specialize warp_id to have a start warp ID");

    auto loc = op.getLoc();
    TritonLLVMIRRewriter b(loc, op);

    // Keep `ttg.warp_id` for NVGPUToLLVM and only make it relative here.
    Value absWarpId =
        mlir::triton::gpu::WarpIdOp::create(b, loc, op.getOmitUniformHint());
    Value relWarpId =
        LLVM::SubOp::create(b, loc, absWarpId, b.i32_val(*startWarpId));
    b.replaceOp(op, relWarpId);
  }
}

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

  // Determine how many registers the worker warps can surrender before they
  // begin execution.
  auto maxnreg = func->getParentOfType<ModuleOp>()->getAttrOfType<IntegerAttr>(
      AttrMaxRegistersName);
  // Disable dynamic register allocation if requested
  bool emitDynamicRegRealloc = llvm::none_of(wsOps, [](WarpSpecializeOp ws) {
    return ws->hasAttr(kDisableSetMaxRegisterAttr);
  });

  int lowRegs = -1;
  int defRegs = -1;
  if (emitDynamicRegRealloc && maxnreg) {
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
  bool direct = canDirectDispatch(wsOps);

  TritonLLVMIRRewriter b(func.getLoc(), func.getContext());
  // Keep the kernel arguments in the header so they dominate every warp group.
  Block *header = &func.getBody().front();
  Block *entry = b.splitBlock(header, header->begin());
  Block *switchLoop = b.createBlock(entry);
  b.setInsertionPointToStart(header);

  // This is the absolute thread ID.
  Value tid = NVVM::ThreadIdXOp::create(b, b.getLoc(), b.getI32Type());
  Value wid = b.udiv(tid, b.i32_val(threadsPerWarp));
  // Tell PTXAS this value is warp-uniform.
  wid = targetInfo.shuffleIdx(b, b.getLoc(), wid, 0);
  Value isDefault = b.icmp_ult(wid, b.i32_val(defaultNumWarps));
  LLVM::CondBrOp::create(b, b.getLoc(), isDefault, entry, switchLoop);

  b.setInsertionPointToStart(entry);
  if (emitDynamicRegRealloc && maxnreg)
    createRegRealloc(b, maxnreg.getInt(), defRegs);

  WarpSpecializeCallbacks callbacks;
  if (auto regs = wsOps.front().getActualRegisters();
      direct && emitDynamicRegRealloc && regs) {
    // AllocateWarpGroups assigns a single limit to each hardware warpgroup.
    llvm::DenseMap<int, int> representatives;
    representatives.try_emplace(lowRegs, -1);
    for (auto [i, cap] : llvm::enumerate(regs->drop_front()))
      callbacks.workerRegisterPartitions.push_back(
          representatives.try_emplace(std::min(256, cap), i).first->second);
  }
  callbacks.createAllBarrier = [](TritonLLVMIRRewriter &b, unsigned barIdx) {
    assert(barIdx < kNumBarriers && "not enough barriers");
    NVVM::BarrierOp::create(b, b.getLoc(), b.i32_val(barIdx), Value{},
                            /*aligned=*/false);
  };

  callbacks.reallocRegisters = [&](TritonLLVMIRRewriter &b, WarpSpecializeOp ws,
                                   RegisterReallocPhase phase,
                                   unsigned regionNumber) {
    if (!emitDynamicRegRealloc)
      return;

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

  return lowerWarpSpecializeCommon(func, wsOps, direct, switchLoop, wid,
                                   defaultNumWarps, totalNumWarpsAttr.getInt(),
                                   targetInfo, callbacks,
                                   LLVM::NVIDIA::kWarpSpecializeCTABarrierId);
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

    rewriteWarpSpecializeWarpIdsOnce(mod);

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
