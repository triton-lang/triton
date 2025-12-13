#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUToLLVM/TypeConverter.h"
#include "Utility.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/WarpSpecializeUtility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONAMDGPUCONVERTWARPSPECIALIZETOLLVM
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

enum BarrierIndex {
  kNullBarrierIdx,
  kDefaultWarpGroupBarrierIdx,
  kNumReservedBarriers,
  kNumBarriers = 17
};

static void createBarrier(TritonLLVMIRRewriter &b, unsigned barIdx,
                          unsigned numWarps,
                          const AMD::TargetInfo &targetInfo) {
  assert(barIdx < kNumBarriers && "not enough barriers");
  Location loc = b.getLoc();
  auto moduleOp = b.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto ctx = moduleOp.getContext();
  const std::string namedBarrierName = "nbar" + std::to_string(barIdx);
  auto nbarAttr = StringAttr::get(ctx, namedBarrierName);
  auto nbarTy = LLVM::LLVMTargetExtType::get(
      ctx, "amdgcn.named.barrier", ArrayRef<Type>{}, ArrayRef<unsigned>{0});

  LLVM::GlobalOp nbarGV;
  Operation *nbarGlobalOp = SymbolTable::lookupSymbolIn(moduleOp, nbarAttr);
  if (!nbarGlobalOp) {
    RewriterBase::InsertionGuard guard(b);
    Location uloc = UnknownLoc::get(ctx);
    b.setInsertionPointToStart(moduleOp.getBody());
    nbarGV = LLVM::GlobalOp::create(b, uloc, nbarTy,
                                    /*isConstant=*/false,
                                    LLVM::Linkage::Internal, namedBarrierName,
                                    /*value=*/Attribute(), /*alignment=*/0,
                                    targetInfo.getSharedAddressSpace());
    // Add initializer region that returns 'poison'
    Block *initBlock = b.createBlock(&nbarGV.getInitializerRegion());
    b.setInsertionPointToStart(initBlock);
    Value poison = LLVM::PoisonOp::create(b, uloc, nbarTy);
    LLVM::ReturnOp::create(b, uloc, poison);
  } else {
    nbarGV = cast<LLVM::GlobalOp>(*nbarGlobalOp);
  }

  auto nbarPtr = LLVM::AddressOfOp::create(b, loc, nbarGV);
  auto smemObj = SharedMemoryObject(nbarPtr, nbarTy, 1, loc, b);
  ROCDL::BarrierJoinOp::create(b, loc, smemObj.getBase());
  ROCDL::BarrierSignalVarOp::create(b, loc, smemObj.getBase(), numWarps);
  ROCDL::BarrierWaitOp::create(b, loc, 1);
}

static void createAllBarrier(TritonLLVMIRRewriter &b) {
  Location loc = b.getLoc();
  ROCDL::BarrierOp::create(b, loc);
}

//===----------------------------------------------------------------------===//
// lowerWarpSpecialize
//===----------------------------------------------------------------------===//

// Assign hardware barriers to each warp group and rewrite warp group barriers
// into named barrier instructions. There is a maximum number of named barriers.
static LogicalResult rewriteWarpGroupBarriers(
    LLVM::LLVMFuncOp func, ArrayRef<WarpSpecializeOp> wsOps,
    unsigned defaultNumWarps, const AMD::TargetInfo &targetInfo) {
  // HACK: Turn all `rocdl.barrier` ops into warp group barriers.
  func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Walk into default regions but not partition regions.
    if (isa<WarpSpecializePartitionsOp>(op))
      return WalkResult::skip();

    if (auto bar = dyn_cast<ROCDL::BarrierOp>(op)) {
      TritonLLVMIRRewriter b(bar.getLoc(), bar);
      createBarrier(b, kDefaultWarpGroupBarrierIdx, defaultNumWarps,
                    targetInfo);
      bar.erase();
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });

  // Each partition executes simultaneously, so each will get a different
  // barrier ID, but note this means there is a maximum of 16 barriers.
  for (WarpSpecializeOp op : wsOps) {
    for (auto partitionTuple : llvm::enumerate(op.getPartitionRegions())) {
      auto idx = partitionTuple.index();
      auto partition = partitionTuple.value();
      unsigned barIdx = idx + kNumReservedBarriers;
      if (barIdx >= kNumBarriers) {
        return func.emitError("cannot support more than ")
               << (kNumBarriers - kNumReservedBarriers)
               << " warp group partitions";
      }

      partition->walk([&](ROCDL::BarrierOp bar) {
        TritonLLVMIRRewriter b(bar.getLoc(), bar);
        unsigned partitionNumWarps = op.getPartitionNumWarps()[idx];
        createBarrier(b, barIdx, partitionNumWarps, targetInfo);
        bar.erase();
      });
    }
  }

  return success();
}

static LogicalResult lowerWarpSpecialize(LLVM::LLVMFuncOp func,
                                         const AMD::TargetInfo &targetInfo) {
  SmallVector<WarpSpecializeOp> wsOps;
  func.walk([&](WarpSpecializeOp op) { wsOps.push_back(op); });
  // Nothing to do. This kernel is not warp specialized.
  if (wsOps.empty())
    return success();

  auto module = cast<ModuleOp>(func->getParentOp());
  unsigned threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(module);
  unsigned defaultNumWarps = lookupNumWarps(func);
  if (failed(
          rewriteWarpGroupBarriers(func, wsOps, defaultNumWarps, targetInfo)))
    return failure();

  auto totalNumWarpsAttr =
      module->getAttrOfType<IntegerAttr>("ttg.total-num-warps");
  if (!totalNumWarpsAttr) {
    return mlir::emitError(module.getLoc(),
                           "module missing 'ttg.total-num-warps' attribute");
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

  // This is the absolute warp ID.
  auto warpIdOp = LLVM::createLLVMIntrinsicCallOp(
      b, b.getLoc(), "llvm.amdgcn.wave.id", {i32_ty}, ValueRange{});
  Value wid = warpIdOp.getResult(0);
  Value isDefault = b.icmp_ult(wid, b.i32_val(defaultNumWarps));
  LLVM::CondBrOp::create(b, b.getLoc(), isDefault, entry, switchLoop);

  // Forward arguments from the header into the old entry block.
  for (auto [arg, oldArg] :
       llvm::zip(header->getArguments(), entry->getArguments()))
    oldArg.replaceAllUsesWith(arg);
  entry->eraseArguments([](auto) { return true; });

  WarpSpecializeCallbacks callbacks;
  callbacks.createAllBarrier = [](TritonLLVMIRRewriter &b, unsigned) {
    createAllBarrier(b);
  };

  callbacks.reallocRegisters = [](TritonLLVMIRRewriter &, WarpSpecializeOp,
                                  RegisterReallocPhase, unsigned) {};

  return lowerWarpSpecializeCommon(
      func, wsOps, entry, header, switchLoop, wid, ctx, defaultNumWarps,
      totalNumWarpsAttr.getInt(), targetInfo, callbacks, 0);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct TritonAMDGPUConvertWarpSpecializeToLLVM
    : public mlir::triton::impl::TritonAMDGPUConvertWarpSpecializeToLLVMBase<
          TritonAMDGPUConvertWarpSpecializeToLLVM> {

  TritonAMDGPUConvertWarpSpecializeToLLVM(StringRef arch)
      : TritonAMDGPUConvertWarpSpecializeToLLVMBase<
            TritonAMDGPUConvertWarpSpecializeToLLVM>() {
    this->arch = arch;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    SmallVector<Operation *> wsOps;
    mod.walk([&](Operation *op) {
      if (isa<WarpSpecializeOp, WarpSpecializePartitionsOp, WarpYieldOp>(op))
        wsOps.push_back(op);
    });

    // If no warp specialization ops, this pass is a no-op
    if (wsOps.empty())
      return;

    // Use the arch parameter if provided, otherwise get from module
    std::string archStr = this->arch;
    if (archStr.empty()) {
      auto arch = getAMDArch(mod);
      if (!arch.has_value()) {
        mod.emitError(
            "Warp specialization requires AMD architecture to be specified");
        return signalPassFailure();
      }
      archStr = arch->str();
    }

    if (archStr != "gfx1250") {
      mod.emitError("Warp specialization is only supported on gfx1250, got ")
          << archStr;
      return signalPassFailure();
    }
    AMD::TargetInfo targetInfo(archStr.c_str());

    // Convert types and cleanup unrealized conversions.
    mlir::LowerToLLVMOptions option(&getContext());
    option.overrideIndexBitwidth(32);
    TritonAMDGPUToLLVMTypeConverter typeConverter(&getContext(), option,
                                                  targetInfo);
    for (Operation *op : wsOps) {
      convertOpTypes(op, typeConverter);
    }
    OpPassManager pm;
    pm.addPass(createReconcileUnrealizedCastsPass());
    if (failed(runPipeline(pm, mod)))
      return signalPassFailure();

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

namespace mlir::triton::AMD {

std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPUConvertWarpSpecializeToLLVMPass(StringRef arch) {
  return std::make_unique<TritonAMDGPUConvertWarpSpecializeToLLVM>(arch);
}

} // namespace mlir::triton::AMD
