#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUToLLVM/TypeConverter.h"
#include "Utility.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
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

class AMDWarpSpecializeBarrierHelper : public WarpSpecializeBarrierHelper {
public:
  AMDWarpSpecializeBarrierHelper(ModuleOp module,
                                 const AMD::TargetInfo &targetInfo)
      : module(module), targetInfo(targetInfo) {}

  bool isBarrierOp(Operation *op) const override {
    return isa<ROCDL::BarrierOp>(op);
  }

  Type getBarrierHandleType(MLIRContext *ctx) const override {
    return LLVM::LLVMPointerType::get(ctx, targetInfo.getSharedAddressSpace());
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

    auto nbarAttr = b.getStringAttr("nbar" + Twine(barIdx));
    auto nbarTy = LLVM::LLVMTargetExtType::get(b.getContext(),
                                               "amdgcn.named.barrier", {}, {0});

    LLVM::GlobalOp nbarGV;
    Operation *nbarGlobalOp = SymbolTable::lookupSymbolIn(module, nbarAttr);
    if (!nbarGlobalOp) {
      RewriterBase::InsertionGuard guard(b);
      Location uloc = b.getUnknownLoc();
      b.setInsertionPointToStart(module.getBody());
      nbarGV = LLVM::GlobalOp::create(
          b, uloc, nbarTy, /*isConstant=*/false, LLVM::Linkage::Internal,
          nbarAttr.getValue(), /*value=*/Attribute(), /*alignment=*/0,
          targetInfo.getSharedAddressSpace());
      // Add initializer region that returns 'poison'
      Block *initBlock = b.createBlock(&nbarGV.getInitializerRegion());
      b.setInsertionPointToStart(initBlock);
      Value poison = LLVM::PoisonOp::create(b, uloc, nbarTy);
      LLVM::ReturnOp::create(b, uloc, poison);
    } else {
      nbarGV = cast<LLVM::GlobalOp>(*nbarGlobalOp);
    }

    return Value(LLVM::AddressOfOp::create(b, b.getLoc(), nbarGV));
  }

  void createBarrier(TritonLLVMIRRewriter &b, unsigned numWarps,
                     Value handle) override {
    Location loc = b.getLoc();
    auto nbarTy = LLVM::LLVMTargetExtType::get(b.getContext(),
                                               "amdgcn.named.barrier", {}, {0});
    auto smemObj = SharedMemoryObject(handle, nbarTy, 1, loc, b);
    ROCDL::BarrierJoinOp::create(b, loc, smemObj.getBase());
    ROCDL::BarrierSignalVarOp::create(b, loc, smemObj.getBase(), numWarps);
    ROCDL::BarrierWaitOp::create(b, loc, 1);
  }

private:
  ModuleOp module;
  const AMD::TargetInfo &targetInfo;
};

//===----------------------------------------------------------------------===//
// lowerWarpSpecialize
//===----------------------------------------------------------------------===//

static LogicalResult lowerWarpSpecialize(LLVM::LLVMFuncOp func,
                                         const AMD::TargetInfo &targetInfo) {
  SmallVector<WarpSpecializeOp> wsOps;
  func.walk([&](WarpSpecializeOp op) { wsOps.push_back(op); });
  // Nothing to do. This kernel is not warp specialized.
  if (wsOps.empty())
    return success();

  auto module = cast<ModuleOp>(func->getParentOp());
  unsigned defaultNumWarps = lookupNumWarps(func);

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
    Location loc = b.getLoc();
    ROCDL::BarrierOp::create(b, loc);
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

    AMDWarpSpecializeBarrierHelper barrierHelper(mod, targetInfo);
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

namespace mlir::triton::AMD {

std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPUConvertWarpSpecializeToLLVMPass(StringRef arch) {
  return std::make_unique<TritonAMDGPUConvertWarpSpecializeToLLVM>(arch);
}

} // namespace mlir::triton::AMD
