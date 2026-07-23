#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUToLLVM/TypeConverter.h"
#include "Utility.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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
    return isa<ROCDL::BarrierOp, ROCDL::SBarrierOp>(op);
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
  Value wid = ROCDL::WaveId::create(b, b.getLoc(), i32_ty);
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
// lowerWarpPredicate
//===----------------------------------------------------------------------===//

// Lower a single `ttg.warp_predicate` to a divergent branch. By the time this
// runs (right after convert-to-llvm), the region body is already LLVM-dialect
// and every value crossing the op boundary is bridged by an
// `unrealized_conversion_cast` to/from the tensor type the op still carries.
// We follow those casts to the underlying LLVM structs and rebuild:
//
//   curBlock:
//     %p0.. = extractvalue <predicate struct>          ; per-lane predicate
//     %lane = or %p0, %p1, ...                          ; any element set?
//     cf.cond_br %lane, body, merge(<init structs>)
//   body:                       ; runs with exec = lanes that need it
//     ... (already-lowered body) ...
//     cf.br merge(<yield structs>)
//   merge(%r0.. : structs):     ; phi: yields (true) vs inits (false)
//     ...
//
// A per-lane `cf.cond_br` is turned by the AMDGPU backend into
// `s_and_saveexec` + `s_cbranch_execz`, i.e. the whole wavefront skips `body`
// when no lane has the predicate set -- the per-wave skip we want.
static LogicalResult lowerOneWarpPredicate(WarpPredicateOp op) {
  Location loc = op.getLoc();
  IRRewriter rw(op->getContext());
  rw.setInsertionPoint(op);

  // Follow a 1:1 unrealized_conversion_cast to its source value.
  auto asLLVM = [](Value v) -> Value {
    if (auto c = v.getDefiningOp<UnrealizedConversionCastOp>())
      if (c.getNumOperands() == 1)
        return c.getOperand(0);
    return v;
  };

  // Per-lane predicate: OR together the i1 elements this lane holds.
  Value predStruct = asLLVM(op.getPredicate());
  SmallVector<Value> predElems = unpackLLElements(loc, predStruct, rw);
  if (predElems.empty())
    return op.emitError("warp_predicate: empty predicate");
  Value lanePred;
  for (Value e : predElems)
    lanePred = lanePred ? LLVM::OrOp::create(rw, loc, lanePred, e).getResult()
                        : e;

  // True-path (yield) and false-path (init) values, as LLVM structs.
  auto yield = cast<PredicateYieldOp>(op.getRegion().front().getTerminator());
  SmallVector<Value> initStructs, yieldStructs;
  SmallVector<Type> resTypes;
  for (Value in : op.getInits())
    initStructs.push_back(asLLVM(in));
  for (Value y : yield.getValues()) {
    Value s = asLLVM(y);
    yieldStructs.push_back(s);
    resTypes.push_back(s.getType());
  }

  // Split off everything from `op` onward into the merge block, which receives
  // the results as block arguments (the phi).
  Block *curBlock = rw.getInsertionBlock();
  Block *mergeBlock = rw.splitBlock(curBlock, Block::iterator(op));
  SmallVector<Location> argLocs(resTypes.size(), loc);
  SmallVector<Value> mergeArgs;
  for (auto [t, l] : llvm::zip(resTypes, argLocs))
    mergeArgs.push_back(mergeBlock->addArgument(t, l));

  // Inline the (already-lowered) body between curBlock and mergeBlock.
  Block *bodyBlock = &op.getRegion().front();
  rw.inlineRegionBefore(op.getRegion(), mergeBlock);

  // predicate_yield -> branch to merge with the yielded structs.
  rw.setInsertionPoint(yield);
  cf::BranchOp::create(rw, loc, mergeBlock, yieldStructs);
  rw.eraseOp(yield);

  // Rewire result uses. Downstream consumes the results through
  // tensor->struct casts; point those at the merge block args instead.
  for (auto [res, arg] : llvm::zip(op.getResults(), mergeArgs)) {
    for (OpOperand &use : llvm::make_early_inc_range(res.getUses())) {
      auto c = dyn_cast<UnrealizedConversionCastOp>(use.getOwner());
      if (!c || c.getNumResults() != 1)
        return op.emitError("warp_predicate: unexpected use of result");
      rw.replaceAllUsesWith(c.getResult(0), arg);
      rw.eraseOp(c);
    }
  }
  rw.eraseOp(op);

  // Divergent branch: true lanes run the body, false lanes carry inits.
  rw.setInsertionPointToEnd(curBlock);
  cf::CondBranchOp::create(rw, loc, lanePred, bodyBlock, ValueRange{},
                           mergeBlock, ValueRange(initStructs));
  return success();
}

static LogicalResult lowerWarpPredicateOps(ModuleOp mod) {
  SmallVector<WarpPredicateOp> ops;
  mod.walk([&](WarpPredicateOp op) { ops.push_back(op); });
  for (WarpPredicateOp op : ops)
    if (failed(lowerOneWarpPredicate(op)))
      return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct TritonAMDGPUConvertWarpSpecializeToLLVM
    : public mlir::triton::impl::TritonAMDGPUConvertWarpSpecializeToLLVMBase<
          TritonAMDGPUConvertWarpSpecializeToLLVM> {

  TritonAMDGPUConvertWarpSpecializeToLLVM(StringRef gfxArch)
      : TritonAMDGPUConvertWarpSpecializeToLLVMBase<
            TritonAMDGPUConvertWarpSpecializeToLLVM>() {
    this->gfxArch = gfxArch;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cf::ControlFlowDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Lower ttg.warp_predicate to divergent control flow. Independent of warp
    // specialization and supported on all AMD targets.
    if (failed(lowerWarpPredicateOps(mod)))
      return signalPassFailure();

    SmallVector<Operation *> wsOps;
    mod.walk([&](Operation *op) {
      if (isa<WarpSpecializeOp, WarpSpecializePartitionsOp, WarpYieldOp>(op))
        wsOps.push_back(op);
    });

    // If no warp specialization ops, this pass is a no-op
    if (wsOps.empty())
      return;

    // Use the arch parameter if provided, otherwise get from module
    std::string archStr = this->gfxArch;
    if (archStr.empty()) {
      auto arch = getAMDArch(mod);
      if (!arch.has_value()) {
        mod.emitError(
            "Warp specialization requires AMD architecture to be specified");
        return signalPassFailure();
      }
      archStr = arch->str();
    }

    AMD::TargetInfo targetInfo(archStr.c_str());
    if (targetInfo.getISAFamily() != triton::amdgpu::ISAFamily::GFX1250) {
      mod.emitError("Warp specialization is only supported on gfx1250, got ")
          << archStr;
      return signalPassFailure();
    }

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
createTritonAMDGPUConvertWarpSpecializeToLLVMPass(StringRef gfxArch) {
  return std::make_unique<TritonAMDGPUConvertWarpSpecializeToLLVM>(gfxArch);
}

} // namespace mlir::triton::AMD
