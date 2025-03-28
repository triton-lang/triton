#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

//===----------------------------------------------------------------------===//
// relayoutWarps
//===----------------------------------------------------------------------===//

using RunPipelineFn = function_ref<LogicalResult(OpPassManager &, ModuleOp)>;

// Take the body of a partition into a new `tt.func`. We can use this to run a
// full compiler pipeline on the partition.
static OwningOpRef<ModuleOp> takeIntoFunction(ModuleAxisInfoAnalysis &axisInfo,
                                              Region *partition,
                                              unsigned numWarps) {
  // Forward the module attributes (target, number of threads per warp, etc.)
  // onto the container module.
  ModuleOp mod = axisInfo.getModuleOp();
  OwningOpRef<ModuleOp> container = ModuleOp::create(mod.getLoc());
  Block *containerBlock = container->getBody();

  auto b = OpBuilder::atBlockBegin(containerBlock);
  FunctionType funcType = b.getFunctionType(partition->getArgumentTypes(), {});
  auto containerFunc = b.create<FuncOp>(mod.getLoc(), "container", funcType);
  containerFunc.getBody().takeBody(*partition);
  container.get()->setAttrs(mod->getAttrs());
  container.get()->setAttr(AttrNumWarpsName, b.getI32IntegerAttr(numWarps));

  // Replace `ttg.warp_return` with `tt.return` to make the IR valid.
  containerFunc.walk([&](WarpReturnOp op) {
    b.setInsertionPoint(op);
    b.create<ReturnOp>(op.getLoc());
    op.erase();
  });

  // This should make valid IR.
  if (failed(mlir::verify(*container)))
    llvm::report_fatal_error("expected partition region to make valid IR");

  // Attach axis info properties.
  auto wsOp = partition->getParentOfType<WarpSpecializeOp>();
  auto *funcInfo =
      axisInfo.getFuncData(wsOp->getParentOfType<FunctionOpInterface>());
  assert(funcInfo && "expected to find function axis info");
  for (auto [i, capture] : llvm::enumerate(wsOp.getExplicitCaptures())) {
    AxisInfo info = funcInfo->lookup(capture);
    containerFunc.setArgAttr(i, "tt.contiguity",
                             b.getI64IntegerAttr(info.getContiguity(0)));
    containerFunc.setArgAttr(i, "tt.divisibility",
                             b.getI64IntegerAttr(info.getDivisibility(0)));
    containerFunc.setArgAttr(i, "tt.constancy",
                             b.getI64IntegerAttr(info.getConstancy(0)));
  }

  return container;
}

// Take the partition body out of the container module and function.
static void extractPartitionBody(OwningOpRef<ModuleOp> container,
                                 Region *partition) {
  auto containerFunc = cast<FuncOp>(container->lookupSymbol("container"));

  // Rewrite the returns.
  containerFunc.walk([](ReturnOp op) {
    OpBuilder b(op);
    b.create<WarpReturnOp>(op.getLoc());
    op.erase();
  });

  partition->takeBody(containerFunc.getBody());
}

// Reset the layouts of operations in a region and re-run layout assignment.
static LogicalResult relayoutWarps(ModuleAxisInfoAnalysis &axisInfo,
                                   Region *partition, unsigned prevNumWarps,
                                   unsigned newNumWarps,
                                   RunPipelineFn runPipeline) {
  OwningOpRef<ModuleOp> container =
      takeIntoFunction(axisInfo, partition, prevNumWarps);

  // Start by removing all tensor encodings.
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([](RankedTensorType ty) {
    return RankedTensorType::get(ty.getShape(), ty.getElementType());
  });
  // But don't remove them from the tensors inside descriptors.
  replacer.addReplacement([](TensorDescType ty) -> std::pair<Type, WalkResult> {
    return {ty, WalkResult::skip()};
  });
  replacer.recursivelyReplaceElementsIn(*container, /*replaceAttrs=*/false,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);

  ModuleOp mod = axisInfo.getModuleOp();
  auto target = mod->getAttrOfType<StringAttr>(AttrTargetName);
  if (!target)
    return mlir::emitError(mod.getLoc(), "module missing target specification");
  int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
  int numCTAs = TritonGPUDialect::getNumCTAs(mod);

  // Enable `convert-triton-to-tritongpu` to rematerialize source layouts for
  // TTG dialect operations. They will get cleared later.
  OpPassManager pm;
  pm.addPass(createConvertTritonToTritonGPUPass(target.str(), newNumWarps,
                                                threadsPerWarp, numCTAs,
                                                /*enableSourceRemat=*/true));
  if (failed(runPipeline(pm, *container)))
    return failure();
  // Clear source rematerializations by propagating the source layout.
  container->walk([](UnrealizedConversionCastOp op) {
    op.getResult(0).replaceAllUsesWith(op.getOperand(0));
    op.erase();
  });

  pm.clear();
  pm.addPass(createTritonGPUCoalesce());
  pm.addPass(createTritonGPURemoveLayoutConversions());
  pm.addPass(createTritonGPUOptimizeThreadLocality());
  pm.addPass(createTritonGPURemoveLayoutConversions());
  if (failed(runPipeline(pm, *container)))
    return failure();

  extractPartitionBody(std::move(container), partition);
  return success();
}

//===----------------------------------------------------------------------===//
// optimizePartitionWarps
//===----------------------------------------------------------------------===//

// Get the number of i32 registers required to store a tensor, assuming no
// redundancy in the layout.
static unsigned getTensorNumI32RegsPerWarp(RankedTensorType ty) {
  unsigned numElems = getTotalElemsPerThread(ty) *
                      product(getThreadsPerWarp(ty)) *
                      product(getWarpsPerCTA(ty));
  unsigned elSize =
      isa<PointerType>(ty.getElementType()) ? 64 : ty.getElementTypeBitWidth();
  return numElems * elSize / 32;
}

static LogicalResult optimizePartitionNumWarps(ModuleAxisInfoAnalysis &axisInfo,
                                               WarpSpecializeOp wsOp,
                                               RunPipelineFn runPipeline) {
  // Extremely rough estimate of the number of registers needed per partition.
  // For each partition, get the number of i32 registers used by the largest
  // tensor value.
  //
  // Because the partition region is isolated from above, we could in theory
  // compile it to PTX and read the number of registers that got allocated.
  SmallVector<unsigned> maxTensorRegs;
  for (Region *partition : wsOp.getPartitionRegions()) {
    unsigned &tensorRegs = maxTensorRegs.emplace_back(0);
    partition->walk([&](Operation *op) {
      for (Type type :
           llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
        if (auto tensor = dyn_cast<RankedTensorType>(type))
          tensorRegs = std::max(tensorRegs, getTensorNumI32RegsPerWarp(tensor));
      }
    });
    // Assume that the largest tensor accounts for half of the registers used
    // by a warpgroup.
    tensorRegs *= 2;
  }

  // Reduce the number of warps used by partitions. For partitions with no
  // tensor computations, always reduce them to 1 warp.
  //
  // We can't use `nvvm.setmaxnreg` because this requires a known value for
  // `maxnreg` on the kernel, which is currently controlled by the frontend.
  // Thus, assume PTXAS will evenly distribute the total pool of registers
  // across all warps.
  //
  // If the compiler could control that, then we could allow non-uniform
  // register distributions, mostly beneficial for single-warp warpgroups that
  // just do some artihmetic.
  constexpr unsigned nTotalRegs = 65536; // for Blackwell SMs
  const unsigned threadsPerWarp =
      TritonGPUDialect::getThreadsPerWarp(axisInfo.getModuleOp());
  const unsigned defaultNumWarps = lookupNumWarps(wsOp);

  SmallVector<int32_t> partitionNumWarps =
      llvm::to_vector(wsOp.getPartitionNumWarps());

  // Figure out the number of registers per thread and then how many warps each
  // partition needs. If this reduces the number of warps, it increases the
  // number of threads per warp, which can further reduce warps, etc. Iterate
  // until this converges.
  //
  // The algorithm is generally conservative and won't drop the number of warps
  // unless there is big gap between the number of registers used and needed,
  // since the estimate is quite rough.
  bool changed;
  do {
    changed = false;

    int32_t totalNumWarps = std::accumulate(
        partitionNumWarps.begin(), partitionNumWarps.end(), defaultNumWarps);
    // Given the number of remaining registers per thread, figure out how many
    // warps are needed to capture the tensor regs. Assume each thread uses a
    // minimum number of registers for other things.
    unsigned regsPerThread = nTotalRegs / totalNumWarps / threadsPerWarp;
    for (auto [numWarps, tensorRegs] :
         llvm::zip(partitionNumWarps, maxTensorRegs)) {
      int32_t nextNumWarps = 1;
      if (tensorRegs) {
        unsigned fittedWarps = llvm::PowerOf2Ceil(llvm::divideCeil(
            llvm::divideCeil(tensorRegs, regsPerThread), threadsPerWarp));
        // Never increase the number of warps.
        nextNumWarps = std::min<int32_t>(numWarps, fittedWarps);
        assert(nextNumWarps > 0);
      }
      changed |= (nextNumWarps != numWarps);
      numWarps = nextNumWarps;
    }
  } while (changed);

  for (auto [partition, newNumWarps, prevNumWarps, tensorRegs] :
       llvm::zip(wsOp.getPartitionRegions(), partitionNumWarps,
                 wsOp.getPartitionNumWarps(), maxTensorRegs)) {
    if (newNumWarps == prevNumWarps || !tensorRegs)
      continue;
    // We need to reassign layouts.
    if (failed(relayoutWarps(axisInfo, partition, prevNumWarps, newNumWarps,
                             runPipeline)))
      return failure();
  }
  wsOp.setPartitionNumWarps(partitionNumWarps);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUOPTIMIZEPARTITIONWARPS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct OptimizePartitionWarps
    : triton::gpu::impl::TritonGPUOptimizePartitionWarpsBase<
          OptimizePartitionWarps> {
  using TritonGPUOptimizePartitionWarpsBase::
      TritonGPUOptimizePartitionWarpsBase;

  void runOnOperation() override;
};
} // namespace

void OptimizePartitionWarps::runOnOperation() {
  ModuleAxisInfoAnalysis axisInfo(getOperation());
  auto runPipelineFn = [&](OpPassManager &pm, ModuleOp container) {
    getOperation().push_back(container);
    auto remove = llvm::make_scope_exit([&] { container->remove(); });
    return runPipeline(pm, container);
  };
  WalkResult result = getOperation().walk([&](WarpSpecializeOp wsOp) {
    if (failed(optimizePartitionNumWarps(axisInfo, wsOp, runPipelineFn)))
      return WalkResult::interrupt();
    return WalkResult::skip();
  });
  if (result.wasInterrupted())
    return signalPassFailure();
}
