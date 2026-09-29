#include <memory>

#include "Allocation.h"
#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonInstrument/IR/ConSanConstants.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ALLOCATESHAREDMEMORYNV
#define GEN_PASS_DEF_SETMINIMUMSHAREDMEMORY
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {
// Track the physical context and the affine CTA offset separately: subslices
// compose by XOR within their allocation's context, while nested contexts have
// an additive physical start. This analysis runs after device-function
// inlining.
static FailureOr<std::pair<int32_t, int32_t>> getCTAOrigin(Value value) {
  namespace ttg = triton::gpu;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (auto split =
            dyn_cast<ttg::CTASpecializeOp>(arg.getOwner()->getParentOp())) {
      if (arg.getOwner()->isEntryBlock())
        return getCTAOrigin(split.getExplicitCaptures()[arg.getArgNumber()]);
    }
    return failure();
  }
  Operation *op = value.getDefiningOp();
  if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op))
    return std::make_pair(ttg::lookupCTAStart(alloc), int32_t(0));
  if (auto rebase = dyn_cast<ttg::MemDescCTARebaseOp>(op))
    return std::make_pair(int32_t(rebase.getCtaStart()), int32_t(0));
  if (auto slice = dyn_cast<ttg::MemDescSubsliceOp>(op)) {
    auto origin = getCTAOrigin(slice.getSrc());
    if (failed(origin))
      return failure();
    auto srcTy = slice.getSrc().getType();
    auto offsets =
        ttg::dropPipeliningDim(slice.getOffsets(), srcTy.getEncoding());
    auto dims = standardOutDimNames(op->getContext(), offsets.size());
    SmallVector<std::pair<StringAttr, int32_t>> coordinates;
    for (auto [dim, offset] : llvm::zip(dims, offsets))
      coordinates.emplace_back(dim, offset);
    auto inverse = ttg::toLinearLayoutIgnoringPadding(srcTy).pseudoinvert();
    auto block = StringAttr::get(op->getContext(), "block");
    for (auto [dim, offset] : inverse.apply(coordinates))
      if (dim == block)
        origin->second ^= offset;
    return origin;
  }
  if (isa<ttg::MemDescIndexOp, ttg::MemDescTransOp, ttg::MemDescReshapeOp,
          ttg::MemDescReinterpretOp>(op))
    return getCTAOrigin(op->getOperand(0));
  return failure();
}

static LogicalResult verifyCTACapture(triton::gpu::MemDescCTARebaseOp op) {
  namespace ttg = triton::gpu;
  auto origin = getCTAOrigin(op.getSrc());
  if (failed(origin))
    return op.emitOpError("cannot prove the shared-memory slice's CTA owners");
  auto srcTy = op.getSrc().getType();
  auto ll = ttg::toLinearLayoutIgnoringPadding(srcTy);
  auto inverse = ll.pseudoinvert();
  auto block = StringAttr::get(op.getContext(), "block");
  auto shape = ttg::dropPipeliningDim(srcTy.getShape(), srcTy.getEncoding());
  auto dims = standardOutDimNames(op.getContext(), shape.size());
  llvm::SmallSetVector<int32_t, 16> owners;
  owners.insert(0);
  auto addBasis = [&](int32_t basis) {
    for (int32_t owner : llvm::to_vector(owners))
      owners.insert(owner ^ basis);
  };
  for (auto [dim, size] : llvm::zip(dims, shape))
    for (int bit = 0; (1 << bit) < size; ++bit)
      addBasis(inverse.getBasis(dim, bit, block));
  // Replicated copies also belong to the view; a logical slice cannot select
  // only some of the CTAs on which the same elements are replicated.
  unsigned freeBlocks = ll.getFreeVariableMasks().lookup(block);
  for (int bit = 0; (1u << bit) <= freeBlocks; ++bit)
    if (freeBlocks & (1u << bit))
      addBasis(1 << bit);
  int count = ttg::getNumCTAs(op.getType().getEncoding());
  if (owners.size() != count || llvm::any_of(owners, [&](int32_t owner) {
        int physical = origin->first + (origin->second ^ owner);
        return owner >= count || physical < op.getCtaStart() ||
               physical >= op.getCtaStart() + count;
      }))
    return op.emitOpError(
        "shared-memory slice must reside on exactly the receiving CTA range");
  return success();
}

// Run after inlining, while all execution regions and source operations are
// present. Keep prototype restrictions here, out of individual lowerings.
static LogicalResult prepareCTASpecialization(ModuleOp mod, int capability) {
  SmallVector<triton::gpu::CTASpecializeOp> splits;
  mod.walk([&](triton::gpu::CTASpecializeOp op) { splits.push_back(op); });
  if (splits.empty())
    return success();
  if (capability < 90)
    return splits.front().emitOpError("requires NVIDIA Hopper or newer");
  auto captures = mod.walk([](triton::gpu::MemDescCTARebaseOp op) {
    return failed(verifyCTACapture(op)) ? WalkResult::interrupt()
                                        : WalkResult::advance();
  });
  if (captures.wasInterrupted())
    return failure();
  auto invalid = mod.walk([](Operation *op) {
    if (isa<triton::gpu::WarpSpecializeOp>(op)) {
      op->emitOpError(
          "mixing warp and CTA specialization is not yet supported");
      return WalkResult::interrupt();
    }
    if (!op->getParentOfType<triton::gpu::CTASpecializeOp>())
      return WalkResult::advance();
    StringRef name = op->getName().getStringRef();
    if (isa<CallOpInterface>(op) || name.contains("async") ||
        name.starts_with("tti.") ||
        (name.starts_with("ttng.") &&
         !isa<nvidia_gpu::ClusterBarrierOp, nvidia_gpu::PackedArithOp>(op))) {
      op->emitOpError(
          "operation is not yet supported inside CTA specialization; "
          "use inline functions and synchronous operations");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (invalid.wasInterrupted())
    return failure();
  for (triton::gpu::CTASpecializeOp op : splits) {
    int start = triton::gpu::lookupCTAStart(op);
    for (auto [region, count] :
         llvm::zip(op.getPartitionRegions(), op.getPartitionNumCTAs())) {
      for (auto [arg, capture] :
           llvm::zip(region.getArguments(), op.getExplicitCaptures())) {
        if (arg.use_empty() || !isa<triton::gpu::MemDescType>(arg.getType()))
          continue;
        auto rebase = capture.getDefiningOp<triton::gpu::MemDescCTARebaseOp>();
        if (!rebase || rebase.getCtaStart() != start)
          return op.emitOpError("shared-memory capture must be rebased to the "
                                "receiving CTA range");
      }
      start += count;
    }
    auto fn = op->getParentOfType<FunctionOpInterface>();
    if (!triton::isKernel(fn))
      return op.emitOpError(
          "CTA specialization requires inlining into the kernel");
    OpBuilder b(op);
    auto synchronize = [&] {
      if (triton::gpu::lookupNumCTAs(op) == 1)
        triton::gpu::BarrierOp::create(b, op.getLoc(),
                                       triton::gpu::AddrSpace::All);
      else
        nvidia_gpu::ClusterBarrierOp::create(b, op.getLoc());
    };
    synchronize();
    b.setInsertionPointAfter(op);
    synchronize();
  }
  return success();
}

struct AllocateSharedMemoryNv
    : public mlir::triton::impl::AllocateSharedMemoryNvBase<
          AllocateSharedMemoryNv> {
  using AllocateSharedMemoryNvBase::AllocateSharedMemoryNvBase;

  AllocateSharedMemoryNv(int32_t computeCapability, int32_t ptxVersion)
      : AllocateSharedMemoryNvBase({computeCapability, ptxVersion}) {}

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (failed(prepareCTASpecialization(mod, computeCapability)))
      return signalPassFailure();
    mlir::triton::NVIDIA::TargetInfo targetInfo(computeCapability, ptxVersion);
    ModuleAllocation allocation(
        mod, mlir::triton::nvidia_gpu::getNvidiaAllocationAnalysisScratchSizeFn(
                 targetInfo));
    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};

struct SetMinimumSharedMemory
    : public mlir::triton::impl::SetMinimumSharedMemoryBase<
          SetMinimumSharedMemory> {
  using SetMinimumSharedMemoryBase::SetMinimumSharedMemoryBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (minimumSize < 0) {
      mod.emitError("minimum shared memory size must be non-negative");
      return signalPassFailure();
    }
    auto sharedAttr = mod->getAttrOfType<IntegerAttr>("ttg.shared");
    int64_t sharedSize = sharedAttr ? sharedAttr.getInt() : 0;
    if (sharedSize < minimumSize)
      mod->setAttr("ttg.shared",
                   IntegerAttr::get(IntegerType::get(mod.getContext(), 32),
                                    minimumSize));
  }
};
} // namespace

namespace mlir::triton::nvidia_gpu {

static unsigned getNumScratchElemsSwizzledCvt(RankedTensorType srcTy,
                                              RankedTensorType dstTy,
                                              TargetInfoBase &targetInfo) {
  auto *ctx = srcTy.getContext();
  auto srcLayout = triton::gpu::toLinearLayout(srcTy);
  auto dstLayout = triton::gpu::toLinearLayout(dstTy);
  srcLayout = actionRemoveBroadcastedRegs(srcLayout).apply(srcLayout);
  dstLayout = actionRemoveBroadcastedRegs(dstLayout).apply(dstLayout);
  auto bitwidth = getBitwidth(srcTy);
  auto kBlock = StringAttr::get(ctx, "block");
  bool crossCTA =
      !dstLayout.invertAndCompose(srcLayout).isTrivialOver({kBlock});
  auto [srcTiles, dstTiles] =
      gpu::getSrcDstTiles(targetInfo, bitwidth, crossCTA);
  auto [smem, _] = triton::gpu::optimalSwizzling(srcLayout, dstLayout, srcTiles,
                                                 dstTiles, bitwidth);
  auto reps = smem.getInDimSize(StringAttr::get(ctx, "reps"));
  // The smem has the same CGA layout as srcLayout, so use that instead.
  // Remove the number of elements duplicated in the CGA layout.
  auto nBlocks = product(triton::gpu::getCTASplitNum(srcTy.getEncoding()));
  return smem.getTotalOutDimSize() / (reps * nBlocks);
}

std::function<unsigned(Operation *)>
getNvidiaAllocationAnalysisScratchSizeFn(TargetInfoBase &targetInfo) {
  auto allocation = [&targetInfo](Operation *op) -> unsigned {
    if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      auto srcTy = cvtOp.getSrc().getType();
      auto dstTy = cvtOp.getType();
      if (!cvtNeedsSharedMemory(cvtOp))
        return 0;
      // In cuda we always swizzle
      auto elems = getNumScratchElemsSwizzledCvt(srcTy, dstTy, targetInfo);
      return elems * getBitwidth(srcTy) / 8;
    }
    if (auto ws = dyn_cast<triton::gpu::WarpSpecializeOp>(op)) {
      unsigned captureSize = defaultAllocationAnalysisScratchSizeFn(op);
      // ConSan adds captures after allocation; reserve space pre-computed by
      // the common TritonInstrumentPrepareConSanCaptures pass.
      if (auto extra = ws->getAttrOfType<IntegerAttr>(
              mlir::triton::instrument::kConSanExtraCaptureBytesAttr))
        captureSize += extra.getInt();
      return captureSize;
    }
    return defaultAllocationAnalysisScratchSizeFn(op);
  };
  return allocation;
}
} // namespace mlir::triton::nvidia_gpu

namespace mlir::triton {
std::unique_ptr<OperationPass<ModuleOp>>
createAllocateSharedMemoryNvPass(int32_t computeCapability,
                                 int32_t ptxVersion) {
  return std::make_unique<AllocateSharedMemoryNv>(computeCapability,
                                                  ptxVersion);
}
} // namespace mlir::triton
