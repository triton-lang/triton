#include "MembarCFG.h"
#include "triton/Analysis/MembarPlacement.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::nvidia_gpu;

namespace mlir::triton::membar {

TMemModel::TMemModel(ModuleOp module, BufferRegionAnalysis &regions)
    : regions(regions) {
  module.walk([&](TMEMAllocOp alloc) {
    Value root = alloc.getResult();
    // A noinline function can execute in more than one warp role.
    Region *owner = isKernel(alloc->getParentOfType<FunctionOpInterface>()) &&
                            hasRegionLocalUses(root)
                        ? alloc->getParentRegion()
                        : nullptr;
    allocations.emplace_back(regions.getFootprint(root), owner);
  });
}

bool TMemModel::isPrivate(const BufferRegionFootprint *footprint,
                          Region *issuer) const {
  // Different descriptor roots can reuse the same physical TMEM words.
  return footprint && llvm::none_of(allocations, [&](const auto &allocation) {
           auto [other, owner] = allocation;
           return owner != issuer && mayOverlap(footprint, other);
         });
}

struct TMemModel::Info {
  // (row, word column) -> warp.
  LinearLayout warps;
  const BufferRegionFootprint *footprint;

  uint32_t getWarp(uint32_t address) const {
    auto dims = llvm::to_vector<2>(warps.getInDimNames());
    return warps
        .apply({{dims[0], int32_t(address >> 16)},
                {dims[1], int32_t(address & 0xffff)}})
        .front()
        .second;
  }
};

TMemModel::~TMemModel() = default;

bool TMemModel::isSameWarp(Operation *lhsOp, Operation *rhsOp) {
  const Info *lhs = getInfo(lhsOp);
  const Info *rhs = getInfo(rhsOp);
  if (!lhs || !rhs)
    return false;
  for (const auto &a : lhs->footprint->regionInfo.views)
    for (const auto &b : rhs->footprint->regionInfo.views) {
      assert(a.allocationFrame && a.allocationFrame == b.allocationFrame &&
             "TMEM allocations share a module frame");
      assert(!a.affineCTAOffset && !b.affineCTAOffset &&
             "TMEM views do not shift CTA coordinates");
      if (a.region.baseOffset == b.region.baseOffset &&
          lhs->warps == rhs->warps)
        continue;
      for (const auto &[cta, addresses] : a.region.ctaAddresses)
        for (const auto &[otherCTA, otherAddresses] : b.region.ctaAddresses) {
          if (cta != otherCTA)
            continue;
          auto overlap = addresses.intersection(otherAddresses);
          // Descriptor offsets use integer addition, not layout XOR.
          for (uint32_t address : overlap) {
            if (lhs->getWarp(address - a.region.baseOffset) !=
                rhs->getWarp(address - b.region.baseOffset))
              return false;
          }
        }
    }
  return true;
}

static LinearLayout getWarpLayout(RankedTensorType regTy,
                                  gpu::MemDescType memTy) {
  auto layout = computeTMemLdStLayout(regTy, memTy);
  assert(succeeded(layout) && "TMEM layout must have been verified");
  auto *ctx = regTy.getContext();
  auto warp = StringAttr::get(ctx, "warp");
  auto row = StringAttr::get(ctx, "row");
  auto col = StringAttr::get(ctx, "col");

  // Match the footprint's 32-bit words. Packed elements may share a word.
  unsigned bitwidth = memTy.getElementTypeBitWidth();
  unsigned elementsPerWord = 32 / bitwidth;
  auto rows = layout->getOutDimSize(row);
  auto cols = layout->getOutDimSize(col) / elementsPerWord;
  auto toWords = LinearLayout::identity1D(rows, row, row) *
                 LinearLayout::zeros1D(elementsPerWord, col, col) *
                 LinearLayout::identity1D(cols, col, col);
  auto accesses = layout->compose(toWords);

  // Complete the image after the real inputs so the inverse prefers them.
  auto padding = StringAttr::get(ctx, "padding");
  auto completion = LinearLayout::identity1D(rows, padding, row) *
                    LinearLayout::identity1D(cols, padding, col);
  return accesses.concatIns(completion)
      .pseudoinvert()
      .sublayout({row, col}, {warp});
}

const TMemModel::Info *TMemModel::getInfo(Operation *op) {
  auto [it, inserted] = info.try_emplace(op);
  if (!inserted)
    return it->second.get();
  RankedTensorType regTy;
  Value mem;
  if (auto load = dyn_cast<TMEMLoadOp>(op)) {
    regTy = load.getType();
    mem = load.getSrc();
  } else if (auto store = dyn_cast<TMEMStoreOp>(op)) {
    regTy = store.getSrc().getType();
    mem = store.getDst();
  } else if (auto alloc = dyn_cast<TMEMAllocOp>(op)) {
    regTy = alloc.getSrc().getType();
    mem = alloc.getResult();
  } else {
    return nullptr;
  }
  auto *footprint = regions.getFootprint(mem);
  if (!footprint)
    return nullptr;
  auto warps = getWarpLayout(regTy, cast<gpu::MemDescType>(mem.getType()));
  // A non-surjective inverse hides other warps accessing the same word.
  // TMEM waits only order accesses within their issuing warp.
  if (!warps.isSurjective())
    return nullptr;
  it->second = std::make_unique<Info>(Info{std::move(warps), footprint});
  return it->second.get();
}

namespace {

static bool isTensorMemory(Value value) {
  auto memDescType = dyn_cast<gpu::MemDescType>(value.getType());
  return memDescType &&
         isa<TensorMemorySpaceAttr>(memDescType.getMemorySpace());
}

static Boundary getTMemBoundary(Operation *op) {
  // Assumptions do not publish pending TMEM accesses.
  if (isa<LLVM::AssumeOp>(op))
    return Boundary::None;
  // An acquire does not publish earlier TMEM accesses. Keep them pending.
  if (isa<WaitBarrierOp>(op))
    return Boundary::None;
  // Defer completion at CTA barriers until a hazard or boundary needs it.
  // Complete each warp's accesses before its rendezvous, retaining
  // dependencies on other warps for later hazards and publications.
  if (auto barrier = dyn_cast<gpu::BarrierOp>(op))
    return barrier.isWarp() ? Boundary::WaitAll : Boundary::None;
  // Drain the issuing warps before a role handoff or region exit.
  if (isa<gpu::WarpSpecializeOp, gpu::WarpYieldOp, gpu::WarpReturnOp>(op))
    return Boundary::WaitAll;
  // A following 2CTA MMA needs completion before the cluster rendezvous.
  if (auto barrier = dyn_cast<ClusterBarrierOp>(op))
    return barrier.getRelaxed() ? Boundary::None : Boundary::WaitAll;
  if (auto atomic = dyn_cast<AtomicOpInterface>(op))
    if (atomic.getMemSemantic() == MemSemantic::RELEASE ||
        atomic.getMemSemantic() == MemSemantic::ACQUIRE_RELEASE)
      return Boundary::Publication;
  if (auto barrier = dyn_cast<gpu::MBarrierOpInterface>(op)) {
    if (barrier.isPerWarp() || isa<InitBarrierOp, InvalBarrierOp>(op))
      return Boundary::WaitAll;
    return barrier.getBarriers().empty() ? Boundary::None
                                         : Boundary::Publication;
  }
  // Volatile loads still access only global memory.
  if (isa<triton::LoadOp>(op))
    return Boundary::None;
  // Shared-memory fences and copy-group waits do not publish TMEM accesses.
  if (isa<FenceAsyncSharedOp, FenceMBarrierInitReleaseClusterOp,
          gpu::AsyncCommitGroupOp, gpu::AsyncWaitOp, TMAStoreWaitOp>(op))
    return Boundary::None;

  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memory)
    return isMemoryEffectFree(op) ? Boundary::None : Boundary::Publication;
  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  // Opaque effects can hide synchronization in inline assembly or externs.
  bool publishes = llvm::any_of(effects, [](const auto &effect) {
    return !effect.getValue() &&
           isa<SideEffects::DefaultResource>(effect.getResource());
  });
  return publishes ? Boundary::Publication : Boundary::None;
}

} // namespace

bool usesTensorMemory(ModuleOp module) {
  if (auto size = module->getAttrOfType<IntegerAttr>("ttg.tensor_memory_size"))
    return size.getInt() != 0;
  // Standalone Membar can run before allocation attaches the module size.
  return module
      .walk([](Operation *op) {
        bool usesTMEM = isa<TMEMAllocOp>(op) ||
                        llvm::any_of(op->getOperands(), isTensorMemory);
        return usesTMEM ? WalkResult::interrupt() : WalkResult::advance();
      })
      .wasInterrupted();
}

void TMemModel::describe(Operation *op, OpModel &model,
                         ArrayRef<MemoryAccess> accesses) {
  // A verified scalar callback cannot access or publish TMEM.
  if (isa<MapElementwiseOp>(op))
    return;
  if (auto wait = dyn_cast<TMEMWaitOp>(op)) {
    Action kind = wait.getKind() == TMEMWaitKind::LOAD ? Action::WaitLoad
                                                       : Action::WaitStore;
    model.fixedAfter |= coverage(kind);
    return;
  }
  // Footprints preserve TMEM rows and columns. Unknown footprints alias all
  // TMEM.
  for (const MemoryAccess &access : accesses)
    if (isTensorMemory(access.value))
      model.tmemAccesses.push_back(
          {regions.getFootprint(access.value), access.isWrite});
  model.boundary = getTMemBoundary(op);
  // MMA and copies complete through mbarriers. Other TMEM effects need
  // ordering, but only loads and stores need tmem_wait.
  if (model.tmemAccesses.empty() || isa<MMAv5OpInterface, TMEMCopyOp>(op))
    return;
  if (isa<TMEMLoadOp, TMEMStoreOp, TMEMAllocOp>(op)) {
    const auto &access = llvm::getSingleElement(model.tmemAccesses);
    model.tmemProducerNeed = access.isWrite ? Need::WaitStore : Need::WaitLoad;
    model.privateTMem = isPrivate(access.footprint, op->getParentRegion());
  } else {
    model.tmemProducerNeed = Need::RegionOrder;
  }
}

} // namespace mlir::triton::membar
