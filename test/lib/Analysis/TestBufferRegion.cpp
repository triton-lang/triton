#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

using namespace mlir;

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

struct TestBufferRegionPass
    : public PassWrapper<TestBufferRegionPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBufferRegionPass);

  static void emitRegionInfo(Location loc, StringRef name,
                             const tt::RegionInfo &regionInfo) {
    InFlightDiagnostic diag = mlir::emitRemark(loc);
    diag << name << ": ";
    regionInfo.print(diag);
  }

  static void emitRegionList(Location loc, StringRef name,
                             llvm::ArrayRef<tt::BufferRegion> regions) {
    if (regions.empty())
      return;

    InFlightDiagnostic diag = mlir::emitRemark(loc);
    diag << name << ": ";
    llvm::interleaveComma(regions, diag, [&](const tt::BufferRegion &region) {
      region.print(diag);
    });
  }

  StringRef getArgument() const final { return "test-print-buffer-region"; }
  StringRef getDescription() const final {
    return "print the result of the buffer region analysis pass";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    triton::BufferRegionAnalysis *analysis =
        solver->load<triton::BufferRegionAnalysis>();
    if (failed(solver->initializeAndRun(moduleOp)))
      return signalPassFailure();
    analysis->calculateUsedBufferRegions(moduleOp);

    moduleOp.walk([&](Operation *op) {
      for (const auto &access : triton::getMemoryAccesses(op)) {
        if (!llvm::is_contained(op->getOperands(), access.value))
          continue;
        emitRegionInfo(op->getLoc(), "Buffers",
                       analysis->getLatticeElement(access.value)->getValue());
        break;
      }
    });

    llvm::SmallVector<Operation *> anchors;
    moduleOp.walk([&](Operation *op) {
      if (op->hasAttr("test.print_all_used_regions"))
        anchors.push_back(op);
    });

    for (Operation *anchor : anchors) {
      auto emitAllRegions = [&](tt::BufferRegionAnalysis::RegionType type,
                                StringRef label) {
        emitRegionList(anchor->getLoc(), label,
                       analysis->getAllUsedBufferRegions(type));
      };

      emitAllRegions(tt::BufferRegionAnalysis::SHARED_MEMORY,
                     "All Shared Regions");
      emitAllRegions(tt::BufferRegionAnalysis::TENSOR_MEMORY,
                     "All Tensor Regions");
      emitAllRegions(tt::BufferRegionAnalysis::BARRIER, "All Barrier Regions");
    }
  }
};

struct TestBufferRegionAliasPass
    : public PassWrapper<TestBufferRegionAliasPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBufferRegionAliasPass);

  using NamedRegion = std::pair<std::string, tt::BufferRegionFootprint>;

  StringRef getArgument() const final { return "test-buffer-region-alias"; }
  StringRef getDescription() const final {
    return "test exact buffer-region alias and containment analysis";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ttg::TritonGPUDialect>();
  }

  static Value getTaggedMemDesc(Operation *op) {
    for (Value value : llvm::concat<Value>(op->getOperands(), op->getResults()))
      if (isa<ttg::MemDescType>(value.getType()))
        return value;
    return {};
  }

  static FailureOr<tt::BufferRegionFootprint>
  getTaggedFootprint(Operation *op, tt::BufferRegionAnalysis *analysis) {
    if (auto addressesAttr =
            op->getAttrOfType<DenseI32ArrayAttr>("test.region_addresses")) {
      SmallVector<uint32_t> addresses;
      for (int32_t address : addressesAttr.asArrayRef()) {
        if (address < 0) {
          op->emitError("test.region_addresses must be non-negative");
          return failure();
        }
        addresses.push_back(static_cast<uint32_t>(address));
      }
      uint32_t base = 0;
      uint32_t length = 0;
      if (!addresses.empty()) {
        auto [min, max] =
            std::minmax_element(addresses.begin(), addresses.end());
        base = *min;
        length = *max - *min + 1;
      }
      tt::BufferRegion region{base, length};
      if (!addresses.empty()) {
        tt::AddressSet addressSet;
        for (uint32_t address : addresses)
          addressSet.set(address);
        region.ctaAddresses.emplace_back(0, std::move(addressSet));
      }
      tt::BufferRegionView view{std::move(region), /*storageBase=*/base,
                                /*affineOffset=*/0};
      // Explicit address sets share one synthetic allocation frame.
      view.allocationFrame =
          analysis->getOperationId(op->getParentOfType<ModuleOp>());
      return tt::BufferRegionFootprint{
          ttg::SharedMemorySpaceAttr::get(op->getContext()),
          tt::RegionInfo({std::move(view)})};
    }

    Value memdesc = getTaggedMemDesc(op);
    if (!memdesc) {
      op->emitError("test.region_name requires test.region_addresses or a "
                    "memdesc operand/result");
      return failure();
    }
    if (const auto *footprint = analysis->getFootprint(memdesc))
      return *footprint;
    return tt::BufferRegionFootprint{{}, analysis->getRegionInfo(memdesc)};
  }

  static bool contains(const tt::BufferRegionFootprint &container,
                       const tt::BufferRegionFootprint &contained) {
    if (!container.memorySpace ||
        container.memorySpace != contained.memorySpace ||
        container.regionInfo.isUnknown() || contained.regionInfo.isUnknown())
      return false;
    return llvm::all_of(contained.regionInfo.views, [&](const auto &b) {
      return llvm::any_of(container.regionInfo.views,
                          [&](const auto &a) { return a.contains(b); });
    });
  }

  static void printMask(InFlightDiagnostic &diag,
                        const llvm::SmallBitVector &mask) {
    diag << "{";
    llvm::interleave(mask.set_bits(), diag, ",");
    diag << "}";
  }

  static void emitStatePlan(ModuleOp module,
                            ArrayRef<NamedRegion> namedRegions) {
    SmallVector<tt::BufferRegion> regions;
    for (const auto &[name, footprint] : namedRegions)
      for (const tt::BufferRegionView &view : footprint.regionInfo.views)
        regions.push_back(view.region);
    llvm::sort(regions);
    regions.erase(std::unique(regions.begin(), regions.end()), regions.end());

    bool hasUnknown = llvm::any_of(namedRegions, [](const auto &named) {
      return named.second.regionInfo.isUnknown();
    });
    tt::BufferStatePlan plan = tt::createBufferStatePlan(regions, hasUnknown);
    InFlightDiagnostic summary = module.emitRemark();
    summary << "state-plan: lanes=" << plan.numLanes;

    for (const auto &[name, footprint] : namedRegions) {
      const tt::RegionInfo &info = footprint.regionInfo;
      if (info.isUnknown()) {
        InFlightDiagnostic diag = module.emitRemark();
        diag << name << " case unknown: mask=";
        printMask(diag, plan.unknownMask);
        continue;
      }
      SmallVector<tt::BufferRegion> candidates;
      for (const tt::BufferRegionView &view : info.views)
        candidates.push_back(view.region);
      llvm::sort(candidates);
      for (const tt::BufferRegion &candidate : candidates) {
        auto it = llvm::lower_bound(regions, candidate);
        assert(it != regions.end() && *it == candidate);
        const llvm::SmallBitVector &mask =
            plan.regionMasks[std::distance(regions.begin(), it)];
        InFlightDiagnostic diag = module.emitRemark();
        diag << name << " case ";
        candidate.print(diag);
        diag << ": mask=";
        printMask(diag, mask);
      }
    }
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    tt::BufferRegionAnalysis *analysis =
        solver->load<tt::BufferRegionAnalysis>();
    if (failed(solver->initializeAndRun(module)))
      return signalPassFailure();

    SmallVector<NamedRegion> namedRegions;
    module.walk([&](Operation *op) {
      auto name = op->getAttrOfType<StringAttr>("test.region_name");
      if (!name)
        return;
      auto footprint = getTaggedFootprint(op, analysis);
      if (failed(footprint)) {
        return signalPassFailure();
      }
      namedRegions.push_back({name.str(), std::move(*footprint)});
    });
    llvm::sort(namedRegions, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });

    if (!module->hasAttr("test.state_plan_only")) {
      for (size_t i = 0; i < namedRegions.size(); ++i) {
        for (size_t j = i; j < namedRegions.size(); ++j) {
          const auto &[lhsName, lhs] = namedRegions[i];
          const auto &[rhsName, rhs] = namedRegions[j];
          module.emitRemark()
              << lhsName << " vs " << rhsName
              << ": alias=" << (tt::mayOverlap(&lhs, &rhs) ? "true" : "false")
              << ", lhs_contains_rhs="
              << (contains(lhs, rhs) ? "true" : "false")
              << ", rhs_contains_lhs="
              << (contains(rhs, lhs) ? "true" : "false");
        }
      }
    }

    if (module->hasAttr("test.print_state_plan"))
      emitStatePlan(module, namedRegions);
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestBufferRegionPass() {
  PassRegistration<TestBufferRegionPass>();
  PassRegistration<TestBufferRegionAliasPass>();
}
} // namespace test
} // namespace mlir
