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

    tt::UsedBufferRegions used =
        tt::calculateUsedBufferRegions(moduleOp, *analysis);
    moduleOp.walk([&](Operation *op) {
      for (const auto &access :
           triton::BufferRegionAnalysis::getMemoryAccesses(op)) {
        if (!llvm::is_contained(op->getOperands(), access.value))
          continue;
        emitRegionInfo(op->getLoc(), "Buffers",
                       analysis->getRegionInfo(access.value));
        break;
      }
    });

    llvm::SmallVector<Operation *> anchors;
    moduleOp.walk([&](Operation *op) {
      if (op->hasAttr("test.print_all_used_regions"))
        anchors.push_back(op);
    });

    for (Operation *anchor : anchors) {
      auto emitAllRegions = [&](tt::BufferRegionType type, StringRef label) {
        emitRegionList(anchor->getLoc(), label, used.getRegions(type));
      };

      emitAllRegions(tt::BufferRegionType::Shared, "All Shared Regions");
      emitAllRegions(tt::BufferRegionType::Tensor, "All Tensor Regions");
      emitAllRegions(tt::BufferRegionType::Barrier, "All Barrier Regions");
    }
  }
};

struct TestBufferRegionAliasPass
    : public PassWrapper<TestBufferRegionAliasPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBufferRegionAliasPass);

  StringRef getArgument() const final { return "test-buffer-region-alias"; }
  StringRef getDescription() const final {
    return "test exact buffer-region alias and containment analysis";
  }

  static std::optional<Value> getTaggedMemDesc(Operation *op) {
    for (Value operand : op->getOperands())
      if (isa<ttg::MemDescType>(operand.getType()))
        return operand;
    for (Value result : op->getResults())
      if (isa<ttg::MemDescType>(result.getType()))
        return result;
    return std::nullopt;
  }

  static FailureOr<tt::RegionInfo>
  getTaggedRegionInfo(Operation *op, tt::BufferRegionAnalysis *analysis) {
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
      tt::RegionInfo info(tt::RegionInfo::ViewList{});
      info.views.insert(
          {std::move(region), /*storageBase=*/base, /*affineOffset=*/0});
      return info;
    }

    std::optional<Value> memdesc = getTaggedMemDesc(op);
    if (!memdesc) {
      op->emitError("test.region_name requires test.region_addresses or a "
                    "memdesc operand/result");
      return failure();
    }
    return analysis->getRegionInfo(*memdesc);
  }

  static bool mayAlias(const tt::RegionInfo &lhs, const tt::RegionInfo &rhs) {
    if (lhs.isUnknown() || rhs.isUnknown())
      return true;
    return llvm::any_of(lhs.views, [&](const tt::BufferRegionView &a) {
      return llvm::any_of(rhs.views, [&](const tt::BufferRegionView &b) {
        return a.region.intersects(b.region);
      });
    });
  }

  static bool contains(const tt::RegionInfo &container,
                       const tt::RegionInfo &contained) {
    if (container.isUnknown() || contained.isUnknown())
      return false;
    return llvm::all_of(contained.views, [&](const tt::BufferRegionView &b) {
      return llvm::any_of(container.views, [&](const tt::BufferRegionView &a) {
        return a.region.contains(b.region);
      });
    });
  }

  static void printMask(InFlightDiagnostic &diag,
                        const llvm::SmallBitVector &mask) {
    diag << "{";
    bool first = true;
    for (unsigned bit = 0; bit < mask.size(); ++bit) {
      if (!mask.test(bit))
        continue;
      if (!first)
        diag << ",";
      diag << bit;
      first = false;
    }
    diag << "}";
  }

  static void
  emitStatePlan(ModuleOp module,
                ArrayRef<std::pair<std::string, tt::RegionInfo>> namedRegions) {
    SmallVector<tt::BufferRegion> regions;
    for (const auto &[name, info] : namedRegions)
      for (const tt::BufferRegionView &view : info.views)
        regions.push_back(view.region);
    llvm::sort(regions);
    regions.erase(std::unique(regions.begin(), regions.end()), regions.end());

    bool hasUnknown = llvm::any_of(namedRegions, [](const auto &named) {
      return named.second.isUnknown();
    });
    tt::BufferStatePlan plan = tt::createBufferStatePlan(regions, hasUnknown);
    InFlightDiagnostic summary = module.emitRemark();
    summary << "state-plan: lanes=" << plan.numLanes;

    for (const auto &[name, info] : namedRegions) {
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

    SmallVector<std::pair<std::string, tt::RegionInfo>> namedRegions;
    module.walk([&](Operation *op) {
      auto name = op->getAttrOfType<StringAttr>("test.region_name");
      if (!name)
        return;
      FailureOr<tt::RegionInfo> regionInfo = getTaggedRegionInfo(op, analysis);
      if (failed(regionInfo)) {
        return signalPassFailure();
      }
      namedRegions.push_back({name.str(), std::move(*regionInfo)});
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
              << ": alias=" << (mayAlias(lhs, rhs) ? "true" : "false")
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
