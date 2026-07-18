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
      if (!triton::BufferRegionAnalysis::isMemoryAccessOperation(op))
        return;

      auto maybeMemDesc = llvm::find_if(op->getOperands(), [](Value operand) {
        return isa<ttg::MemDescType>(operand.getType());
      });

      if (maybeMemDesc == op->operand_end())
        return;

      emitRegionInfo(op->getLoc(), "Buffers",
                     analysis->getLatticeElement(*maybeMemDesc)->getValue());
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
      if (auto baseAttr = op->getAttrOfType<IntegerAttr>("test.region_base"))
        base = static_cast<uint32_t>(baseAttr.getInt());
      uint32_t length = 0;
      if (auto lengthAttr =
              op->getAttrOfType<IntegerAttr>("test.region_length")) {
        length = static_cast<uint32_t>(lengthAttr.getInt());
      } else if (!addresses.empty()) {
        auto [min, max] =
            std::minmax_element(addresses.begin(), addresses.end());
        base = *min;
        length = *max - *min + 1;
      }
      tt::RegionInfo info;
      info.regions.insert(tt::BufferRegion(
          base, length, tt::AddressSet::fromAddresses(addresses),
          /*storageBase=*/base, /*affineOffset=*/0));
      return info;
    }

    std::optional<Value> memdesc = getTaggedMemDesc(op);
    if (!memdesc) {
      op->emitError("test.region_name requires test.region_addresses or a "
                    "memdesc operand/result");
      return failure();
    }
    return analysis->getLatticeElement(*memdesc)->getValue();
  }

  static bool mayAlias(const tt::RegionInfo &lhs, const tt::RegionInfo &rhs) {
    return llvm::any_of(lhs.regions, [&](const tt::BufferRegion &a) {
      return llvm::any_of(rhs.regions, [&](const tt::BufferRegion &b) {
        return a.intersects(b);
      });
    });
  }

  static bool contains(const tt::RegionInfo &container,
                       const tt::RegionInfo &contained) {
    return llvm::all_of(contained.regions, [&](const tt::BufferRegion &b) {
      return llvm::any_of(container.regions, [&](const tt::BufferRegion &a) {
        return a.contains(b);
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
      llvm::append_range(regions, info.regions);
    llvm::sort(regions);
    regions.erase(std::unique(regions.begin(), regions.end()), regions.end());

    tt::BufferStatePlan plan = tt::createBufferStatePlan(regions);
    InFlightDiagnostic summary = module.emitRemark();
    summary << "state-plan: lanes=" << plan.numLanes << ", components=";
    llvm::interleaveComma(
        plan.components, summary, [&](const tt::BufferStateComponent &comp) {
          summary << (comp.basis == tt::BufferStateBasis::Atoms ? "atoms"
                                                                : "views")
                  << "(" << comp.laneCount << ")";
        });

    for (const auto &[name, info] : namedRegions) {
      SmallVector<tt::BufferRegion> candidates(info.regions.begin(),
                                               info.regions.end());
      llvm::sort(candidates);
      for (const tt::BufferRegion &candidate : candidates) {
        auto it = llvm::lower_bound(regions, candidate);
        assert(it != regions.end() && *it == candidate);
        const tt::BufferStateMasks &masks =
            plan.regionMasks[std::distance(regions.begin(), it)];
        InFlightDiagnostic diag = module.emitRemark();
        diag << name << " case ";
        candidate.print(diag);
        diag << ": update=";
        printMask(diag, masks.update);
        diag << ", check=";
        printMask(diag, masks.check);
        diag << ", complete=";
        printMask(diag, masks.complete);
      }
    }
  }

  LogicalResult runExhaustiveAddressSetTest(ModuleOp module) {
    constexpr unsigned kUniverse = 8;
    constexpr unsigned kSetCount = 1u << kUniverse;
    for (unsigned lhsMask = 0; lhsMask < kSetCount; ++lhsMask) {
      SmallVector<uint32_t> lhsAddresses;
      for (unsigned bit = 0; bit < kUniverse; ++bit)
        if (lhsMask & (1u << bit))
          lhsAddresses.push_back(bit);
      tt::AddressSet lhs = tt::AddressSet::fromAddresses(lhsAddresses);
      for (unsigned rhsMask = 0; rhsMask < kSetCount; ++rhsMask) {
        SmallVector<uint32_t> rhsAddresses;
        for (unsigned bit = 0; bit < kUniverse; ++bit)
          if (rhsMask & (1u << bit))
            rhsAddresses.push_back(bit);
        tt::AddressSet rhs = tt::AddressSet::fromAddresses(rhsAddresses);
        bool expectedIntersection = (lhsMask & rhsMask) != 0;
        bool expectedContainment = (rhsMask & ~lhsMask) == 0;
        if (lhs.intersects(rhs) != expectedIntersection ||
            lhs.contains(rhs) != expectedContainment) {
          module.emitError() << "AddressSet oracle mismatch for masks "
                             << lhsMask << ", " << rhsMask;
          return failure();
        }
      }
    }
    module.emitRemark()
        << "exhaustive AddressSet oracle passed: 65536 ordered pairs";
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr("test.exhaustive_address_sets") &&
        failed(runExhaustiveAddressSetTest(module)))
      return signalPassFailure();

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
