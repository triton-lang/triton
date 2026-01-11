#ifndef PROTON_DATA_TREE_DATA_H_
#define PROTON_DATA_TREE_DATA_H_

#include "Context/Context.h"
#include "Data.h"
#include "nlohmann/json.hpp"
#include <stdexcept>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

namespace proton {

class TreeData : public Data {
public:
  struct PhaseSummary {
    // Kernel activity timing (nanoseconds, from CUPTI start/end timestamps).
    uint64_t kernel_span_ns{0}; // max(end) - min(start)
    uint64_t kernel_sum_ns{0};  // sum(duration) across nodes (may overcount overlaps)
    uint64_t kernel_invocations{0};
    // Optional raw bounds for debugging; 0 if no kernel activities were observed.
    uint64_t kernel_min_start_ns{0};
    uint64_t kernel_max_end_ns{0};
  };

  TreeData(const std::string &path, ContextSource *contextSource);
  virtual ~TreeData();

  TreeData(const std::string &path) : TreeData(path, nullptr) {}

  std::string toJsonString(size_t phase) const override;

  std::vector<uint8_t> toMsgPack(size_t phase) const override;
  // Compute a small summary for a given phase without serializing the full tree.
  PhaseSummary summarizePhase(size_t phase) const;
  // Compute per-node-path average durations (ms) for kernels whose *node name*
  // starts with `prefix`.
  //
  // The key is the full tree path (ROOT excluded), e.g. "moduleA/opB/_p_matmul_X".
  // This disambiguates kernels that share a leaf name but occur under different
  // module/op paths.
  std::map<std::string, double>
  summarizeKernelPathsAvgDurationMsByPrefix(size_t phase,
                                            const std::string &prefix) const;
  // Sum a named flexible metric (e.g. "flops8", "bytes") for nodes whose *node name*
  // starts with `prefix`, grouped by full tree path (ROOT excluded).
  //
  // The returned value is the summed metric value for that path in the phase.
  std::map<std::string, double>
  summarizeKernelPathsSumFlexibleMetricByPrefix(size_t phase,
                                                const std::string &prefix,
                                                const std::string &metricName) const;

  DataEntry addOp(const std::string &name) override;

  DataEntry addOp(size_t phase, size_t contextId,
                  const std::vector<Context> &contexts) override;

  void addScopeMetrics(
      size_t scopeId,
      const std::map<std::string, MetricValueType> &metrics) override;

  void addEntryMetrics(
      size_t phase, size_t entryId,
      const std::map<std::string, MetricValueType> &metrics) override;

protected:
  // ScopeInterface
  void enterScope(const Scope &scope) override;

  void exitScope(const Scope &scope) override;

private:
  // `tree` and `scopeIdToContextId` can be accessed by both the user thread and
  // the background threads concurrently, so methods that access them should be
  // protected by a (shared) mutex.
  class Tree;
  json buildHatchetJson(TreeData::Tree *tree) const;
  std::vector<uint8_t> buildHatchetMsgPack(TreeData::Tree *tree) const;

  // Data
  void doDump(std::ostream &os, OutputFormat outputFormat,
              size_t phase) const override;

  OutputFormat getDefaultOutputFormat() const override {
    return OutputFormat::Hatchet;
  }

  void dumpHatchet(std::ostream &os, size_t phase) const;
  void dumpHatchetMsgPack(std::ostream &os, size_t phase) const;

  PhaseStore<Tree> treePhases;
  // ScopeId -> ContextId
  std::unordered_map<size_t, size_t> scopeIdToContextId;
};

} // namespace proton

#endif // PROTON_DATA_TREE_DATA_H_
