#ifndef PROTON_DATA_DATA_H_
#define PROTON_DATA_DATA_H_

#include "Context/Context.h"
#include "Metric.h"
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <utility>

namespace proton {

enum class OutputFormat { Hatchet, ChromeTrace, Count };

/// An "entry" is a data specific unit of operation, e.g., a node in a tree
/// data structure or an event in a trace data structure.
struct DataEntry {
  /// `entryId` is a unique identifier for the entry in the data.
  size_t id{Scope::DummyScopeId};
  /// `metrics` is a map from metric kind to metric accumulator associated
  /// with the entry.
  /// Flexible metrics cannot be directly stored here since they maybe added by
  /// both the frontend and the backend.
  /// Use `Data::addScopeMetrics` and `Data::addEntryMetrics` to add flexible
  /// metrics.
  std::reference_wrapper<std::map<MetricKind, std::unique_ptr<Metric>>> metrics;

  explicit DataEntry(size_t id,
                     std::map<MetricKind, std::unique_ptr<Metric>> &metrics)
      : id(id), metrics(metrics) {}

  void upsertMetric(std::unique_ptr<Metric> metric) {
    if (!metric)
      return;
    auto &metricsMap = metrics.get();
    auto it = metricsMap.find(metric->getKind());
    if (it == metricsMap.end()) {
      metricsMap.emplace(metric->getKind(), std::move(metric));
    } else {
      it->second->updateMetric(*metric);
    }
  }
};

class Data : public ScopeInterface {
public:
  Data(const std::string &path, ContextSource *contextSource = nullptr)
      : path(path), contextSource(contextSource) {}
  virtual ~Data() = default;

  /// Add an op to the data.
  /// Otherwise obtain the current context and append `opName` to it if `opName`
  /// is not empty. Return the entry id of the added op.
  virtual DataEntry addOp(const std::string &opName = {}) = 0;

  /// Add an op with custom contexts to the data.
  /// This is often used when context source is not available or when
  /// the profiler itself needs to supply the contexts, such as
  /// instruction samples in GPUs whose contexts are
  /// synthesized from the instruction address (no unwinder).
  virtual DataEntry addOp(size_t entryId,
                          const std::vector<Context> &contexts) = 0;

  /// Record a batch of named metrics for a scope.
  ///
  /// This is primarily intended for user-defined metrics defined in Python and
  /// directly associated with a scope.
  /// `metrics` is a map from metric name to value to be applied to `scopeId`.
  virtual void
  addScopeMetrics(size_t scopeId,
                  const std::map<std::string, MetricValueType> &metrics) = 0;

  /// Record a batch of named metrics for an entry.
  ///
  /// This is primarily intended for user-defined metrics defined in Python and
  /// added lazily by the backend profiler.
  /// `metrics` is a map from metric name to value to be applied to `entryId`.
  virtual void
  addEntryMetrics(size_t entryId,
                  const std::map<std::string, MetricValueType> &metrics) = 0;

  /// Clear all non-persistent fields in the data.
  virtual void clear() = 0;

  /// To Json
  virtual std::string toJsonString() const = 0;

  /// To MsgPack
  virtual std::vector<uint8_t> toMsgPack() const = 0;

  /// Dump the data to the given output format.
  void dump(const std::string &outputFormat);

  /// Get the contexts associated with the data.
  std::vector<Context> getContexts() const {
    return contextSource->getContexts();
  }

protected:
  /// The actual implementation of the dump operation.
  virtual void doDump(std::ostream &os, OutputFormat outputFormat) const = 0;

  virtual OutputFormat getDefaultOutputFormat() const = 0;

  mutable std::shared_mutex mutex;
  const std::string path{};
  ContextSource *contextSource{};
};

typedef std::map<Data *, DataEntry> DataToEntryMap;

OutputFormat parseOutputFormat(const std::string &outputFormat);

const std::string outputFormatToString(OutputFormat outputFormat);

} // namespace proton

#endif // PROTON_DATA_DATA_H_
