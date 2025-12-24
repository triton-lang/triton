#ifndef PROTON_DATA_DATA_H_
#define PROTON_DATA_DATA_H_

#include "Context/Context.h"
#include "Metric.h"
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>

namespace proton {

enum class OutputFormat { Hatchet, ChromeTrace, Count };

class Data : public ScopeInterface {
public:
  Data(const std::string &path, ContextSource *contextSource = nullptr)
      : path(path), contextSource(contextSource) {}
  virtual ~Data() = default;

  /// Add an op to the data.
  /// Otherwise obtain the current context and append `opName` to it if `opName`
  /// is not empty. Return the entry id of the added op.
  /// An "entry" is a data specific unit of operation, e.g., a node in a tree
  /// data structure or an event in a trace data structure.
  virtual size_t addOp(const std::string &opName = {}) = 0;

  /// Add an op with custom contexts to the data.
  /// This is often used when context source is not available or when
  /// the profiler itself needs to supply the contexts, such as
  /// instruction samples in GPUs whose contexts are
  /// synthesized from the instruction address (no unwinder).
  /// `entryId` is an anchor node to indicate where to add the new contexts.
  /// Return the new entry id of the added op, which may be different from
  /// `entryId`.
  virtual size_t addOp(size_t entryId,
                       const std::vector<Context> &contexts) = 0;

  /// Add a single metric to the data.
  virtual void addMetric(size_t entryId, std::shared_ptr<Metric> metric) = 0;

  /// Add a flexible metric to the data.
  virtual void addMetric(size_t entryId, const FlexibleMetric &metric) = 0;

  /// Add an op and a metric with one call.
  /// The default implementation forwards to addOp + addMetric.
  virtual void addOpAndMetric(size_t entryId, const std::string &opName,
                              std::shared_ptr<Metric> metric) {
    entryId = this->addOp(entryId, {Context(opName)});
    this->addMetric(entryId, metric);
  }

  virtual void
  addOpAndMetricBatch(std::vector<std::tuple<size_t, const std::string &,
                                             std::shared_ptr<Metric>>> &batch) {
    for (auto &[entryId, opName, metric] : batch) {
      this->addOpAndMetric(entryId, opName, metric);
    }
  }

  /// Add multiple metrics to the data.
  /// This metric is only used for flexible metrics passed from the inside.
  virtual void
  addMetrics(size_t entryId,
             const std::map<std::string, MetricValueType> &metrics) = 0;

  /// Add multiple metrics to the data.
  /// This metric is only used for flexible metrics passed from the outside.
  /// Note that the index here is `scopeId` instead of `entryId`.
  virtual void addMetricsByScopeId(
      size_t scopeId,
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

OutputFormat parseOutputFormat(const std::string &outputFormat);

const std::string outputFormatToString(OutputFormat outputFormat);

} // namespace proton

#endif // PROTON_DATA_DATA_H_
