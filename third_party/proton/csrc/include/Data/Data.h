#ifndef PROTON_DATA_DATA_H_
#define PROTON_DATA_DATA_H_

#include "Context/Context.h"
#include "Metric.h"
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>

namespace proton {

enum class OutputFormat { Hatchet, Count };

class Data : public ThreadLocalOpInterface {
public:
  Data(const std::string &path, ContextSource *contextSource = nullptr)
      : path(path), contextSource(contextSource) {}
  virtual ~Data() = default;

  /// Add a new scope to the data.
  /// [MT] The implementation must be thread-safe.
  virtual void addScope(size_t scopeId, const std::string &name = "") = 0;

  /// Add a single metric to the data.
  /// [MT] The implementation must be thread-safe.
  virtual void addMetric(size_t scopeId, std::shared_ptr<Metric> metric) = 0;

  /// Add multiple metrics to the data.
  /// [MT] The implementation must be thread-safe.
  virtual void addMetrics(size_t scopeId,
                          const std::map<std::string, MetricValueType> &metrics,
                          bool aggregable) = 0;

  /// Dump the data to the given output format.
  /// [MT] Thread-safe.
  void dump(OutputFormat outputFormat);

protected:
  /// The actual implementation of the dump operation.
  /// [MT] Thread-safe.
  virtual void doDump(std::ostream &os, OutputFormat outputFormat) const = 0;

  mutable std::shared_mutex mutex;
  const std::string path{};
  ContextSource *contextSource{};

  class Profiler;
  friend class Profiler;
};

OutputFormat parseOutputFormat(const std::string &outputFormat);

const std::string outputFormatToString(OutputFormat outputFormat);

} // namespace proton

#endif // PROTON_DATA_DATA_H_
