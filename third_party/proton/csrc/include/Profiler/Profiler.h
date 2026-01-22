#ifndef PROTON_PROFILER_PROFILER_H_
#define PROTON_PROFILER_PROFILER_H_

#include "Data/Data.h"
#include "Data/Metric.h"
#include "Utility/Singleton.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <vector>

namespace proton {

/// A profiler contains utilities provided by the profiler library to
/// collect and analyze performance data.
class Profiler : public MetricInterface {
public:
  Profiler() = default;

  virtual ~Profiler() = default;

  /// Start the profiler.
  /// If the profiler is already started, this function does nothing.
  Profiler *start() {
    if (!this->started) {
      this->started = true;
      this->doStart();
    }
    return this;
  }

  /// Flush the profiler's data from the device to the host.
  /// It doesn't stop the profiler.
  Profiler *flush() {
    this->doFlush();
    // Treat all phases up to currentPhase - 1 as flushed, even if a phase has
    // no GPU activity records (i.e., nothing to flush from device to host).
    for (auto *data : this->getDataSet()) {
      const auto phaseInfo = data->getPhaseInfo();
      if (phaseInfo.current == 0)
        continue;
      data->completePhase(phaseInfo.current - 1);
    }
    return this;
  }

  /// Stop the profiler.
  /// Do real stop if there's no data to collect.
  Profiler *stop() {
    if (!this->started) {
      return this;
    }
    if (this->dataSet.empty()) {
      this->started = false;
      this->doStop();
    }
    return this;
  }

  /// Register a data object to the profiler.
  /// A profiler can yield metrics to multiple data objects.
  Profiler *registerData(Data *data) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    dataSet.insert(data);
    return this;
  }

  /// Unregister a data object from the profiler.
  Profiler *unregisterData(Data *data) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    dataSet.erase(data);
    return this;
  }

  /// Get the set of data objects registered to the profiler.
  std::set<Data *> getDataSet() const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return dataSet;
  }

  Profiler *setMode(const std::vector<std::string> &modeAndOptions) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    this->modeAndOptions = modeAndOptions;
    this->doSetMode(modeAndOptions);
    return this;
  }

  std::vector<std::string> getMode() const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return modeAndOptions;
  }

  void addMetrics(
      size_t scopeId,
      const std::map<std::string, MetricValueType> &scalarMetrics,
      const std::map<std::string, TensorMetric> &tensorMetrics) override {
    std::unique_lock<std::shared_mutex> lock(mutex);
    this->doAddMetrics(scopeId, scalarMetrics, tensorMetrics);
  }

  /// These fields are not persistent, function pointers will be changed
  /// when modules and contexts are switched.
  /// So we just set them as thread local storage before the application kernel
  /// starts or after the application kernel ends.
  void setMetricKernels(void *tensorMetricKernel, void *scalarMetricKernel,
                        void *stream) override {
    this->tensorMetricKernel = tensorMetricKernel;
    this->scalarMetricKernel = scalarMetricKernel;
    this->metricKernelStream = stream;
  }

protected:
  virtual void doStart() = 0;
  virtual void doFlush() = 0;
  virtual void doStop() = 0;
  virtual void doSetMode(const std::vector<std::string> &modeAndOptions) = 0;
  virtual void
  doAddMetrics(size_t scopeId,
               const std::map<std::string, MetricValueType> &scalarMetrics,
               const std::map<std::string, TensorMetric> &tensorMetrics) = 0;

  mutable std::shared_mutex mutex;
  std::set<Data *> dataSet;
  static thread_local void *tensorMetricKernel;
  static thread_local void *scalarMetricKernel;
  static thread_local void *metricKernelStream;

private:
  bool started{};
  std::vector<std::string> modeAndOptions{};
};

} // namespace proton

#endif // PROTON_PROFILER_PROFILER_H_
