#ifndef PROTON_PROFILER_PROFILER_H_
#define PROTON_PROFILER_PROFILER_H_

#include "Data/Data.h"
#include "Utility/Singleton.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>

namespace proton {

/// A profiler contains utilities provided by the profiler library to
/// collect and analyze performance data.
class Profiler {
public:
  Profiler() = default;

  virtual ~Profiler() = default;

  /// Start the profiler.
  /// If the profiler is already started, this function does nothing.
  Profiler *start() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    if (this->initializedCount == 0)
      this->doStart();
    this->initializedCount++;
    return this;
  }

  /// Flush the profiler's data from the device to the host.
  /// It doesn't stop the profiler.
  Profiler *flush() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    this->doFlush();
    return this;
  }

  /// Stop the profiler.
  Profiler *stop() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    if (this->initializedCount == 0)
      return this;
    this->initializedCount--;
    if (this->initializedCount == 0)
      this->doStop();
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

protected:
  virtual void doStart() = 0;
  virtual void doFlush() = 0;
  virtual void doStop() = 0;

  mutable std::shared_mutex mutex;
  std::set<Data *> dataSet;

private:
  int initializedCount{};
};

} // namespace proton

#endif // PROTON_PROFILER_PROFILER_H_
