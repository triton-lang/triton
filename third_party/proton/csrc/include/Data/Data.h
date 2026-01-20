#ifndef PROTON_DATA_DATA_H_
#define PROTON_DATA_DATA_H_

#include "Context/Context.h"
#include "Metric.h"
#include "PhaseStore.h"
#include <atomic>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace proton {

enum class OutputFormat { Hatchet, HatchetMsgPack, ChromeTrace, Count };

/// An "entry" is a data specific unit of operation, e.g., a node in a tree
/// data structure or an event in a trace data structure.
struct DataEntry {
  /// `entryId` is a unique identifier for the entry in the data.
  size_t id{Scope::DummyScopeId};
  /// `phase` indicates which phase the entry belongs to.
  size_t phase{0};
  /// `metrics` is a map from metric kind to metric accumulator associated
  /// with the entry.
  /// Flexible metrics cannot be directly stored here since they maybe added by
  /// both the frontend and the backend.
  /// Use `Data::addMetrics` and `Data::addMetrics` to add flexible
  /// metrics.
  std::reference_wrapper<std::map<MetricKind, std::unique_ptr<Metric>>> metrics;

  explicit DataEntry(size_t id, size_t phase,
                     std::map<MetricKind, std::unique_ptr<Metric>> &metrics)
      : id(id), phase(phase), metrics(metrics) {}

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
  static constexpr size_t kNoCompletePhase = std::numeric_limits<size_t>::max();

  struct PhaseInfo {
    size_t current{0};
    size_t completeUpTo{kNoCompletePhase};

    bool isComplete(size_t phase) const {
      return completeUpTo != kNoCompletePhase && completeUpTo >= phase;
    }
  };

  Data(const std::string &path, ContextSource *contextSource = nullptr)
      : path(path), contextSource(contextSource) {}
  virtual ~Data() = default;

  /// Get the path associated with the data.
  const std::string &getPath() const { return path; }

  /// Get the contexts associated with the data.
  std::vector<Context> getContexts() const {
    return contextSource->getContexts();
  }

  /// Dump the data to the given output format.
  void dump(const std::string &outputFormat);

  /// Clear all non-persistent fields in the data.
  /// If `clearUpToPhase` is false, clear the given phase only.
  /// Otherwise, clear all phases up to and including the given phase.
  void clear(size_t phase, bool clearUpToPhase = false);

  /// Advance to the next phase.
  size_t advancePhase();

  /// Mark phases up to `phase` as complete.
  void completePhase(size_t phase);

  /// Atomically get current and complete phases.
  PhaseInfo getPhaseInfo() const;

  /// Add an op to the data of the current phase.
  /// If `opName` is empty, just use the current context as is.
  /// Otherwise obtain the current context and append `opName` to it. Return the
  /// entry id of the added op.
  virtual DataEntry addOp(const std::string &opName = {}) = 0;

  /// Add an op with custom contexts to the data.
  /// This is often used when context source is not available or when
  /// the profiler itself needs to supply the contexts, such as
  /// instruction samples in GPUs whose contexts are
  /// synthesized from the instruction address (no unwinder).
  ///
  /// `phase` is the phase the op should be added to. This is important for
  /// asynchronous profilers, where the current phase may have advanced by the
  /// time the profiler needs to attach a child op.
  virtual DataEntry addOp(size_t phase, size_t entryId,
                          const std::vector<Context> &contexts) = 0;

  /// Record a batch of named metrics for a scope to the data of the current
  /// phase.
  ///
  /// This is primarily intended for user-defined metrics defined in Python and
  /// directly associated with a scope.
  /// `metrics` is a map from metric name to value to be applied to `scopeId`.
  virtual void
  addMetrics(size_t scopeId,
             const std::map<std::string, MetricValueType> &metrics) = 0;

  /// Record a batch of named metrics for an entry.
  ///
  /// This is primarily intended for user-defined metrics defined in Python and
  /// added lazily by the backend profiler.
  /// `metrics` is a map from metric name to value to be applied to `entryId`.
  ///
  /// The same as `addOp`, `phase` is important for asynchronous profilers.
  virtual void
  addMetrics(size_t phase, size_t entryId,
             const std::map<std::string, MetricValueType> &metrics) = 0;

  /// To Json
  virtual std::string toJsonString(size_t phase) const = 0;

  /// To MsgPack
  virtual std::vector<uint8_t> toMsgPack(size_t phase) const = 0;

protected:
  /// The actual implementations
  virtual void doDump(std::ostream &os, OutputFormat outputFormat,
                      size_t phase) const = 0;
  virtual OutputFormat getDefaultOutputFormat() const = 0;

  void initPhaseStore(PhaseStoreBase &store);

  template <typename T> T *currentPhasePtrAs() {
    return static_cast<T *>(currentPhasePtr);
  }

  template <typename T> T *phasePtrAs(size_t phase) {
    return static_cast<T *>(phaseStore->getPtr(phase));
  }

  [[nodiscard]] std::unique_lock<std::shared_mutex>
  lockIfCurrentPhase(size_t phase) {
    std::unique_lock<std::shared_mutex> lock(mutex, std::defer_lock);
    const auto currentPhaseValue = currentPhase.load(std::memory_order_relaxed);
    // Note that currentPhase is not locked here and can get incremented after
    // this point. Correctness can still be guaranteed as no threads other than
    // the profiler thread will access the data after phase advancement.
    if (phase == currentPhaseValue) {
      lock.lock();
    }
    // Otherwise, no need to lock for other phases since they won't be updated
    // by the application thread
    return lock;
  }

  std::atomic<std::size_t> currentPhase{0};
  std::size_t completeUpToPhase{kNoCompletePhase};
  std::set<size_t> activePhases{};

  mutable std::shared_mutex mutex;
  const std::string path{};
  ContextSource *contextSource{};

private:
  PhaseStoreBase *phaseStore{};
  void *currentPhasePtr{};
};

typedef std::map<Data *, DataEntry> DataToEntryMap;

OutputFormat parseOutputFormat(const std::string &outputFormat);

const std::string outputFormatToString(OutputFormat outputFormat);

} // namespace proton

#endif // PROTON_DATA_DATA_H_
