#ifndef PROTON_PROFILER_GPU_PROFILER_H_
#define PROTON_PROFILER_GPU_PROFILER_H_

#include "Context/Context.h"
#include "Data/Metric.h"
#include "Profiler.h"
#include "Profiler/Graph.h"
#include "Session/Session.h"
#include "Utility/Map.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace proton {

using DataPhases = std::map<Data *, std::pair</*start_phase=*/size_t,
                                              /*end_phase=*/size_t>>;

using CorrIdToExternIdMap =
    ThreadSafeMap</*correlation_id=*/uint64_t, /*extern_id=*/size_t,
                  std::unordered_map<uint64_t, size_t>>;

struct ExternIdState {
  // ----non-graph launch fields----
  DataToEntryMap dataToEntry;
  // Sometimes the kernel name cannot be retrieved in application threads
  // for reasons like uninitialize CUDA context.
  bool isMissingName{true};
  // ----graph launch fields----
  // For graph launches, the launch correlation id fans out into multiple
  // kernel activity records. We track the expected fanout here and keep
  // updating it when we have processed each kernel activity record.
  size_t numNodes{1};
  DataToEntryMap dataToGraphEntry;
  GraphState::NodeIdToStateMap *nodeIdToState{nullptr};
};

using ExternIdToStateMap =
    ThreadSafeMap<size_t, ExternIdState,
                  std::unordered_map<size_t, ExternIdState>>;

class GPUCorrelation {
public:
  void submit(uint64_t numTasks,
              uint64_t correlationId = Scope::DummyScopeId);
  void complete(uint64_t numTasks, uint64_t correlationId);
  void complete(uint64_t correlationId);
  void correlate(uint64_t correlationId, size_t externId, size_t numNodes,
                 bool isMissingName, const DataToEntryMap &dataToEntry);
  void flush(uint64_t maxRetries, uint64_t sleepUs,
             const std::function<void()> &flushFn);
  void clear();

  // These maps are consumed directly by backend-specific activity processors.
  CorrIdToExternIdMap corrIdToExternId;
  ExternIdToStateMap externIdToState;

private:
  std::atomic<uint64_t> numSubmittedTasks{0};
  std::atomic<uint64_t> numCompletedTasks{0};
  std::atomic<uint64_t> maxSubmittedCorrelationId{0};
  std::atomic<uint64_t> maxCompletedCorrelationId{0};
};

namespace detail {

void flushDataPhasesImpl(const bool periodicFlushEnabled,
                         const std::string &periodicFlushingFormat,
                         const DataPhases &dataPhases,
                         PendingGraphPool *pendingGraphPool);

void updateDataPhases(DataPhases &dataPhases, Data *data, size_t phase);

void setPeriodicFlushingMode(bool &periodicFlushingEnabled,
                             std::string &periodicFlushingFormat,
                             const std::vector<std::string> &modeAndOptions,
                             const char *profilerName);

int64_t
computeTimestampOffsetNs(const std::function<void(uint64_t *)> &getTimestamp);

size_t prepareGraphLaunch(ThreadSafeMap<uint64_t, GraphState> &graphStates,
                          uint64_t graphExecId, size_t externId,
                          const DataToEntryMap &dataToEntry,
                          ExternIdToStateMap &externIdToState,
                          PendingGraphPool *pendingGraphPool,
                          bool flushMetricBuffer);
} // namespace detail

// Singleton<ConcreteProfilerT>: Each concrete GPU profiler, e.g.,
// CuptiProfiler, should be a singleton.
template <typename ConcreteProfilerT>
class GPUProfiler : public Profiler,
                    public OpInterface,
                    public TimestampAlignmentInterface,
                    public Singleton<ConcreteProfilerT> {
public:
  GPUProfiler() = default;
  ~GPUProfiler() override = default;

  int64_t getTimestampOffsetNs() const override final {
    return timestampOffsetNs.value_or(0);
  }

protected:
  // OpInterface
  void startOp(const Scope &scope) override {
    this->threadState.scopeStack.push_back(scope);
    for (auto *data : dataSet) {
      auto entry = data->addOp(scope.name);
      threadState.dataToEntry.insert_or_assign(data, entry);
    }
  }

  void stopOp(const Scope &) override {
    this->threadState.scopeStack.pop_back();
    threadState.dataToEntry.clear();
  }

  void flushDataPhases(const DataPhases &dataPhases,
                       PendingGraphPool *pendingGraphPool) {
    detail::flushDataPhasesImpl(periodicFlushingEnabled, periodicFlushingFormat,
                                dataPhases, pendingGraphPool);
  }

  // Profiler
  void doStart() override { pImpl->doStart(); }
  void doFlush() override { pImpl->doFlush(); }
  void doStop() override { pImpl->doStop(); }
  void addMetrics(
      size_t scopeId,
      const std::map<std::string, MetricValueType> &scalarMetrics,
      const std::map<std::string, TensorMetric> &tensorMetrics) override {
    pImpl->doAddMetrics(scopeId, scalarMetrics, tensorMetrics);
  }

  struct ThreadState {
    ConcreteProfilerT &profiler;
    SessionManager &sessionManager = SessionManager::instance();
    std::vector<Scope> scopeStack; // Used for nvtx range or triton op tracking
    DataToEntryMap dataToEntry;
    bool isApiExternOp{false};
    bool isStreamCapturing{false};
    bool isMetricKernelLaunching{false};
    struct MetricKernelLaunchInfo {
      uint64_t seqId{};
      uint64_t metricId{};
      size_t numWords{};
    };
    std::deque<MetricKernelLaunchInfo> metricKernelLaunchInfoQueue;
    explicit ThreadState(ConcreteProfilerT &profiler) : profiler(profiler) {}

    void enterOp(const Scope &scope) {
      if (profiler.isOpInProgress()) // Already in a triton op
        return;
      // Enter a new GPU API op
      isApiExternOp = true;
      profiler.enterOp(scope);
    }

    void exitOp() {
      if (!profiler.isOpInProgress() || !isApiExternOp)
        return;
      profiler.exitOp(scopeStack.back());
      isApiExternOp = false;
    }

    void enterScope(const std::string &name) {
      Scope scope(name);
      scopeStack.push_back(scope);
      sessionManager.enterScope(scope);
    }

    void exitScope() {
      sessionManager.exitScope(scopeStack.back());
      scopeStack.pop_back();
    }

  };

  static thread_local ThreadState threadState;

  std::unique_ptr<MetricBuffer> metricBuffer;
  std::unique_ptr<PendingGraphPool> pendingGraphPool;

  GPUCorrelation correlation;

  std::optional<int64_t> timestampOffsetNs;

  // Use the pimpl idiom to hide the implementation details. This lets us avoid
  // including the cupti header from this header. The cupti header and the
  // equivalent header from AMD define conflicting macros, so we want to use
  // those headers only within cpp files.
  class GPUProfilerPimplInterface {
  public:
    explicit GPUProfilerPimplInterface(ConcreteProfilerT &profiler)
        : profiler(profiler) {}
    virtual ~GPUProfilerPimplInterface() = default;

    virtual void doStart() = 0;
    virtual void doFlush() = 0;
    virtual void doStop() = 0;

    void
    doAddMetrics(size_t scopeId,
                 const std::map<std::string, MetricValueType> &scalarMetrics,
                 const std::map<std::string, TensorMetric> &tensorMetrics) {
      if (threadState.isStreamCapturing) { // Graph capture mode
        // Launch metric kernels
        auto &metricKernelLaunchState = profiler.metricKernelLaunchState;
        threadState.isMetricKernelLaunching = true;
        profiler.metricBuffer->receive(
            tensorMetrics, scalarMetrics, metricKernelLaunchState,
            [&](uint64_t seqId, uint64_t metricId, size_t numWords) {
              threadState.metricKernelLaunchInfoQueue.push_back(
                  {seqId, metricId, numWords});
            });
        threadState.isMetricKernelLaunching = false;
      } else { // Eager mode, directly copy
        // Populate tensor metrics
        auto tensorMetricsHost = collectTensorMetrics(
            profiler.metricBuffer->getRuntime(), tensorMetrics,
            profiler.metricKernelLaunchState.tensor.stream);
        auto &dataToEntry = threadState.dataToEntry;
        if (dataToEntry.empty()) {
          // Add metrics to a specific scope
          for (auto *data : profiler.dataSet) {
            data->addMetrics(scopeId, scalarMetrics);
            data->addMetrics(scopeId, tensorMetricsHost);
          }
        } else {
          // Add metrics to the current op
          for (const auto &entryIt : dataToEntry) {
            const auto &entry = entryIt.second;
            entry.upsertFlexibleMetrics(scalarMetrics);
            entry.upsertFlexibleMetrics(tensorMetricsHost);
          }
        }
      }
    }

  protected:
    ConcreteProfilerT &profiler;
  };

  std::unique_ptr<GPUProfilerPimplInterface> pImpl;

  bool pcSamplingEnabled{false};
  bool periodicFlushingEnabled{false};
  std::string periodicFlushingFormat{};
};

} // namespace proton

#endif // PROTON_PROFILER_GPU_PROFILER_H_
