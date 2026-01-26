#ifndef PROTON_PROFILER_GPU_PROFILER_H_
#define PROTON_PROFILER_GPU_PROFILER_H_

#include "Context/Context.h"
#include "Data/Metric.h"
#include "Profiler.h"
#include "Profiler/Graph.h"
#include "Session/Session.h"
#include "Utility/Atomic.h"
#include "Utility/Env.h"
#include "Utility/Map.h"
#include "Utility/Table.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <thread>
#include <unordered_map>
#include <vector>

namespace proton {

namespace detail {

void flushDataPhasesImpl(
    const bool periodicFlushEnabled, const std::string &periodicFlushingFormat,
    std::map<Data *, size_t> &dataFlushedPhases,
    const std::map<Data *,
                   std::pair</*start_phase=*/size_t, /*end_phase=*/size_t>>
        &dataPhases,
    PendingGraphPool *pendingGraphPool);

void updateDataPhases(
    std::map<Data *, std::pair</*start_phase=*/size_t, /*end_phase=*/size_t>>
        &dataPhases,
    Data *data, size_t phase);

void setPeriodicFlushingMode(bool &periodicFlushingEnabled,
                             std::string &periodicFlushingFormat,
                             const std::vector<std::string> &modeAndOptions,
                             const char *profilerName);
} // namespace detail

// Singleton<ConcreteProfilerT>: Each concrete GPU profiler, e.g.,
// CuptiProfiler, should be a singleton.
template <typename ConcreteProfilerT>
class GPUProfiler : public Profiler,
                    public OpInterface,
                    public Singleton<ConcreteProfilerT> {
public:
  GPUProfiler() = default;
  virtual ~GPUProfiler() = default;

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

    struct GraphNodeState {
      // If the node is launched as a metric kernel, ignore it's timing data.
      bool isMetricNode{false};
      bool isMissingName{true};

      void setEntry(Data *data, const DataEntry &entry) {
        dataToEntry.insert_or_assign(data, entry);
      }

      const DataEntry *findEntry(Data *data) const {
        auto it = dataToEntry.find(data);
        if (it == dataToEntry.end())
          return nullptr;
        return &it->second;
      }

      template <typename FnT> void forEachEntry(FnT &&fn) {
        for (auto &[data, entry] : dataToEntry)
          fn(data, entry);
      }

      DataToEntryMap dataToEntry;
    };

    using GraphNodeStateTable = RangeTable<GraphNodeState>;

    // graphNodeId -> (per-Data entry)
    GraphNodeStateTable graphNodeIdToState;
  };

  using ExternIdToStateMap =
      ThreadSafeMap<size_t, ExternIdState,
                    std::unordered_map<size_t, ExternIdState>>;

protected:
  // OpInterface
  void startOp(const Scope &scope) override {
    this->threadState.scopeStack.push_back(scope);
    for (auto *data : dataSet) {
      auto entry = data->addOp(scope.name);
      threadState.dataToEntry.insert_or_assign(data, entry);
    }
  }

  void stopOp(const Scope &scope) override {
    this->threadState.scopeStack.pop_back();
    threadState.dataToEntry.clear();
  }

  void flushDataPhases(
      std::map<Data *, size_t> &dataFlushedPhases,
      const std::map<Data *,
                     std::pair</*start_phase=*/size_t, /*end_phase=*/size_t>>
          &dataPhases,
      PendingGraphPool *pendingGraphPool) {
    detail::flushDataPhasesImpl(periodicFlushingEnabled, periodicFlushingFormat,
                                dataFlushedPhases, dataPhases,
                                pendingGraphPool);
  }

  // Profiler
  virtual void doStart() override { pImpl->doStart(); }
  virtual void doFlush() override { pImpl->doFlush(); }
  virtual void doStop() override { pImpl->doStop(); }
  virtual void doAddMetrics(
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

    ThreadState(ConcreteProfilerT &profiler) : profiler(profiler) {}

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

  struct Correlation {
    std::atomic<uint64_t> maxSubmittedCorrelationId{0};
    std::atomic<uint64_t> maxCompletedCorrelationId{0};
    // Mapping from a native profiler correlation id to an external id.
    CorrIdToExternIdMap corrIdToExternId;
    // Mapping from an external id to graph-node states
    ExternIdToStateMap externIdToState;

    Correlation() = default;

    void submit(uint64_t correlationId) {
      atomicMax(maxSubmittedCorrelationId, correlationId);
    }

    void complete(uint64_t correlationId) {
      atomicMax(maxCompletedCorrelationId, correlationId);
    }

    // Correlate the correlationId with the last externId
    void correlate(uint64_t correlationId, size_t externId, size_t numNodes,
                   bool isMissingName, const DataToEntryMap &dataToEntry) {
      corrIdToExternId.insert(correlationId, externId);
      externIdToState.upsert(externId, [&](ExternIdState &state) {
        state.numNodes = numNodes;
        state.dataToEntry = dataToEntry;
        state.isMissingName = isMissingName;
      });
    }

    template <typename FlushFnT>
    void flush(uint64_t maxRetries, uint64_t sleepUs, FlushFnT &&flushFn) {
      flushFn();
      auto submittedId = maxSubmittedCorrelationId.load();
      auto completedId = maxCompletedCorrelationId.load();
      auto retries = maxRetries;
      while ((completedId < submittedId) && retries > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(sleepUs));
        flushFn();
        completedId = maxCompletedCorrelationId.load();
        --retries;
      }
    }
  };

  static thread_local ThreadState threadState;

  std::unique_ptr<MetricBuffer> metricBuffer;
  std::unique_ptr<PendingGraphPool> pendingGraphPool;

  Correlation correlation;

  // Use the pimpl idiom to hide the implementation details. This lets us avoid
  // including the cupti header from this header. The cupti header and the
  // equivalent header from AMD define conflicting macros, so we want to use
  // those headers only within cpp files.
  class GPUProfilerPimplInterface {
  public:
    GPUProfilerPimplInterface(ConcreteProfilerT &profiler)
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
        threadState.isMetricKernelLaunching = true;
        // Launch metric kernels
        profiler.metricBuffer->receive(
            scalarMetrics, tensorMetrics, profiler.tensorMetricKernel,
            profiler.scalarMetricKernel, profiler.metricKernelStream);
        threadState.isMetricKernelLaunching = false;
      } else { // Eager mode, directly copy
        // Populate tensor metrics
        auto tensorMetricsHost =
            collectTensorMetrics(profiler.metricBuffer->getRuntime(),
                                 tensorMetrics, profiler.metricKernelStream);
        auto &dataToEntry = threadState.dataToEntry;
        if (dataToEntry.empty()) {
          // Add metrics to a specific scope
          for (auto *data : profiler.dataSet) {
            data->addMetrics(scopeId, scalarMetrics);
            data->addMetrics(scopeId, tensorMetricsHost);
          }
        } else {
          // Add metrics to the current op
          for (auto [data, entry] : dataToEntry) {
            data->addMetrics(entry.phase, entry.id, scalarMetrics);
            data->addMetrics(entry.phase, entry.id, tensorMetricsHost);
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
