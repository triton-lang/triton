#ifndef PROTON_PROFILER_GPU_PROFILER_H_
#define PROTON_PROFILER_GPU_PROFILER_H_

#include "Context/Context.h"
#include "Data/Metric.h"
#include "Profiler.h"
#include "Session/Session.h"
#include "Utility/Atomic.h"
#include "Utility/Map.h"

#include <atomic>
#include <chrono>
#include <deque>
#include <thread>
#include <unordered_map>

namespace proton {

using DataEntryMap = std::map<Data *, size_t>;

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
    DataEntryMap dataToEntryId;
    // If graphNodeIdToScopes is empty, this externId is non-graph launch.
    // For non-graph launches, we only need to track whether the externId
    // itself is API-originated.
    bool isApiExternId{false};
    // ----graph launch fields----
    // For graph launches, the launch correlation id fans out into multiple
    // kernel activity records. We track the expected fanout here and keep
    // updating it when we have processed each kernel activity record.
    size_t numNodes{1};

    struct GraphNodeState {
      // If the node is launched as a metric kernel, ignore it's timing data.
      bool isMetricNode{false};
      bool isApiNode{false};

      void setEntryId(Data *data, size_t entryId) {
        dataToEntryId.insert_or_assign(data, entryId);
      }

      const size_t *findEntryId(Data *data) const {
        auto it = dataToEntryId.find(data);
        if (it == dataToEntryId.end())
          return nullptr;
        return &it->second;
      }

      template <typename FnT> void forEachEntryId(FnT &&fn) const {
        for (const auto &[data, entryId] : dataToEntryId)
          fn(data, entryId);
      }

      DataEntryMap dataToEntryId;
    };

    // graphNodeId -> (per-Data entry id + API-originated flag)
    std::unordered_map<uint64_t, GraphNodeState> graphNodeIdToState;
  };

  // TODO(Keren): replace `Data *` with `dataId` to avoid pointer recycling
  // issue.
  using ExternIdToStateMap =
      ThreadSafeMap<size_t, ExternIdState,
                    std::unordered_map<size_t, ExternIdState>>;

protected:
  // OpInterface
  void startOp(const Scope &scope) override {
    this->threadState.scopeStack.push_back(scope);
    for (auto *data : dataSet) {
      auto entryId = data->addOp(scope.name);
      threadState.dataToEntryId.insert_or_assign(data, entryId);
    }
  }

  void stopOp(const Scope &scope) override {
    this->threadState.scopeStack.pop_back();
    threadState.dataToEntryId.clear();
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
    DataEntryMap dataToEntryId;
    bool isApiExternId{false};
    bool isStreamCapturing{false};
    bool isMetricKernelLaunching{false};

    ThreadState(ConcreteProfilerT &profiler) : profiler(profiler) {}

    void enterOp() {
      if (profiler.isOpInProgress()) // Already in a triton op
        return;
      // Enter a new GPU API op
      isApiExternId = true;
      auto opId = Scope::getNewScopeId();
      profiler.enterOp(Scope(opId));
      profiler.correlation.setApiExternId(opId);
    }

    void exitOp() {
      if (!profiler.isOpInProgress() || !isApiExternId)
        return;
      profiler.exitOp(scopeStack.back());
      isApiExternId = false;
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
    // Mapping from an external id to graph-node scopes + API-extern-id flags.
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
                   const DataEntryMap &dataToEntryId) {
      corrIdToExternId.insert(correlationId, externId);
      externIdToState.upsert(externId, [&](ExternIdState &state) {
        state.numNodes = numNodes;
        state.dataToEntryId = dataToEntryId;
      });
    }

    bool isApiExternId(size_t externId) const {
      bool isApi = false;
      externIdToState.withRead(externId, [&](const ExternIdState &state) {
        isApi = state.isApiExternId;
      });
      return isApi;
    }

    void setApiExternId(size_t externId) {
      externIdToState.upsert(
          externId, [&](ExternIdState &state) { state.isApiExternId = true; });
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
        metricBuffer->receive(
            scalarMetrics, tensorMetrics, profiler.tensorMetricKernel,
            profiler.scalarMetricKernel, profiler.metricKernelStream);
        threadState.isMetricKernelLaunching = false;
      } else { // Eager mode, directly copy
        // Populate tensor metrics
        auto tensorMetricsHost = metricBuffer->collectTensorMetrics(
            tensorMetrics, profiler.metricKernelStream);
        auto &dataToEntryId = threadState.dataToEntryId;
        for (auto *data : profiler.dataSet) {
          if (dataToEntryId.empty()) {
            data->addMetricsByScopeId(scopeId, scalarMetrics);
            data->addMetricsByScopeId(scopeId, tensorMetricsHost);
          } else {
            data->addMetrics(dataToEntryId[data], scalarMetrics);
            data->addMetrics(dataToEntryId[data], tensorMetricsHost);
          }
        }
      }
    }

  protected:
    ConcreteProfilerT &profiler;
    std::unique_ptr<MetricBuffer> metricBuffer;
    Runtime *runtime{nullptr};
  };

  std::unique_ptr<GPUProfilerPimplInterface> pImpl;

  bool pcSamplingEnabled{false};
};

} // namespace proton

#endif // PROTON_PROFILER_GPU_PROFILER_H_
