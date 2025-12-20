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
    // If graphNodeIdToScopes is empty, this externId is non-graph launch.
    // For non-graph launches, we only need to track whether the externId
    // itself is API-originated.
    bool isApiExternId{false};
    // ----graph launch fields----
    // For graph launches, the launch correlation id fans out into multiple
    // kernel activity records. We track the expected fanout here and keep
    // updating it when we have processed each kernel activity record.
    size_t numNodes{1};

    struct GraphNodeScopes {
      bool isApiExternId{false};

      void setScopeId(Data *data, size_t scopeId) {
        if (singleData == nullptr || singleData == data) {
          singleData = data;
          singleScopeId = scopeId;
          return;
        }
        if (multiDataToScopeId.empty())
          multiDataToScopeId.reserve(2);
        multiDataToScopeId[data] = scopeId;
      }

      const size_t *findScopeId(Data *data) const {
        if (singleData == data)
          return &singleScopeId;
        if (multiDataToScopeId.empty())
          return nullptr;
        auto it = multiDataToScopeId.find(data);
        if (it == multiDataToScopeId.end())
          return nullptr;
        return &it->second;
      }

      template <typename FnT> void forEachScopeId(FnT &&fn) const {
        if (singleData != nullptr)
          fn(singleData, singleScopeId);
        for (const auto &[data, scopeId] : multiDataToScopeId)
          fn(data, scopeId);
      }

    private:
      // In most cases, a graph node is only associated with one Data object.
      // So we optimize the hot path here.
      Data *singleData{nullptr};
      size_t singleScopeId{0};
      std::unordered_map<Data *, size_t> multiDataToScopeId;
    };

    // graphNodeId -> (per-Data scopeId + API-originated flag)
    std::unordered_map<uint64_t, GraphNodeScopes> graphNodeIdToScopes;
  };

  // TODO(Keren): replace `Data *` with `dataId` to avoid pointer recycling
  // issue.
  using ExternIdToStateMap =
      ThreadSafeMap<size_t, ExternIdState,
                    std::unordered_map<size_t, ExternIdState>>;

protected:
  // OpInterface
  void startOp(const Scope &scope) override {
    this->correlation.pushExternId(scope.scopeId);
    this->threadState.scopeStack.push_back(scope);
    for (auto data : getDataSet())
      data->addOp(scope.scopeId, scope.name);
  }
  void stopOp(const Scope &scope) override {
    this->threadState.scopeStack.pop_back();
    this->correlation.popExternId();
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
    size_t opId{Scope::DummyScopeId};
    std::vector<Scope> scopeStack; // Used for nvtx range or triton op tracking
    bool isApiExternId{false};
    bool isStreamCapturing{false};
    bool isMetricKernelLaunching{false};

    ThreadState(ConcreteProfilerT &profiler) : profiler(profiler) {}

    void enterOp() {
      if (profiler.isOpInProgress()) // Already in a triton op
        return;
      // Enter a new GPU API op
      isApiExternId = true;
      opId = Scope::getNewScopeId();
      profiler.enterOp(Scope(opId));
      profiler.correlation.setApiExternId(opId);
    }

    void exitOp() {
      if (!profiler.isOpInProgress() || !isApiExternId)
        return;
      profiler.exitOp(Scope(opId));
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
    static thread_local std::deque<size_t> externIdQueue;

    Correlation() = default;

    void submit(const uint64_t correlationId) {
      atomicMax(maxSubmittedCorrelationId, correlationId);
    }

    void complete(const uint64_t correlationId) {
      atomicMax(maxCompletedCorrelationId, correlationId);
    }

    void pushExternId(size_t externId) { externIdQueue.push_back(externId); }

    void popExternId() { externIdQueue.pop_front(); }

    // Correlate the correlationId with the last externId
    void correlate(uint64_t correlationId, size_t numNodes) {
      if (externIdQueue.empty())
        return;
      const auto externId = externIdQueue.back();
      corrIdToExternId.insert(correlationId, externId);
      externIdToState.upsert(
          externId, [&](ExternIdState &state) { state.numNodes = numNodes; });
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
        for (auto *data : profiler.getDataSet()) {
          data->addMetrics(scopeId, scalarMetrics);
          data->addMetrics(scopeId, tensorMetricsHost);
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
