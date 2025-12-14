#ifndef PROTON_PROFILER_GPU_PROFILER_H_
#define PROTON_PROFILER_GPU_PROFILER_H_

#include "Context/Context.h"
#include "Data/Metric.h"
#include "Profiler.h"
#include "Session/Session.h"
#include "Utility/Atomic.h"
#include "Utility/Map.h"
#include "Utility/Set.h"

#include <atomic>
#include <deque>
#include <thread>
#include <unordered_map>
#include <unordered_set>

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

  struct CorrIdState {
    size_t externId{Scope::DummyScopeId};
    size_t numKernels{1};
    bool isApiExternId{false};
  };

  using CorrIdToExternIdMap =
      ThreadSafeMap<uint64_t, CorrIdState,
                    std::unordered_map<uint64_t, CorrIdState>>;

  struct ExternIdState {
    std::map<Data *, std::unordered_map<uint64_t, std::pair<bool, size_t>>>
        dataToGraphNodeScopeId;
  };

  // TODO(Keren): replace `Data *` with `dataId` to avoid pointer recycling
  // issue.
  using ExternIdToStateMap =
      ThreadSafeMap<size_t, /*extern_id*/ ExternIdState,
                    std::unordered_map<size_t, ExternIdState>>;

protected:
  // OpInterface
  void startOp(const Scope &scope) override {
    this->correlation.pushExternId(scope.scopeId, this->threadState.createdOp);
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
  void clearCache() override {
    if (!getDataSet().empty()) {
      return;
    }
    correlation.clear();
  }
  virtual void doAddMetrics(
      size_t scopeId,
      const std::map<std::string, MetricValueType> &scalarMetrics,
      const std::map<std::string, TensorMetric> &tensorMetrics) override {
    pImpl->doAddMetrics(scopeId, scalarMetrics, tensorMetrics);
  }

  struct ThreadState {
    ConcreteProfilerT &profiler;
    SessionManager &sessionManager = SessionManager::instance();
    std::vector<Scope>
        scopeStack; // Used for nvtx range tracking or triton op tracking
    size_t opId{Scope::DummyScopeId};
    bool isStreamCapturing{false};
    bool isMetricKernelLaunching{false};
    bool createdOp{false};

    ThreadState(ConcreteProfilerT &profiler) : profiler(profiler) {}

    void enterOp() {
      createdOp = false;
      if (profiler.isOpInProgress()) {
        return;
      }
      opId = Scope::getNewScopeId();
      createdOp = true;
      profiler.enterOp(Scope(opId));
      createdOp = false;
    }

    void exitOp() {
      if (!profiler.isOpInProgress())
        return;
      profiler.exitOp(Scope(opId));
    }

    void enterScope(const std::string &name) {
      auto scope = Scope(name);
      scopeStack.push_back(scope);
      sessionManager.enterScope(scope);
    }

    void exitScope() {
      if (scopeStack.empty()) {
        return;
      }
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
    struct ExternIdFrame {
      size_t externId;
      bool isApiExternId;
    };
    static thread_local std::deque<ExternIdFrame> externIdQueue;

    Correlation() = default;

    void clear() {
      corrIdToExternId.clear();
      externIdToState.clear();
    }

    void submit(const uint64_t correlationId) {
      atomicMax(maxSubmittedCorrelationId, correlationId);
    }

    void complete(const uint64_t correlationId) {
      atomicMax(maxCompletedCorrelationId, correlationId);
    }

    void pushExternId(size_t externId, bool isApiExternId) {
      externIdQueue.push_back(ExternIdFrame{externId, isApiExternId});
    }

    void popExternId() {
      externIdQueue.pop_front();
    }

    // Correlate the correlationId with the last externId
    void correlate(uint64_t correlationId, size_t numInstances = 1) {
      if (externIdQueue.empty())
        return;
      corrIdToExternId.insert(
          correlationId, CorrIdState{externIdQueue.back().externId, numInstances,
                                     externIdQueue.back().isApiExternId});
    }

    void setGraphNodeScope(size_t externId, Data *data, uint64_t nodeId,
                           bool isApi, size_t scopeId) {
      externIdToState.upsert(externId, [&](ExternIdState &state) {
        state.dataToGraphNodeScopeId[data][nodeId] = {isApi, scopeId};
      });
    }

    void mergeGraphNodeScopes(
        size_t externId,
        const std::map<Data *, std::unordered_map<uint64_t, std::pair<bool, size_t>>>
            &scopes) {
      externIdToState.upsert(externId, [&](ExternIdState &state) {
        for (const auto &[data, nodeIdToScope] : scopes) {
          auto &dst = state.dataToGraphNodeScopeId[data];
          dst.insert(nodeIdToScope.begin(), nodeIdToScope.end());
        }
      });
    }

    bool getGraphNodeScope(size_t externId, Data *data, uint64_t nodeId,
                           bool &isApi, size_t &scopeId) const {
      bool found = false;
      externIdToState.withRead(externId, [&](const ExternIdState &state) {
        auto dataIt = state.dataToGraphNodeScopeId.find(data);
        if (dataIt == state.dataToGraphNodeScopeId.end()) {
          return;
        }
        auto nodeIt = dataIt->second.find(nodeId);
        if (nodeIt == dataIt->second.end()) {
          return;
        }
        isApi = nodeIt->second.first;
        scopeId = nodeIt->second.second;
        found = true;
      });
      return found;
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
