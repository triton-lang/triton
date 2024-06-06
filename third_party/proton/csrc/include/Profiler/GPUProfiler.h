#ifndef PROTON_PROFILER_GPU_PROFILER_H_
#define PROTON_PROFILER_GPU_PROFILER_H_

#include "Context/Context.h"
#include "Profiler.h"
#include "Utility/Atomic.h"
#include "Utility/Map.h"
#include "Utility/Set.h"

#include <atomic>
#include <deque>
#include <map>
#include <thread>
#include <vector>

namespace proton {

// Singleton<ConcreteProfilerT>: Each concrete GPU profiler, e.g.,
// CuptiProfiler, should be a singleton.
template <typename ConcreteProfilerT>
class GPUProfiler : public Profiler,
                    public ThreadLocalOpInterface,
                    public Singleton<ConcreteProfilerT> {
public:
  GPUProfiler() = default;
  virtual ~GPUProfiler() = default;

protected:
  // OpInterface
  void startOp(const Scope &scope) override { pImpl->startOp(scope); }
  void stopOp(const Scope &scope) override { pImpl->stopOp(scope); }

  // Profiler
  virtual void doStart() override { pImpl->doStart(); }
  virtual void doFlush() override { pImpl->doFlush(); }
  virtual void doStop() override { pImpl->doStop(); }

  struct ProfilerState {
    ConcreteProfilerT &profiler;

    ProfilerState(ConcreteProfilerT &profiler) : profiler(profiler) {}

    void record(size_t scopeId) {
      if (profiler.isOpInProgress())
        return;
      std::set<Data *> dataSet = profiler.getDataSet();
      for (auto data : dataSet)
        data->addScope(scopeId);
      profiler.correlation.apiExternIds.insert(scopeId);
    }

    void enterOp(size_t scopeId) {
      if (profiler.isOpInProgress())
        return;
      profiler.correlation.pushExternId(scopeId);
      profiler.setOpInProgress(true);
    }

    void exitOp() {
      if (!profiler.isOpInProgress())
        return;
      profiler.correlation.popExternId();
      profiler.setOpInProgress(false);
    }
  };

  struct Correlation {
    std::atomic<uint64_t> maxSubmittedCorrelationId{0};
    std::atomic<uint64_t> maxCompletedCorrelationId{0};
    ThreadSafeMap<uint64_t, size_t> corrIdToExternId;
    ThreadSafeSet<size_t> apiExternIds;
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
    void correlate(uint64_t correlationId) {
      if (externIdQueue.empty())
        return;
      corrIdToExternId[correlationId] = externIdQueue.back();
    }

    template <typename FlushFnT>
    void flush(uint64_t maxRetries, uint64_t sleepMs, FlushFnT &&flushFn) {
      flushFn();
      auto submittedId = maxSubmittedCorrelationId.load();
      auto completedId = maxCompletedCorrelationId.load();
      auto retries = maxRetries;
      while ((completedId < submittedId) && retries > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(sleepMs));
        flushFn();
        completedId = maxCompletedCorrelationId.load();
        --retries;
      }
    }
  };

  static thread_local ProfilerState profilerState;
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

    virtual void startOp(const Scope &scope) = 0;
    virtual void stopOp(const Scope &scope) = 0;
    virtual void doStart() = 0;
    virtual void doFlush() = 0;
    virtual void doStop() = 0;

  protected:
    ConcreteProfilerT &profiler;
  };
  std::unique_ptr<GPUProfilerPimplInterface> pImpl;
};

} // namespace proton

#endif // PROTON_PROFILER_GPU_PROFILER_H_
