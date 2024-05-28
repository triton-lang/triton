#ifndef PROTON_PROFILER_ROCTRACER_PROFILER_H_
#define PROTON_PROFILER_ROCTRACER_PROFILER_H_

#include "Context/Context.h"
#include "Profiler.h"
#include "roctracer/roctracer.h"

#include <map>

namespace proton {

class RoctracerProfiler : public Profiler,
                          public OpInterface,
                          public Singleton<RoctracerProfiler> {
public:
  RoctracerProfiler() = default;
  virtual ~RoctracerProfiler() = default;

  // External Correlation
  enum CorrelationDomain {
    begin,
    Default = begin,
    Domain0 = begin,
    Domain1,
    end,
    size = end
  };
  static void pushCorrelationID(uint64_t id, CorrelationDomain type);
  static void popCorrelationID(CorrelationDomain type);

protected:
  // OpInterface
  void startOp(const Scope &scope) override final;
  void stopOp(const Scope &scope) override final;
  void setOpInProgress(bool value) override final;
  bool isOpInProgress() override final;

  // Profiler
  void doStart() override;
  void doFlush() override;
  void doStop() override;

private:
  static void apiCallback(uint32_t domain, uint32_t cid,
                          const void *callback_data, void *arg);
  static void activityCallback(const char *begin, const char *end, void *arg);
  static void processActivity(std::map<uint32_t, size_t> &correlation,
                              std::set<Data *> &dataSet,
                              const roctracer_record_t *activity);

  const inline static size_t AlignSize = 8;
  const inline static size_t BufferSize = 64 * 1024 * 1024;

  std::map<uint32_t, size_t> correlation;
  std::mutex correlationLock;

  bool externalCorrelationEnabled{true};

  struct RoctracerState {
    RoctracerProfiler &profiler;
    std::set<Data *> dataSet;
    size_t level{0};
    bool isRecording{false};
    Scope scope{};

    RoctracerState(RoctracerProfiler &profiler) : profiler(profiler) {}

    void record(const Scope &scope, const std::set<Data *> &dataSet) {
      this->scope = scope;
      this->dataSet.insert(dataSet.begin(), dataSet.end());
    }

    void reset() {
      dataSet.clear();
      level = 0;
      scope = Scope();
    }

    void enterOp() {
      profiler.enterOp(scope);
      for (auto data : dataSet) {
        data->enterOp(scope);
      }
    }

    void exitOp() {
      profiler.exitOp(scope);
      for (auto data : dataSet) {
        data->exitOp(this->scope);
      }
    }
  };

  static inline thread_local RoctracerState roctracerState{
      RoctracerProfiler::instance()};
};

} // namespace proton

#endif // PROTON_PROFILER_ROCTRACER_PROFILER_H_
