#ifndef PROTON_PROFILER_CUPTI_PROFILER_H_
#define PROTON_PROFILER_CUPTI_PROFILER_H_

#include "Context/Context.h"
#include "Driver/GPU/Cupti.h"
#include "Profiler.h"

#include <atomic>
#include <map>

namespace proton {

class CuptiProfiler : public Profiler,
                      public OpInterface,
                      public Singleton<CuptiProfiler> {
public:
  CuptiProfiler() = default;
  virtual ~CuptiProfiler() = default;

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
  static void allocBuffer(uint8_t **buffer, size_t *bufferSize,
                          size_t *maxNumRecords);
  static void completeBuffer(CUcontext context, uint32_t streamId,
                             uint8_t *buffer, size_t size, size_t validSize);
  static void processActivity(std::map<uint32_t, size_t> &correlation,
                              std::set<Data *> &dataSet,
                              CUpti_Activity *activity);
  static void callback(void *userData, CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbId, const void *cbData);

  const inline static size_t AlignSize = 8;
  const inline static size_t BufferSize = 64 * 1024 * 1024;

  std::map<uint32_t, size_t> correlation;
  CUpti_SubscriberHandle subscriber{};
  struct CuptiState {
    CuptiProfiler &profiler;
    std::set<Data *> dataSet;
    size_t level{0};
    bool isRecording{false};
    Scope scope{};

    CuptiState(CuptiProfiler &profiler) : profiler(profiler) {}

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

  static inline thread_local CuptiState cuptiState{CuptiProfiler::instance()};
};

} // namespace proton

#endif // PROTON_PROFILER_CUPTI_PROFILER_H_
