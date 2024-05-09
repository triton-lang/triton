#ifndef PROTON_PROFILER_CUPTI_PROFILER_H_
#define PROTON_PROFILER_CUPTI_PROFILER_H_

#include "Context/Context.h"
#include "Profiler.h"

#include <atomic>
#include <map>

struct CUctx_st;

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
  static void completeBuffer(struct CUctx_st *context, uint32_t streamId,
                             uint8_t *buffer, size_t size, size_t validSize);

  const inline static size_t AlignSize = 8;
  const inline static size_t BufferSize = 64 * 1024 * 1024;

  std::map<uint32_t, size_t> correlation;
  void *subscriber{};
};

} // namespace proton

#endif // PROTON_PROFILER_CUPTI_PROFILER_H_
