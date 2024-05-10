#ifndef PROTON_PROFILER_CUPTI_PROFILER_H_
#define PROTON_PROFILER_CUPTI_PROFILER_H_

#include "Context/Context.h"
#include "Profiler.h"

#include <atomic>
#include <map>

namespace proton {

class CuptiProfiler : public Profiler,
                      public OpInterface,
                      public Singleton<CuptiProfiler> {
public:
  CuptiProfiler();
  virtual ~CuptiProfiler();

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
  // pimpl-idiom to hide the implementation details
  struct CuptiCallback;
  std::unique_ptr<CuptiCallback> cuptiCallback;
};

} // namespace proton

#endif // PROTON_PROFILER_CUPTI_PROFILER_H_
