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
  // Use the pimpl idiom to hide the implementation details. This lets us avoid
  // including the cupti header from this header. The cupti header and the
  // equivalent header from AMD define conflicting macros, so we want to use
  // those headers only within cc files.
  struct CuptiProfilerPimpl;
  std::unique_ptr<CuptiProfilerPimpl> pImpl;
};

} // namespace proton

#endif // PROTON_PROFILER_CUPTI_PROFILER_H_
