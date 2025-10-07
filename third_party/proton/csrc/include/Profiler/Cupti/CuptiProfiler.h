#ifndef PROTON_PROFILER_CUPTI_PROFILER_H_
#define PROTON_PROFILER_CUPTI_PROFILER_H_

#include "Profiler/GPUProfiler.h"

namespace proton {

class CuptiProfiler : public GPUProfiler<CuptiProfiler> {
public:
  CuptiProfiler();
  virtual ~CuptiProfiler();

private:
  struct CuptiProfilerPimpl;

  virtual void
  doSetMode(const std::vector<std::string> &modeAndOptions) override;
};

} // namespace proton

#endif // PROTON_PROFILER_CUPTI_PROFILER_H_
