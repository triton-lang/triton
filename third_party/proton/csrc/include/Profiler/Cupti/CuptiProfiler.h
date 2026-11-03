#ifndef PROTON_PROFILER_CUPTI_PROFILER_H_
#define PROTON_PROFILER_CUPTI_PROFILER_H_

#include "Profiler/GPUProfiler.h"

namespace proton {

class CuptiProfiler final : public GPUProfiler<CuptiProfiler> {
public:
  ~CuptiProfiler() override;

private:
  friend class Singleton<CuptiProfiler>;

  CuptiProfiler();

  struct CuptiProfilerPimpl;

  void doSetMode(const std::vector<std::string> &modeAndOptions) override;
};

} // namespace proton

#endif // PROTON_PROFILER_CUPTI_PROFILER_H_
