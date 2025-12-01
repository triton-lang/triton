#ifndef PROTON_PROFILER_ROCM_ROCPROFILER_PROFILER_H_
#define PROTON_PROFILER_ROCM_ROCPROFILER_PROFILER_H_

#include "Profiler/GPUProfiler.h"

namespace proton {

class RocprofProfiler : public GPUProfiler<RocprofProfiler> {
public:
  RocprofProfiler();
  virtual ~RocprofProfiler();

  struct RocprofProfilerPimpl;

private:
  virtual void
  doSetMode(const std::vector<std::string> &modeAndOptions) override;
};

} // namespace proton

#endif // PROTON_PROFILER_ROCM_ROCPROFILER_PROFILER_H_
