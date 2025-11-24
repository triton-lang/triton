#ifndef PROTON_PROFILER_ROCM_ROCPROFILER_PROFILER_H_
#define PROTON_PROFILER_ROCM_ROCPROFILER_PROFILER_H_

#include "Profiler/GPUProfiler.h"

namespace proton {

class RocprofilerProfiler : public GPUProfiler<RocprofilerProfiler> {
public:
  RocprofilerProfiler();
  virtual ~RocprofilerProfiler();

  struct RocprofilerProfilerPimpl;

private:
  virtual void
  doSetMode(const std::vector<std::string> &modeAndOptions) override;
};

} // namespace proton

#endif // PROTON_PROFILER_ROCM_ROCPROFILER_PROFILER_H_
