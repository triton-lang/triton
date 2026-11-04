#ifndef PROTON_PROFILER_ROCPROFSDK_PROFILER_H_
#define PROTON_PROFILER_ROCPROFSDK_PROFILER_H_

#include "Profiler/GPUProfiler.h"

namespace proton {

class RocprofSDKProfiler final : public GPUProfiler<RocprofSDKProfiler> {
public:
  ~RocprofSDKProfiler() override;

  struct RocprofSDKProfilerPimpl;

private:
  friend class Singleton<RocprofSDKProfiler>;

  RocprofSDKProfiler();

  void doSetMode(const std::vector<std::string> &modeAndOptions) override;
};

} // namespace proton

#endif // PROTON_PROFILER_ROCPROFSDK_PROFILER_H_
