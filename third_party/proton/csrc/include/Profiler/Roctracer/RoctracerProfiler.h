#ifndef PROTON_PROFILER_ROCTRACER_PROFILER_H_
#define PROTON_PROFILER_ROCTRACER_PROFILER_H_

#include "Profiler/GPUProfiler.h"

namespace proton {

class RoctracerProfiler final : public GPUProfiler<RoctracerProfiler> {
public:
  ~RoctracerProfiler() override;

private:
  friend class Singleton<RoctracerProfiler>;

  RoctracerProfiler();

  struct RoctracerProfilerPimpl;

  void doSetMode(const std::vector<std::string> &modeAndOptions) override;
};

} // namespace proton

#endif // PROTON_PROFILER_ROCTRACER_PROFILER_H_
