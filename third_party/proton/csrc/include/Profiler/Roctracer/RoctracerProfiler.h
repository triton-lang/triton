#ifndef PROTON_PROFILER_ROCTRACER_PROFILER_H_
#define PROTON_PROFILER_ROCTRACER_PROFILER_H_

#include "Profiler/GPUProfiler.h"

namespace proton {

class RoctracerProfiler : public GPUProfiler<RoctracerProfiler> {
public:
  RoctracerProfiler();
  virtual ~RoctracerProfiler();

private:
  struct RoctracerProfilerPimpl;

  // XXX(Keren): RocTracer is deprecated, so we don't fix problems
  int64_t getTimestampOffsetNs() const override { return 0; }

  virtual void
  doSetMode(const std::vector<std::string> &modeAndOptions) override;
};

} // namespace proton

#endif // PROTON_PROFILER_ROCTRACER_PROFILER_H_
