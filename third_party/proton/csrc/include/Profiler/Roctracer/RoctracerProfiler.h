#ifndef PROTON_PROFILER_ROCTRACER_PROFILER_H_
#define PROTON_PROFILER_ROCTRACER_PROFILER_H_

#include "Profiler/GPUProfiler.h"

namespace proton {

class RoctracerProfiler : public GPUProfiler<RoctracerProfiler> {
public:
  RoctracerProfiler();
  virtual ~RoctracerProfiler();

  int64_t getTimestampOffsetNs() const override { return timestampOffsetNs; }

private:
  struct RoctracerProfilerPimpl;
  int64_t timestampOffsetNs{};
  bool isTimestampCalibrated{false};

  virtual void
  doSetMode(const std::vector<std::string> &modeAndOptions) override;
};

} // namespace proton

#endif // PROTON_PROFILER_ROCTRACER_PROFILER_H_
