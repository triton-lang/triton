#ifndef PROTON_PROFILER_ROCPROFSDK_PROFILER_H_
#define PROTON_PROFILER_ROCPROFSDK_PROFILER_H_

#include "Profiler/GPUProfiler.h"

namespace proton {

class RocprofSDKProfiler : public GPUProfiler<RocprofSDKProfiler> {
public:
  RocprofSDKProfiler();
  virtual ~RocprofSDKProfiler();

  int64_t getTimestampOffsetNs() const override { return timestampOffsetNs; }

  struct RocprofSDKProfilerPimpl;

private:
  int64_t timestampOffsetNs{};
  bool isTimestampCalibrated{false};

  virtual void
  doSetMode(const std::vector<std::string> &modeAndOptions) override;
};

} // namespace proton

#endif // PROTON_PROFILER_ROCPROFSDK_PROFILER_H_
