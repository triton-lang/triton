#include "Profiler/RocprofSDK/RocprofSDKProfiler.h"

#include <stdexcept>

namespace proton {

template <>
thread_local GPUProfiler<RocprofSDKProfiler>::ThreadState
    GPUProfiler<RocprofSDKProfiler>::threadState(
        RocprofSDKProfiler::instance());

RocprofSDKProfiler::RocprofSDKProfiler() {
  throw std::runtime_error(
      "RocprofSDKProfiler is not available (rocprofiler-sdk headers were not "
      "found at build time)");
}

RocprofSDKProfiler::~RocprofSDKProfiler() = default;

void RocprofSDKProfiler::doSetMode(
    const std::vector<std::string> & /*modeAndOptions*/) {
  throw std::runtime_error(
      "RocprofSDKProfiler is not available (rocprofiler-sdk headers were not "
      "found at build time)");
}

} // namespace proton
