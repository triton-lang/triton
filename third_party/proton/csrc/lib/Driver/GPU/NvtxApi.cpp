#include "Driver/GPU/NvtxApi.h"
#include "Driver/GPU/CuptiApi.h"

#include <cstdint>
#include <cstdlib>

namespace proton {

namespace {

// Declare nvtx function params without including the nvtx header
struct RangePushAParams {
  const char *message;
};

} // namespace

namespace nvtx {

void enable() {
  // Get cupti lib path and append it to NVTX_INJECTION64_PATH
  const std::string cuptiLibPath = cupti::getLibPath();
  if (!cuptiLibPath.empty()) {
    setenv("NVTX_INJECTION64_PATH", cuptiLibPath.c_str(), 1);
  }
}

void disable() { unsetenv("NVTX_INJECTION64_PATH"); }

std::string getMessageFromRangePushA(const void *params) {
  if (const auto *p = static_cast<const RangePushAParams *>(params))
    return std::string(p->message ? p->message : "");
  return "";
}

} // namespace nvtx

} // namespace proton
