#include "Driver/GPU/Hip.h"
#include "Driver/Dispatch.h"

namespace proton {

namespace hip {

struct ExternLibHip : public ExternLibBase {
  using RetType = hipError_t;
  static constexpr const char *name = "libamdhip64.so";
  static constexpr RetType success = hipSuccess;
  static void *lib;
};

void *ExternLibHip::lib = nullptr;

DEFINE_DISPATCH(ExternLibHip, deviceSynchronize, hipDeviceSynchronize)

} // namespace hip

} // namespace proton
