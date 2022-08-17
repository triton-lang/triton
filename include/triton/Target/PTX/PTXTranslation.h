#ifndef TRITON_TARGET_PTXTRANSLATION_H
#define TRITON_TARGET_PTXTRANSLATION_H

#include "triton/driver/dispatch.h"

#include <string>

namespace mlir {

class ModuleOp;

} // namespace mlir

namespace triton {

template <CUdevice_attribute attr> int cuGetInfo(CUdevice device) {
  int res;
  driver::dispatch::cuDeviceGetAttribute(&res, attr, device);
  return res;
}

void getCuCCAndVersionFromDevice(uint64_t device, int *cc, int *version,
                                 std::string *ptxasPath);

// Translate TritonGPU IR to PTX code.
std::tuple<std::string, // ptx code
           size_t,      // PTX cc
           int,         // PTX version
           std::string  // ptxas path
           >
translateTritonGPUToPTX(mlir::ModuleOp module, uint64_t device);

} // namespace triton

#endif
