#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;

namespace mlir::LLVM::AMD {

ElemLocationKey getElemCoordinatesFromRegisters(tt::LinearLayout ll,
                                                unsigned regId,
                                                MLIRContext *ctx) {
  StringAttr kReg = StringAttr::get(ctx, "register");
  SmallVector<std::pair<StringAttr, int32_t>> hardwareLocation;
  for (auto dimName : ll.getInDimNames()) {
    if (dimName == kReg)
      hardwareLocation.push_back({dimName, regId});
    else
      hardwareLocation.push_back({dimName, 0});
  }
  return ll.apply(hardwareLocation);
}

std::optional<int> getRegFromCoordinates(tt::LinearLayout ll,
                                         ElemLocationKey coordinates,
                                         MLIRContext *ctx) {
  auto hardwareLocation = ll.pseudoinvert().apply(coordinates);
  llvm::MapVector<ElemLocationKey, unsigned> elemToReg;
  StringAttr kReg = StringAttr::get(ctx, "register");
  for (auto location : hardwareLocation) {
    if (location.first == kReg)
      return location.second;
  }

  return {};
} // namespace mlir::triton

} // namespace mlir::LLVM::AMD
