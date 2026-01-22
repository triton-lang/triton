#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {
ElemLocationKey getElemCoordinatesFromRegisters(triton::LinearLayout ll,
                                                unsigned regId,
                                                MLIRContext *ctx) {
  StringAttr kReg = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  StringAttr kBlock = StringAttr::get(ctx, "block");

  SmallVector<std::pair<StringAttr, int32_t>> hardwareLocation = {
      {kReg, static_cast<int32_t>(regId)},
      {kLane, 0},
      {kWarp, 0},
      {kBlock, 0},
  };

  return ll.apply(hardwareLocation);
}

std::optional<int> getRegFromCoordinates(triton::LinearLayout ll,
                                         ElemLocationKey coordinates,
                                         MLIRContext *ctx) {
  auto dims = ll.pseudoinvert().apply(coordinates);
  StringAttr kReg = StringAttr::get(ctx, "register");
  assert(dims[0].first == kReg && "First dimension must be 'register'");

  int regId = dims[0].second; // "register"
  if (dims[1].second != 0 || dims[2].second != 0 || dims[3].second != 0)
    return std::nullopt;
  return regId;
}
} // namespace mlir::triton::AMD
