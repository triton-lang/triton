#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;

namespace mlir::LLVM::AMD {

ElemLocationKey getElemCoordsFromReg(tt::LinearLayout ll, unsigned regId,
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

llvm::MapVector<ElemLocationKey, unsigned>
mapRegToCoordinates(tt::LinearLayout ll, MLIRContext *ctx) {
  llvm::MapVector<ElemLocationKey, unsigned> elemToReg;
  StringAttr kReg = StringAttr::get(ctx, "register");
  auto regBases = ll.getBases().lookup(kReg);
  int regNum = 1 << regBases.size();
  for (int regId = 0; regId < regNum; ++regId) {
    elemToReg[getElemCoordsFromReg(ll, regId, ctx)] = regId;
  }
  return elemToReg;
} // namespace mlir::triton
} // namespace mlir::LLVM::AMD
