#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonInstrument/IR/Ops.cpp.inc"

#include "triton/Dialect/TritonInstrument/IR/OpsEnums.cpp.inc"

namespace mlir {
namespace triton {
namespace instrument {

OpFoldResult ExperimentalFPSanEmbedOp::fold(FoldAdaptor adaptor) {
  if (auto unembed = getVal().getDefiningOp<ExperimentalFPSanUnembedOp>())
    if (unembed.getVal().getType() == getType())
      return unembed.getVal();
  return {};
}

OpFoldResult ExperimentalFPSanUnembedOp::fold(FoldAdaptor adaptor) {
  if (auto embed = getVal().getDefiningOp<ExperimentalFPSanEmbedOp>())
    if (embed.getVal().getType() == getType())
      return embed.getVal();
  return {};
}

} // namespace instrument
} // namespace triton
} // namespace mlir
