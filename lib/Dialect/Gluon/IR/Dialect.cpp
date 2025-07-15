#include "triton/Dialect/Gluon/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton::gpu;
namespace gluon = mlir::triton::gluon;

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/Gluon/IR/Dialect.cpp.inc"
#include "triton/Dialect/Gluon/IR/GluonAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/Gluon/IR/Ops.cpp.inc"

namespace {

// Layout inference for AutoEncodingAttr -> always propagate AutoEncodingAttr to
// results
struct GluonInferLayoutInterface : public triton::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  LogicalResult inferAutoEncoding(Attribute operandEncoding,
                                  Attribute &resultEncoding) const {
    assert(isa<gluon::AutoEncodingAttr>(operandEncoding));
    resultEncoding = operandEncoding;
    return success();
  }

  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding,
                        std::optional<Location> loc) const override {
    return inferAutoEncoding(operandEncoding, resultEncoding);
  }

  LogicalResult
  inferTransOpEncoding(Attribute operandEncoding, ArrayRef<int64_t> shape,
                       ArrayRef<int32_t> order, Attribute &resultEncoding,
                       std::optional<Location> loc) const override {
    return inferAutoEncoding(operandEncoding, resultEncoding);
  }

  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> location) const override {
    return inferAutoEncoding(operandEncoding, resultEncoding);
  }

  LogicalResult
  inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                     Attribute resultEncoding,
                     std::optional<Location> location) const override {
    return inferAutoEncoding(operandEncoding, resultEncoding);
  }

  LogicalResult
  verifyDotOpEncodingCompatibility(Operation *op, Attribute operandEncodingA,
                                   Attribute operandEncodingB) const override {
    return success();
  }

  LogicalResult
  verifyLayoutsAreEqual(ArrayRef<int64_t> shape, Attribute expected,
                        Attribute got,
                        std::optional<Location> loc) const override {
    return success(expected == got);
  }

  LogicalResult
  inferReshapeOpEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                         ArrayRef<int64_t> dstShape, Attribute &dstEnc,
                         std::optional<Location> loc) const override {
    return inferAutoEncoding(srcEnc, dstEnc);
  }

  LogicalResult
  inferDefaultJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                             ArrayRef<int64_t> shape,
                             std::optional<Location> loc) const override {
    return inferAutoEncoding(srcEnc, dstEnc);
  }

  LogicalResult
  inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                       ArrayRef<int64_t> shape,
                       std::optional<Location> loc) const override {
    return inferAutoEncoding(srcEnc, dstEnc);
  }

  LogicalResult
  inferFp4ToFpOpEncoding(ArrayRef<int64_t> shape, int axis, Attribute srcEnc,
                         Attribute &dstEnc, bool fwdInference,
                         std::optional<Location> loc) const override {
    return inferAutoEncoding(srcEnc, dstEnc);
  }
};
} // namespace

namespace mlir::triton::gluon {

void GluonDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/Gluon/IR/GluonAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/Gluon/IR/Ops.cpp.inc"
      >();
  addInterfaces<GluonInferLayoutInterface>();
}

void SetAutoLayoutOp::build(OpBuilder &builder, OperationState &state,
                            Attribute enc, Value value) {
  auto resTy = cast<RankedTensorType>(value.getType()).cloneWithEncoding(enc);
  return build(builder, state, resTy, value);
}

LogicalResult SetAutoLayoutOp::verify() {
  if (!isa<gluon::AutoEncodingAttr>(getSrc().getType().getEncoding())) {
    return emitOpError("input tensor must have an auto layout type");
  }
  auto dstEncoding = getType().getEncoding();
  if (!dstEncoding)
    return emitOpError("result tensor must have an encoding");
  if (isa<gluon::AutoEncodingAttr>(dstEncoding))
    return emitOpError("result type must not be auto layout");
  return success();
}

} // namespace mlir::triton::gluon
