#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::triton::gpu;

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/Types.cpp.inc"

static constexpr llvm::StringRef kMutableMemory = "mutable";

Type MemDescType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return Type();

  SmallVector<int64_t> dimensions; // required
  if (failed(parser.parseDimensionList(dimensions, /*allowDynamic=*/false)))
    return Type();

  Type elementType; // required
  if (failed(parser.parseType(elementType)))
    return Type();

  Attribute encoding; // required
  if (failed(parser.parseComma()) || failed(parser.parseAttribute(encoding)))
    return Type();

  Attribute memorySpace; // required
  if (failed(parser.parseComma()) || failed(parser.parseAttribute(memorySpace)))
    return Type();

  bool mutableMemory = false;      // optional
  SmallVector<int64_t> allocShape; // optional
  if (succeeded(parser.parseOptionalComma())) {
    if (succeeded(parser.parseOptionalKeyword(kMutableMemory))) {
      mutableMemory = true;
      if (succeeded(parser.parseOptionalComma())) {
        if (failed(parser.parseDimensionList(allocShape, /*allowDynamic=*/false,
                                             /*withTrailingX=*/false))) {
          return Type();
        }
      }
    } else if (failed(parser.parseDimensionList(allocShape,
                                                /*allowDynamic=*/false,
                                                /*withTrailingX=*/false))) {
      return Type();
    }
  }

  if (parser.parseGreater())
    return Type();

  if (allocShape.size() > 0)
    return MemDescType::get(parser.getContext(), dimensions, elementType,
                            encoding, memorySpace, mutableMemory, allocShape);

  return MemDescType::get(parser.getContext(), dimensions, elementType,
                          encoding, memorySpace, mutableMemory, dimensions);
}

void MemDescType::print(AsmPrinter &printer) const {
  printer << "<";
  auto shape = getShape();
  for (auto dim : shape)
    printer << dim << "x";
  printer << getElementType();
  if (getEncoding())
    printer << ", " << getEncoding();
  if (getMemorySpace())
    printer << ", " << getMemorySpace();
  if (getMutableMemory())
    printer << ", " << kMutableMemory;
  auto allocShape = getAllocShape();
  if (allocShape != shape) {
    printer << ", " << allocShape[0];
    for (auto dim : allocShape.drop_front(1)) {
      printer << "x" << dim;
    }
  }
  printer << ">";
}

LogicalResult MemDescType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<int64_t> shape, Type elementType,
                                  Attribute encoding, Attribute memorySpace,
                                  bool mutableMemory,
                                  ArrayRef<int64_t> allocShape) {
  if (allocShape.size() < shape.size())
    emitError() << "alloc shape must have at least as many dimensions as shape";
  if (llvm::any_of(shape, [](int64_t dim) { return dim < 0; }))
    emitError() << "shape must have non-negative dimensions";
  if (llvm::any_of(llvm::zip(shape, allocShape), [](auto pair) {
        return std::get<0>(pair) > std::get<1>(pair);
      }))
    emitError() << "shape must be less than or equal to allocShape";
  auto ctx = encoding.getContext();
  if (auto enc = dyn_cast<nvidia_gpu::TensorMemoryEncodingAttr>(encoding)) {
    if (shape.size() != 2) {
      return emitError() << "shape must be a 2-element array";
    }
    if (memorySpace != nvidia_gpu::TensorMemorySpaceAttr::get(ctx)) {
      return emitError() << "memorySpace must be TensorMemorySpace";
    }
    if (shape != allocShape) {
      return emitError() << "shape must be equal to allocShape";
    }
    if (shape[0] < enc.getBlockM() * enc.getCTASplitM() ||
        shape[1] < enc.getBlockN() * enc.getCTASplitN() *
                       (enc.getUnpacked() ? 2 : 1)) {
      return emitError() << "shape must be at least "
                         << enc.getBlockM() * enc.getCTASplitM() << "x"
                         << enc.getBlockN() * enc.getCTASplitN() *
                                (enc.getUnpacked() ? 2 : 1);
    }
    auto ll = toLinearLayout(shape, enc, {});
    auto dims = standardOutDimNames(ctx, 2);
    if (ll.getInDimSize(dims[0]) != shape[0] ||
        ll.getInDimSize(dims[1]) != shape[1]) {
      return emitError() << "shape must be equal to "
                         << ll.getInDimSize(dims[0]) << "x"
                         << ll.getInDimSize(dims[1]);
    }
  } else if (auto enc = dyn_cast<SharedEncodingTrait>(encoding)) {
    // TODO Add verifier
  } else {
    return emitError() << "unsupported encoding";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Triton Dialect
//===----------------------------------------------------------------------===//
void ::mlir::triton::gpu::TritonGPUDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton/Dialect/TritonGPU/IR/Types.cpp.inc"
      >();
}
