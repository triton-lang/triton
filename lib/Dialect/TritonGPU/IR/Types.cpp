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
  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
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

  if (!allocShape.empty())
    return MemDescType::getChecked(loc, parser.getContext(), dimensions,
                                   elementType, encoding, memorySpace,
                                   mutableMemory, allocShape);

  return MemDescType::getChecked(loc, parser.getContext(), dimensions,
                                 elementType, encoding, memorySpace,
                                 mutableMemory, dimensions);
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
  if (shape.empty()) {
    return emitError() << "rank 0 memdesc is not allowed";
  }
  // Every dimension but the first (to allow for pipelining) must be a power of
  // 2
  if (!isa<PaddedSharedEncodingAttr>(encoding) &&
      llvm::any_of(shape.drop_front(1),
                   [](int64_t dim) { return !llvm::isPowerOf2_64(dim); }))
    return emitError() << "shape must have power-of-2 dimensions; got "
                       << shape;
  if (allocShape.size() < shape.size())
    return emitError()
           << "alloc shape must have at least as many dimensions as shape";
  if (llvm::any_of(
          llvm::zip(shape, allocShape.take_back(shape.size())),
          [](auto pair) { return std::get<0>(pair) > std::get<1>(pair); }))
    return emitError() << "shape must be less than or equal to allocShape. "
                       << "shape = " << shape
                       << ", allocShape = " << allocShape;
  auto ctx = encoding.getContext();
  if (auto enc = dyn_cast<nvidia_gpu::TensorMemoryEncodingAttr>(encoding)) {
    if (memorySpace != nvidia_gpu::TensorMemorySpaceAttr::get(ctx)) {
      return emitError() << "memorySpace must be TensorMemorySpace";
    }
    if (shape.size() != 2 && shape.size() != 3) {
      return emitError() << "rank must be 2 or 3";
    }
    auto bitwidth = elementType.getIntOrFloatBitWidth();
    if (!enc.getUnpacked() && bitwidth > 16) {
      return emitError() << "bitwidth must be <= 16 for packed tensor memory";
    }
    if (enc.getUnpacked() && (16 != bitwidth && 32 != bitwidth)) {
      return emitError()
             << "bitwidth must be either 16 or 32 for unpacked tensor memory";
    }
    shape = shape.take_back(2);
    allocShape = allocShape.take_back(2);
    if (allocShape[0] < enc.getBlockM() * enc.getCTASplitM() ||
        allocShape[1] < enc.getBlockN() * enc.getCTASplitN()) {
      return emitError() << "the allocation shape must be at least "
                         << enc.getBlockM() * enc.getCTASplitM() << "x"
                         << enc.getBlockN() * enc.getCTASplitN() << ". Got "
                         << allocShape;
    }
    auto ll = toLinearLayout(allocShape, enc);
    auto dims = standardOutDimNames(ctx, 2);
    if (ll.getOutDimSize(dims[0]) != allocShape[0] ||
        ll.getOutDimSize(dims[1]) != allocShape[1]) {
      return emitError() << "allocation shape must be equal to "
                         << ll.getOutDimSize(dims[0]) << "x"
                         << ll.getOutDimSize(dims[1]);
    }
  } else if (auto enc = dyn_cast<SharedEncodingTrait>(encoding)) {
    if (memorySpace != SharedMemorySpaceAttr::get(ctx)) {
      return emitError()
             << "memorySpace must be SharedMemorySpace for shared encoding. "
             << "Got " << memorySpace;
    }
  } else if (auto enc = dyn_cast<nvidia_gpu::TensorMemoryScalesEncodingAttr>(
                 encoding)) {
    if (memorySpace != nvidia_gpu::TensorMemorySpaceAttr::get(ctx)) {
      return emitError() << "memorySpace must be TensorMemorySpace";
    }
    // TODO Add rest of verifier
  } else {
    return emitError() << encoding << " is not a valid encoding";
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
