#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "triton/Dialect/Triton/IR/Utility.h"
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
  unsigned bitwidth = getIntOrFloatOrPtrBitWidth(elementType);
  if (bitwidth != 1 && bitwidth < 8)
    return emitError() << "element type bit width must be 1 or at least 8; got "
                       << bitwidth;
  if (llvm::is_contained(shape, 0))
    return emitError() << "shape has 0 dimension";
  if (llvm::is_contained(allocShape, 0))
    return emitError() << "alloc shape has 0 dimension";
  if (allocShape.size() < shape.size())
    return emitError()
           << "alloc shape must have at least as many dimensions as shape";
  auto layoutEncoding = dyn_cast_if_present<LayoutEncodingTrait>(encoding);
  if (!layoutEncoding ||
      !isa<nvidia_gpu::TensorMemoryEncodingAttr, SharedEncodingTrait,
           nvidia_gpu::TensorMemoryScalesEncodingAttr>(encoding))
    return emitError() << encoding << " is not a valid encoding";
  auto rank = layoutEncoding.getRank();
  if (isa<nvidia_gpu::TensorMemoryEncodingAttr>(encoding)) {
    if (shape.size() != 2 && shape.size() != 3)
      return emitError() << "rank must be 2 or 3";
  } else if (isa<SharedEncodingTrait>(encoding)) {
    if (!(rank == shape.size() || rank == shape.size() - 1))
      return emitError() << "rank must be equal to or one less than "
                         << "the shape size. Got " << rank << " and "
                         << shape.size();
  } else {
    assert(isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(encoding) &&
           "expected tensor-memory scales encoding");
    if (shape.size() != 2)
      return emitError() << "tensor-memory scale descriptors must have rank 2; "
                         << "got " << shape.size();
  }
  // Every layout dimension must be a power of 2; only a leading pipeline
  // dimension may have another positive size.
  ArrayRef<int64_t> layoutShape = dropPipeliningDim(shape, encoding);
  ArrayRef<int64_t> layoutAllocShape = dropPipeliningDim(allocShape, encoding);
  if (!llvm::all_of(layoutShape, llvm::isPowerOf2_64))
    return emitError()
           << "shape must have power-of-2 and non-zero dimensions; got "
           << shape;
  if (!llvm::all_of(layoutAllocShape, llvm::isPowerOf2_64))
    return emitError()
           << "alloc shape must have power-of-2 and non-zero dimensions; got "
           << allocShape;
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
    unsigned bitwidth = elementType.getIntOrFloatBitWidth();
    if (bitwidth < 8)
      return emitError()
             << "tensor-memory element type bit width must be at least 8; got "
             << bitwidth;
    if (bitwidth * enc.getColStride() > 32) {
      return emitError()
             << "bitwidth * colStride must be less than or equal to 32. Got "
             << bitwidth << " and " << enc.getColStride();
    }
    // Takes subslices into account and figures out whether we can construct
    // the linear layout at all
    allocShape = dropPipeliningDim(allocShape, enc);
    auto ctaSplit = enc.getCGALayout().getCTASplitNum();
    auto blockN = std::min<int32_t>(enc.getBlockN(), shape.back());
    if (shape[shape.size() - 2] < enc.getBlockM() * ctaSplit[0] ||
        shape[shape.size() - 1] < blockN * ctaSplit[1]) {
      return emitError() << "the tensor shape must be at least "
                         << enc.getBlockM() * ctaSplit[0] << "x"
                         << blockN * ctaSplit[1] << ". Got " << shape;
    }
    // Checks the layout of the allocation
    auto ll = toLinearLayout(allocShape, enc);
    // Sanity check that the layout is of the right shape
    auto dims = standardOutDimNames(ctx, 2);
    if (ll.getOutDimSize(dims[0]) != allocShape[0] ||
        ll.getOutDimSize(dims[1]) != allocShape[1]) {
      return emitError() << "allocation shape must be equal to "
                         << ll.getOutDimSize(dims[0]) << "x"
                         << ll.getOutDimSize(dims[1]);
    }
  } else if (isa<SharedEncodingTrait>(encoding)) {
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
    if (allocShape.size() != 2) {
      return emitError() << "Scales don't currently support multibuffering";
    }
    auto bitwidth = elementType.getIntOrFloatBitWidth();
    if (bitwidth != 8) {
      return emitError() << "bitwidth must be 8";
    }
  }

  // PaddedSharedEncodingAttr is also a SharedEncodingTrait but we have some
  // additional rules to verify.
  if (auto enc = dyn_cast<PaddedSharedEncodingAttr>(encoding)) {
    auto rank = enc.getRank();
    // Ensure linear component's outDims match the alloc size ignoring
    // pipelining dimension
    auto outDims = standardOutDimNames(ctx, rank);
    const auto &ll = enc.getLinearComponent();
    auto expectedShape = dropPipeliningDim(allocShape, enc);

    for (auto d = 0; d < rank; d++) {
      if (ll.getOutDimSize(outDims[d]) != expectedShape[d]) {
        return emitError() << "Mismatch in expected shape for dimension " << d
                           << ". Expected: " << expectedShape[d]
                           << ", got: " << ll.getOutDimSize(outDims[d]);
      }
    }
  } else if (auto enc = dyn_cast<NVMMASharedEncodingAttr>(encoding)) {
    SmallVector<int64_t> shapePerCTA(getShapePerCTA(enc, allocShape));
    auto blockShape = dropPipeliningDim(ArrayRef(shapePerCTA), enc);
    if (failed(getTMABlockShape(blockShape, enc.getElementBitWidth(),
                                enc.getSwizzlingByteWidth(), enc.getFp4Padded(),
                                enc.getTransposed(), /*packedSize=*/false,
                                emitError, TMAMode::Tiled)))
      return failure();
  } else if (auto enc = dyn_cast<SharedLinearEncodingAttr>(encoding)) {
    auto blockShape = dropPipeliningDim(allocShape, enc);
    const LinearLayout &ll = enc.getLinearLayout();
    for (auto [dim, size, llSize] :
         llvm::enumerate(blockShape, ll.getOutDimSizes())) {
      if (size == llSize)
        continue;
      return emitError() << "Mismatch in expected shape for dimension " << dim
                         << ". Expected: " << size << ", got: " << llSize;
    }
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
