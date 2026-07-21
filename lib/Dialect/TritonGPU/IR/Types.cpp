#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`
#include <numeric>

using namespace mlir;
using namespace mlir::triton::gpu;

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/Types.cpp.inc"

static constexpr llvm::StringRef kMutableMemory = "mutable";

static LogicalResult verifyNonPow2InvertedLayoutInvariant(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<int64_t> blockShape,
    const mlir::triton::LinearLayout &ll) {
  // Pow2-only shapes already satisfy the regular SharedLinear assumptions.
  if (isPositivePowerOfTwoShape(blockShape))
    return success();

  auto *ctx = ll.getOutDimNames().begin()->getContext();
  auto kOffset = StringAttr::get(ctx, "offset");
  auto outDims = mlir::triton::standardOutDimNames(ctx, blockShape.size());
  auto llInv = ll.pseudoinvert();
  if (!llInv.hasOutDim(kOffset))
    return emitError()
           << "non-power-of-two shapes require an inverted layout view "
              "(logical indices -> offset) with an offset basis";

  bool seenNonPow2Dim = false;
  for (auto [dim, m] : llvm::enumerate(blockShape)) {
    if (llvm::isPowerOf2_64(m))
      continue;
    if (seenNonPow2Dim)
      return emitError()
             << "at most one non-power-of-two dimension is currently "
                "supported for shared_linear memdesc shapes in shape "
             << blockShape;
    seenNonPow2Dim = true;

    // Project the inverted layout to a single logical-dim -> offset view.
    // This enforces that the non-pow2 logical-count dimension is tiled by
    // homogeneous physical blocks.
    auto dimName = outDims[dim];
    auto dimToOffset = llInv.sublayout({dimName}, {kOffset});
    int64_t m2 = dimToOffset.getInDimSize(dimName);
    int64_t projectedOffsetRange = dimToOffset.getOutDimSize(kOffset);
    int64_t c = std::gcd(m, m2);
    unsigned log2C = llvm::Log2_64(c);
    unsigned log2M2 = llvm::Log2_64(m2);
    // From the gcd boundary onward, adjacent bases must double. This means
    // the layout past the inner C-sized region repeats the same tile.
    for (unsigned i = log2C; i + 1 < log2M2; ++i) {
      int32_t curr = dimToOffset.getBasis(dimName, i, kOffset);
      int32_t next = dimToOffset.getBasis(dimName, i + 1, kOffset);
      if (next != 2 * curr) {
        return emitError()
               << "non-power-of-two dimension " << m
               << " has unexpected basis in inverted layout view (logical "
                  "indices -> offset) at dim "
               << dim << ": b[" << (i + 1) << "] (" << next << ") != 2 * b["
               << i << "] (" << (2 * curr) << ")";
      }
    }

    // The MSB must span half of this projected offset range, making the
    // non-pow2 dimension the slowest-moving dimension of the view.
    int32_t msbBasis = dimToOffset.getBasis(dimName, log2M2 - 1, kOffset);
    int32_t expectedMsbBasis = static_cast<int32_t>(projectedOffsetRange / 2);
    if (msbBasis != expectedMsbBasis) {
      return emitError() << "non-power-of-two dimension " << m
                         << " has unexpected MSB basis in inverted layout "
                            "view (logical indices -> offset) at dim "
                         << dim << ": b[" << (log2M2 - 1) << "] (" << msbBasis
                         << ") != offsetRange/2 (" << expectedMsbBasis << ")";
    }
  }
  return success();
}

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
  bool hasNonPow2Shape = !isPositivePowerOfTwoShape(shape.drop_front(1));
  // Non-pow2 logical dimensions are only meaningful for shared-memory
  // encodings, where we can validate them against layout invariants below.
  if (hasNonPow2Shape && !isa<SharedEncodingTrait>(encoding))
    return emitError() << "shape must have power-of-2 and non-zero dimensions; "
                          "got "
                       << shape;
  if (llvm::any_of(shape.drop_front(1), [](int64_t dim) { return dim <= 0; }))
    return emitError() << "shape dimensions must be positive; got " << shape;
  if (shape.front() == 0)
    return emitError() << "shape has 0 dimension";
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
    auto isPowerOfTwo = [](int64_t dim) {
      return llvm::isPowerOf2_64(dim) && dim > 0;
    };
    if (!llvm::all_of(shape.take_back(2), isPowerOfTwo)) {
      return emitError()
             << "shape must have power-of-2 and non-zero dimensions; got "
             << shape;
    }
    if (!llvm::all_of(allocShape.take_back(2), isPowerOfTwo)) {
      return emitError()
             << "alloc shape must have power-of-2 and non-zero dimensions; got "
             << allocShape;
    }
    unsigned bitwidth = elementType.getIntOrFloatBitWidth();
    if (bitwidth * enc.getColStride() > 32) {
      return emitError()
             << "bitwidth * colStride must be less than or equal to 32. Got "
             << bitwidth << " and " << enc.getColStride();
    }
    // Takes subslices into account and figures out whether we can construct
    // the linear layout at all
    allocShape = allocShape.take_back(2);
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
  } else if (auto enc = dyn_cast<SharedEncodingTrait>(encoding)) {
    if (memorySpace != SharedMemorySpaceAttr::get(ctx)) {
      return emitError()
             << "memorySpace must be SharedMemorySpace for shared encoding. "
             << "Got " << memorySpace;
    }
    auto rank = cast<LayoutEncodingTrait>(enc).getRank();
    if (!(rank == shape.size() || rank == shape.size() - 1)) {
      return emitError() << "rank must be equal to or one less than "
                         << "the shape size. Got " << rank << " and "
                         << shape.size();
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
  } else if (auto enc = dyn_cast<nvidia_gpu::TensorMemoryLUTEncodingAttr>(
                 encoding)) {
    if (memorySpace != nvidia_gpu::TensorMemorySpaceAttr::get(ctx)) {
      return emitError() << "memorySpace must be TensorMemorySpace";
    }
    if (allocShape.size() != 2) {
      return emitError() << "LUT doesn't currently support multibuffering";
    }
    if (elementType.getIntOrFloatBitWidth() != 8) {
      return emitError() << "bitwidth must be 8";
    }
  } else {
    return emitError() << encoding << " is not a valid encoding";
  }

  // PaddedSharedEncodingAttr is also a SharedEncodingTrait but we have some
  // additional rules to verify.
  if (auto enc = dyn_cast<PaddedSharedEncodingAttr>(encoding)) {
    auto rank = enc.getRank();
    // Ensure linear component's outDims match the alloc size ignoring
    // pipelining dimension
    auto outDims = standardOutDimNames(ctx, rank);
    const auto &ll = enc.getLinearComponent();
    auto expectedShape = allocShape;
    if (rank == allocShape.size() - 1)
      expectedShape = expectedShape.drop_front(1);

    for (auto d = 0; d < rank; d++) {
      if (ll.getOutDimSize(outDims[d]) != expectedShape[d]) {
        return emitError() << "Mismatch in expected shape for dimension " << d
                           << ". Expected: " << expectedShape[d]
                           << ", got: " << ll.getOutDimSize(outDims[d]);
      }
    }
  } else if (auto enc = dyn_cast<NVMMASharedEncodingAttr>(encoding)) {
    SmallVector<int64_t> shapePerCTA(getShapePerCTA(enc, allocShape));
    auto blockShape = ArrayRef(shapePerCTA).take_back(enc.getRank());
    if (failed(getTMABlockShape(blockShape, enc.getElementBitWidth(),
                                enc.getSwizzlingByteWidth(), enc.getFp4Padded(),
                                enc.getTransposed(), /*packedSize=*/false,
                                emitError, TMAMode::Tiled)))
      return failure();
    auto packedTMABlockShape = getTMABlockShape(
        blockShape, enc.getElementBitWidth(), enc.getSwizzlingByteWidth(),
        enc.getFp4Padded(), enc.getTransposed(), /*packedSize=*/true,
        TMAMode::Tiled);
    for (auto [dim, dimShapes] :
         llvm::enumerate(llvm::zip(blockShape, packedTMABlockShape))) {
      auto [dimSize, logicalBlockSize] = dimShapes;
      if (dimSize % logicalBlockSize != 0)
        return emitError() << "shapePerCTA size " << dimSize
                           << " must be divisible by its logical block size "
                           << logicalBlockSize << " (dim " << dim << ")";
      int64_t logicalBlocks = dimSize / logicalBlockSize;
      if (!llvm::isPowerOf2_64(logicalBlocks))
        return emitError() << "number of logical blocks per CTA ("
                           << logicalBlocks << ") must be a power of two (dim "
                           << dim << ", dimSize=" << dimSize
                           << ", logicalBlockSize=" << logicalBlockSize << ")";
      if (logicalBlocks > 1 && !llvm::isPowerOf2_64(dimSize))
        return emitError()
               << "non-power-of-two dimension " << dimSize
               << " is unsupported when split across multiple messages (dim "
               << dim << ", logicalBlocks=" << logicalBlocks << ")";
    }
  } else if (auto enc = dyn_cast<SharedLinearEncodingAttr>(encoding)) {
    auto blockShape = ArrayRef(allocShape).take_back(enc.getRank());
    const LinearLayout &ll = enc.getLinearLayout();
    for (auto [dim, size, llSize] :
         llvm::enumerate(blockShape, ll.getOutDimSizes())) {
      if (llvm::PowerOf2Ceil(size) == llSize)
        continue;
      return emitError() << "Mismatch in expected shape for dimension " << dim
                         << ". Expected in [1, " << llSize
                         << "], got: " << size;
    }
  }

  if (isa<SharedLinearEncodingAttr, NVMMASharedEncodingAttr>(encoding)) {
    auto shared = cast<SharedEncodingTrait>(encoding);
    auto rank = cast<LayoutEncodingTrait>(shared).getRank();
    auto blockShape = ArrayRef(allocShape).take_back(rank);
    if (!isPositivePowerOfTwoShape(blockShape)) {
      auto ll = toLinearLayout(normalizeShapeToPowerOf2(blockShape), encoding);
      if (failed(
              verifyNonPow2InvertedLayoutInvariant(emitError, blockShape, ll)))
        return failure();
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
