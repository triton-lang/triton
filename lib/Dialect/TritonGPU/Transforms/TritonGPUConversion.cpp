#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <algorithm>

using namespace mlir;

//
// TypeConverter
//
TritonGPUTypeConverter::TritonGPUTypeConverter(MLIRContext *context, 
                                               int numThreads)
    : context(context), numThreads(numThreads) {
  // TODO: how does MLIR pick the right conversion?
  addConversion([](Type type) { return type; });
  addConversion([this](RankedTensorType tensorType) -> RankedTensorType {
    MLIRContext *context = this->context;
    int numThreads = this->numThreads;

    llvm::ArrayRef<int64_t> shape = tensorType.getShape();
    Type elementType = tensorType.getElementType();
    int64_t rank = tensorType.getRank();
    int64_t numElements = tensorType.getNumElements();

    // TODO: are there any better ways to raise this error?
    if (!(numElements >= numThreads)) {
      SmallVector<char> buffer;
      llvm::raw_svector_ostream os(buffer);
      os << tensorType << " has " << numElements << " numElements "
         << " smaller than numThreads (" << numThreads << ")\n"
         << "consider using smaller num-warps\n";
      llvm::report_fatal_error(os.str());
    }
    assert(numElements % numThreads == 0);

    // or assert no encoding?

    // Now we assume:
    //   contiguous = 1, order = 0, 1, 2, ..., 
    llvm::SmallVector<unsigned> threadTileSize(rank, 1); // naive layout
    llvm::SmallVector<unsigned> warpTileSize(rank, 1);
    llvm::SmallVector<unsigned> blockTileSize(rank);
    llvm::SmallVector<unsigned> order(rank);
    int remainingThreads = numThreads;
    int remainingLanes = /*warp size*/32;
    for (int64_t dim = 0; dim < rank; ++dim) {
      blockTileSize[dim] = std::clamp(remainingThreads, 1, int(shape[dim]));
      warpTileSize[dim] = std::clamp(remainingLanes, 1, int(shape[dim]));
      order[dim] = dim;

      remainingThreads /= blockTileSize[dim];
      remainingLanes /= warpTileSize[dim];
      // TODO: will we need repetition?
    }
    Attribute encoding = triton::gpu::TritonGPUBlockedEncodingAttr::get(
        context, threadTileSize, warpTileSize, blockTileSize, order);
    return RankedTensorType::get(shape, elementType, encoding);
  });

  //
  // materailizations
  //
  // This will be called when (newArgType != origArgType)
  // This will create newArg, and map(origArg, newArg)
  addArgumentMaterialization([&](OpBuilder &builder, RankedTensorType tensorType,
                                 ValueRange inputs, Location loc) {
    llvm_unreachable("Not implemented");
    return llvm::None;
  });

  // If the origValue still has live user(s), use this to
  // convert origValue to newValue
  addSourceMaterialization([&](OpBuilder &builder, RankedTensorType tensorType,
                                 ValueRange inputs, Location loc) {
    llvm_unreachable("Not implemented");
    return llvm::None;
  });

  // This will be called when (desiredType != newOperandType)
  // where, desiredType = typeConverter->convertType(origType)
  // NOTE: only for remapped values.
  addTargetMaterialization([&](OpBuilder &builder, RankedTensorType tensorType,
                                ValueRange inputs, Location loc) {
    assert(inputs.size() == 1);
    llvm_unreachable("Not implemented");
    return llvm::None;
  });
}

//
// TritonGPUConversion
//
TritonGPUConversionTarget::TritonGPUConversionTarget(
  MLIRContext &context, TritonGPUTypeConverter &typeConverter)
    : ConversionTarget(context), typeConverter(typeConverter) {
  // TODO: we should also verify ops of TritonGPUDialect
  addLegalDialect<triton::gpu::TritonGPUDialect>();

  // Some ops from SCF are illegal
  addIllegalOp<scf::ExecuteRegionOp, scf::ParallelOp, 
               scf::ReduceOp, scf::ReduceReturnOp>();

  addDynamicallyLegalDialect<arith::ArithmeticDialect,
                             triton::TritonDialect,
                             StandardOpsDialect,
                             scf::SCFDialect>([&](Operation *op) {
    if (typeConverter.isLegal(op))
      return true;
    return false;
  });


  // We have requirements for the data layouts
  addDynamicallyLegalOp<triton::DotOp>([this](triton::DotOp dotOp) -> bool {
    Attribute aEncoding = dotOp.a().getType().cast<RankedTensorType>().getEncoding();
    Attribute bEncoding = dotOp.b().getType().cast<RankedTensorType>().getEncoding();
    if (aEncoding && aEncoding.isa<triton::gpu::TritonGPUSharedEncodingAttr>() &&
        bEncoding && bEncoding.isa<triton::gpu::TritonGPUSharedEncodingAttr>())
      return true;
    // // TODO: we should delete this
    // if (this->typeConverter.isLegal(dotOp))
    //   return true;
    return false;
  });

}
