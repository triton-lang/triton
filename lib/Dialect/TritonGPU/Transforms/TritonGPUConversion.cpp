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
  addConversion([&](RankedTensorType tensorType) -> RankedTensorType {
    llvm::ArrayRef<int64_t> shape = tensorType.getShape();
    Type elementType = tensorType.getElementType();
    int64_t rank = tensorType.getRank();
    int64_t numElements = tensorType.getNumElements();

    // TODO: we should raise exception here.
    assert(numElements > numThreads);
    assert(numElements % numThreads == 0);

    // assert no encoding?

    // Now we assume:
    //   contiguous = 1, order = 0, 1, 2, ..., 
    llvm::SmallVector<unsigned> threadTileSize(rank, 1); // naive layout
    llvm::SmallVector<unsigned> blockTileSize(rank);
    llvm::SmallVector<unsigned> order(rank);
    int remainingThreads = numThreads;
    for (int64_t dim = 0; dim < rank; ++dim) {
      blockTileSize[dim] = std::clamp(remainingThreads, 1, int(shape[dim]));
      order[dim] = dim;

      remainingThreads /= blockTileSize[dim];
      // TODO: will we need repetition?
    }
    Attribute encoding = triton::gpu::TritonGPUDistributedEncodingAttr::get(
        context, threadTileSize, blockTileSize, order);
    return RankedTensorType::get(shape, elementType, encoding);
  });
}

//
// TritonGPUConversion
//
TritonGPUConversionTarget::TritonGPUConversionTarget(MLIRContext &context)
    : ConversionTarget(context) {
  addLegalDialect<triton::TritonDialect,
                  arith::ArithmeticDialect,
                  scf::SCFDialect>();

  // Some ops from SCF are illegal
  addIllegalOp<scf::ExecuteRegionOp, scf::ParallelOp, 
               scf::ReduceOp, scf::ReduceReturnOp>();

  // // We have requirements for the data layouts
  // addDynamicallyLegalOp<triton::DotOp>([](triton::DotOp dotOp) -> bool {
  //   Attribute aEncoding = dotOp.a().getType().cast<RankedTensorType>().getEncoding();
  //   Attribute bEncoding = dotOp.b().getType().cast<RankedTensorType>().getEncoding();
  //   if (aEncoding && aEncoding.isa<triton::gpu::TritonGPUSharedEncodingAttr>() &&
  //       bEncoding && bEncoding.isa<triton::gpu::TritonGPUSharedEncodingAttr>())
  //     return true;
  //   return false;
  // });

}
