#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::triton::gpu;

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
    llvm::SmallVector<unsigned> broadcastAxis;
    int remainingThreads = numThreads;
    int remainingLanes = /*warp size*/ 32;
    for (int64_t dim = 0; dim < rank; ++dim) {
      blockTileSize[dim] = std::clamp(remainingThreads, 1, int(shape[dim]));
      warpTileSize[dim] = std::clamp(remainingLanes, 1, int(shape[dim]));
      order[dim] = dim;

      remainingThreads /= blockTileSize[dim];
      remainingLanes /= warpTileSize[dim];
      // TODO: will we need repetition?
    }
    Attribute encoding = triton::gpu::TritonGPUBlockedEncodingAttr::get(
        context, threadTileSize, warpTileSize, blockTileSize, order,
        broadcastAxis);
    return RankedTensorType::get(shape, elementType, encoding);
  });

  //
  // materailizations
  //
  // This will be called when (newArgType != origArgType)
  // This will create newArg, and map(origArg, newArg)
  addArgumentMaterialization([&](OpBuilder &builder,
                                 RankedTensorType tensorType, ValueRange inputs,
                                 Location loc) {
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
  addIllegalOp<scf::ExecuteRegionOp, scf::ParallelOp, scf::ReduceOp,
               scf::ReduceReturnOp>();

  addDynamicallyLegalDialect<arith::ArithmeticDialect, triton::TritonDialect,
                             StandardOpsDialect, scf::SCFDialect>(
      [&](Operation *op) {
        if (typeConverter.isLegal(op))
          return true;
        return false;
      });

  // We have requirements for the data layouts
  addDynamicallyLegalOp<triton::DotOp>([this](triton::DotOp dotOp) -> bool {
    Attribute aEncoding =
        dotOp.a().getType().cast<RankedTensorType>().getEncoding();
    Attribute bEncoding =
        dotOp.b().getType().cast<RankedTensorType>().getEncoding();
    if (aEncoding &&
        aEncoding.isa<triton::gpu::TritonGPUSharedEncodingAttr>() &&
        bEncoding && bEncoding.isa<triton::gpu::TritonGPUSharedEncodingAttr>())
      return true;
    // // TODO: we should delete this
    // if (this->typeConverter.isLegal(dotOp))
    //   return true;
    return false;
  });
}

// %dst = tt.broadcast %src
//   =>
// %newSrc = convert_layout %src
// %bcst = tt.broadcast %newSrc
// %dst = convert_layout %bcst
LogicalResult TritonGPUConversionTarget::refineLayouts(ModuleOp mod,
                                                       int numThreads) {
  // collect broadcasts
  SmallVector<triton::BroadcastOp> broadcasts;
  mod.walk([&](triton::BroadcastOp op) { broadcasts.push_back(op); });

  BlockAndValueMapping mapping;
  for (auto broadcast : broadcasts) {
    OpBuilder builder(broadcast);
    Value src = mapping.lookupOrDefault(broadcast.src());
    Type originSrcType = src.getType();
    Type originDstType = broadcast.getType();
    auto originDstTensorType = originDstType.dyn_cast<RankedTensorType>();
    unsigned dstRank = originDstTensorType.getRank();

    // compute newSrcType & broadcastAxis
    Type newSrcType;
    SmallVector<unsigned> broadcastAxis;
    bool isSrcScalar = false;
    if (auto tensorType = originSrcType.dyn_cast<RankedTensorType>()) {
      assert(tensorType.getRank() == dstRank &&
             "src & dst should have same rank (verifier should catch this)");
      for (unsigned ax = 0; ax < dstRank; ++ax)
        if (tensorType.getShape()[ax] < originDstTensorType.getShape()[ax])
          broadcastAxis.push_back(ax);

      Attribute originSrcEnc = tensorType.getEncoding();
      if (auto blockedEnc =
              originSrcEnc.dyn_cast<TritonGPUBlockedEncodingAttr>()) {
        auto newSrcEnc = TritonGPUBlockedMulticastEncodingAttr::get(
            blockedEnc.getContext(), blockedEnc.getThreadTileSize(),
            blockedEnc.getWarpTileSize(), blockedEnc.getBlockTileSize(),
            blockedEnc.getOrder(), broadcastAxis);
        newSrcType = RankedTensorType::get(
            tensorType.getShape(), tensorType.getElementType(), newSrcEnc);
      } else
        llvm_unreachable("src of broadcast should have blocked encoding");
    } else {
      for (unsigned ax = 0; ax < dstRank; ++ax)
        broadcastAxis.push_back(ax);
      newSrcType = originSrcType;
      isSrcScalar = true;
    }

    // create new src
    if (!isSrcScalar) // we don't need to convert layout for scalar values
      src = builder.create<triton::gpu::ConvertLayoutOp>(src.getLoc(),
                                                         newSrcType, src);

    // create new broadcast
    // compute new type (encoding)
    auto originDstEnc = originDstTensorType.getEncoding()
                            .dyn_cast<TritonGPUBlockedEncodingAttr>();
    auto newEnc = TritonGPUBlockedEncodingAttr::get(
        originDstEnc.getContext(), originDstEnc.getThreadTileSize(),
        originDstEnc.getWarpTileSize(), originDstEnc.getBlockTileSize(),
        originDstEnc.getOrder(), broadcastAxis);
    auto newType =
        RankedTensorType::get(originDstTensorType.getShape(),
                              originDstTensorType.getElementType(), newEnc);
    Value newBroadcast =
        builder.create<triton::BroadcastOp>(broadcast.getLoc(), newType, src);
    // we don't want to change the encoding of the result
    Value newDst = builder.create<triton::gpu::ConvertLayoutOp>(
        broadcast.getLoc(), originDstType, newBroadcast);

    broadcast.replaceAllUsesWith(newDst);
    mapping.map(broadcast, newDst);
  }

  return success();
}
