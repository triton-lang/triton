#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/NvidiaPatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/TargetInfo.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

// Circular strategy memory layout of profiled data (total: N bytes).
// Assuming we record data from warp 0, 2, 7 so buffer looks like:
//  +-----------------------------------------------+
//  | warp 0 data (N/3 bytes)                       |
//  +-----------------------------------------------+
//  | warp 2 data (N/3 bytes)                       |
//  +-----------------------------------------------+
//  | warp 7 data (N/3 bytes)                       |
//  +-----------------------------------------------+

struct CircularStoreOpConversion
    : public ConvertOpToLLVMPattern<
          mlir::triton::proton::gpu::CircularStoreOp> {
  explicit CircularStoreOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<
            mlir::triton::proton::gpu::CircularStoreOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::CircularStoreOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    const int bytesPerEntry = proton::gpu::getBytesPerClockEntry();
    const int wordsPerEntry = bytesPerEntry / 4; // 1 word = 4 bytes

    Value indexPtr = adaptor.getIndexPtr();
    Value dataStruct = adaptor.getData();
    auto bufferPtrTy =
        mlir::cast<LLVM::LLVMStructType>(dataStruct.getType()).getBody()[0];

    Value bufferDataBasePtr = b.extract_val(bufferPtrTy, dataStruct, 0);

    // Update the index (could be register promoted).
    Value curIdx = b.load(i32_ty, indexPtr);
    Value newIdx = b.add(curIdx, b.i32_val(wordsPerEntry));
    b.store(newIdx, indexPtr);

    // Compute the segment size in word (4 bytes).
    auto segbaseOp =
        mlir::cast<proton::gpu::SegmentBaseOp>(op.getSeg().getDefiningOp());
    int selectedWarpNum = mlir::triton::gpu::lookupNumWarps(mod);
    auto selectedIds = segbaseOp.getSelectIdsAttr().asArrayRef();
    if (!selectedIds.empty())
      selectedWarpNum = selectedIds.size();
    auto memDescTy =
        mlir::cast<triton::gpu::MemDescType>(op.getData().getType());
    const int bufferSizeInBytes =
        mlir::ShapedType::getNumElements(memDescTy.getShape()) *
        memDescTy.getElementType().getIntOrFloatBitWidth() / 8;
    const int segmentWordSize = bufferSizeInBytes / selectedWarpNum / 4;

    // Compute the actual base offset (with urem as circular buffer).
    Value segmentBase = adaptor.getSeg();
    Value tagOffset =
        b.add(segmentBase, b.urem(curIdx, b.i32_val(segmentWordSize)));

    // Store the counter into buffer.
    Value vecPtr = b.gep(bufferPtrTy, i32_ty, bufferDataBasePtr, tagOffset);
    Value tag = op.getIsStart() ? b.i32_val(op.getScopeId())
                                : b.i32_val(1 << 31 | op.getScopeId());
    Value clock = op.getCounter();
    Value valsVec = packLLVector(loc, {tag, clock}, rewriter);

    // Compute the predicate for the writer.
    const int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    Value curThreadId = getThreadId(rewriter, loc);
    Value isWarpMaster =
        b.icmp_eq(b.urem(curThreadId, b.i32_val(warpSize)), b.i32_val(0));
    Value isWriter;

    auto granularity = segbaseOp.getGranularity();
    if (selectedIds.empty()) {
      if (granularity == proton::gpu::Granularity::WARP) {
        isWriter = isWarpMaster;
      } else {
        llvm::report_fatal_error(
            "segment address specialization not implemented yet");
      }
    } else {
      Value isCurWarpEnabled = b.icmp_ne(segmentBase, b.i32_val(-1));
      isWriter = b.and_(isCurWarpEnabled, isWarpMaster);
    }
    uint32_t AddrSpace =
        cast<LLVM::LLVMPointerType>(bufferPtrTy).getAddressSpace();
    if (AddrSpace == 1) {
      llvm::report_fatal_error("unimplemented");
    } else if (AddrSpace == 3) {
      targetInfo.getTritonTargetInfo().storeDShared(rewriter, loc, vecPtr,
                                                    std::nullopt, valsVec,
                                                    /*pred=*/isWriter);
    } else {
      llvm::report_fatal_error("unsupported address space in circular store");
    }
    rewriter.eraseOp(op);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

} // namespace

namespace mlir::triton::proton::gpu::NVIDIA {
void populateProtonGPUOpNvidiaPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       const TargetInfo &targetInfo,
                                       PatternBenefit benefit) {
  patterns.add<CircularStoreOpConversion>(typeConverter, targetInfo, benefit);
}
} // namespace mlir::triton::proton::gpu::NVIDIA
