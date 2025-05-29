#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/AMDPatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/TargetInfo.h"
#include "Conversion/ProtonGPUToLLVM/Utility.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

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
    auto segment = adaptor.getSegment();
    auto segmentObj = LLVM::SegmentObject::fromStruct(loc, segment, rewriter);
    Value indexPtr = segmentObj.indexPtr;
    Value bufferBase = segmentObj.base;
    Value segmentBase = segmentObj.segmentBase;
    auto bufferBaseType = bufferBase.getType();

    Value curIdx = b.load(i32_ty, indexPtr);
    Value newIdx = b.add(curIdx, b.i32_val(wordsPerEntry));
    b.store(newIdx, indexPtr);

    int selectedWarpNum = mlir::triton::gpu::lookupNumWarps(mod);
    auto segmentType = op.getSegment().getType();
    auto selectedIds = segmentType.getSelectIds();
    if (!selectedIds.empty())
      selectedWarpNum = selectedIds.size();
    const int bufferSizeInBytes = segmentType.getNBytes();
    const int segmentWordSize = bufferSizeInBytes / selectedWarpNum / 4;
    Value tagOffset =
        b.add(segmentBase, b.urem(curIdx, b.i32_val(segmentWordSize)));
    Value vecPtr = b.gep(bufferBaseType, i32_ty, bufferBase, tagOffset);
    Value tag = op.getIsStart() ? b.i32_val(op.getScopeId())
                                : b.i32_val(1 << 31 | op.getScopeId());
    Value clock = op.getCounter();
    Value valsVec = packLLVector(loc, {tag, clock}, rewriter);

    uint32_t addrSpace =
        cast<LLVM::LLVMPointerType>(bufferBaseType).getAddressSpace();
    if (addrSpace == 1) {
      // TODO(crobeck): see what buffer ops performance looks like here for
      // stack mem (address space 1) compared to predicated ops to shared memory
      llvm::report_fatal_error("unimplemented");
    } else if (addrSpace == 3) {
      // Setting predicate always true has bank conflicts but it is
      // expected and stable.
      targetInfo.getTritonTargetInfo().storeDShared(rewriter, loc, vecPtr,
                                                    std::nullopt, valsVec,
                                                    /*predicate=*/b.true_val());
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

namespace mlir::triton::proton::gpu::AMD {
void populateProtonGPUOpAMDPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    const TargetInfo &targetInfo,
                                    PatternBenefit benefit) {
  patterns.add<CircularStoreOpConversion>(typeConverter, targetInfo, benefit);
}
} // namespace mlir::triton::proton::gpu::AMD
