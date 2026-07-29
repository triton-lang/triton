#include "Conversion/ProtonGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir {

Value getRawThreadId(OpBuilder &rewriter, Location loc) {
  Value tid =
      ::mlir::gpu::ThreadIdOp::create(rewriter, loc, ::mlir::gpu::Dimension::x);
  Value threadId = arith::IndexCastOp::create(rewriter, loc, i32_ty, tid);
  return threadId;
}

namespace LLVM {

LLVMStructType SegmentObject::getStructType(MLIRContext *ctx, int memorySpace,
                                            int indexPtrAddrSpace) {
  SmallVector<Type, 4> types;
  // ------------
  // Memory descriptor
  // ------------
  auto ptrType = LLVM::LLVMPointerType::get(ctx, memorySpace);
  types.push_back(ptrType);
  // ------------
  // Segment base
  // ------------
  auto SegmentAllocType = IntegerType::get(ctx, 32);
  types.push_back(SegmentAllocType);
  // ------------
  // Index ptr
  // ------------
  auto indexPtrType = LLVM::LLVMPointerType::get(ctx, indexPtrAddrSpace);
  types.push_back(indexPtrType);
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}

Value SegmentObject::getStruct(Location loc,
                               ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  int memorySpace =
      mlir::cast<LLVM::LLVMPointerType>(base.getType()).getAddressSpace();
  int indexPtrAddrSpace =
      mlir::cast<LLVM::LLVMPointerType>(indexPtr.getType()).getAddressSpace();
  auto structTy =
      getStructType(loc.getContext(), memorySpace, indexPtrAddrSpace);
  Value segmentStruct = LLVM::UndefOp::create(rewriter, loc, structTy);
  segmentStruct = b.insert_val(structTy, segmentStruct, base, 0);
  segmentStruct = b.insert_val(structTy, segmentStruct, segmentBase, 1);
  segmentStruct = b.insert_val(structTy, segmentStruct, indexPtr, 2);
  return segmentStruct;
}

SegmentObject SegmentObject::fromStruct(Location loc, Value segmentStruct,
                                        ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto structTy = mlir::cast<LLVM::LLVMStructType>(segmentStruct.getType());
  Value memoryDescriptorPtr =
      b.extract_val(structTy.getBody()[0], segmentStruct, 0);
  Value segmentBase = b.extract_val(structTy.getBody()[1], segmentStruct, 1);
  Value indexPtr = b.extract_val(structTy.getBody()[2], segmentStruct, 2);
  return SegmentObject(memoryDescriptorPtr, segmentBase, indexPtr);
}

} // namespace LLVM

namespace triton {
namespace proton::gpu {

namespace {

Value encodeMetric(Value metric, MetricValueType metricType,
                   TritonLLVMOpBuilder &b) {
  auto i32Type = IntegerType::get(metric.getContext(), 32);
  auto i16Type = IntegerType::get(metric.getContext(), 16);
  if (metricType == MetricValueType::F32)
    return b.bitcast(metric, i32Type);
  if (metricType == MetricValueType::F16 || metricType == MetricValueType::BF16)
    return b.zext(i32Type, b.bitcast(metric, i16Type));

  auto intType = cast<IntegerType>(metric.getType());
  if (intType.getWidth() == 32)
    return metric;
  return b.zext(i32Type, metric);
}

SmallVector<CircularStoreDataPack>
lowerCircularEvent(Operation *op, SegmentType segmentType, Value segmentStruct,
                   Value counter, Value tag, Value metric, Value metricTag,
                   MetricValueType metricType,
                   ConversionPatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto mod = op->getParentOfType<ModuleOp>();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  const int bytesPerEntry = proton::gpu::getBytesPerClockEntry();
  const int wordsPerEntry = bytesPerEntry / 4; // 1 word = 4 bytes
  const int numEntries = metric ? 2 : 1;

  auto segmentObj =
      LLVM::SegmentObject::fromStruct(loc, segmentStruct, rewriter);
  Value indexPtr = segmentObj.indexPtr;
  Value bufferBase = segmentObj.base;
  Value segmentBase = segmentObj.segmentBase;

  // Update the index (could be register promoted).
  Value curIdx = b.load(i32_ty, indexPtr);
  Value newIdx = b.add(curIdx, b.i32_val(wordsPerEntry * numEntries));

  // Compute the segment size in word (4 bytes).
  int selectedWarpNum = getTotalNumWarps(mod);
  auto selectedIds = segmentType.getSelectIds();
  if (!selectedIds.empty())
    selectedWarpNum = selectedIds.size();
  const int bufferSizeInBytes = segmentType.getNBytes();
  const int segmentWordSize = bufferSizeInBytes / selectedWarpNum / 4;

  auto bufferBaseType = bufferBase.getType();

  // Constructing the tag and clock (8 byte)
  // =======================================
  // tag and upper clock (4 bytes):
  // 31: start or end (1 bit)
  // 30:23 scope id (8 bits)
  // 22:20 event type (3 bits)
  // 19:16 metric type (4 bits)
  // 15:11 reserved (5 bits)
  // 10:0  64-bit clock bit 32:42 (11 bits)
  // =======================================
  // lower clock (4 bytes):
  // 31:0 64-bit clock bit 0:31
  // =======================================
  Value clock = counter;
  auto clkTy = mlir::cast<IntegerType>(clock.getType());
  Value valsVec;
  if (clkTy.getWidth() == 64) {
    auto clkVecTy = vec_ty(i32_ty, 2);
    auto clkVec = b.bitcast(clock, clkVecTy);
    Value clkLower = b.extract_element(i32_ty, clkVec, b.i32_val(0));
    Value clkUpper = b.extract_element(i32_ty, clkVec, b.i32_val(1));
    Value tagClkUpper = b.or_(tag, b.and_(clkUpper, b.i32_val(0x7ff)));
    valsVec = packLLVector(loc, {tagClkUpper, clkLower}, rewriter);
  } else {
    valsVec = packLLVector(loc, {tag, clock}, rewriter);
  }

  // Compute the predicate for the writer.
  const int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  Value curThreadId = getThreadId(rewriter, loc);
  Value isWarpMaster =
      b.icmp_eq(b.urem(curThreadId, b.i32_val(warpSize)), b.i32_val(0));
  Value isWriter;

  Value idxToStore = newIdx;
  auto granularity = segmentType.getGranularity();
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
    idxToStore = b.select(isCurWarpEnabled, newIdx, curIdx);
  }

  b.store(idxToStore, indexPtr);

  uint32_t addrSpace =
      cast<LLVM::LLVMPointerType>(bufferBaseType).getAddressSpace();

  auto getRecordPtr = [&](int entryIndex) {
    Value entryIdx = entryIndex == 0
                         ? curIdx
                         : b.add(curIdx, b.i32_val(entryIndex * wordsPerEntry));
    Value tagOffset =
        b.add(segmentBase, b.urem(entryIdx, b.i32_val(segmentWordSize)));
    return b.gep(bufferBaseType, i32_ty, bufferBase, tagOffset);
  };

  SmallVector<CircularStoreDataPack> dataPacks;
  dataPacks.push_back(
      {isWriter, valsVec, getRecordPtr(/*entryIndex=*/0), addrSpace});
  if (metric) {
    Value metricRecord = packLLVector(
        loc, {metricTag, encodeMetric(metric, metricType, b)}, rewriter);
    dataPacks.push_back(
        {isWriter, metricRecord, getRecordPtr(/*entryIndex=*/1), addrSpace});
  }
  return dataPacks;
}

} // namespace

SmallVector<CircularStoreDataPack>
lowerCircularStore(CircularStoreOp op, Value segmentStruct, Value counter,
                   Value dynamicScopeId, Value metric,
                   ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
  Value tag;
  Value metricTag;
  if (dynamicScopeId) {
    tag = op.getIsStart() ? dynamicScopeId
                          : b.or_(dynamicScopeId, b.i32_val(1u << 31));
  } else {
    auto scopeId = op.getScopeIdAttr().getInt();
    tag = b.i32_val(encodeEventTag(scopeId, op.getIsStart(), EventType::SCOPE,
                                   op.getMetricType()));
    if (metric)
      metricTag = b.i32_val(encodeEventTag(
          scopeId, /*isStart=*/true, EventType::METRIC, op.getMetricType()));
  }
  return lowerCircularEvent(op, op.getSegment().getType(), segmentStruct,
                            counter, tag, metric, metricTag, op.getMetricType(),
                            rewriter);
}

SmallVector<CircularStoreDataPack>
lowerCircularMarkOpHelper(CircularMarkOp op, Value segmentStruct, Value counter,
                          ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
  Value tag = b.i32_val(encodeEventTag(op.getScopeIdAttr().getInt(),
                                       /*isStart=*/true, EventType::MARK,
                                       MetricValueType::NONE));
  return lowerCircularEvent(op, op.getSegment().getType(), segmentStruct,
                            counter, tag, Value(), Value(),
                            MetricValueType::NONE, rewriter);
}

SmallVector<FunctionOpInterface> getTritonFunctions(ModuleOp mod) {
  SmallVector<FunctionOpInterface> funcOps;
  mod.walk([&](FunctionOpInterface funcOp) {
    // Ignore any intrinsic functions which have an empty body.
    // For example, on AMD the predicate load/store ops are currently pseudo
    // instructions at this point and may get picked up here and trigger the
    // FunctionOpInterface range based assert below.
    if (funcOp.empty())
      return;
    funcOps.push_back(funcOp);
  });
  return funcOps;
}

} // namespace proton::gpu
} // namespace triton

} // namespace mlir
