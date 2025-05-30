#include "Conversion/ProtonGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::LLVM {

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
  Value segmentStruct = rewriter.create<LLVM::UndefOp>(loc, structTy);
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

} // namespace mlir::LLVM
