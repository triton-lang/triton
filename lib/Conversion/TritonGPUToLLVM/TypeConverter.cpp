#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::MemDescType;

TritonGPUToLLVMTypeConverter::TritonGPUToLLVMTypeConverter(
    MLIRContext *ctx, const TargetInfoBase &targetInfo,
    const DataLayoutAnalysis *analysis)
    : TritonGPUToLLVMTypeConverter(ctx, LowerToLLVMOptions(ctx), targetInfo,
                                   analysis) {}

TritonGPUToLLVMTypeConverter::TritonGPUToLLVMTypeConverter(
    MLIRContext *ctx, const LowerToLLVMOptions &options,
    const TargetInfoBase &targetInfo, const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, options, analysis) {
  addConversion([ctx](triton::PointerType type) -> std::optional<Type> {
    return LLVM::LLVMPointerType::get(ctx, type.getAddressSpace());
  });
  addConversion([ctx](TensorDescType type) -> std::optional<Type> {
    return LLVM::LLVMPointerType::get(ctx, 0);
  });
  addConversion(
      [ctx](nvidia_gpu::TensorDescIm2ColType type) -> std::optional<Type> {
        return LLVM::LLVMPointerType::get(ctx, 0);
      });
  addConversion([&](RankedTensorType type) -> std::optional<Type> {
    return convertTritonTensorType(type, targetInfo);
  });
  addConversion([&](MemDescType type) -> std::optional<Type> {
    return convertMemDescType(type, targetInfo);
  });
  addConversion([&](triton::gpu::AsyncTokenType type) -> std::optional<Type> {
    return convertAsyncTokenType(type);
  });

  convertFP8Type<mlir::Float8E4M3FNUZType, mlir::Float8E4M3FNType,
                 mlir::Float8E5M2Type, mlir::Float8E5M2FNUZType>();
}

Type TritonGPUToLLVMTypeConverter::convertTritonTensorType(
    RankedTensorType type, const TargetInfoBase &targetInfo) {
  auto ctx = type.getContext();
  Type eltType = convertType(type.getElementType());
  unsigned numElementsPerThread = getTotalElemsPerThread(type);
  SmallVector<Type, 4> types(numElementsPerThread, eltType);
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}

Type TritonGPUToLLVMTypeConverter::convertMemDescType(
    MemDescType type, const TargetInfoBase &targetInfo) {
  auto ctx = type.getContext();
  // base ptr
  auto ptrType = LLVM::LLVMPointerType::get(
      ctx, targetInfo.getAddressSpace(type.getMemorySpace()));

  if (isa<triton::nvidia_gpu::TensorMemoryEncodingAttr,
          triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(
          type.getEncoding())) {
    return ptrType;
  }

  SmallVector<Type, 4> types;

  // Determine number of base pointers based on encoding.
  // For partitioned tensors, we need one base pointer per partition.
  size_t numBases = 1;
  if (auto partitioned = dyn_cast<triton::gpu::PartitionedSharedEncodingAttr>(
          type.getEncoding())) {
    numBases = partitioned.getNumPartitions();
  }

  // Add base pointer(s)
  for (size_t i = 0; i < numBases; ++i) {
    types.push_back(ptrType);
  }

  auto rank = type.getRank();
  // offsets
  for (auto i = 0; i < rank; i++) {
    types.push_back(IntegerType::get(ctx, 32));
  }
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}

Type TritonGPUToLLVMTypeConverter::convertAsyncTokenType(
    triton::gpu::AsyncTokenType type) {
  return IntegerType::get(type.getContext(), 32);
}
