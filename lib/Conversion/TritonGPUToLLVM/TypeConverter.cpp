#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::MemDescType;
using ::mlir::triton::gpu::SharedEncodingAttr;

TritonGPUToLLVMTypeConverter::TritonGPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &options,
    const TargetInfoBase &targetInfo, const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, options, analysis) {
  addConversion([ctx](triton::PointerType type) -> std::optional<Type> {
    return LLVM::LLVMPointerType::get(ctx, type.getAddressSpace());
  });
  addConversion([ctx](TensorDescType type) -> std::optional<Type> {
    return LLVM::LLVMPointerType::get(ctx, 1);
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

namespace {
LLVM::LLVMStructType getSharedMemoryType(MLIRContext *ctx, int64_t rank,
                                         const TargetInfoBase &targetInfo) {
  SmallVector<Type, 4> types;
  // base ptr
  auto ptrType =
      LLVM::LLVMPointerType::get(ctx, targetInfo.getSharedAddressSpace());
  types.push_back(ptrType);
  // offsets
  for (auto i = 0; i < rank; i++) {
    types.push_back(IntegerType::get(ctx, 32));
  }
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}
} // namespace

Type TritonGPUToLLVMTypeConverter::convertTritonTensorType(
    RankedTensorType type, const TargetInfoBase &targetInfo) {
  auto ctx = type.getContext();
  if (isa<SharedEncodingAttr>(type.getEncoding())) {
    return getSharedMemoryType(ctx, type.getRank(), targetInfo);
  }

  Type eltType = convertType(type.getElementType());
  unsigned numElementsPerThread = getTotalElemsPerThread(type);
  SmallVector<Type, 4> types(numElementsPerThread, eltType);
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}

Type TritonGPUToLLVMTypeConverter::convertMemDescType(
    MemDescType type, const TargetInfoBase &targetInfo) {
  auto ctx = type.getContext();
  if (isa<SharedEncodingAttr>(type.getEncoding())) {
    return getSharedMemoryType(ctx, type.getRank(), targetInfo);
  }
  emitError(UnknownLoc::get(ctx), "unsupported MemDescType encoding");
  return nullptr;
}

Type TritonGPUToLLVMTypeConverter::convertAsyncTokenType(
    triton::gpu::AsyncTokenType type) {
  return IntegerType::get(type.getContext(), 32);
}
