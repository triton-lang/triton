#ifndef TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_TYPECONVERTER_H

#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "llvm/Support/AMDGPUAddrSpace.h"

using namespace mlir;
using namespace mlir::triton;

class TritonAMDGPUToLLVMTypeConverter : public TritonGPUToLLVMTypeConverter {
public:
  TritonAMDGPUToLLVMTypeConverter(MLIRContext *ctx,
                                  const LowerToLLVMOptions &options,
                                  const TargetInfoBase &targetInfo,
                                  const DataLayoutAnalysis *analysis = nullptr)
      : TritonGPUToLLVMTypeConverter(ctx, options, targetInfo, analysis) {
    // Override the base conversion (last registration wins) with the AMDGPU
    // address spaces.
    addConversion([ctx](triton::PointerType type) -> std::optional<Type> {
      switch (type.getAddressSpace()) {
      case triton::PtrAddrSpace::Global:
        return LLVM::LLVMPointerType::get(ctx, llvm::AMDGPUAS::GLOBAL_ADDRESS);
      case triton::PtrAddrSpace::Descriptor:
        return LLVM::LLVMPointerType::get(ctx, llvm::AMDGPUAS::FLAT_ADDRESS);
      }
      llvm_unreachable("unknown PtrAddrSpace");
    });
    addConversion([&](TensorDescType type) -> std::optional<Type> {
      return convertTensorDescType(type);
    });
  }

  Type convertTensorDescType(triton::TensorDescType type) {
    auto ctx = type.getContext();
    int numDwords = amdgpu::getTensorDescNumDwords(type);

    // Keep the MLIR-visible descriptor type as a flat `{i32 × N}` struct so
    // that its in-memory ABI matches the host-side `TDMDescriptor` layout in
    // third_party/amd/backend/driver.c (N consecutive uint32_t).  Inside the
    // lowering we pack these scalars into vector groups (<4 × i32>,
    // <8 × i32>) for cleaner intra-function code.
    auto types = SmallVector<Type>(numDwords, IntegerType::get(ctx, 32));
    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }
};

#endif
