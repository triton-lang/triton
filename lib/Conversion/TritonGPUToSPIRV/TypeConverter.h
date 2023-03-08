#ifndef TRITON_CONVERSION_TRITONGPU_TO_SPIRV_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONGPU_TO_SPIRV_TYPECONVERTER_H

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "triton/Conversion/MLIRTypes.h"

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

/// Mapping between SPIR-V storage classes to Triton memory spaces.
///
#define STORAGE_SPACE_MAP_LIST(MAP_FN)                                         \
  MAP_FN(spirv::StorageClass::CrossWorkgroup, 1)                               \
  MAP_FN(spirv::StorageClass::Workgroup, 3)                                    \

#if 0
  MAP_FN(spirv::StorageClass::StorageBuffer, 0)                                \
  MAP_FN(spirv::StorageClass::Uniform, 4)                                      \
  MAP_FN(spirv::StorageClass::Private, 5)                                      \
  MAP_FN(spirv::StorageClass::Function, 6)                                     \
  MAP_FN(spirv::StorageClass::PushConstant, 7)                                 \
  MAP_FN(spirv::StorageClass::UniformConstant, 8)                              \
  MAP_FN(spirv::StorageClass::Input, 9)                                        \
  MAP_FN(spirv::StorageClass::Output, 10)                                      \
  MAP_FN(spirv::StorageClass::CrossWorkgroup, 11)                              \
  MAP_FN(spirv::StorageClass::AtomicCounter, 12)                               \
  MAP_FN(spirv::StorageClass::Image, 13)                                       \
  MAP_FN(spirv::StorageClass::CallableDataKHR, 14)                             \
  MAP_FN(spirv::StorageClass::IncomingCallableDataKHR, 15)                     \
  MAP_FN(spirv::StorageClass::RayPayloadKHR, 16)                               \
  MAP_FN(spirv::StorageClass::HitAttributeKHR, 17)                             \
  MAP_FN(spirv::StorageClass::IncomingRayPayloadKHR, 18)                       \
  MAP_FN(spirv::StorageClass::ShaderRecordBufferKHR, 19)                       \
  MAP_FN(spirv::StorageClass::PhysicalStorageBuffer, 20)                       \
  MAP_FN(spirv::StorageClass::CodeSectionINTEL, 21)                            \
  MAP_FN(spirv::StorageClass::DeviceOnlyINTEL, 22)                             \
  MAP_FN(spirv::StorageClass::HostOnlyINTEL, 23)
#endif


class TritonGPUToSPIRVTypeConverter : public SPIRVTypeConverter {
public:
  using TypeConverter::convertType;

  TritonGPUToSPIRVTypeConverter(spirv::TargetEnvAttr &targetAttr, SPIRVConversionOptions &option,
                                const DataLayoutAnalysis *analysis = nullptr)
          : SPIRVTypeConverter(targetAttr, option/*, analysis*/) {
    addConversion([&](triton::PointerType type) -> llvm::Optional<Type> {
      return convertTritonPointerType(type);
    });
    addConversion([&](RankedTensorType type) -> llvm::Optional<Type> {
      return convertTritonTensorType(type);
    });
    // Internally store float8 as int8
    addConversion([&](triton::Float8Type type) -> llvm::Optional<Type> {
      return IntegerType::get(type.getContext(), 8);
    });
    addConversion([&](IndexType type) -> llvm::Optional<Type> {
      return getIndexType();
    });

    // Add generic source and target materializations to handle cases where
    // non-SPIRV types persist after an SPIRV conversion.
    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> Optional<Value> {
      if (inputs.size() != 1)
        return std::nullopt;

      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
              .getResult(0);
    });
    addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> Optional<Value> {
      if (inputs.size() != 1)
        return std::nullopt;

      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
              .getResult(0);
    });
  }

  Type convertTritonPointerType(triton::PointerType type) {
    // Recursively translate pointee type
    Optional<spirv::StorageClass> storageClass = getStorageClassForMemorySpace(
            type.getAddressSpace());
    assert(storageClass && "uncompatible pointer address type in SPIRV");
    return spirv::PointerType::get(convertType(type.getPointeeType()), *storageClass);
  }

  Optional<spirv::StorageClass>
  getStorageClassForMemorySpace(unsigned space) {
#define STORAGE_SPACE_MAP_FN(storage, space)                                   \
  case space:                                                                  \
    return storage;

    switch (space) {
      STORAGE_SPACE_MAP_LIST(STORAGE_SPACE_MAP_FN)
      default:
        return std::nullopt;
    }
#undef STORAGE_SPACE_MAP_FN
  }

  llvm::Optional<Type> convertTritonTensorType(RankedTensorType type) {
    auto ctx = type.getContext();
    Attribute layout = type.getEncoding();
    auto shape = type.getShape();
    if (layout &&
        (layout.isa<BlockedEncodingAttr>() || layout.isa<SliceEncodingAttr>() ||
         layout.isa<MmaEncodingAttr>())) {
      unsigned numElementsPerThread = getElemsPerThread(type);
      SmallVector<Type, 4> types(numElementsPerThread,
                                 convertType(type.getElementType()));
      return spirv::StructType::get(types);
    } else if (auto shared_layout =
            layout.dyn_cast_or_null<SharedEncodingAttr>()) {
      assert(0 && "doesn't support shared encoding yet");
    } else if (auto dotOpLayout =
            layout.dyn_cast_or_null<DotOperandEncodingAttr>()) {
      assert(0 && "doesn't support dot encoding yet");

      llvm::errs() << "Unexpected dot operand layout detected in "
                      "TritonToLLVMTypeConverter";
      return std::nullopt;
    }
    return std::nullopt;
  }
};

#endif
