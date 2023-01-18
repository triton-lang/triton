#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TYPECONVERTER_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"

#include "DotOpHelpers.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::DotOpFMAConversionHelper;
using ::mlir::LLVM::DotOpMmaV1ConversionHelper;
using ::mlir::LLVM::MMA16816ConversionHelper;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

class TritonGPUToLLVMTypeConverter : public LLVMTypeConverter {
public:
  using TypeConverter::convertType;

  TritonGPUToLLVMTypeConverter(MLIRContext *ctx, LowerToLLVMOptions &option,
                               const DataLayoutAnalysis *analysis = nullptr)
      : LLVMTypeConverter(ctx, option, analysis) {
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
    // Internally store bfloat16 as int16
    addConversion([&](BFloat16Type type) -> llvm::Optional<Type> {
      return IntegerType::get(type.getContext(), 16);
    });
  }

  Type convertTritonPointerType(triton::PointerType type) {
    // Recursively translate pointee type
    return LLVM::LLVMPointerType::get(convertType(type.getPointeeType()),
                                      type.getAddressSpace());
  }

  llvm::Optional<Type> convertTritonTensorType(RankedTensorType type) {
    auto ctx = type.getContext();
    Attribute layout = type.getEncoding();
    SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());

    if (layout &&
        (layout.isa<BlockedEncodingAttr>() || layout.isa<SliceEncodingAttr>() ||
         layout.isa<MmaEncodingAttr>())) {
      unsigned numElementsPerThread = getElemsPerThread(type);
      SmallVector<Type, 4> types(numElementsPerThread,
                                 convertType(type.getElementType()));
      return LLVM::LLVMStructType::getLiteral(ctx, types);
    } else if (auto shared_layout =
                   layout.dyn_cast_or_null<SharedEncodingAttr>()) {
      SmallVector<Type, 4> types;
      // base ptr
      auto ptrType =
          LLVM::LLVMPointerType::get(convertType(type.getElementType()), 3);
      types.push_back(ptrType);
      // shape dims
      auto rank = type.getRank();
      // offsets + strides
      for (auto i = 0; i < rank * 2; i++) {
        types.push_back(IntegerType::get(ctx, 32));
      }
      return LLVM::LLVMStructType::getLiteral(ctx, types);
    } else if (auto dotOpLayout =
                   layout.dyn_cast_or_null<DotOperandEncodingAttr>()) {
      if (dotOpLayout.getParent()
              .isa<BlockedEncodingAttr>()) { // for parent is blocked layout
        int numElemsPerThread =
            DotOpFMAConversionHelper::getNumElemsPerThread(shape, dotOpLayout);

        return LLVM::LLVMStructType::getLiteral(
            ctx, SmallVector<Type>(numElemsPerThread, type::f32Ty(ctx)));
      } else { // for parent is MMA layout
        auto mmaLayout = dotOpLayout.getParent().cast<MmaEncodingAttr>();
        auto wpt = mmaLayout.getWarpsPerCTA();
        Type elemTy = convertType(type.getElementType());
        if (mmaLayout.isAmpere()) {
          const llvm::DenseMap<int, Type> targetTyMap = {
              {32, vec_ty(elemTy, 1)},
              {16, vec_ty(elemTy, 2)},
              {8, vec_ty(elemTy, 4)},
          };
          Type targetTy;
          if (targetTyMap.count(elemTy.getIntOrFloatBitWidth())) {
            targetTy = targetTyMap.lookup(elemTy.getIntOrFloatBitWidth());
          } else {
            assert(false && "Unsupported element type");
          }
          if (dotOpLayout.getOpIdx() == 0) { // $a
            auto elems =
                MMA16816ConversionHelper::getANumElemsPerThread(type, wpt[0]);
            return struct_ty(SmallVector<Type>(elems, targetTy));
          }
          if (dotOpLayout.getOpIdx() == 1) { // $b
            auto elems =
                MMA16816ConversionHelper::getBNumElemsPerThread(type, wpt[1]);
            return struct_ty(SmallVector<Type>(elems, targetTy));
          }
        }

        if (mmaLayout.isVolta()) {
          DotOpMmaV1ConversionHelper helper(mmaLayout);

          // TODO[Superjomn]: Both transA and transB are not available here.
          bool trans = false;
          // TODO[Superjomn]: The order of A and B are not available here.
          SmallVector<unsigned> order({1, 0});
          if (trans) {
            std::swap(shape[0], shape[1]);
            std::swap(order[0], order[1]);
          }

          if (dotOpLayout.getOpIdx() == 0) { // $a
            int elems = helper.numElemsPerThreadA(shape, order);
            Type x2Ty = vec_ty(elemTy, 2);
            return struct_ty(SmallVector<Type>(elems, x2Ty));
          }
          if (dotOpLayout.getOpIdx() == 1) { // $b
            int elems = helper.numElemsPerThreadB(shape, order);
            Type x2Ty = vec_ty(elemTy, 2);
            return struct_ty(SmallVector<Type>(elems, x2Ty));
          }
        }
      }

      llvm::errs() << "Unexpected dot operand layout detected in "
                      "TritonToLLVMTypeConverter";
      return llvm::None;
    }

    return llvm::None;
  }
};

#endif
