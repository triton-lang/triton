#include "TypeConverter.h"
#include "DotOpHelpers.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"

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

TritonGPUToLLVMTypeConverter::TritonGPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, option, analysis) {
  addConversion([&](triton::PointerType type) -> llvm::Optional<Type> {
    return convertTritonPointerType(type);
  });
  addConversion([&](RankedTensorType type) -> llvm::Optional<Type> {
    return convertTritonTensorType(type);
  });
  // Internally store float8 as int8
  addConversion([&](mlir::Float8E4M3FNType type) -> llvm::Optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E5M2Type type) -> llvm::Optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  // Internally store bfloat16 as int16
  addConversion([&](BFloat16Type type) -> llvm::Optional<Type> {
    return IntegerType::get(type.getContext(), 16);
  });
}

Type TritonGPUToLLVMTypeConverter::convertTritonPointerType(
    triton::PointerType type) {
  // Recursively translate pointee type
  return LLVM::LLVMPointerType::get(convertType(type.getPointeeType()),
                                    type.getAddressSpace());
}

Value TritonGPUToLLVMTypeConverter::packLLElements(
    Location loc, ValueRange resultVals, ConversionPatternRewriter &rewriter,
    Type type) {
  auto structType = this->convertType(type);
  if (!structType.isa<LLVM::LLVMStructType>()) {
    return *resultVals.begin();
  }

  Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structType);
  // llvm::outs() << structType << "\n";
  for (const auto &v : llvm::enumerate(resultVals)) {
    assert(v.value() && "can not insert null values");
    llvmStruct = insert_val(structType, llvmStruct, v.value(), v.index());
  }
  return llvmStruct;
}

SmallVector<Value> TritonGPUToLLVMTypeConverter::unpackLLElements(
    Location loc, Value llvmStruct, ConversionPatternRewriter &rewriter,
    Type type) {
  assert(bool(llvmStruct) && "can not unpack null values");
  if (llvmStruct.getType().isIntOrIndexOrFloat() ||
      llvmStruct.getType().isa<triton::PointerType>() ||
      llvmStruct.getType().isa<LLVM::LLVMPointerType>())
    return {llvmStruct};
  ArrayRef<Type> types =
      llvmStruct.getType().cast<LLVM::LLVMStructType>().getBody();
  SmallVector<Value> results(types.size());
  for (unsigned i = 0; i < types.size(); ++i) {
    Type type = types[i];
    results[i] = extract_val(type, llvmStruct, i);
  }
  return results;
}

llvm::Optional<Type>
TritonGPUToLLVMTypeConverter::convertTritonTensorType(RankedTensorType type) {
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
          // <2xi16>/<4xi8> => i32
          // We are doing this because NVPTX inserts extra integer instrs to
          // pack & unpack vectors of sub-word integers
          // Note: this needs to be synced with
          //       DotOpMmaV2ConversionHelper::loadX4
          if (elemTy.isa<IntegerType>() &&
              (elemTy.getIntOrFloatBitWidth() == 8 ||
               elemTy.getIntOrFloatBitWidth() == 16))
            targetTy = IntegerType::get(ctx, 32);
        } else {
          assert(false && "Unsupported element type");
        }
        auto elems = getElemsPerThread(type);
        return struct_ty(SmallVector<Type>(elems, targetTy));
      }

      if (mmaLayout.isVolta()) {
        int elems = getElemsPerThread(type);
        Type x2Ty = vec_ty(elemTy, 2);
        return struct_ty(SmallVector<Type>(elems, x2Ty));
      }
    }

    llvm::errs() << "Unexpected dot operand layout detected in "
                    "TritonToLLVMTypeConverter";
    return std::nullopt;
  }

  return std::nullopt;
}