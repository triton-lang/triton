#ifndef TRITON_CONVERSION_TRITONGPU_TO_SPIRV_UTILITY_H
#define TRITON_CONVERSION_TRITONGPU_TO_SPIRV_UTILITY_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/PTXAsmFormat.h"

// Shortcuts for some commonly used LLVM ops to keep code simple and intuitive
// Operators
#define inttoptr(...) rewriter.create<spirv::IntToPtrOp>(loc, __VA_ARGS__)
#define ptrtoint(...) rewriter.create<spirv::PtrToIntOp>(loc, __VA_ARGS__)
#define zext(...) rewriter.create<spirv::UConvertOp>(loc, __VA_ARGS__)
#define udiv(...) rewriter.create<spirv::UDivOp>(loc, __VA_ARGS__)
#define urem(...) rewriter.create<spirv::UModOp>(loc, __VA_ARGS__)
#define add(...) rewriter.create<spirv::IAddOp>(loc, __VA_ARGS__)
#define sub(...) rewriter.create<spirv::ISubOp>(loc, __VA_ARGS__)
#define fadd(...) rewriter.create<spirv::FAddOp>(loc, __VA_ARGS__)
#define mul(...) rewriter.create<spirv::IMulOp>(loc, __VA_ARGS__)
#define fmul(...) rewriter.create<spirv::FMulOp>(loc, __VA_ARGS__)
#define smax(...) rewriter.create<spirv::CLSMaxOp>(loc, __VA_ARGS__)
#define umax(...) rewriter.create<spirv::CLUMaxOp>(loc, __VA_ARGS__)
#define fmax(...) rewriter.create<spirv::CLFMaxOp>(loc, __VA_ARGS__)
#define smin(...) rewriter.create<spirv::CLSMinOp>(loc, __VA_ARGS__)
#define umin(...) rewriter.create<spirv::CLUMinOp>(loc, __VA_ARGS__)
#define fmin(...) rewriter.create<spirv::CLFMinOp>(loc, __VA_ARGS__)
#define and_(...) rewriter.create<spirv::BitwiseAndOp>(loc, __VA_ARGS__)
#define xor_(...) rewriter.create<spirv::BitwiseXorOp>(loc, __VA_ARGS__)
#define bitcast(val__, type__)                                                 \
  rewriter.create<spirv::BitcastOp>(loc, type__, val__)
#define gep(...) rewriter.create<spirv::PtrAccessChainOp>(loc, __VA_ARGS__, ValueRange{})
#define ptr_ty(...) spirv::PointerType::get(__VA_ARGS__)
#define insert_val(...) rewriter.create<spirv::CompositeInsertOp>(loc, __VA_ARGS__)
#define extract_val(...) rewriter.create<spirv::CompositeExtractOp>(loc, __VA_ARGS__)
#define insert_element(...)                                                    \
  rewriter.create<spirv::VectorInsertDynamicOp>(loc, __VA_ARGS__)
#define extract_element(...)                                                   \
  rewriter.create<spirv::VectorExtractDynamicOp>(loc, __VA_ARGS__)
#define load(...) rewriter.create<spirv::LoadOp>(loc, __VA_ARGS__)
#define store(val, ptr) rewriter.create<spirv::StoreOp>(loc, ptr, val)
#define fcmp_ogt(lhs, rhs)                                                     \
  rewriter.create<spirv::FOrdGreaterThanOp>(loc, lhs, rhs)
#define fcmp_olt(lhs, rhs)                                                     \
  rewriter.create<spirv::FOrdLessThanOp>(loc, lhs, rhs)
#define icmp_eq(...)                                                           \
  rewriter.create<spirv::IEqualOp>(loc, __VA_ARGS__)
#define icmp_ne(...)                                                           \
  rewriter.create<spirv::INotEqualOp>(loc, __VA_ARGS__)
#define icmp_slt(...)                                                          \
  rewriter.create<spirv::SLessThanOp>(loc, __VA_ARGS__)
#define icmp_sle(...)                                                          \
  rewriter.create<spirv::SLessThanEqualOp>(loc, __VA_ARGS__)
#define icmp_sgt(...)                                                          \
  rewriter.create<spirv::SGreaterThanOp>(loc, __VA_ARGS__)
#define icmp_sge(...)                                                          \
  rewriter.create<spirv::SGreaterThanEqualOp>(loc, __VA_ARGS__)
#define icmp_ult(...)                                                          \
  rewriter.create<spirv::FUnordLessThanOp>(loc, __VA_ARGS__)
#define icmp_ule(...)                                                          \
  rewriter.create<spirv::FUnordLessThanEqualOp>(loc, __VA_ARGS__)
#define icmp_ugt(...)                                                          \
  rewriter.create<spirv::FUnordGreaterThanOp>(loc, __VA_ARGS__)
#define icmp_uge(...)                                                          \
  rewriter.create<spirv::FUnordGreaterThanEqualOp>(loc, __VA_ARGS__)
#define select(...) rewriter.create<spirv::SelectOp>(loc, __VA_ARGS__)
#define address_of(...) rewriter.create<spirv::AddressOfOp>(loc, __VA_ARGS__)
#define barrier() rewriter.create<mlir::gpu::BarrierOp>(loc)
#define undef(...) rewriter.create<spirv::UndefOp>(loc, __VA_ARGS__)

// Types
#define i32_ty rewriter.getIntegerType(32)
#define i16_ty rewriter.getIntegerType(16)
#define ui32_ty rewriter.getIntegerType(32, false)
#define f16_ty rewriter.getF16Type()
#define bf16_ty rewriter.getBF16Type()
#define i8_ty rewriter.getIntegerType(8)
#define f32_ty rewriter.getF32Type()
#define f64_ty rewriter.getF64Type()
#define vec_ty(type, num) VectorType::get(num, type)
#define f32_val(...) spirv::createConstantF32(loc, rewriter, __VA_ARGS__)
#define f64_val(...) spirv::createConstantF64(loc, rewriter, __VA_ARGS__)
#define void_ty(ctx) spirv::LLVMVoidType::get(ctx)
#define struct_ty(...) spirv::StructType::get(__VA_ARGS__)
#define array_ty(elemTy, count) spirv::LLVMArrayType::get(elemTy, count)

// Constants
#define i32_val(...) spirv::createConstantI32(loc, rewriter, __VA_ARGS__)
#define int_val(width, val)                                                    \
  spirv::createSPIRVIntegerConstant(rewriter, loc, width, val)
#define idx_val(...)                                                           \
  spirv::createIndexConstant(rewriter, loc, this->getTypeConverter(),           \
                            __VA_ARGS__)
#define tid_val() getThreadId(rewriter, loc)

namespace mlir {
namespace triton {

// Delinearize supposing order is [0, 1, .. , n]
template <typename T>
llvm::SmallVector<T> getMultiDimIndexImpl(T linearIndex,
                                          llvm::ArrayRef<T> shape) {
  // shape: {a, b, c, d}  ->  accMul: {1, a, a*b, a*b*c}
  size_t rank = shape.size();
  T accMul = product(shape.drop_back());
  T linearRemain = linearIndex;
  llvm::SmallVector<T> multiDimIndex(rank);
  for (int i = rank - 1; i >= 0; --i) {
    multiDimIndex[i] = linearRemain / accMul;
    linearRemain = linearRemain % accMul;
    if (i != 0) {
      accMul = accMul / shape[i - 1];
    }
  }
  return multiDimIndex;
}

template <typename T>
llvm::SmallVector<T> getMultiDimIndex(T linearIndex, llvm::ArrayRef<T> shape,
                                      llvm::ArrayRef<unsigned> order) {
  size_t rank = shape.size();
  assert(rank == order.size());
  auto reordered = reorder(shape, order);
  auto reorderedMultiDim = getMultiDimIndexImpl<T>(linearIndex, reordered);
  llvm::SmallVector<T> multiDim(rank);
  for (unsigned i = 0; i < rank; ++i) {
    multiDim[order[i]] = reorderedMultiDim[i];
  }
  return multiDim;
}

// Linearize supposing order is [0, 1, .. , n]
template <typename T>
static T getLinearIndexImpl(llvm::ArrayRef<T> multiDimIndex,
                            llvm::ArrayRef<T> shape) {
  assert(multiDimIndex.size() == shape.size());
  // shape: {a, b, c, d}  ->  accMul: {1, a, a*b, a*b*c}
  size_t rank = shape.size();
  T accMul = product(shape.drop_back());
  T linearIndex = 0;
  for (int i = rank - 1; i >= 0; --i) {
    linearIndex += multiDimIndex[i] * accMul;
    if (i != 0) {
      accMul = accMul / shape[i - 1];
    }
  }
  return linearIndex;
}

template <typename T>
static T getLinearIndex(llvm::ArrayRef<T> multiDimIndex,
                        llvm::ArrayRef<T> shape,
                        llvm::ArrayRef<unsigned> order) {
  assert(shape.size() == order.size());
  return getLinearIndexImpl<T>(reorder(multiDimIndex, order),
                               reorder(shape, order));
}

} // namespace triton

namespace spirv {
using namespace mlir::triton;

static Value getStructFromElements(Location loc, ValueRange resultVals,
                                   ConversionPatternRewriter &rewriter,
                                   Type structType) {
  if (!structType.isa<spirv::StructType>()) {
    return *resultVals.begin();
  }

  Value spirvStruct = rewriter.create<spirv::UndefOp>(loc, structType);
  for (const auto &v : llvm::enumerate(resultVals)) {
    assert(v.value() && "can not insert null values");
    spirvStruct = insert_val(structType, v.value(), spirvStruct,
                            rewriter.getI32ArrayAttr(v.index()));
  }
  return spirvStruct;
}

static SmallVector<Value>
getElementsFromStruct(Location loc, Value spirvStruct,
                      ConversionPatternRewriter &rewriter) {
  if (spirvStruct.getType().isIntOrIndexOrFloat() ||
      spirvStruct.getType().isa<triton::PointerType>() ||
      spirvStruct.getType().isa<spirv::PointerType>())
    return {spirvStruct};
  auto spirvType = spirvStruct.getType();
  if (spirvType.isa<spirv::StructType>()) {
    auto types =
            spirvStruct.getType().cast<spirv::StructType>().getElementTypes();
    SmallVector<Value> results(types.size());
    for (unsigned i = 0; i < types.size(); ++i) {
      Type type = types[i];
      results[i] = extract_val(type, spirvStruct, rewriter.getI32ArrayAttr(i));
    }
    return results;
  } else if (spirvType.isa<mlir::VectorType>()){
    auto vecType = spirvStruct.getType().cast<mlir::VectorType>();
    SmallVector<Value> results(vecType.getNumElements());
    for (unsigned i = 0; i < vecType.getNumElements(); ++i) {
      Type type = vecType.getElementType();
      results[i] = extract_val(type, spirvStruct, rewriter.getI32ArrayAttr(i));
    }
    return results;
  } else {
    assert(0);
  }
}

// Create a 32-bit integer constant.
static Value createConstantI32(Location loc, PatternRewriter &rewriter,
                               int32_t v) {
  auto i32ty = rewriter.getIntegerType(32);
  return rewriter.create<spirv::ConstantOp>(loc, i32ty,
                                           IntegerAttr::get(i32ty, v));
}

static Value createConstantF32(Location loc, PatternRewriter &rewriter,
                               float v) {
  auto type = type::f32Ty(rewriter.getContext());
  return rewriter.create<spirv::ConstantOp>(loc, type,
                                           rewriter.getF32FloatAttr(v));
}

static Value createConstantF64(Location loc, PatternRewriter &rewriter,
                               float v) {
  auto type = type::f64Ty(rewriter.getContext());
  return rewriter.create<spirv::ConstantOp>(loc, type,
                                           rewriter.getF64FloatAttr(v));
}

// Create an index type constant.
static Value createIndexConstant(OpBuilder &builder, Location loc,
                                 TypeConverter *converter, int64_t value) {
  Type ty = converter->convertType(builder.getIndexType());
  Value v = builder.create<spirv::ConstantOp>(loc, ty,
                                          builder.getIntegerAttr(ty, value));

  return v;
}

// Create an integer constant of \param width bits.
static Value createSPIRVIntegerConstant(OpBuilder &builder, Location loc,
                                       short width, int64_t value) {
  Type ty = builder.getIntegerType(width);
  return builder.create<spirv::ConstantOp>(loc, ty,
                                          builder.getIntegerAttr(ty, value));
}

/// Helper function to get strides from a given shape and its order
static SmallVector<Value>
getStridesFromShapeAndOrder(ArrayRef<int64_t> shape, ArrayRef<unsigned> order,
                            Location loc, ConversionPatternRewriter &rewriter) {
  auto rank = shape.size();
  SmallVector<Value> strides(rank);
  int64_t stride = 1;
  for (auto idx : order) {
    strides[idx] = i32_val(stride);
    stride *= shape[idx];
  }
  return strides;
}

struct SharedMemoryObject {
  Value base; // i32 ptr. The start address of the shared memory object.
  // We need to store strides as Values but not integers because the
  // extract_slice instruction can take a slice at arbitrary offsets.
  // Take $a[16:32, 16:32] as an example, though we know the stride of $a[0] is
  // 32, we need to let the instruction that uses $a to be aware of that.
  // Otherwise, when we use $a, we only know that the shape of $a is 16x16. If
  // we store strides into an attribute array of integers, the information
  // cannot pass through block argument assignment because attributes are
  // associated with operations but not Values.
  // TODO(Keren): We may need to figure out a way to store strides as integers
  // if we want to support more optimizations.
  SmallVector<Value>
      strides; // i32 int. The strides of the shared memory object.
  SmallVector<Value> offsets; // i32 int. The offsets of the shared memory
  // objects from the originally allocated object.

  SharedMemoryObject(Value base, ArrayRef<Value> strides,
                     ArrayRef<Value> offsets)
      : base(base), strides(strides.begin(), strides.end()),
        offsets(offsets.begin(), offsets.end()) {}

  SharedMemoryObject(Value base, ArrayRef<int64_t> shape,
                     ArrayRef<unsigned> order, Location loc,
                     ConversionPatternRewriter &rewriter)
      : base(base) {
    strides = getStridesFromShapeAndOrder(shape, order, loc, rewriter);

    for (auto idx : order) {
      offsets.emplace_back(i32_val(0));
    }
  }

  SmallVector<Value> getElems() const {
    SmallVector<Value> elems;
    elems.push_back(base);
    elems.append(strides.begin(), strides.end());
    elems.append(offsets.begin(), offsets.end());
    return elems;
  }

  SmallVector<Type> getTypes() const {
    SmallVector<Type> types;
    types.push_back(base.getType());
    types.append(strides.size(), IntegerType::get(base.getContext(), 32));
    types.append(offsets.size(), IntegerType::get(base.getContext(), 32));
    return types;
  }

  Value getCSwizzleOffset(int order) const {
    assert(order >= 0 && order < strides.size());
    return offsets[order];
  }

  Value getBaseBeforeSwizzle(int order, Location loc,
                             ConversionPatternRewriter &rewriter) const {
    Value cSwizzleOffset = getCSwizzleOffset(order);
    Value offset = sub(i32_val(0), cSwizzleOffset);
    Type type = base.getType();
    return gep(type, base, offset);
  }
};

static SharedMemoryObject
getSharedMemoryObjectFromStruct(Location loc, Value llvmStruct,
                                ConversionPatternRewriter &rewriter) {
  auto elems = getElementsFromStruct(loc, llvmStruct, rewriter);
  auto rank = (elems.size() - 1) / 2;
  return {/*base=*/elems[0],
          /*strides=*/{elems.begin() + 1, elems.begin() + 1 + rank},
          /*offsets=*/{elems.begin() + 1 + rank, elems.end()}};
}

static Value storeShared(ConversionPatternRewriter &rewriter, Location loc,
                         Value ptr, Value val, Value pred) {
  assert(0 && "no storeShared");
}

static Value shflSync(Location loc, ConversionPatternRewriter &rewriter,
                      Value val, int i) {
  assert(0 && "no shfl sync");
}

} // namespace spirv
} // namespace mlir

#endif
