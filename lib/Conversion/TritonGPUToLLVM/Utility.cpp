#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
namespace mlir {

namespace LLVM {
using namespace mlir::triton;

Value createConstantI32(Location loc, OpBuilder &rewriter, int32_t v) {
  auto i32ty = rewriter.getIntegerType(32);
  return rewriter.create<LLVM::ConstantOp>(loc, i32ty,
                                           IntegerAttr::get(i32ty, v));
}

Value createConstantF32(Location loc, OpBuilder &rewriter, float v) {
  auto type = type::f32Ty(rewriter.getContext());
  return rewriter.create<LLVM::ConstantOp>(loc, type,
                                           rewriter.getF32FloatAttr(v));
}

Value createConstantF64(Location loc, OpBuilder &rewriter, double v) {
  auto type = type::f64Ty(rewriter.getContext());
  return rewriter.create<LLVM::ConstantOp>(loc, type,
                                           rewriter.getF64FloatAttr(v));
}

Value createNaNConstant(Location loc, OpBuilder &rewriter, Type type) {
  if (!type.isa<FloatType>()) {
    llvm::report_fatal_error("Creating NaN constant for non-float type!");
  }
  return rewriter.create<LLVM::ConstantOp>(
      loc, type, APFloat::getNaN(type.cast<FloatType>().getFloatSemantics()));
}

// Create an index type constant.
Value createIndexConstant(OpBuilder &builder, Location loc,
                          TypeConverter *converter, int64_t value) {
  Type ty = converter->convertType(builder.getIndexType());
  return builder.create<LLVM::ConstantOp>(loc, ty,
                                          builder.getIntegerAttr(ty, value));
}

// Create an integer constant of \param width bits.
Value createLLVMIntegerConstant(OpBuilder &builder, Location loc, short width,
                                int64_t value) {
  Type ty = builder.getIntegerType(width);
  return builder.create<LLVM::ConstantOp>(loc, ty,
                                          builder.getIntegerAttr(ty, value));
}

// A wrapper of LoadDSmemOp when vec = 1
// (1) Get bitwidth from elemTy
// (2) Create LoadDSmemOp
// (3) Bitcast result from dataTy (u16/u32/u64) back to elemTy
Value createLoadDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Type elemTy) {
  assert(addr.getType().isa<LLVMPointerType>() &&
         "addr must be a pointer type");
  auto ptrTy = addr.getType().cast<LLVMPointerType>();
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  Value ret =
      rewriter.create<triton::nvgpu::LoadDSmemOp>(loc, addr, ctaId, bitwidth);
  return bitcast(ret, elemTy);
}

// A wrapper of LoadDSmemOp when vec > 1
// (1) Get bitwidth from elemTy
// (2) Create LoadDSmemOp and extract results from retStruct
// (3) Bitcast results from dataTy (u16/u32/u64) back to elemTy
SmallVector<Value> createLoadDSmem(Location loc, PatternRewriter &rewriter,
                                   Value addr, Value ctaId, unsigned vec,
                                   Type elemTy) {
  assert(addr.getType().isa<LLVMPointerType>() &&
         "addr must be a pointer type");
  auto ptrTy = addr.getType().cast<LLVMPointerType>();
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  Value retStruct = rewriter.create<triton::nvgpu::LoadDSmemOp>(
      loc, addr, ctaId, bitwidth, vec);
  SmallVector<Value> retVals;
  for (unsigned i = 0; i < vec; ++i) {
    auto dataTy = rewriter.getIntegerType(bitwidth);
    Value data = extract_val(dataTy, retStruct, i);
    retVals.push_back(bitcast(data, elemTy));
  }
  return retVals;
}

// A wrapper of StoreDSmemOp when vec = 1
// (1) Get bitwidth from elemTy
// (2) Bitcast value from elemTy to dataTy (u16/u32/u64)
// (3) Create StoreDSmemOp
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Value value, Value pred) {
  assert(addr.getType().isa<LLVMPointerType>() &&
         "addr must be a pointer type");
  auto ptrTy = addr.getType().cast<LLVMPointerType>();
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  unsigned bitwidth = value.getType().getIntOrFloatBitWidth();
  auto dataTy = rewriter.getIntegerType(bitwidth);
  Value data = bitcast(value, dataTy);
  rewriter.create<triton::nvgpu::StoreDSmemOp>(loc, addr, ctaId, data, pred);
}

// A wrapper of StoreDSmemOp when vec = 1 and pred = 1
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Value value) {
  Value pred = int_val(/*width=*/1, 1);
  createStoreDSmem(loc, rewriter, addr, ctaId, value, pred);
}

// A wrapper of StoreDSmemOp when vec > 1
// (1) Get bitwidth from elemTy
// (2) Bitcast values from elemTy to dataTy (u16/u32/u64)
// (3) Create StoreDSmemOp
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, ArrayRef<Value> values, Value pred) {
  assert(addr.getType().isa<LLVMPointerType>() &&
         "addr must be a pointer type");
  auto ptrTy = addr.getType().cast<LLVMPointerType>();
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  unsigned bitwidth = 0;
  if (!values.empty()) {
    bitwidth = values.back().getType().getIntOrFloatBitWidth();
  }
  auto dataTy = rewriter.getIntegerType(bitwidth);
  SmallVector<Value> data;
  for (unsigned i = 0; i < values.size(); ++i)
    data.push_back(bitcast(values[i], dataTy));
  rewriter.create<triton::nvgpu::StoreDSmemOp>(loc, addr, ctaId, data, pred);
}

// A wrapper of StoreDSmemOp when vec > 1 and pred = 1
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, ArrayRef<Value> values) {
  Value pred = int_val(/*width=*/1, 1);
  createStoreDSmem(loc, rewriter, addr, ctaId, values, pred);
}

SharedMemoryObject
getSharedMemoryObjectFromStruct(Location loc, Value llvmStruct, Type elemTy,
                                ConversionPatternRewriter &rewriter) {
  ArrayRef<Type> types =
      llvmStruct.getType().cast<LLVM::LLVMStructType>().getBody();
  SmallVector<Value> elems(types.size());
  for (unsigned i = 0; i < types.size(); ++i) {
    Type type = types[i];
    elems[i] = extract_val(type, llvmStruct, i);
  }

  auto rank = (elems.size() - 1) / 2;
  return {/*base=*/elems[0],
          /*baseElemType=*/elemTy,
          /*strides=*/{elems.begin() + 1, elems.begin() + 1 + rank},
          /*offsets=*/{elems.begin() + 1 + rank, elems.end()}};
}

SmallVector<Value> getStridesFromShapeAndOrder(ArrayRef<int64_t> shape,
                                               ArrayRef<unsigned> order,
                                               Location loc,
                                               RewriterBase &rewriter) {
  auto rank = shape.size();
  SmallVector<Value> strides(rank);
  int64_t stride = 1;
  for (auto idx : order) {
    strides[idx] = i32_val(stride);
    stride *= shape[idx];
  }
  return strides;
}

// Convert an \param index to a multi-dim coordinate given \param shape and
// \param order.
SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               Value linear, ArrayRef<unsigned> shape,
                               ArrayRef<unsigned> order) {
  unsigned rank = shape.size();
  assert(rank == order.size());
  auto reordered = applyPermutation(shape, order);
  SmallVector<Value> reorderedMultiDim(rank);
  if (auto constantOp = linear.getDefiningOp<arith::ConstantOp>()) {
    unsigned intVal =
        constantOp.getValue().cast<IntegerAttr>().getValue().getSExtValue();
    reorderedMultiDim = delinearize(rewriter, loc, intVal, reordered);
  } else {
    reorderedMultiDim = delinearize(rewriter, loc, linear, reordered);
  }
  SmallVector<Value> multiDim(rank);
  for (unsigned i = 0; i < rank; ++i) {
    multiDim[order[i]] = reorderedMultiDim[i];
  }
  return multiDim;
}

SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               unsigned linear, ArrayRef<unsigned> shape) {
  unsigned rank = shape.size();
  assert(rank > 0);
  SmallVector<Value> multiDim(rank);
  unsigned remained = linear;
  for (auto &&en : llvm::enumerate(shape)) {
    unsigned dimSize = en.value();
    multiDim[en.index()] = i32_val(remained % dimSize);
    remained = remained / dimSize;
  }
  return multiDim;
}

SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               Value linear, ArrayRef<unsigned> shape) {
  unsigned rank = shape.size();
  assert(rank > 0);
  SmallVector<Value> multiDim(rank);
  Value remained = linear;
  for (auto &&en : llvm::enumerate(shape)) {
    Value dimSize = i32_val(en.value());
    multiDim[en.index()] = urem(remained, dimSize);
    remained = udiv(remained, dimSize);
  }
  return multiDim;
}

Value linearize(ConversionPatternRewriter &rewriter, Location loc,
                ArrayRef<Value> multiDim, ArrayRef<unsigned> shape,
                ArrayRef<unsigned> order) {
  return linearize(rewriter, loc, applyPermutation(multiDim, order),
                   applyPermutation(shape, order));
}

Value linearize(ConversionPatternRewriter &rewriter, Location loc,
                ArrayRef<Value> multiDim, ArrayRef<unsigned> shape) {
  auto rank = multiDim.size();
  Value linear = i32_val(0);
  if (rank > 0) {
    linear = multiDim.back();
    for (auto [dim, dimShape] :
         llvm::reverse(llvm::zip(multiDim.drop_back(), shape.drop_back()))) {
      Value dimSize = i32_val(dimShape);
      linear = add(mul(linear, dimSize), dim);
    }
  }
  return linear;
}

Value addStringToModule(Location loc, ConversionPatternRewriter &rewriter,
                        StringRef key, StringRef content) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto ctx = moduleOp.getContext();
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (key + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));

  llvm::SmallString<64> contentStr(content);
  size_t contentSize = contentStr.size_in_bytes();
  auto globalType = LLVM::LLVMArrayType::get(i8_ty, contentSize);

  LLVM::GlobalOp global;
  {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    global = rewriter.create<LLVM::GlobalOp>(
        UnknownLoc::get(ctx), globalType,
        /*isConstant=*/true, LLVM::Linkage::Internal, stringConstName,
        rewriter.getStringAttr(contentStr));
  }

  Value zero = i32_val(0);
  Type globalPtrType = LLVM::LLVMPointerType::get(ctx, global.getAddrSpace());
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(
      UnknownLoc::get(ctx), globalPtrType, global.getSymName());
  Value stringStart =
      gep(ptr_ty(ctx), i8_ty, globalPtr, SmallVector<Value>({zero}));
  return stringStart;
}

} // namespace LLVM
} // namespace mlir
