#include "Utility.h"
#include "TypeConverter.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"

#if USE_ROCM
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#endif
namespace mlir {

namespace LLVM {
using namespace mlir::triton;

Value createConstantI32(Location loc, OpBuilder &rewriter, int32_t v) {
  auto i32ty = rewriter.getIntegerType(32);
  return rewriter.create<LLVM::ConstantOp>(loc, i32ty,
                                           IntegerAttr::get(i32ty, v));
}

Value createConstantBF16(Location loc, OpBuilder &rewriter, float v) {
  auto type = type::bf16Ty(rewriter.getContext());
  return rewriter.create<LLVM::ConstantOp>(loc, type,
                                           rewriter.getFloatAttr(type, v));
}

Value createConstantF16(Location loc, OpBuilder &rewriter, float v) {
  auto type = type::f16Ty(rewriter.getContext());
  return rewriter.create<LLVM::ConstantOp>(loc, type,
                                           rewriter.getF16FloatAttr(v));
}

Value createConstantF32(Location loc, OpBuilder &rewriter, float v) {
  auto type = type::f32Ty(rewriter.getContext());
  return rewriter.create<LLVM::ConstantOp>(loc, type,
                                           rewriter.getF32FloatAttr(v));
}

Value createConstantF64(Location loc, OpBuilder &rewriter, float v) {
  auto type = type::f64Ty(rewriter.getContext());
  return rewriter.create<LLVM::ConstantOp>(loc, type,
                                           rewriter.getF64FloatAttr(v));
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
                      Value ctaId) {
  assert(addr.getType().isa<LLVMPointerType>() &&
         "addr must be a pointer type");
  auto ptrTy = addr.getType().cast<LLVMPointerType>();
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  auto elemTy = ptrTy.getElementType();
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
                                   Value addr, Value ctaId, unsigned vec) {
  assert(addr.getType().isa<LLVMPointerType>() &&
         "addr must be a pointer type");
  auto ptrTy = addr.getType().cast<LLVMPointerType>();
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  auto elemTy = ptrTy.getElementType();
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
  auto elemTy = ptrTy.getElementType();
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
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
  auto elemTy = ptrTy.getElementType();
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
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
getSharedMemoryObjectFromStruct(Location loc, Value llvmStruct,
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
          /*strides=*/{elems.begin() + 1, elems.begin() + 1 + rank},
          /*offsets=*/{elems.begin() + 1 + rank, elems.end()}};
}

SmallVector<Value>
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

// Convert an \param index to a multi-dim coordinate given \param shape and
// \param order.
SmallVector<Value> delinearize(ConversionPatternRewriter &rewriter,
                               Location loc, Value linear,
                               ArrayRef<unsigned> shape,
                               ArrayRef<unsigned> order) {
  unsigned rank = shape.size();
  assert(rank == order.size());
  auto reordered = reorder(shape, order);
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

SmallVector<Value> delinearize(ConversionPatternRewriter &rewriter,
                               Location loc, unsigned linear,
                               ArrayRef<unsigned> shape) {
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

SmallVector<Value> delinearize(ConversionPatternRewriter &rewriter,
                               Location loc, Value linear,
                               ArrayRef<unsigned> shape) {
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
  return linearize(rewriter, loc, reorder<Value>(multiDim, order),
                   reorder<unsigned>(shape, order));
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

Value storeShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                  Value val, Value pred) {
#if USE_ROCM
  store(val, ptr);
  return val;
#else
  MLIRContext *ctx = rewriter.getContext();
  unsigned bits = std::max(8u, val.getType().getIntOrFloatBitWidth());
  const char *c = bits == 64 ? "l" : (bits == 16 ? "h" : "r");

  PTXBuilder builder;
  auto *ptrOpr = builder.newAddrOperand(ptr, "r");
  auto *valOpr = builder.newOperand(val, c);
  auto &st = builder.create<>("st")->shared().b(bits);
  st(ptrOpr, valOpr).predicate(pred, "b");
  return builder.launch(rewriter, loc, void_ty(ctx));
#endif
}

Value loadShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                 Value pred) {
#if USE_ROCM
  return load(ptr);
#else
  MLIRContext *ctx = rewriter.getContext();
  auto ptrTy = ptr.getType().cast<LLVMPointerType>();
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for loadShared");
  auto elemTy = ptrTy.getElementType();
  unsigned bitwidth = std::max(8u, elemTy.getIntOrFloatBitWidth());

  const char *c = bitwidth == 64 ? "=l" : (bitwidth == 16 ? "=h" : "=r");

  PTXBuilder builder;
  auto *dOpr = builder.newOperand(c);
  auto *ptrOpr = builder.newAddrOperand(ptr, "r");
  auto &ld = builder.create<>("ld")->shared().b(bitwidth);
  ld(dOpr, ptrOpr).predicate(pred, "b");
  return builder.launch(rewriter, loc, elemTy);
#endif
}

static Value commonShflSync(Location loc, ConversionPatternRewriter &rewriter,
                            Value val, Value i, int strideInt, NVVM::ShflKind mode,
                            Value clamp, Value laneId = Value()) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();

#ifdef USE_ROCM
  //On AMD, the ds_swizzle_b32 and ds_permute_b32 instructions work on 32bit/dwords
  //so we need promote to 32 here.
  auto valType = val.getType();
  if (!valType.isInteger(32) && bits <= 32) {
    if (!valType.isIntOrIndex())
      val = bitcast(val, int_ty(bits));
    if (bits < 32)
      val = sext(i32_ty, val);

    val = commonShflSync(loc, rewriter, val, i, strideInt, mode, clamp, laneId);

    if (bits < 32)
      val = trunc(int_ty(bits), val);
    if (!valType.isIntOrIndex())
      val = bitcast(val, valType);
    return val;
  }
#endif

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = bitcast(val, vecTy);
    Value val0 = extract_element(f32_ty, vec, i32_val(0));
    Value val1 = extract_element(f32_ty, vec, i32_val(1));
    val0 = commonShflSync(loc, rewriter, val0, i, strideInt, mode, clamp, laneId);
    val1 = commonShflSync(loc, rewriter, val1, i, strideInt, mode, clamp, laneId);
    vec = undef(vecTy);
    vec = insert_element(vecTy, vec, val0, i32_val(0));
    vec = insert_element(vecTy, vec, val1, i32_val(1));
    return bitcast(vec, val.getType());
  }

#ifdef USE_ROCM
  auto bpermute = [&](Value lane) {
    // Multiple lineId by 4. (More on permute instruction semantics:
    // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf#page=180
    Value byteOffset = i32_val(2);
    Value permuteAddr = shl(lane, byteOffset);
    return rewriter.create<ROCDL::DsBpermuteOp>(loc, valType, permuteAddr, val);
  };

  switch (mode) {
  case NVVM::ShflKind::bfly:
    if (strideInt > 16) {
      Value threadId =
          rewriter
              .create<UnrealizedConversionCastOp>(
                  loc, TypeRange{i32_ty},
                  ValueRange{rewriter.create<::mlir::gpu::ThreadIdOp>(
                      loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x)})
              .getResult(0);
      Value stride = i32_val(32);
      Value lineId = xor_(threadId, stride);
      return bpermute(lineId);
    } else {
      // This map facilates the butterfly shuffle pattern for a stride less
      // than 16. The pattern stride is the key of the map.
      DenseMap<short, unsigned int> masks{
          {16, 0x401F}, {8, 0x201F}, {4, 0x101F}, {2, 0x081F}, {1, 0x041F}};
      Value offset = i32_val(masks[strideInt]);
      return rewriter.create<ROCDL::DsSwizzleOp>(loc, valType, val, offset);
    }
    break;
  case NVVM::ShflKind::up: {
    Value mask = icmp_slt(laneId, i);
    Value delta = sub(laneId, i);
    Value index = select(mask, laneId, delta);
    return bpermute(index);
  }
  case NVVM::ShflKind::idx:
    return bpermute(i);
  default:
    assert(false && "Unsupported ShflKind");
    break;
  }
  return Value();
#else
  Type type = val.getType();
  if (type != i32_ty) {
    val = bitcast(val, int_ty(bits));
    if (bits < 32)
      val = zext(i32_ty, val);
  }
  Value mask = i32_val(0xFFFFFFFF);
  Value result = rewriter.create<NVVM::ShflOp>(loc, i32_ty, mask, val, i, clamp,
                                               mode, UnitAttr());
  if (type != i32_ty) {
    if (bits < 32)
      result = trunc(int_ty(bits), result);
    result = bitcast(result, type);
  }
  return result;
#endif
}

Value shflSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
               int i) {
  return commonShflSync(loc, rewriter, val, i32_val(i), i, NVVM::ShflKind::bfly,
                        i32_val(0x1f));
}

Value shflUpSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i, Value laneId) {
  return commonShflSync(loc, rewriter, val, i32_val(i), i, NVVM::ShflKind::up,
		  i32_val(0x0), laneId);
}

Value shflIdxSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                  int i) {
  return shflIdxSync(loc, rewriter, val, i32_val(i));
}

Value shflIdxSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                  Value i) {
  return commonShflSync(loc, rewriter, val, i, 0, NVVM::ShflKind::idx,
                        i32_val(0x1f));
}

Value getSRegValue(OpBuilder &b, Location loc, const std::string &sRegStr) {
  PTXBuilder builder;
  auto &mov = builder.create("mov")->o("u32");
  auto *destOpr = builder.newOperand("=r");
  auto *sRegOpr = builder.newConstantOperand(sRegStr);
  mov(destOpr, sRegOpr);
  Value val = builder.launch(b, loc, b.getIntegerType(32), false);
  return val;
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
  Type globalPtrType =
      LLVM::LLVMPointerType::get(globalType, global.getAddrSpace());
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(
      UnknownLoc::get(ctx), globalPtrType, global.getSymName());
  Value stringStart =
      rewriter.create<LLVM::GEPOp>(UnknownLoc::get(ctx), ptr_ty(i8_ty),
                                   globalPtr, SmallVector<Value>({zero, zero}));
  return stringStart;
}

} // namespace LLVM

bool isF8(Type eType) {
  return eType.isFloat8E4M3FNUZ() or eType.isFloat8E4M3FN() or
         eType.isFloat8E5M2() or eType.isFloat8E5M2FNUZ();
}

} // namespace mlir
