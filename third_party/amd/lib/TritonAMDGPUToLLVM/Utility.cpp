#include "Utility.h"
#include "ConvertLayoutOpToLLVM/SharedToDotOperandHelper.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include <utility>

using mlir::triton::gpu::appendOrGetExternFuncOp;
using mlir::triton::gpu::getFunctionType;

namespace {

// Utility function to traverse a struct, get to the GEP contained
// in the struct at position `pos` and extract its base pointer and offset
mlir::FailureOr<std::pair<Value, Value>> getBaseAndOffset(Value ptr,
                                                          int64_t pos = 0) {
  // Typedef the return type here, to have more coincise code
  using ReturnType = mlir::FailureOr<std::pair<Value, Value>>;

  Operation *currentOp = ptr.getDefiningOp();
  auto res =
      llvm::TypeSwitch<Operation *, ReturnType>(currentOp)
          .Case<LLVM::GEPOp>([&](auto gepOp) -> ReturnType {
            SmallVector<Value> indices =
                llvm::to_vector(gepOp.getDynamicIndices());
            if (indices.size() == 1)
              return std::make_pair(gepOp.getBase(), indices[0]);
            return failure();
          })
          .Case<LLVM::AddrSpaceCastOp>([&](auto addrspaceCastOp) -> ReturnType {
            return getBaseAndOffset(addrspaceCastOp.getArg(), pos);
          })
          .Case<LLVM::ExtractValueOp>([&](auto extractValOp) -> ReturnType {
            ArrayRef<int64_t> position = extractValOp.getPosition();
            if (position.size() > 1)
              return failure();
            return getBaseAndOffset(extractValOp.getContainer(), position[0]);
          })
          .Case<LLVM::InsertValueOp>([&](auto insertValOp) -> ReturnType {
            ArrayRef<int64_t> position = insertValOp.getPosition();
            if (position.size() > 1)
              return failure();
            if (position[0] == pos)
              return getBaseAndOffset(insertValOp.getValue(), 0);
            return getBaseAndOffset(insertValOp.getContainer(), pos);
          })
          .Default([&](Operation *op) -> ReturnType { return failure(); });
  return res;
}

// Utility class to take care of buffer operation emission. We may add more
// emitters into this as needed.
struct BufferEmitter {

  BufferEmitter(RewriterBase &rw, Location loc, AMD::TargetInfo ti)
      : rewriter(rw), loc(loc), targetInfo(ti) {}

  // Emit a predicated rocdl.raw.ptr.buffer.load. `type` needs to be a
  // `VectorType`
  Value emitMaskedBufferLoad(Type type, Value basePtr, Value offset, Value pred,
                             Value falseVal, bool nt = false) {
    SmallVector<Value, 6> args;
    fillBufferArgs(type, basePtr, offset, pred, nt, args);
    Type bufferType = getBufferOpType(type);
    Value data = rewriter.create<ROCDL::RawPtrBufferLoadOp>(
        loc, bufferType, args, ArrayRef<NamedAttribute>());
    data = bitcast(data, type);
    return data;
  }

  // Emit a predicated rocdl.raw.ptr.buffer.store. `type` needs to be a
  // `VectorType`
  void emitMaskedBufferStore(Value data, Value basePtr, Value offset,
                             Value pred, bool nt = false) {
    // We only support vector types. So the caller needs to ensure we have a
    // vector type here
    VectorType vecTy = cast<VectorType>(data.getType());
    Type bufferType = getBufferOpType(vecTy);
    if (vecTy != bufferType)
      data = bitcast(data, bufferType);
    SmallVector<Value, 6> args{data};
    fillBufferArgs(vecTy, basePtr, offset, pred, nt, args);
    rewriter.create<ROCDL::RawPtrBufferStoreOp>(loc, TypeRange{}, args,
                                                ArrayRef<NamedAttribute>());
  }

private:
  // Given a type, the buffer type can be either the same type
  // or a packed version. E.g., a vector of 8xfp16 can be bitcasted to
  // a vector of 4xi32. This usually makes the life of the backend easier
  Type getBufferOpType(Type type) {
    int64_t vecSize = 1;
    Type elementType = type;
    if (auto vecType = dyn_cast<VectorType>(type)) {
      vecSize = vecType.getNumElements();
      elementType = vecType.getElementType();
    }

    const int valueElemNBits =
        std::max(8u, elementType.getIntOrFloatBitWidth());
    const size_t totalWidthBits = valueElemNBits * vecSize;

    // For bf16, always convert to i16
    Type bufferElementType = elementType;
    if (elementType.isBF16())
      bufferElementType = rewriter.getI16Type();

    // If we are dealing with a subword type (e.g., i8 or f16) but we
    // still need multiple words, then pack the subwords into 32bit integers
    // and update the vector length and the type
    int64_t bufferVecSize = vecSize;
    if (valueElemNBits < 32) {
      if (totalWidthBits > 32) {
        bufferElementType = rewriter.getI32Type();
        bufferVecSize = totalWidthBits / 32;
      } else {
        bufferElementType = rewriter.getIntegerType(totalWidthBits);
        bufferVecSize = 1;
      }
    }

    // This is the buffer type that the buffer operation will use. It
    // will be bitcast-able to the original type. So if the types
    // ended up different, we simply have to emit a `bitcastOp` to convert
    Type bufferType = type;
    if (bufferVecSize != vecSize)
      bufferType = VectorType::get(bufferVecSize, bufferElementType);
    if (bufferVecSize == 1)
      bufferType = getElementTypeOrSelf(bufferType);

    return bufferType;
  }

  // Fill common buffer operation arguments. A large part of this function is
  // courtesy of: mlir/lib/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp
  void fillBufferArgs(Type type, Value basePtr, Value offset, Value pred,
                      bool nt, SmallVector<Value> &args) {
    // 1. Create the resource descriptor
    // bits 0-11: dst sel, ignored by these intrinsics
    // bits 12-14: data format (ignored, must be nonzero, 7=float)
    // bits 15-18: data format (ignored, must be nonzero, 4=32bit)
    // bit 19: In nested heap (0 here)
    // bit 20: Behavior on unmap (0 means  "return 0 / ignore")
    // bits 21-22: Index stride for swizzles (N/A)
    // bit 23: Add thread ID (0)
    // bit 24: Reserved to 1 (RDNA) or 0 (CDNA)
    // bits 25-26: Reserved (0)
    // bit 27: Buffer is non-volatile (CDNA only)
    // bits 28-29: Out of bounds select (RDNA only)
    //             (0 = structured,
    //              1 = check index,
    //              2 = none,
    //              3 = either swizzles or testing against offset field)
    // bits 30-31: Type (must be 0)
    uint32_t flags = (7 << 12) | (4 << 15);
    if (targetInfo.getISAFamily() == AMD::ISAFamily::RDNA2 ||
        targetInfo.getISAFamily() == AMD::ISAFamily::RDNA3) {
      flags |= (1 << 24);
      uint32_t oob = 3;
      flags |= (oob << 28);
    }
    Value stride = int_val(16, 0);
    Value flagsConst = int_val(32, flags);
    Type rsrcType = LLVM::LLVMPointerType::get(rewriter.getContext(), 8);
    Value numRecordsByte = int_val(32, std::numeric_limits<int>::max() - 1);

    Value resource = rewriter.createOrFold<ROCDL::MakeBufferRsrcOp>(
        loc, rsrcType, basePtr, stride, numRecordsByte, flagsConst);

    // 2. Create the (masked) offset
    Type elementType = getElementTypeOrSelf(type);
    const int valueElemNBits =
        std::max(8u, elementType.getIntOrFloatBitWidth());
    const int elementByteWidth = valueElemNBits / 8;
    // Please note: the index passed to GEP is not in bytes, but in number of
    // elements In order to pass the index to the buffer operation, we need to
    // convert in bytes (i.e., we need to multiply by `elementByteWidth`)
    Value vOffsetOutOfBunds = int_val(
        32, static_cast<int>(std::numeric_limits<int>::max() + int64_t(1)));
    Value vRealOffset = mul(int_val(32, elementByteWidth), offset);
    Value maskedOffset = select(pred, vRealOffset, vOffsetOutOfBunds);

    // 3. Set the sgprOffset to 0
    Value sgprOffset = int_val(32, 0);

    // 4. Create the cache modifiers word
    // bit 0: GLC = 0 (atomics drop value, less coherency)
    // bits 1-2: SLC, DLC = 0 (similarly)
    // bit 3: swizzled (0 for raw)
    Value cacheModifiers = int_val(32, nt << 1);

    // 5. Add the arguments
    args.push_back(resource);
    args.push_back(maskedOffset);
    args.push_back(sgprOffset);
    args.push_back(cacheModifiers);
  }

  // Rewriter utilities
  RewriterBase &rewriter;
  Location loc;
  AMD::TargetInfo targetInfo;
};

enum class ShflKind : uint32_t {
  bfly = 0,
  up = 1,
  down = 2,
  idx = 3,
};

std::string getTypeString(Type ty) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  ty.print(rso);
  rso.flush();
  return str;
}

std::string mangleFunc(std::string name, Type type) {
  auto funcType = dyn_cast<LLVM::LLVMFunctionType>(type);
  assert(funcType && "Expecting an LLVMFunctionType");
  std::string mangled = name + "_";
  auto retTy = funcType.getReturnType();
  mangled += getTypeString(retTy) + "_";
  auto params = funcType.getParams();
  for (auto paramType : params) {
    mangled += getTypeString(paramType) + "_";
  }
  return mangled;
}

// Utility function to create a constant vector mask of length `vecSize` with
// the same `pred` value
Value createVectorMaskFromPredicate(RewriterBase &rewriter, Location loc,
                                    Value pred, int64_t vecSize) {
  auto vecMaskTy = LLVM::getFixedVectorType(rewriter.getI1Type(), vecSize);
  Value maskVal = undef(vecMaskTy);
  for (size_t s = 0; s < vecSize; ++s) {
    Value indexVal =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(s));
    maskVal = insert_element(vecMaskTy, maskVal, pred, indexVal);
  }
  return maskVal;
}

// Utility function to get the number of elements of a vector or a scalar
int64_t getNumElements(Type ty) {
  if (auto vecType = dyn_cast<VectorType>(ty))
    return vecType.getNumElements();
  return 1;
}

// Utility function to cast the given scalar or vector type to a vector type
Type castToVectorType(Type ty) {
  if (isa<VectorType>(ty))
    return ty;
  return LLVM::getFixedVectorType(ty, 1);
}

} // namespace

namespace mlir::LLVM::AMD {
static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value i, int strideInt, ShflKind mode, Value clamp) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  // On AMD, the ds_swizzle_b32 and ds_permute_b32 instructions work on
  // 32bit/dwords so we need promote to 32 here.
  auto valType = val.getType();
  if (!valType.isInteger(32) && bits <= 32) {
    if (!valType.isIntOrIndex())
      val = bitcast(val, int_ty(bits));
    if (bits < 32)
      val = sext(i32_ty, val);

    val = shuffleCommon(loc, rewriter, val, i, strideInt, mode, clamp);

    if (bits < 32)
      val = trunc(int_ty(bits), val);
    if (!valType.isIntOrIndex())
      val = bitcast(val, valType);
    return val;
  }

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = bitcast(val, vecTy);
    Value val0 = extract_element(f32_ty, vec, i32_val(0));
    Value val1 = extract_element(f32_ty, vec, i32_val(1));
    val0 = shuffleCommon(loc, rewriter, val0, i, strideInt, mode, clamp);
    val1 = shuffleCommon(loc, rewriter, val1, i, strideInt, mode, clamp);
    vec = undef(vecTy);
    vec = insert_element(vecTy, vec, val0, i32_val(0));
    vec = insert_element(vecTy, vec, val1, i32_val(1));
    return bitcast(vec, val.getType());
  }

  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  Value threadId =
      rewriter.create<::mlir::gpu::ThreadIdOp>(loc, ::mlir::gpu::Dimension::x);
  threadId = rewriter.create<arith::IndexCastOp>(loc, i32_ty, threadId);
  unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  Value warpSize = i32_val(iWarpSize);
  Value laneId = urem(threadId, warpSize);
  auto bpermute = [&](Value lane) {
    // Multiple lineId by 4. (More on permute instruction semantics:
    // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf#page=180
    Value byteOffset = i32_val(2);
    Value permuteAddr = shl(lane, byteOffset);
    return rewriter.create<ROCDL::DsBpermuteOp>(loc, valType, permuteAddr, val);
  };

  switch (mode) {
  case ShflKind::bfly:
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
  case ShflKind::up: {
    Value mask = icmp_slt(laneId, i);
    Value delta = sub(laneId, i);
    Value index = select(mask, laneId, delta);
    return bpermute(index);
  }
  case ShflKind::idx:
    return bpermute(i);
  default:
    assert(false && "Unsupported ShflKind");
    break;
  }
  return Value();
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), i, ShflKind::bfly,
                       i32_val(0x1f));
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), i, ShflKind::up,
                       i32_val(0x0));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleIdx(loc, rewriter, val, i32_val(i));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i) {
  return shuffleCommon(loc, rewriter, val, i, 0, ShflKind::idx, i32_val(0x1f));
}

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               int axis) {
  assert(axis >= 0);
  assert(axis < 3);
  assert(moduleOp);
  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
  Value blockId = rewriter.create<::mlir::gpu::BlockIdOp>(loc, dims[axis]);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, blockId);
}

Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal, triton::AMD::TargetInfo targetInfo,
             int64_t alignmentBytes, triton::CacheModifier cm,
             bool useBufferOps) {

  bool noCacheModifiers = (cm == triton::CacheModifier::NONE);
  // Use a predicated buffer load intrinsic if we can. This should be optimal,
  // since we don't have to emit any branch. Also, in this way the hardware
  // is automatically doing the pointer arithmetic, so we should save in VALU
  // arithmetic and registers.
  if (useBufferOps && noCacheModifiers) {
    auto maybeBaseAndOffset = getBaseAndOffset(ptr);
    if (!failed(maybeBaseAndOffset)) {
      BufferEmitter bufferEmitter(rewriter, loc, targetInfo);
      Value basePtr = maybeBaseAndOffset->first;
      Value offset = maybeBaseAndOffset->second;
      Value vecData = bufferEmitter.emitMaskedBufferLoad(
          elemTy, basePtr, offset, pred, falseVal);
      vecData = bitcast(vecData, elemTy);
      return vecData;
    }
  }

  // Alternatively, try to emit llvm.intr.masked.load if we can. In theory the
  // backend should be happier because we emit less branchy code to optimize.
  // The backend will lower it down however it wants at some point.
  bool nonTemporal = (cm == triton::CacheModifier::CG);
  if (alignmentBytes && (noCacheModifiers || nonTemporal)) {
    // `llvm.intr.masked.load` only accepts vectors. If we see a scalar we
    // need to bitcast to `vector<1xelemTy>` (and back)
    int64_t vecSize = getNumElements(elemTy);
    Type vecType = castToVectorType(elemTy);
    falseVal = bitcast(falseVal, vecType);
    Value maskVal = createVectorMaskFromPredicate(rewriter, loc, pred, vecSize);
    bool nt = (cm == triton::CacheModifier::CG);
    Value vecData = rewriter.create<LLVM::MaskedLoadOp>(
        loc, vecType, ptr, maskVal, falseVal, alignmentBytes, nonTemporal);
    // If it is not a vector, remember to bitcast back to a scalar
    vecData = bitcast(vecData, elemTy);
    return vecData;
  }

  // Default strategy: emit a branch in MLIR.
  Type funcType = getFunctionType(elemTy, ValueRange({ptr, pred, falseVal}));
  auto parent = ptr.getParentRegion()->getParentOfType<LLVM::LLVMFuncOp>();
  auto getLoadNameRaw = [](triton::CacheModifier cm) {
    switch (cm) {
    case triton::CacheModifier::CA:
      return predicatedLoadCA;
    case triton::CacheModifier::CG:
      return predicatedLoadCG;
    case triton::CacheModifier::CV:
      return predicatedLoadCV;
    default:
      // Do not fail in compile time in the case of unsupported modifier.
      // Just apply default config.
      return predicatedLoad;
    }
  };

  auto funcName = mangleFunc(getLoadNameRaw(cm), funcType);
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, parent, funcName, funcType);
  return LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                ValueRange({ptr, pred, falseVal}))
      .getResult();
}

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, triton::AMD::TargetInfo targetInfo,
             int64_t alignmentBytes, triton::CacheModifier cm,
             bool useBufferOps) {
  // Use a predicated buffer store intrinsic if we can. This should be optimal,
  // since we don't have to emit any branch, ever.
  if (useBufferOps && cm == triton::CacheModifier::NONE) {
    auto maybeBaseAndOffset = getBaseAndOffset(ptr);
    if (!failed(maybeBaseAndOffset)) {
      BufferEmitter bufferEmitter(rewriter, loc, targetInfo);
      Type elemTy = val.getType();
      int64_t vecSize = getNumElements(elemTy);
      Type vecType = castToVectorType(elemTy);
      Value basePtr = maybeBaseAndOffset->first;
      Value offset = maybeBaseAndOffset->second;
      val = bitcast(val, vecType);
      bufferEmitter.emitMaskedBufferStore(val, basePtr, offset, pred);
      return;
    }
  }

  // Alternatively, try to emit llvm.intr.masked.store if we can. In theory
  // the backend should be happier because we emit less branchy code to
  // optimize. The backend will lower it down however it wants at some point.
  if (alignmentBytes && cm == triton::CacheModifier::NONE) {
    // `llvm.intr.masked.store` only accepts vectors. If we see a scalar we need
    // to bitcast to `vector<1xelemTy>`
    Type elemTy = val.getType();
    int64_t vecSize = getNumElements(elemTy);
    Type vecType = castToVectorType(elemTy);
    val = bitcast(val, vecType);
    Value maskVal = createVectorMaskFromPredicate(rewriter, loc, pred, vecSize);
    auto op = rewriter.create<LLVM::MaskedStoreOp>(loc, val, ptr, maskVal,
                                                   alignmentBytes);
    return;
  }

  // Default strategy: emit a branch in MLIR.
  auto ctx = ptr.getContext();
  Type funcType = getFunctionType(void_ty(ctx), ValueRange({ptr, val, pred}));
  auto parent = ptr.getParentRegion()->getParentOfType<LLVM::LLVMFuncOp>();
  auto getStoreNameRaw = [](triton::CacheModifier cm) {
    switch (cm) {
    case triton::CacheModifier::WT:
      return predicatedStoreWT;
    case triton::CacheModifier::CG:
      return predicatedStoreCG;
    case triton::CacheModifier::CS:
      return predicatedStoreCS;
    default:
      // Do not fail in compile time in the case of unsupported modifier.
      // Just apply default config.
      return predicatedStore;
    }
  };
  auto funcName = mangleFunc(getStoreNameRaw(cm), funcType);
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, parent, funcName, funcType);
  LLVM::createLLVMCallOp(rewriter, loc, funcOp, ValueRange({ptr, val, pred}));
}

} // namespace mlir::LLVM::AMD
