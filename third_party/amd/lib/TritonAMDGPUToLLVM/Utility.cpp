#include "Utility.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using mlir::triton::ModuleAxisInfoAnalysis;
using mlir::triton::AMD::DppCtrl;
using mlir::triton::AMD::ISAFamily;
using mlir::triton::gpu::appendOrGetExternFuncOp;
using mlir::triton::gpu::getFunctionType;

namespace {
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
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto vecMaskTy = LLVM::getFixedVectorType(rewriter.getI1Type(), vecSize);
  Value maskVal = b.undef(vecMaskTy);
  for (size_t s = 0; s < vecSize; ++s) {
    Value indexVal =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(s));
    maskVal = b.insert_element(vecMaskTy, maskVal, pred, indexVal);
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
static Value shuffleCommonImpl(Location loc, RewriterBase &rewriter,
                               ISAFamily isaFamily, Value val, Value i,
                               int strideInt, ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  // On AMD, the ds_swizzle_b32 and ds_permute_b32 instructions work on
  // 32bit/dwords so we need promote to 32 here.
  auto valType = val.getType();
  if (!valType.isInteger(32) && bits <= 32) {
    if (!valType.isIntOrIndex())
      val = b.bitcast(val, int_ty(bits));
    if (bits < 32)
      val = b.sext(i32_ty, val);

    val = shuffleCommonImpl(loc, rewriter, isaFamily, val, i, strideInt, mode,
                            clamp);

    if (bits < 32)
      val = b.trunc(int_ty(bits), val);
    if (!valType.isIntOrIndex())
      val = b.bitcast(val, valType);
    return val;
  }

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = b.bitcast(val, vecTy);
    Value val0 = b.extract_element(f32_ty, vec, b.i32_val(0));
    Value val1 = b.extract_element(f32_ty, vec, b.i32_val(1));
    val0 = shuffleCommonImpl(loc, rewriter, isaFamily, val0, i, strideInt, mode,
                             clamp);
    val1 = shuffleCommonImpl(loc, rewriter, isaFamily, val1, i, strideInt, mode,
                             clamp);
    vec = b.undef(vecTy);
    vec = b.insert_element(vecTy, vec, val0, b.i32_val(0));
    vec = b.insert_element(vecTy, vec, val1, b.i32_val(1));
    return b.bitcast(vec, val.getType());
  }

  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  Value threadId =
      rewriter.create<::mlir::gpu::ThreadIdOp>(loc, ::mlir::gpu::Dimension::x);
  threadId = rewriter.create<arith::IndexCastOp>(loc, i32_ty, threadId);
  unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  Value warpSize = b.i32_val(iWarpSize);
  Value laneId = b.urem(threadId, warpSize);
  auto bpermute = [&](Value lane) {
    // Multiple lineId by 4. (More on permute instruction semantics:
    // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf#page=180
    Value byteOffset = b.i32_val(2);
    Value permuteAddr = b.shl(lane, byteOffset);
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
      Value stride = b.i32_val(32);
      Value lineId = b.xor_(threadId, stride);
      return bpermute(lineId);
    } else if (strideInt == 16) {
      Value offset = b.i32_val(0x401F);
      return rewriter.create<ROCDL::DsSwizzleOp>(loc, valType, val, offset);
    } else {
      if (isaFamily != ISAFamily::CDNA2 && isaFamily != ISAFamily::CDNA3) {
        // DPP is only supportted for CDNA2 and CDNA3 right now, so we fallback
        // to ds_swizzle for other archs.
        //
        // This map facilates the butterfly shuffle pattern for a stride less
        // than 16. The pattern stride is the key of the map.
        DenseMap<short, unsigned int> masks{
            {16, 0x401F}, {8, 0x201F}, {4, 0x101F}, {2, 0x081F}, {1, 0x041F}};
        Value offset = b.i32_val(masks[strideInt]);
        return rewriter.create<ROCDL::DsSwizzleOp>(loc, valType, val, offset);
      }

      auto createDppOpWithoutBoundCtrl = [&](Value &old, Value &src,
                                             uint32_t dppCtrl, uint32_t rowMask,
                                             uint32_t bankMask) {
        return rewriter.create<ROCDL::DPPUpdateOp>(
            loc, valType, old, src, rewriter.getI32IntegerAttr(dppCtrl),
            rewriter.getI32IntegerAttr(rowMask),
            rewriter.getI32IntegerAttr(bankMask), rewriter.getBoolAttr(false));
      };

      const int allRows = 0xf;
      const int allBanks = 0xf;

      switch (strideInt) {
      case 1: {
        // quad_perm: 1, 0, 3, 2
        uint32_t dppCtrl = static_cast<uint32_t>(DppCtrl::QUAD_PERM_FIRST);
        std::array<uint32_t, 4> mask = {1, 0, 3, 2};
        for (int i = 0; i < mask.size(); i++) {
          dppCtrl |= mask[i] << (i * 2);
        }
        return createDppOpWithoutBoundCtrl(val, val, dppCtrl, allRows,
                                           allBanks);
      }
      case 2: {
        // quad_perm: 2, 3, 0, 1
        uint32_t dppCtrl = static_cast<uint32_t>(DppCtrl::QUAD_PERM_FIRST);
        std::array<uint32_t, 4> mask = {2, 3, 0, 1};
        for (int i = 0; i < mask.size(); i++) {
          dppCtrl |= mask[i] << (i * 2);
        }
        return createDppOpWithoutBoundCtrl(val, val, dppCtrl, allRows,
                                           allBanks);
      }
      case 4: {
        // row_shr:4 bank_mask: 0xa
        auto ret = createDppOpWithoutBoundCtrl(
                       val, val, 4 + static_cast<uint32_t>(DppCtrl::ROW_SHR0),
                       allRows, 0xa)
                       .getRes();

        // row_shl:4 bank_mask: 0x5
        return createDppOpWithoutBoundCtrl(
            ret, val, 4 + static_cast<uint32_t>(DppCtrl::ROW_SHL0), allRows,
            0x5);
      }
      case 8: {
        // row_shr:8 bank_mask: 0xc
        auto ret = createDppOpWithoutBoundCtrl(
                       val, val, 8 + static_cast<uint32_t>(DppCtrl::ROW_SHR0),
                       allRows, 0xc)
                       .getRes();

        // row_shl:8 bank_mask: 0x3
        return createDppOpWithoutBoundCtrl(
            ret, val, 8 + static_cast<uint32_t>(DppCtrl::ROW_SHL0), allRows,
            0x3);
      }
      default:
        assert(false &&
               "bfly shfl with stride >= 16 should not be handled by dpp.");
      }
    }
    break;
  case ShflKind::up: {
    Value mask = b.icmp_slt(laneId, i);
    Value delta = b.sub(laneId, i);
    Value index = b.select(mask, laneId, delta);
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

static Value shuffleCommon(Location loc, RewriterBase &rewriter,
                           ISAFamily isaFamily, Value val, Value i,
                           int strideInt, ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // To shuffle pointers, convert them to i64.
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = b.ptrtoint(i64_ty, val);
  Value result = shuffleCommonImpl(loc, rewriter, isaFamily, val, i, strideInt,
                                   mode, clamp);
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = b.inttoptr(valTy, result);
  return result;
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i,
                 ISAFamily isaFamily) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, isaFamily, val, b.i32_val(i), i,
                       ShflKind::bfly, b.i32_val(0x1f));
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i,
                ISAFamily isaFamily) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, isaFamily, val, b.i32_val(i), i,
                       ShflKind::up, b.i32_val(0x0));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i,
                 ISAFamily isaFamily) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleIdx(loc, rewriter, val, b.i32_val(i), isaFamily);
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i,
                 ISAFamily isaFamily) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, isaFamily, val, i, 0, ShflKind::idx,
                       b.i32_val(0x1f));
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
             Value pred, Value falseVal, int64_t alignmentBytes,
             triton::CacheModifier cm) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // Try to emit llvm.intr.masked.load if we can. In theory the backend should
  // be happier because we emit less branchy code to optimize. The backend will
  // lower it down however it wants at some point.
  if (alignmentBytes &&
      (cm == triton::CacheModifier::CG || cm == triton::CacheModifier::NONE)) {
    // `llvm.intr.masked.load` only accepts vectors. If we see a scalar we need
    // to bitcast to `vector<1xelemTy>` (and back)
    int64_t vecSize = getNumElements(elemTy);
    Type vecType = castToVectorType(elemTy);
    falseVal = b.bitcast(falseVal, vecType);
    Value maskVal = createVectorMaskFromPredicate(rewriter, loc, pred, vecSize);
    bool nt = (cm == triton::CacheModifier::CG);
    Value vecData = rewriter.create<LLVM::MaskedLoadOp>(
        loc, vecType, ptr, maskVal, falseVal, alignmentBytes, nt);
    // If it is not a vector, remember to bitcast back to a scalar
    vecData = b.bitcast(vecData, elemTy);
    return vecData;
  }

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
             Value pred, int64_t alignmentBytes, triton::CacheModifier cm) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // Try to emit llvm.intr.masked.store if we can. In theory the backend should
  // be happier because we emit less branchy code to optimize. The backend will
  // lower it down however it wants at some point.
  if (alignmentBytes && cm == triton::CacheModifier::NONE) {
    // `llvm.intr.masked.store` only accepts vectors. If we see a scalar we need
    // to bitcast to `vector<1xelemTy>`
    Type elemTy = val.getType();
    int64_t vecSize = getNumElements(elemTy);
    Type vecType = castToVectorType(elemTy);
    val = b.bitcast(val, vecType);
    Value maskVal = createVectorMaskFromPredicate(rewriter, loc, pred, vecSize);
    auto op = rewriter.create<LLVM::MaskedStoreOp>(loc, val, ptr, maskVal,
                                                   alignmentBytes);
    return;
  }

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

static bool isPredicatedLoadCA(LLVM::CallOp callOp) {
  return callOp.getCallee().value().contains(mlir::LLVM::AMD::predicatedLoadCA);
}

static bool isPredicatedLoadCG(LLVM::CallOp callOp) {
  return callOp.getCallee().value().contains(mlir::LLVM::AMD::predicatedLoadCG);
}

static bool isPredicatedLoadCV(LLVM::CallOp callOp) {
  return callOp.getCallee().value().contains(mlir::LLVM::AMD::predicatedLoadCV);
}

static bool isPredicatedStoreCS(LLVM::CallOp callOp) {
  return callOp.getCallee().value().contains(
      mlir::LLVM::AMD::predicatedStoreCS);
}

static bool isPredicatedStoreCG(LLVM::CallOp callOp) {
  return callOp.getCallee().value().contains(
      mlir::LLVM::AMD::predicatedStoreCG);
}

static bool isPredicatedStoreWT(LLVM::CallOp callOp) {
  return callOp.getCallee().value().contains(
      mlir::LLVM::AMD::predicatedStoreWT);
}

// Utility function that returns flags <volatile, nontemporal> for a predicated
// Load or Store
// ---------------------------------
// Op   | cm  | volatile | NT
// -----+-----+---------------------
// Load | .ca |   F      | F
//      | .cg |   F      | T
//      | .cs |   F      | T
//      | .cv |   T      | T
// -----+-----+----------+---------
// Store| .wb |   F      | F
//      | .cg |   F      | F
//      | .cs |   F      | T
//      | .wt |   T      | T
// -----+-----+----------+---------
std::pair<bool, bool>
getCacheModifierFlagsForPredicatedCall(LLVM::CallOp callOp) {
  if (isPredicatedLoadCA(callOp))
    return std::make_pair(false, false);
  if (isPredicatedLoadCG(callOp))
    return std::make_pair(false, true);
  if (isPredicatedLoadCV(callOp))
    return std::make_pair(true, true);

  if (isPredicatedStoreCG(callOp))
    return std::make_pair(false, false);
  if (isPredicatedStoreCS(callOp))
    return std::make_pair(false, true);
  if (isPredicatedStoreWT(callOp))
    return std::make_pair(true, true);
  // unsupported modifier
  return std::make_pair(false, false);
}

// Create the auxiliary/cachepolicy value of ROCDL::RawPtrBufferLoad/StoreOp
//   gfx942: bit 0 = sc0, bit 1 = nt, bit 3 = swz, bit 4 = sc1
// GFX942 Vector Memory instructions (Flat, Global, Scratch, and Buffer) have 3
// bits to control scope and cacheability:
// - SC[1:0] System Cache level: 0=wave, 1=group, 2=device, 3=system
// - NT Non-Temporal: 0=expect temporal reuse; 1=do not expect temporal reuse
//
// -------+-----+-----+-----+----+--
// Op     | cm  | SC1 | SC0 | NT |
// -------+-----+-----+-----+----+--
// Load   | .ca |  0  |  0  | 0  |
//        | .cg |  0  |  1  | 1  |
//        | .cs |  0  |  1  | 1  |
//        | .cv |  1  |  1  | x  |
// -------+-----+-----+-----+----+--
// Store  | .wb |  0  |  0  | 0  |
//        | .cg |  0  |  0  | 0  |
//        | .cs |  0  |  1  | 1  |
//        | .wt |  1  |  x  | x  |
// -------+-----+-----+-----+----+--
// Atomic | N/A |  0  |  1  | x  | Setting sc0 returns the pre-op value
//        | N/A |  1  |  0  | x  | Setting sc1 performs a system-scope atomic
// -------+-----+-----+-----+----+--
static int32_t getCtrlBitsForCacheModifierOnGFX942(triton::CacheModifier cm,
                                                   bool isBufferLoad) {
  const int sc0Bit = 0b1, ntBit = 0b10, sc1Bit = 0b1000;
  int32_t aux = 0;
  switch (cm) {
  case triton::CacheModifier::CA:
    aux = 0;
    break;
  case triton::CacheModifier::CG:
    if (isBufferLoad)
      aux |= sc0Bit | ntBit;
    break;
  case triton::CacheModifier::CS:
    aux |= sc0Bit | ntBit;
    break;
  case triton::CacheModifier::CV:
    aux |= sc0Bit | sc1Bit;
    break;
  case triton::CacheModifier::WB:
    aux = 0;
    break;
  case triton::CacheModifier::WT:
    aux |= sc1Bit;
    break;
  default:
    aux = 0;
  }
  return aux;
}

int32_t getCtrlBitsForBufferAtomicsOnGFX942(bool setSC0, bool setSC1,
                                            bool setNT) {
  const int sc0Bit = 0b1, ntBit = 0b10, sc1Bit = 0b1000;
  int32_t aux = 0;
  if (setSC0)
    aux |= sc0Bit;
  if (setSC1)
    aux |= sc1Bit;
  if (setNT)
    aux |= ntBit;
  return aux;
}

static int32_t getDefaultCtrlBitsForCacheModifier(triton::CacheModifier cm) {
  return 0;
}

// Cache modifiers changes how data is managed in the GPU's cache hierarchy:
// .ca: cache at all levels with LRU policy
// .cg: cache at L2, can use .ca or .cs
// .cs: cache streaming, use data once
// .cv: don't cache and fetch again
// .wb: write-back, writes back data at all cache levels
// .wt: write-through, write data directly to system memory
int32_t
getCtrlBitsForCacheModifierOnTarget(triton::CacheModifier cm, bool isBufferLoad,
                                    mlir::triton::AMD::TargetInfo &targetInfo) {
  if (targetInfo.getGPUKind() == llvm::AMDGPU::GK_GFX942) // gfx942
    return getCtrlBitsForCacheModifierOnGFX942(cm, isBufferLoad);
  else
    return getDefaultCtrlBitsForCacheModifier(cm);
}

Value cvtFp32ToFp16(Location loc, RewriterBase &rewriter, const Value &v,
                    triton::RoundingMode rounding) {
  GCNBuilder builder;

  auto &cvt = *builder.create("v_cvt_f16_f32");
  auto res = builder.newOperand("=v");
  auto operand = builder.newOperand(v, "v");
  if (rounding == triton::RoundingMode::RTZ) {
    auto &setRTZ = *builder.create("s_setreg_imm32_b32 0x1801, 0xc");
    setRTZ();
  }
  cvt(res, operand);
  if (rounding == triton::RoundingMode::RTZ) {
    auto &resetRTZ = *builder.create("s_setreg_imm32_b32 0x1801, 0x0");
    resetRTZ();
  }
  return builder.launch(rewriter, loc, f16_ty, false);
}

Type getPointerTypeWithShape(Value basePtr, Value offset) {
  Type basePtrType = basePtr.getType();
  auto offsetType = cast<RankedTensorType>(offset.getType());
  return offsetType.cloneWith(std::nullopt, basePtrType);
}

unsigned getContiguity(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy)
    return 1;
  return axisAnalysisPass.getPtrContiguity(ptr);
}

unsigned getContiguity(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass) {
  // Get contiguity from the offset
  Type type = getPointerTypeWithShape(ptr, offset);
  RankedTensorType tensorTy = cast<RankedTensorType>(type);
  auto layout = tensorTy.getEncoding();
  auto order = triton::gpu::getOrder(layout);
  auto uniqueContigPerThread =
      triton::gpu::getUniqueContigPerThread(layout, tensorTy.getShape());
  assert(order[0] < uniqueContigPerThread.size() &&
         "Unexpected uniqueContigPerThread size");
  unsigned contiguity = uniqueContigPerThread[order[0]];

  // Get alignment from the pointer. Since this is a scalar pointer
  // we should not take the pointer contiguity to consider alignment
  auto *axisInfo = axisAnalysisPass.getAxisInfo(ptr);
  auto maxMultipleBytes = axisInfo->getDivisibility(0);
  auto elemNumBits = triton::getPointeeBitWidth(tensorTy);
  auto elemNumBytes = std::max<unsigned>(elemNumBits / 8, 1);
  auto align = std::max<int64_t>(maxMultipleBytes / elemNumBytes, 1);

  // Final contiguity is a min of the offset contiguity and pointer alignment
  contiguity = std::min<int64_t>(align, contiguity);
  return contiguity;
}

unsigned getVectorSize(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy)
    return 1;
  auto contiguity = getContiguity(ptr, axisAnalysisPass);
  auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
  return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
}

unsigned getVectorSize(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass) {
  auto contiguity = getContiguity(ptr, offset, axisAnalysisPass);
  auto pointeeBitWidth = triton::getPointeeBitWidth(ptr.getType());
  return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
}

} // namespace mlir::LLVM::AMD
