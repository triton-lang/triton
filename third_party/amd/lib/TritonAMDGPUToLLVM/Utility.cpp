#include "Utility.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

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
static Value shuffleCommon(Location loc, RewriterBase &rewriter,
                           ISAFamily isaFamily, Value val, Value i,
                           int strideInt, ShflKind mode, Value clamp) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  // On AMD, the ds_swizzle_b32 and ds_permute_b32 instructions work on
  // 32bit/dwords so we need promote to 32 here.
  auto valType = val.getType();
  if (!valType.isInteger(32) && bits <= 32) {
    if (!valType.isIntOrIndex())
      val = bitcast(val, int_ty(bits));
    if (bits < 32)
      val = sext(i32_ty, val);

    val =
        shuffleCommon(loc, rewriter, isaFamily, val, i, strideInt, mode, clamp);

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
    val0 = shuffleCommon(loc, rewriter, isaFamily, val0, i, strideInt, mode,
                         clamp);
    val1 = shuffleCommon(loc, rewriter, isaFamily, val1, i, strideInt, mode,
                         clamp);
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
    } else if (strideInt == 16) {
      Value offset = i32_val(0x401F);
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
        Value offset = i32_val(masks[strideInt]);
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

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i,
                 ISAFamily isaFamily) {
  return shuffleCommon(loc, rewriter, isaFamily, val, i32_val(i), i,
                       ShflKind::bfly, i32_val(0x1f));
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i,
                ISAFamily isaFamily) {
  return shuffleCommon(loc, rewriter, isaFamily, val, i32_val(i), i,
                       ShflKind::up, i32_val(0x0));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i,
                 ISAFamily isaFamily) {
  return shuffleIdx(loc, rewriter, val, i32_val(i), isaFamily);
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i,
                 ISAFamily isaFamily) {
  return shuffleCommon(loc, rewriter, isaFamily, val, i, 0, ShflKind::idx,
                       i32_val(0x1f));
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

  // Try to emit llvm.intr.masked.load if we can. In theory the backend should
  // be happier because we emit less branchy code to optimize. The backend will
  // lower it down however it wants at some point.
  if (alignmentBytes &&
      (cm == triton::CacheModifier::CG || cm == triton::CacheModifier::NONE)) {
    // `llvm.intr.masked.load` only accepts vectors. If we see a scalar we need
    // to bitcast to `vector<1xelemTy>` (and back)
    int64_t vecSize = getNumElements(elemTy);
    Type vecType = castToVectorType(elemTy);
    falseVal = bitcast(falseVal, vecType);
    Value maskVal = createVectorMaskFromPredicate(rewriter, loc, pred, vecSize);
    bool nt = (cm == triton::CacheModifier::CG);
    Value vecData = rewriter.create<LLVM::MaskedLoadOp>(
        loc, vecType, ptr, maskVal, falseVal, alignmentBytes, nt);
    // If it is not a vector, remember to bitcast back to a scalar
    vecData = bitcast(vecData, elemTy);
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
  // Try to emit llvm.intr.masked.store if we can. In theory the backend should
  // be happier because we emit less branchy code to optimize. The backend will
  // lower it down however it wants at some point.
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

namespace mlir::triton::AMD {
SmallVector<scf::ForOp> getLeafForOps(triton::FuncOp funcOp) {
  SmallVector<scf::ForOp> allOps;
  funcOp->walk([&](scf::ForOp forOp) { allOps.push_back(forOp); });

  SmallVector<scf::ForOp> leafOps;
  for (scf::ForOp forOp : allOps) {
    auto searchResult = forOp.getBody()->walk(
        [](scf::ForOp) { return WalkResult::interrupt(); });
    if (!searchResult.wasInterrupted())
      leafOps.push_back(forOp);
  }
  return leafOps;
}
} // namespace mlir::triton::AMD
