#include "Utility.h"
#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
namespace tt = mlir::triton;
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
  auto vecMaskTy = LLVM::getVectorType(rewriter.getI1Type(), vecSize);
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
  return LLVM::getVectorType(ty, 1);
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
  Value threadId = getThreadId(rewriter, loc);

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
      Value stride = b.i32_val(32);
      Value lineId = b.xor_(threadId, stride);
      return bpermute(lineId);
    } else if (strideInt == 16) {
      if (isRDNA(isaFamily)) {
        // Lane i in the upper 16 lanes reads the value from lane i in the lower
        // 16 lanes and vice versa.
        Value select_lo = b.i32_val(0x76543210);
        Value select_hi = b.i32_val(0xfedcba98);
        return rewriter.create<ROCDL::PermlaneX16Op>(
            loc, valType, val, val, select_lo, select_hi, true, false);
      } else {
        Value offset = b.i32_val(0x401F);
        return rewriter.create<ROCDL::DsSwizzleOp>(loc, valType, val, offset);
      }
    } else {
      if (!llvm::is_contained({ISAFamily::CDNA2, ISAFamily::CDNA3,
                               ISAFamily::CDNA4, ISAFamily::RDNA3},
                              isaFamily)) {
        // DPP is only supported for CDNA2/CDNA3/CDNA4/RDNA3 right now, so we
        // fallback to ds_swizzle for other architectures.
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

Value permute(Location loc, RewriterBase &rewriter, Value x, Value y,
              Value selector) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value prmt_mask = selector;
  // convert from nybble mask to byte mask:
  prmt_mask =
      b.or_(b.and_(prmt_mask, b.i32_val(0x000000ff)),
            b.shl(b.and_(prmt_mask, b.i32_val(0x0000ff00)), b.i32_val(8)));
  prmt_mask =
      b.or_(b.and_(prmt_mask, b.i32_val(0x000f000f)),
            b.shl(b.and_(prmt_mask, b.i32_val(0x00f000f0)), b.i32_val(4)));
  Value args[] = {x, y, prmt_mask};
  auto op = createLLVMIntrinsicCallOp(rewriter, loc, "llvm.amdgcn.perm", i32_ty,
                                      args);
  return op.getResult(0);
}

// Utility function that returns flags <volatile, nontemporal> for a predicated
// Load or Store
// ---------------------------------
// Op   | cm  | volatile | NT
// -----+-----+---------------------
// Load | .ca |   F      | F
//      | .cg |   F      | T
//      | .cs |   F      | T
//      | .cv |   T      | X
// -----+-----+----------+---------
// Store| .wb |   F      | F
//      | .cg |   F      | F
//      | .cs |   F      | T
//      | .wt |   T      | X
// -----+-----+----------+---------
std::pair<bool, bool>
getCacheModifierFlagsForLoadStore(const triton::CacheModifier &cm,
                                  MemoryOp op) {
  switch (op) {
  case MemoryOp::Load: {
    switch (cm) {
    case triton::CacheModifier::CA:
      return std::make_pair(false, false);
    case triton::CacheModifier::CG:
      return std::make_pair(false, true);
    case triton::CacheModifier::CS:
      return std::make_pair(false, true);
    case triton::CacheModifier::CV:
      return std::make_pair(true, true);
    default:
      return std::make_pair(false, false);
    }
  }
  case MemoryOp::Store: {
    switch (cm) {
    case triton::CacheModifier::CG:
      return std::make_pair(false, false);
    case triton::CacheModifier::CS:
      return std::make_pair(false, true);
    case triton::CacheModifier::WT:
      return std::make_pair(true, true);
    default:
      return std::make_pair(false, false);
    }
  }
  }
  return std::make_pair(false, false);
}

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               ProgramIDDim axis) {
  Value blockId =
      rewriter.create<::mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension(axis));
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, blockId);
}

Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal, triton::CacheModifier cm,
             bool forceNoAliasAsyncLoads) {
  return rewriter
      .create<triton::amdgpu::MaskedLoadOp>(loc, elemTy, ptr, pred, falseVal,
                                            cm, forceNoAliasAsyncLoads)
      .getResult();
}

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, triton::CacheModifier cm,
             bool forceNoAliasAsyncLoads) {
  rewriter.create<triton::amdgpu::MaskedStoreOp>(loc, ptr, val, pred, cm,
                                                 forceNoAliasAsyncLoads);
}

// Create the auxiliary/cachepolicy value of ROCDL::RawPtrBufferLoad/StoreOp
//   gfx942 and gfx950: bit 0 = sc0, bit 1 = nt, bit 3 = swz, bit 4 = sc1
// Vector Memory instructions (Flat, Global, Scratch, and Buffer) have 3
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
//        | .wt |  1  |  1  | x  |
// -------+-----+-----+-----+----+--
// Atomic | N/A |  0  |  1  | x  | Setting sc0 returns the pre-op value
//        | N/A |  1  |  0  | x  | Setting sc1 performs a system-scope atomic
// -------+-----+-----+-----+----+--
static int32_t
getCtrlBitsForCacheModifierOnGFX_942_950(triton::CacheModifier cm,
                                         bool isLoad) {
  const int sc0Bit = 0b1, ntBit = 0b10, sc1Bit = 0b10000;
  int32_t aux = 0;
  switch (cm) {
  case triton::CacheModifier::CA:
    aux = 0;
    break;
  case triton::CacheModifier::CG:
    if (isLoad)
      aux |= sc0Bit | ntBit;
    break;
  case triton::CacheModifier::CS:
    aux |= sc0Bit | ntBit;
    break;
  case triton::CacheModifier::CV:
    assert(isLoad);
    aux |= sc0Bit | sc1Bit;
    break;
  case triton::CacheModifier::WB:
    assert(!isLoad);
    aux = 0;
    break;
  case triton::CacheModifier::WT:
    assert(!isLoad);
    aux |= sc0Bit | sc1Bit;
    break;
  default:
    aux = 0;
  }
  return aux;
}

int32_t getCtrlBitsForBufferAtomicsOnGFX_942_950(bool setSC0, bool setSC1,
                                                 bool setNT) {
  const int sc0Bit = 0b1, ntBit = 0b10, sc1Bit = 0b10000;
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
int32_t getCtrlBitsForCacheModifierOnTarget(
    triton::CacheModifier cm, bool isLoad,
    const mlir::triton::AMD::TargetInfo &targetInfo) {
  switch (targetInfo.getGPUKind()) {
  case llvm::AMDGPU::GK_GFX942:
  case llvm::AMDGPU::GK_GFX950:
    return getCtrlBitsForCacheModifierOnGFX_942_950(cm, isLoad);
  default:
    return getDefaultCtrlBitsForCacheModifier(cm);
  }
}

Value cvtFp32ToFp16RTNE_oneValue(Location loc, RewriterBase &rewriter,
                                 const Value &v) {
  LLVM::RoundingMode rm = LLVM::RoundingMode::NearestTiesToEven;
  return rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v);
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
  return axisAnalysisPass.getContiguity(ptr);
}

unsigned getContiguity(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass) {

  Type type = getPointerTypeWithShape(ptr, offset);
  RankedTensorType tensorTy = cast<RankedTensorType>(type);

  // To compute the contiguity of the scalar/warp-uniform ptr and offset pair we
  // need to look at the contiguity of the offsets and the alignment of the ptr
  auto elemNumBits = triton::getPointeeBitWidth(tensorTy);
  auto contiguity = axisAnalysisPass.getContiguity(offset, elemNumBits);

  // To get the alignment of the scalar ptr we need to look at the divisibility
  auto *axisInfo = axisAnalysisPass.getAxisInfo(ptr);
  auto maxMultipleBytes = axisInfo->getDivisibility(0);
  auto elemNumBytes = std::max<unsigned>(elemNumBits / 8, 1);
  auto align = std::max<unsigned>(maxMultipleBytes / elemNumBytes, 1);

  // FIXME (Alex): this should not be needed anymore because it's done inside
  // getContiguity, but we have an order issues with LL, so we keep this
  // until the LL order issue is fixed
  auto linearLayout = triton::gpu::toLinearLayout(tensorTy);
  auto llAttr =
      triton::gpu::LinearEncodingAttr::get(tensorTy.getContext(), linearLayout);
  auto order = triton::gpu::getOrder(tensorTy);
  auto contigPerThread = llAttr.getContigPerThread();
  assert(order[0] < contigPerThread.size() &&
         "Unexpected contigPerThread size");
  contiguity = std::min(contiguity, contigPerThread[order[0]]);

  // Final contiguity is a min of the offset contiguity and pointer alignment
  return std::min(align, contiguity);
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

Type scaleDotElemTypeToMLIRType(MLIRContext *ctx, triton::ScaleDotElemType t) {
  switch (t) {
  case triton::ScaleDotElemType::FP16:
    return Float16Type::get(ctx);
  case triton::ScaleDotElemType::BF16:
    return BFloat16Type::get(ctx);
  case triton::ScaleDotElemType::E4M3:
    return Float8E4M3FNType::get(ctx);
  case triton::ScaleDotElemType::E5M2:
    return Float8E5M2Type::get(ctx);
  case triton::ScaleDotElemType::E3M2:
    return Float6E3M2FNType::get(ctx);
  case triton::ScaleDotElemType::E2M3:
    return Float6E2M3FNType::get(ctx);
  case triton::ScaleDotElemType::E2M1:
    return Float4E2M1FNType::get(ctx);
  default:
    llvm_unreachable("unsupported ScaleDotElemType!");
  }
}

bool canCoalesceWriteIntoSharedMemory(RewriterBase &rewriter,
                                      const LinearLayout &srcToSharedLayout,
                                      unsigned threadsPerWarp) {
  auto contig = srcToSharedLayout.getNumConsecutiveInOut();

  StringAttr kLane = rewriter.getStringAttr("lane");
  for (int inLane : llvm::seq(srcToSharedLayout.getInDimSizeLog2(kLane))) {
    auto basis = srcToSharedLayout.getBasis(kLane, inLane)[0];
    unsigned expected = contig * (1 << inLane);
    if (basis != expected) {
      LDBG("detected uncoalesced layout from blocked to shared in async copy "
           "for lane "
           << 1 + inLane << "; given " << basis << " but expected "
           << expected);
      return false;
    }
  }
  // Additionally we could swizzle based on the warp dimension so we need to
  // check that when all bases are divided by contig, none of the first
  // (log2(warpSize) + 1) bits are set to 1
  assert(llvm::isPowerOf2_32(threadsPerWarp));
  assert(llvm::isPowerOf2_32(contig));
  unsigned mask = (threadsPerWarp * contig) - 1;
  StringAttr kWarp = rewriter.getStringAttr("warp");
  for (int inWarp : llvm::seq(srcToSharedLayout.getInDimSizeLog2(kWarp))) {
    auto basis = srcToSharedLayout.getBasis(kWarp, inWarp)[0];
    if ((basis & mask) != 0) {
      LDBG("detected uncoalesced layout from blocked to shared in async copy "
           "for warp "
           << inWarp);
      return false;
    }
  }

  return true;
}

bool doesSwizzleInsideWarp(RewriterBase &rewriter,
                           const LinearLayout &srcToSharedLayout,
                           unsigned threadsPerWarp) {
  auto contig = srcToSharedLayout.getNumConsecutiveInOut();
  // If all bases in lane dimension are below threadsPerWarp multiplied with the
  // contiguity we do not swizzle across warp boundaries.
  assert(llvm::isPowerOf2_32(threadsPerWarp));
  unsigned upperLimit = threadsPerWarp * contig;

  StringAttr kLane = rewriter.getStringAttr("lane");
  for (int inLane : llvm::seq(srcToSharedLayout.getInDimSizeLog2(kLane))) {
    auto basis = srcToSharedLayout.getBasis(kLane, inLane)[0];
    if (basis >= upperLimit) {
      return false;
    }
  }
  return true;
}

bool isUsedByDotScaledOp(Operation *op) {
  const ForwardSliceOptions fwdOpt;
  SetVector<mlir::Operation *> forwardSliceSet;
  getForwardSlice(op, &forwardSliceSet, fwdOpt);

  return std::any_of(
      forwardSliceSet.begin(), forwardSliceSet.end(), [](auto *operation) {
        return isa<triton::DotScaledOp, triton::amdgpu::UpcastMXFPOp>(
            operation);
      });
}

bool isChainDotHead(tt::DotOpInterface dotOp) {
  auto isInSameRegion = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  ForwardSliceOptions fwdOpt;
  fwdOpt.filter = isInSameRegion;
  SetVector<mlir::Operation *> fwdSlices;
  getForwardSlice(dotOp, &fwdSlices, fwdOpt);
  for (Operation *op : fwdSlices) {
    if (auto dOp = dyn_cast<tt::DotOpInterface>(op)) {
      assert(dOp != dotOp);
      auto opA = dOp.getA().getDefiningOp();
      if (opA && fwdSlices.contains(opA)) {
        return true;
      }
    }
  }
  return false;
}

bool hasTransInDefChain(tt::DotOpInterface dotOp, unsigned opIdx) {
  auto isInSameRegion = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };

  BackwardSliceOptions bwdOpt;
  bwdOpt.omitBlockArguments = true;
  bwdOpt.filter = isInSameRegion;
  SetVector<Operation *> bwdSlices;
  Operation *dotOperand = (opIdx == 0) ? dotOp.getA().getDefiningOp()
                                       : dotOp.getB().getDefiningOp();

  if (!dotOperand)
    return false;
  (void)getBackwardSlice(dotOperand, &bwdSlices, bwdOpt);
  if (llvm::find_if(bwdSlices, [](Operation *op) {
        return isa<tt::TransOp>(op);
      }) != bwdSlices.end())
    return true;
  return false;
}

bool isChainDotTail(tt::DotOpInterface dotOp) {
  auto isInSameRegion = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  BackwardSliceOptions bwdOpt;
  bwdOpt.omitBlockArguments = true;
  bwdOpt.filter = isInSameRegion;
  SetVector<Operation *> bwdSlices;
  Operation *opA = dotOp.getA().getDefiningOp();
  if (!opA)
    return false;
  (void)getBackwardSlice(opA, &bwdSlices, bwdOpt);
  if (llvm::find_if(bwdSlices, [](Operation *op) {
        return isa<tt::DotOpInterface>(op);
      }) != bwdSlices.end())
    return true;
  return false;
}

SmallVector<Value, 4> upcast8xMxfp4_SW(RewriterBase &rewriter, Operation *op,
                                       bool toFp16, Value packedVec) {
  assert((isa<triton::amdgpu::UpcastMXFPOp, triton::gpu::Fp4ToFpOp>(op)) &&
         "Expected UpcastMXFPOp or Fp4ToFpOp");
  Location loc = op->getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // MXFP4 has 4 bits, S.EE.M, for Sign, Exponent, and Mantissa respectively.
  // For a specific S, we have a total of 8 bit patterns. We can encode all
  // these 8 resultant bf16/fp16 bit patterns in a lookup table (LUT). It
  // happens that llvm.amdgcn.perm supports selecting 4 bytes from 8 input bytes
  // using a 4-byte selector. So the overall idea is to use llvm.amdgcn.perm to
  // implement such a LUT; though we need to select the two bytes for the
  // resultant bf16/fp16 bit patterns separately. For the byte containing S, we
  // also need to handle the S and E bits separately.

  // FP4 has 4 bits: S.EE.M. Bf16/fp16 bit patterns for positive values:
  //
  // FP4    | BF16   | FP16   | Value
  // ------ | ------ | ------ | -----
  // 0.00.0 | 0x0000 | 0x0000 | + 0.0
  // 0.00.1 | 0x3f00 | 0x3800 | + 0.5
  // 0.01.0 | 0x3f80 | 0x3c00 | + 1.0
  // 0.01.1 | 0x3fc0 | 0x3e00 | + 1.5
  // 0.10.0 | 0x4000 | 0x4000 | + 2.0
  // 0.10.1 | 0x4040 | 0x4200 | + 3.0
  // 0.11.0 | 0x4080 | 0x4400 | + 4.0
  // 0.11.1 | 0x40c0 | 0x4600 | + 6.0
  //
  // Encode Byte #0 (M) for BF16/FP16 in a LUT.
  Value resB0LutLo = toFp16 ? b.i32_val(0) : b.i32_val(0xc0800000);
  Value resB0LutHi = toFp16 ? b.i32_val(0) : b.i32_val(0xc0804000);
  // Encode Byte #1 (EM, non-S part) for BF16/FP16 in a LUT.
  Value resB1LutLoNoS = toFp16 ? b.i32_val(0x3e3c3800) : b.i32_val(0x3f3f3f00);
  Value resB1LutHiNoS = toFp16 ? b.i32_val(0x46444240) : b.i32_val(0x40404040);

  Type i32Ty = rewriter.getI32Type();
  auto permU32FnTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, i32Ty, i32Ty});
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, op, "llvm.amdgcn.perm", permU32FnTy);

  // Start with 8 mxfp4 elements in a single i32 register
  // | e7e6 | e5e4 | e3e2 | e1e0 |
  Value input = b.bitcast(packedVec, i32Ty);

  // Step 1: extract EM bits for elements 0,2,4,6 and 1,3,5,7 respectively.
  // e2m1_6420_idx = | 0[0e6EM] | 0[0e4EM] | 0[0e2EM] | 0[0e0EM] |
  Value e2m1_6420_idx = b.and_(input, b.i32_val(0x07070707));
  // e2m1_7531_idx = | [0e7EM]0 | [0e5EM]0 | [0e3EM]0 | [0e1EM]0 |
  Value e2m1_7531_idx = b.and_(input, b.i32_val(0x70707070));
  // e2m1_7531_idx = | 0[0e7EM] | 0[0e5EM] | 0[0e3EM] | 0[0e1EM] |
  e2m1_7531_idx = b.lshr(e2m1_7531_idx, b.i32_val(4));

  // Step 2: extract S bit for elements 0,2,4,6 and 1,3,5,7
  // s_6420 = | 0[e6S000] | 0[e4S000] | 0[e2S000] | 0[e0S000] |
  Value s_6420 = b.and_(input, b.i32_val(0x08080808));
  // s_6420 = | [e6S000]0 | [e4S000]0 | [e2S000]0 | [e0S000]0 |
  s_6420 = b.shl(s_6420, b.i32_val(4));
  // s_7531 = | [e7S000]0 | [e5S000]0 | [e3S000]0 | [e1S000]0 |
  Value s_7531 = b.and_(input, b.i32_val(0x80808080));

  // Step 3: Upcast elements 0,2,4,6 to 4 16-bit elements
  // Select Byte #0. It's always 0 if upcasting to fp16.
  // resB0_6420 = | e6B0 | e4B0 | e2B0 | e0B0 |
  Value resB0_6420 = b.i32_val(0);
  if (!toFp16) {
    resB0_6420 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {resB0LutHi, resB0LutLo, e2m1_6420_idx})
                     .getResult();
  }
  // Select Byte #1
  Value resB1NoS_6420 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1LutHiNoS, resB1LutLoNoS, e2m1_6420_idx})
          .getResult();
  // resB1_6420 = | e6B1 | e4B1 | e2B1 | e0B1 |
  Value resB1_6420 = b.or_(resB1NoS_6420, s_6420);
  // Construct 16-bit values of e0 and e2
  // res_20 = | e2B1 | e2B0 | e0B1 | e0B0 | = | e2_f16 | e0_f16 |
  Value res_20 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1_6420, resB0_6420, b.i32_val(0x05010400)})
          .getResult();
  // Construct 16-bit values of e4 and e6
  // res_64 = | e6B1 | e6B0 | e4B1 | e4B0 | = | e6_f16 | e4_f16 |
  Value res_64 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1_6420, resB0_6420, b.i32_val(0x07030602)})
          .getResult();

  // Step 4: Upcast elements 1,3,5,7 to 4 16-bit elements
  // This is a copy of step 3 on different group of elements
  // Select Byte #0. It's always 0 if upcasting to fp16.
  // resB0_7531 = | e7B0 | e5B0 | e3B0 | e1B0 |
  Value resB0_7531 = b.i32_val(0);
  if (!toFp16) {
    resB0_7531 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {resB0LutHi, resB0LutLo, e2m1_7531_idx})
                     .getResult();
  }
  // Select Byte #1
  Value resB1NoS_7531 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1LutHiNoS, resB1LutLoNoS, e2m1_7531_idx})
          .getResult();
  // resB1_7531 = | e7B1 | e5B1 | e3B1 | e1B1 |
  Value resB1_7531 = b.or_(resB1NoS_7531, s_7531);
  // Construct 16-bit values of e1 and e3
  // res_31 = | e3B1 | e3B0 | e1B1 | e1B0 | = | e3_f16 | e1_f16 |
  Value res_31 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1_7531, resB0_7531, b.i32_val(0x05010400)})
          .getResult();
  // Construct 16-bit values of e5 and e7
  // res_75 = | e7B1 | e7B0 | e5B1 | e5B0 | = | e7_f16 | e5_f16 |
  Value res_75 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1_7531, resB0_7531, b.i32_val(0x07030602)})
          .getResult();

  // Step 5: Reorder 16-bit elements to be 0,1,2,3,4,5,6,7
  // res_10 = | e1_f16 | e0_f16 |
  Value res_10 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {res_31, res_20, b.i32_val(0x05040100)})
                     .getResult();
  // res_32 = | e3_f16 | e2_f16 |
  Value res_32 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {res_31, res_20, b.i32_val(0x07060302)})
                     .getResult();
  // res_54 = | e5_f16 | e4_f16 |
  Value res_54 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {res_75, res_64, b.i32_val(0x05040100)})
                     .getResult();
  // res_76 = | e7_f16 | e6_f16 |
  Value res_76 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {res_75, res_64, b.i32_val(0x07060302)})
                     .getResult();

  return {res_10, res_32, res_54, res_76};
}

} // namespace mlir::LLVM::AMD
