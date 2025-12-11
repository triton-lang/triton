#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_UTILITY_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_UTILITY_H_

#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::LLVM::AMD {

enum class MemoryOp { Load, Store };

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i,
                 mlir::triton::AMD::ISAFamily isaFamily =
                     mlir::triton::AMD::ISAFamily::Unknown);
Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i,
                mlir::triton::AMD::ISAFamily isaFamily =
                    mlir::triton::AMD::ISAFamily::Unknown);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i,
                 mlir::triton::AMD::ISAFamily isaFamily =
                     mlir::triton::AMD::ISAFamily::Unknown);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i,
                 mlir::triton::AMD::ISAFamily isaFamily =
                     mlir::triton::AMD::ISAFamily::Unknown);

Value permute(Location loc, RewriterBase &rewriter, Value a, Value b,
              Value selector);

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               ProgramIDDim axis);

// Emit the cta multicast mask for a given cta id based on the src layout
Value emitCtaMulticastMask(RewriterBase &rewriter, Location loc, Value blockId,
                           const LinearLayout &cvt);

std::pair<bool, bool>
getCacheModifierFlagsForLoadStore(const triton::CacheModifier &cm, MemoryOp op);

// Loads from shared or global memory with predication.
// `otherElems` is used to mask out the elements that are not loaded
// forceNoAliasAsyncLoads=true adds alias information to the llvm.load to
// signal its not aliasing with any AsyncCopyGlobalToLocal/BufferLoadToLocal to
// avoid conservative waits. See `addLocalLoadNoAliasScope` for more details
Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal, Value multicastMask,
             triton::CacheModifier cm = triton::CacheModifier::NONE,
             bool forceNoAliasAsyncLoads = false);

// Stores to shared or global memory with predication.
// forceNoAliasAsyncLoads=true adds alias information to the llvm.store to
// signal its not aliasing with any AsyncCopyGlobalToLocal/BufferLoadToLocal to
// avoid conservative waits. See `addLocalLoadNoAliasScope` for more details
void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, triton::CacheModifier cm = triton::CacheModifier::NONE,
             bool forceNoAliasAsyncLoads = false);

// Get cache modifier information for creating load or store instruction
// Get flags <volatile, nontemporal> for a predicated Load or Store
std::pair<bool, bool> getCacheModifierFlagsForLoadStore(LLVM::CallOp);
// Get the cachepolicy value for a cache modifier
int32_t
getCtrlBitsForCacheModifierOnTarget(triton::CacheModifier, bool,
                                    const mlir::triton::AMD::TargetInfo &);

// Get cache modifier information for buffer atomics
int32_t getCtrlBitsForBufferAtomicsOnGFX_942_950(bool setSC0, bool setSC1,
                                                 bool setNT);

Value cvtFp32ToFp16RTNE_oneValue(Location loc, RewriterBase &rewriter,
                                 const Value &v);

// Return a tensor of pointers with the same type of `basePtr` and the same
// shape of `offset`
Type getPointerTypeWithShape(Value basePtr, Value offset);

// Get contiguity for a tensor pointer `ptr`
unsigned getContiguity(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass);

// Get contiguity for a scalar pointer `ptr` and a tensor `offset`
unsigned getContiguity(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass);

// Determine the vector size of a tensor of pointers
unsigned getVectorSize(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass);

// Given a scalar pointer and a tensor of offsets, determine the vector size
unsigned getVectorSize(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass);

Type scaleDotElemTypeToMLIRType(MLIRContext *ctx, triton::ScaleDotElemType t);

// Returns true if we can perform coalesced write from the source encoding to
// the destination encoding for a given vec size.
bool canCoalesceWriteIntoSharedMemory(RewriterBase &rewriter,
                                      const LinearLayout &srcToSharedLayout,
                                      unsigned threadsPerWarp,
                                      unsigned vecSize);

// Returns true if the swizzling pattern does only swizzle the shared memory
// offsets of a warp and does not exchange destination elements across warps
bool doesSwizzleInsideWarp(RewriterBase &rewriter,
                           const LinearLayout &srcToSharedLayout,
                           unsigned threadsPerWarp);

// Return true if op is used by DotScaledOp or UpcastMXFPOp ops.
bool isUsedByDotScaledOp(Operation *op);

// Check if the result of this tl.dot is used as opA or opB of another tl.dot
// in the same region
bool isChainDotHead(mlir::triton::DotOpInterface dotOp, unsigned opIdx = 0);

// Check if the opA of this tl.dot is the result of another tl.dot
// in the same region
bool isChainDotTail(mlir::triton::DotOpInterface dotOp);

// Software implementation of converting an 8-element vector of MXFP4 elements
// to a wider type: BF16 or FP16 for target before CDNA4.
// for CDNA3, we have optimized sequence that can combine scale during the
// conversion
SmallVector<Value> upcast8xMxfp4_SW(RewriterBase &rewriter, Operation *op,
                                    bool toFp16, Value packedVec,
                                    mlir::triton::AMD::ISAFamily isaFamily,
                                    Value scale = nullptr);

template <typename ConvertOp>
SmallVector<Value, 4>
upcast8xMxfp4_HW(RewriterBase &rewriter, Location loc, ArrayRef<Value> xVals,
                 int idx, Value scale, bool useShiftedScale = false) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value packedVec = b.undef(vec_ty(i8_ty, 4));
  for (int i : llvm::seq(4))
    packedVec = b.insert_element(packedVec, xVals[idx + i], b.i32_val(i));
  packedVec = b.bitcast(packedVec, i32_ty);
  Type retElemType = bf16_ty;
  if constexpr (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF16Fp4Op>)
    retElemType = f16_ty;
  Type resType = vec_ty(retElemType, 2);
  // In the DotScaledOp decomposition, the scale has already been left-shifted
  // by 7 to fit the exponent of bf16. So now we only need to further left-shift
  // it by 16
  Value scaleF32;
  if (useShiftedScale) {
    scaleF32 = b.bitcast(
        b.shl(b.zext(i32_ty, b.bitcast(scale, i16_ty)), b.i32_val(16)), f32_ty);
  } else {
    scaleF32 = b.bitcast(b.shl(b.zext(i32_ty, scale), b.i32_val(23)), f32_ty);
  }
  SmallVector<Value, 4> results;
  for (int srcSelIndex : llvm::seq(4))
    results.push_back(ConvertOp::create(rewriter, loc, resType, packedVec,
                                        scaleF32, srcSelIndex));
  return results;
}

template <typename ConvertOp>
SmallVector<Value, 2>
upcast4xMxfp8_HW(RewriterBase &rewriter, Location loc, ArrayRef<Value> xVals,
                 int idx, Value scale, bool useShiftedScale = false) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value packedVec = b.undef(vec_ty(i8_ty, 4));
  for (int i : llvm::seq(4))
    packedVec = b.insert_element(packedVec, xVals[idx + i], b.i32_val(i));
  packedVec = b.bitcast(packedVec, i32_ty);
  Type retElemType = bf16_ty;
  if constexpr (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF16Fp8Op> ||
                std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF16Bf8Op>)
    retElemType = f16_ty;
  Type resType = vec_ty(retElemType, 2);
  // In the DotScaledOp decomposition, the scale has already been left-shifted
  // by 7 to fit the exponent of bf16. So now we only need to further left-shift
  // it by 16
  Value scaleF32;
  if (useShiftedScale) {
    scaleF32 = b.bitcast(
        b.shl(b.zext(i32_ty, b.bitcast(scale, i16_ty)), b.i32_val(16)), f32_ty);
  } else {
    scaleF32 = b.bitcast(b.shl(b.zext(i32_ty, scale), b.i32_val(23)), f32_ty);
  }
  SmallVector<Value, 2> results;
  results.push_back(ConvertOp::create(rewriter, loc, resType, packedVec,
                                      scaleF32,
                                      /*srcLoHiSel=*/false));
  results.push_back(ConvertOp::create(rewriter, loc, resType, packedVec,
                                      scaleF32,
                                      /*srcLoHiSel=*/true));
  return results;
}

// 1) for the parameter `inputVals`
// The fp8 tensor `inputVals` is upcasted to a [b]f16 tensor in the same shape,
// as an operand of 16x16x32_[b]f16 WMMA instruction and the layout is:
// clang-format off
//
// --------------------------------------------------------------------------------------------------------------
// \Row    0,1   2,3   4,5   6,7  |  8,9  10,11  12,13 14,15 | 16,17 18,19 20,21 22,23 | 24,25 26,27  28,29 30,31
// \__
// Col                            |                          |                         |
// 0      t0r0  t0r1  t0r2  t0r3  | t16r0 t16r1  t16r2 t16r3 | t0r4  t0r5  t0r6  t0r7  | t16r4 t16r5  t16r6 t16r7
// 1      t1r0  t1r1  t1r2  t1r3  | t17r0 t17r1  t17r2 t17r3 | t1r4  t1r5  t1r6  t1r7  | t17r4 t17r5  t17r6 t17r7
// ...                            |                           ...... .....
// 15     t15r0 t15r1 t15r2 t15r3 | t31r0 t31r1  t31r2 t31r3 | t15r4 t15r5 t15r6 t15r7 | t31r4 t31r5  t31r6 t31r7
// --------------------------------------------------------------------------------------------------------------
//
// clang-format on

// The points here are:
// Lane and lane+16 co-hold one row
// Input tensor of upcast `inputVals` is with same layout yet element type is
// fp8;
//
// 2) for the parameter `scales`
//   For scale tensor, e.g. if input shape is (32, 4) and block mode is 32,
// it is already transformed via `reshape(broadcast_to(expand_dims(a_scale, 2),
// (32, 4, 32)), (32, 128))` and output layout in the wave is `register = [[0,
// 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[0, 32], [0, 64], [1, 0], [2,
// 0], [4, 0]]` which means every lane will hold continous 32 elements and these
// 32 elements share one scale since the block mode is 32.
//
// 3) for `opSel` used in the rocdl.cvt.scale.pk8
//
// From the SP guide, the `opSel` is defined as:
//
// OPSEL[0:2]  |  Lane0..15 of SRC0         | Lane16..31 of SRC0
// -----------------------------------------------------------
// 000         |  Lane0..15 of Vscale[7:0]  | <-- same
//
// which means if OPSEL is zero, hardware requires every lane and lane+16 share
// the same scale. In the meantime, as comments for parameter `inputVals`,
// `lane` and `lane+16` hold one row of input tile,
//
// In the end, `opSel` is zero.
template <typename ConvertOp>
SmallVector<Value, 8> upcast8xMxfp8fp4_HW(RewriterBase &rewriter, Location loc,
                                          ArrayRef<Value> inputVals, int idx,
                                          ArrayRef<Value> scales) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  bool toFp16 = (std::is_same_v<ConvertOp, ROCDL::CvtPkScalePk8F16Fp8Op> ||
                 std::is_same_v<ConvertOp, ROCDL::CvtPkScalePk8F16Bf8Op> ||
                 std::is_same_v<ConvertOp, ROCDL::CvtPkScalePk8F16Fp4Op>);
  Type resElemType = toFp16 ? f16_ty : bf16_ty;
  Type resType = vec_ty(resElemType, 8);

  bool fromFP4 = (std::is_same_v<ConvertOp, ROCDL::CvtPkScalePk8F16Fp4Op> ||
                  std::is_same_v<ConvertOp, ROCDL::CvtPkScalePk8Bf16Fp4Op>);

  auto packedSize = fromFP4 ? 4 : 8;
  Value packedVec = b.undef(vec_ty(i8_ty, packedSize));
  for (int ii : llvm::seq(packedSize))
    packedVec = b.insert_element(packedVec, inputVals[idx + ii], b.i32_val(ii));
  packedVec =
      fromFP4 ? b.bitcast(packedVec, i32_ty)
              : b.bitcast(packedVec, vec_ty(i32_ty, packedSize / sizeof(int)));

  Value packedScale = b.undef(vec_ty(i8_ty, 4));
  auto scaleIdx = fromFP4 ? (idx + idx) : idx;
  for (int ii : llvm::seq(4))
    packedScale =
        b.insert_element(packedScale, scales[scaleIdx], b.i32_val(ii));
  Value scaleInt32 = b.bitcast(packedScale, i32_ty);
  auto res = ConvertOp::create(rewriter, loc, resType, packedVec, scaleInt32,
                               /*opSel*/ 0)
                 .getRes();
  Value elements = b.bitcast(res, vec_ty(resElemType, 8));

  SmallVector<Value, 8> results;
  for (auto ii : llvm::seq(8)) {
    results.push_back(b.extract_element(elements, b.i32_val(ii)));
  }

  return results;
}
} // namespace mlir::LLVM::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_UTILITY_H_
