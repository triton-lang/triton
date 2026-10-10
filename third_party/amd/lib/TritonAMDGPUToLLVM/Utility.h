#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_UTILITY_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_UTILITY_H_

#include "TargetInfo.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include <cstdint>

namespace mlir::LLVM::AMD {

// Decode the target-independent compatibility payload accepted by Triton
// memory operations. Target-specific cache policy attributes are rejected.
FailureOr<triton::CacheModifier> getCacheModifier(Attribute cachePolicy);

// Here is a partial definition of DppCtrl enums. For the complete definition,
// please check:
// https://github.com/llvm/llvm-project/blob/8c75290/llvm/lib/Target/AMDGPU/SIDefines.h#L939
enum class DppCtrl : uint32_t {
  QUAD_PERM_FIRST = 0,
  ROW_SHL0 = 0x100,
  ROW_SHR0 = 0x110,
  ROW_ROR0 = 0x120,
  ROW_MIRROR = 0x140,
  ROW_HALF_MIRROR = 0x141,
  BCAST15 = 0x142,
  BCAST31 = 0x143,
  ROW_XMASK0 = 0x160,
};

enum class MemoryOp { Load, Store };

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i,
                 mlir::triton::amdgpu::ISAFamily isaFamily =
                     mlir::triton::amdgpu::ISAFamily::Unknown);
Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i,
                mlir::triton::amdgpu::ISAFamily isaFamily =
                    mlir::triton::amdgpu::ISAFamily::Unknown);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i,
                 mlir::triton::amdgpu::ISAFamily isaFamily =
                     mlir::triton::amdgpu::ISAFamily::Unknown);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i,
                 mlir::triton::amdgpu::ISAFamily isaFamily =
                     mlir::triton::amdgpu::ISAFamily::Unknown);

Value permute(Location loc, RewriterBase &rewriter, Value a, Value b,
              Value selector);

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               ProgramIDDim axis);

// Emit the cta multicast mask for a given cta id based on the src layout.
// Groups sharing data among more than maxMaskPopcount CTAs are split into
// smaller subgroups because the hardware would otherwise drop the multicast.
Value emitCtaMulticastMask(RewriterBase &rewriter, Location loc, Value blockId,
                           const LinearLayout &cvt, unsigned maxMaskPopcount);

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
             bool isVolatile = false, bool forceNoAliasAsyncLoads = false);

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

// The vector sizes the pointers/offsets and the mask each allow. Zero means
// unknown, in which case no attribution is made.
struct DirectToLdsVecInfo {
  unsigned fromPtr = 0;
  unsigned fromMask = 0;

  bool maskIsLimiting() const { return fromPtr != 0 && fromMask < fromPtr; }
};

// Returns true if we can load directly from global |srcTy| to shared memory
// |dstEnc| for the given target.
// This function expects the caller to pass in |vectorSize| as the vector size
// reading from global memory, after factoring in axis information and alignment
// hints. It will be updated to factor in shared memory |dstEnc| constraints.
// On failure |*failureReason|, if non-null, is set to an explanation of the
// check that failed and how to satisfy it.
bool canLoadDirectToLDS(const triton::AMD::TargetInfo &targetInfo,
                        RankedTensorType srcTy, Attribute dstEnc,
                        ArrayRef<int64_t> dstAllocShape, unsigned &vectorSize,
                        DirectToLdsVecInfo vecInfo = {},
                        std::string *failureReason = nullptr);

// Check if the result of this tl.dot is used as opA or opB of another tl.dot.
bool isChainDotHead(mlir::triton::DotOpInterface dotOp, unsigned opIdx = 0);

// Check if the opA of this tl.dot is the result of another tl.dot.
bool isChainDotTail(mlir::triton::DotOpInterface dotOp);

// Branchless fp8 -> f32 via multiply trick.
// Handles both E4M3FN (isE4M3FN=true) and E5M2 (isE4M3FN=false) formats.
Value convertF8ToF32_SW(RewriterBase &rewriter, Location loc, Value fp8Val,
                        bool isE4M3FN);

// Software implementation of converting an 8-element vector of MXFP4 elements
// to a wider type: BF16 or FP16 for target before CDNA4.
SmallVector<Value> upcast8xMxfp4_SW(RewriterBase &rewriter, Operation *op,
                                    bool toFp16, Value packedVec,
                                    mlir::triton::amdgpu::ISAFamily isaFamily);

} // namespace mlir::LLVM::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_UTILITY_H_
