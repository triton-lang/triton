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

const char predicatedLoad[] = "__triton_hip_predicated_load";
const char predicatedLoadCA[] = "__triton_hip_predicated_load_CA";
const char predicatedLoadCG[] = "__triton_hip_predicated_load_CG";
const char predicatedLoadCV[] = "__triton_hip_predicated_load_CV";
const char predicatedStore[] = "__triton_hip_predicated_store";
const char predicatedStoreCG[] = "__triton_hip_predicated_store_CG";
const char predicatedStoreCS[] = "__triton_hip_predicated_store_CS";
const char predicatedStoreWT[] = "__triton_hip_predicated_store_WT";
const char noAliasAsyncLoads[] = "__no_alias_async_loads";

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

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               int axis);

// Loads from shared or global memory with predication.
// `otherElems` is used to mask out the elements that are not loaded
Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal,
             triton::CacheModifier cm = triton::CacheModifier::NONE);

// Stores to shared or global memory with predication.
// forceNoAliasAsyncLoads=true adds alias information to the llvm.store to
// signal its not aliasing with any AsyncCopyGlobalToLocal/BufferLoadToLocal to
// avoid conservative waits. See `addLocalLoadNoAliasScope` for more details
void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, triton::CacheModifier cm = triton::CacheModifier::NONE,
             bool forceNoAliasAsyncLoads = false);

// Get cache modifier information for creating load or store instruction
// Get flags <volatile, nontemporal> for a predicated Load or Store
std::pair<bool, bool> getCacheModifierFlagsForPredicatedCall(LLVM::CallOp);
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
// the destination encoding.
bool canCoalesceWriteIntoSharedMemory(RewriterBase &rewriter,
                                      const LinearLayout &srcToSharedLayout,
                                      unsigned threadsPerWarp);

// Returns true if the swizzling pattern does only swizzle the shared memory
// offsets of a warp and does not exchange destination elements across warps
bool doesSwizzleInsideWarp(RewriterBase &rewriter,
                           const LinearLayout &srcToSharedLayout,
                           unsigned threadsPerWarp);

// Return true if op is used by DotScaledOp or UpcastMXFPOp ops.
bool isUsedByDotScaledOp(Operation *op);

// Check if the result of this tl.dot is used as opA of another tl.dot
// in the same region
bool isChainDotHead(mlir::triton::DotOpInterface dotOp);

// Check if the opA of this tl.dot is the result of another tl.dot
// in the same region
bool isChainDotTail(mlir::triton::DotOpInterface dotOp);

// LLVM is unable to deduce dependencies across warps and loop iterations for
// AsyncCopy and LocalLoad and will emit conservative wait counts. In triton the
// dependency is models via AsyncWait, e.g.
//   %token1 = ttg.async_copy_global_to_local/amdgpu.buffer_load_to_local
//   %token2 = ttg.async_wait %token1
//   %1      = ttg.local_load .. token %token2
// For such cases AsyncWait will emit the correct wait and the conservative
// waits are redundant and hindering performance/interleaving.
// To disable the conservative waits two alias scopes are created:
//   1) "amdgpu.AsyncCopies" will contain all AsyncCopy ops
//   2) "amdgpu.LocalLoad" will contain all LocalLoads manually synchronized via
//      AsyncWait
// ALl manually synchronized LocalLoads will additionally have "AsyncCopies" as
// a non alias scope to disable the implicit waits from the LLVM backend

// If localLoadOp has a token from an AsyncWait:
//  - Attaches "amdgpu.LocalLoad" alias scope to llLoadOp
//  - Attaches "amdgpu.AsyncCopies" as *non* alias scope to llLoadOp
void addLocalLoadNoAliasScope(triton::gpu::LocalLoadOp localLoadOp,
                              AliasAnalysisOpInterface llLoadOp);
// Overload from above without checking the AsyncToken
void addLocalLoadNoAliasScope(AliasAnalysisOpInterface llLoadOp);
// Attaches the "AsyncCopies" alias scope to llLoadDirectToLdsOp
void addAsyncCopyAliasScope(AliasAnalysisOpInterface llLoadDirectToLdsOp);

} // namespace mlir::LLVM::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_UTILITY_H_
