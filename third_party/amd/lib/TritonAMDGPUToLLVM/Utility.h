#ifndef TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_UTILITY_H

#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
namespace mlir::LLVM::AMD {

const char predicatedLoad[] = "__predicated_load";
const char predicatedLoadCA[] = "__predicated_load_CA";
const char predicatedLoadCG[] = "__predicated_load_CG";
const char predicatedStore[] = "__predicated_store";
const char predicatedStoreCG[] = "__predicated_store_CG";
const char predicatedStoreCS[] = "__predicated_store_CS";
const char predicatedStoreWT[] = "__predicated_store_WT";

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i);

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               int axis);

// Loads from shared or global memory with predication.
// `otherElems` is used to mask out the elements that are not loaded
Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal,
             triton::CacheModifier cm = triton::CacheModifier::NONE);

// Stores to shared or global memory with predication.
void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred,
             triton::CacheModifier cm = triton::CacheModifier::NONE);
} // namespace mlir::LLVM::AMD

#endif
