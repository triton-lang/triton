#ifndef TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_UTILITY_H

#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
namespace mlir::LLVM::AMD {

const std::string Predicated_Load = "__predicated_load";
const std::string Predicated_Load_CA = Predicated_Load + "_CA";
const std::string Predicated_Load_CG = Predicated_Load + "_CG";
const std::string Predicated_Store = "__predicated_store";
const std::string Predicated_Store_CG = Predicated_Store + "_CG";
const std::string Predicated_Store_CS = Predicated_Store + "_CS";
const std::string Predicated_Store_WT = Predicated_Store + "_WT";

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
