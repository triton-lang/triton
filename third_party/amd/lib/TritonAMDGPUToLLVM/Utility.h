#ifndef TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_UTILITY_H

#include "TargetInfo.h"
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
const char predicatedLoadCV[] = "__predicated_load_CV";
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

// Utility class to take care of buffer operation emission. We may add more
// emitters into this as needed.  Buffer operations accept a memory descriptor
// and an offset.
//
// The memory descriptor is stored in s_gprs and hence needs to
// be uniform across the wave. It contains two fields (among many others):
//
//    - `base_pointer`: represents the (scalar) pointer  to the memory area
//    - `num_records`:  represents the size of the memory region. This is a
//                      32 bit unsigned integer
//
// The offset can be non-uniform across the wave (and hence stored in vgprs).
//
// The high level behaviour of a buffer operation can be described as:
// ```
// def buffer_op(mem_desc, offset):
//     address = splat(mem_desc.base_pointer)
//     address += offset
//     return buffer_op(address)
// ```
// This means we don't need to store the addresses in vgprs and we need less
// VALU operations to compute the final address.
//
// Also note that buffer operations support out-of-boundary memory access.
// I.e., if offset[i] > mem_desc.num_records the operation is a nop for the i-th
// thread.
//
// This can be exploited to support masked operations, like in the following
// snippet:
// ```
// def masked_op(base_ptr, offset, pred)
//     mem_desc.base_ptr = base_ptr
//     mem_desc.num_records = max_int_32
//     oob_offset = max_int_32+1
//     masked_offset = (pred ? offset : oob_offset)
//     buffer_op(mem_desc, masked_offset)
// ```
// To use buffer operations three main requirements need to be met:
//
// 1. The buffer pointer needs to be a scalar, it cannot be non-uniform across
//   threads of the given wave
// 2. The offset needs to be expressed in 32 bits
// 3. The offset needs to be non-negative
//
// Failure to meet 1) will result in a scalarized loop (very poor performance).
// Failure to meet 2) and 3) will result in incorrect memory access.
struct BufferEmitter {
  BufferEmitter(RewriterBase &rw, Location loc,
                mlir::triton::AMD::TargetInfo ti);

  // Create a resource descriptor that points to the area of memory we want to
  // load from
  Value createResourceDescriptor(Value basePtr);

  // Emit a predicated rocdl.raw.ptr.buffer.load
  Value emitLoad(Type type, Value rsrcDesc, Value offset, Value pred,
                 Value falseVal);

  // Emit a predicated rocdl.raw.ptr.buffer.store
  void emitStore(Value rsrcDesc, Value offset, Value data, Value pred);

private:
  // Fill common buffer operation arguments.
  void fillCommonArgs(Type type, Value rsrcDesc, Value vOffsetElems, Value pred,
                      SmallVector<Value> &args);

  // Given a type, the buffer type can be either the same type
  // or a packed version. E.g., a vector of 8xfp16 can be bitcasted to
  // a vector of 4xi32. This usually makes the life of the backend easier
  Type getBufferOpType(Type type);

  // Rewriter utilities
  RewriterBase &rewriter;
  Location loc;
  mlir::triton::AMD::TargetInfo targetInfo;
};

// Loads from shared or global memory with predication.
// `otherElems` is used to mask out the elements that are not loaded
Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal, int64_t alignmentBytes = 0,
             triton::CacheModifier cm = triton::CacheModifier::NONE);

// Stores to shared or global memory with predication.
void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, int64_t alignmentBytes = 0,
             triton::CacheModifier cm = triton::CacheModifier::NONE);
} // namespace mlir::LLVM::AMD

#endif
