#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_BUFFEROPSEMITTER_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_BUFFEROPSEMITTER_H_

#include "TargetInfo.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include <cstdint>

namespace mlir::LLVM::AMD {
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
//     oob_offset = 0xFFFFFFFF  // out of bounds for any 32-bit num_records
//     masked_offset = (pred ? offset : oob_offset)
//     buffer_op(mem_desc, masked_offset)
// ```
// This only works while num_records is at most 2^32. On gfx1250, a buffer
// descriptor can hold a 45-bit num_records, so accesses through one are masked
// by branching around them instead (see `setMaskingFor`).
//
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

  // Create a resource descriptor (V#) for `basePtr`. `numRecords` is the byte
  // bound: an i32 read as unsigned, an i64 (gfx1250 only) read as unsigned and
  // clamped to 2^45 - 1, or null for the default 2^31 - 2 bytes.
  Value createResourceDescriptor(Value basePtr, Value numRecords);

  // Choose how the accesses through `base` (a pointer or a buffer descriptor)
  // mask off lanes. An out-of-bounds offset only fails the range check while
  // num_records is at most 2^32, so on targets whose descriptors can hold more,
  // accesses through a descriptor branch around masked-off lanes instead.
  void setMaskingFor(Value base);

  // Convert the op's optional `soffset`, which counts elements of `elemTy`,
  // to the i32 byte value the instruction takes. Null stays null.
  Value getScalarOffsetBytes(Type elemTy, Value soffset);

  // Emit a predicated rocdl.raw.ptr.buffer.load. `soffsetBytes` may be null.
  // `aux` is the instruction's cache-policy operand.
  Value emitLoad(Type type, Value rsrcDesc, Value offset, Value soffsetBytes,
                 Value pred, Value falseVal, int32_t aux);

  // Emit a predicated llvm.amdgcn.raw.ptr.atomic.buffer.load: a load that
  // LLVM keeps in order with other memory operations and never removes.
  Value emitAtomicLoad(Type type, Value rsrcDesc, Value offset,
                       Value soffsetBytes, Value pred, int32_t aux);

  // Emit a rocdl.raw.ptr.buffer.load.async.lds for direct-to-LDS loads.
  // Always emits the async variant since buffer_load_to_lds is only supported
  // on gfx942/gfx950 which use asyncmark-based synchronization.
  ROCDL::RawPtrBufferLoadAsyncLdsOp emitLoadToLds(Type type, Value byteWidth,
                                                  Value rsrcDesc, Value offset,
                                                  Value soffsetBytes, Value dst,
                                                  Value pred, int32_t aux);

  // Emit a predicated rocdl.raw.ptr.buffer.atomic.* RMWOp
  Value emitAtomicRMW(RMWOp rmwType, Type type, Value rsrcDesc, Value offset,
                      Value data, Value pred, bool hasUsers);

  // Emit a predicated rocdl.raw.ptr.buffer.atomic.cmpswap
  Value emitAtomicCAS(Type type, Value rsrcDesc, Value offset, Value casCmpVal,
                      Value casStoreVal, Value pred, bool hasUsers);

  // Emit a predicated rocdl.raw.ptr.buffer.store
  void emitStore(Value rsrcDesc, Value offset, Value soffsetBytes, Value data,
                 Value pred, int32_t aux);

private:
  // Emit the access built by `emitAccess` for the lanes where `pred` is true,
  // returning its result, or `falseVal` for the other lanes. `emitAccess` gets
  // the predicate to mask the access with: `pred` itself, or true when it
  // runs under a branch on `pred`. `falseVal` is null for accesses with no
  // result.
  Value emitPredicated(Value pred, Value falseVal,
                       function_ref<Value(Value pred)> emitAccess);

  // Fill the descriptor, the per-lane byte offset and the scalar byte offset
  // arguments, masking off lanes where `pred` is false through the hardware
  // range check.
  void fillCommonArgs(Type type, Value rsrcDesc, Value vOffsetElems,
                      Value soffsetBytes, Value pred, bool isAtomicRMW,
                      SmallVector<Value> &args);

  // Given a type, the buffer type can be either the same type
  // or a packed version. E.g., a vector of 8xfp16 can be bitcasted to
  // a vector of 4xi32. This usually makes the life of the backend easier
  Type getBufferOpType(Type type, bool atomicsOp);

  // Rewriter utilities
  RewriterBase &rewriter;
  Location loc;
  mlir::triton::AMD::TargetInfo targetInfo;
  // Whether to mask lanes by branching rather than by the range check.
  bool maskByBranch = false;
};

} // namespace mlir::LLVM::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_BUFFEROPSEMITTER_H_
