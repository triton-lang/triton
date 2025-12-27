#ifndef TRITONINSTRUMENT_OPS
#define TRITONINSTRUMENT_OPS

include "triton/Dialect/TritonInstrument/IR/TritonInstrumentDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton/Dialect/TritonInstrument/IR/TritonInstrumentAttrDefs.td"

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;

//
// Ops
//

class TTI_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonInstrument_Dialect, mnemonic, traits> {
}

def TTI_ExperimentalAssertInThreadOp : TTI_Op<"experimental_assert_in_thread", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "assert the condition within the current thread";
  let description = [{
    Assert that the condition is true given all the values are available in the current thread.
    If the condition is false, the message is printed, and the program is aborted.
    If check_any is true, any of the values in the condition must be true. Otherwise, all the
    values in the condition must be true.
  }];
  let arguments = (ins AnyTypeOf<[I1, I1Tensor]>:$condition, StrAttr:$message, BoolAttr:$check_any);
  let assemblyFormat = "$condition `,` $message attr-dict `:` type($condition)";
}


def TTI_ExperimentalBufferDescriptorsOp
    : TTI_Op<"experimental_buffer_descriptors", [Pure]> {
  let summary = "define an array of buffer descriptors";
  let description = [{
    Create a tensor of buffer descriptors packing 32-bit pointer offsets and
    32-bit lengths into 64-bit elements.
  }];
  let arguments = (ins DenseI32ArrayAttr:$offsets, DenseI32ArrayAttr:$lengths,
                   TT_MemTypeAttr:$memType);
  let results = (outs TT_Tensor:$result);
  let assemblyFormat = [{
    $offsets `,` $lengths `,` $memType attr-dict `:` type($result)
  }];
}

def TTI_ExperimentalMemDescToI32Op : TTI_Op<"experimental_memdesc_to_i32", [Pure]> {
  let summary = "Convert a memdesc into its base pointer as i32";
  let description = [{
    Extract the base pointer from the given memdesc and return it as a 32-bit
    integer. This can be used to compare the memdesc against tensors of barrier
    pointers maintained by the concurrency sanitizer.
  }];
  let arguments = (ins TTG_MemDescType:$memdesc);
  let results = (outs I32:$result);
  let builders = [
    OpBuilder<(ins "Value":$memdesc), [{
      build($_builder, $_state, $_builder.getI32Type(), memdesc);
    }]>
  ];
  let assemblyFormat = "$memdesc attr-dict `:` type($memdesc)";
}


// ===== Critical section lock ops =====


def TTI_ExperimentalLockAcquireOp : TTI_Op<"experimental_lock_acquire", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Acquire a lock.";
  let description = [{
    Enter a critical section by acquiring a lock with single thread.
  }];
  let arguments = (ins TT_PtrLike:$lock, Optional<I1>:$pred);
  let assemblyFormat = [{
    $lock (`,` $pred^)? attr-dict `:` type($lock)
  }];
}


def TTI_ExperimentalLockReleaseOp : TTI_Op<"experimental_lock_release", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Release a lock.";
  let description = [{
    Leave a critical section by releasing a lock with single thread.
  }];
  let arguments = (ins TT_PtrLike:$lock, Optional<I1>:$pred);
  let assemblyFormat = [{
    $lock (`,` $pred^)? attr-dict `:` type($lock)
  }];
}

#endif // TRITONINSTRUMENT_OPS
