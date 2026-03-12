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

def TTI_ExperimentalAssertUniformOp : TTI_Op<"experimental_assert_uniform", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "assert the uniform condition";
  let description = [{
    Assert that the condition is true given all threads in the warp group have
    the same value, so only one thread needs to evaluate the assert and print
    the message.
  }];
  let arguments = (ins I1:$condition, StrAttr:$message);
  let assemblyFormat = "$condition `,` $message attr-dict-with-keyword";
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

def TTI_ExperimentalClusterScratchBaseOp
    : TTI_Op<"experimental_cluster_scratch_base", [Pure]> {
  let summary = "Rebase a profile-scratch pointer to the CTA0 replica";
  let description = [{
    Return the pointer to the CTA0 replica of the same profile-scratch
    allocation. In multi-CTA ConSan kernels this lets every CTA address one
    shared logical scratch slab while still using the default per-CTA profile
    scratch lowering.
  }];
  let arguments = (ins TT_PtrLike:$ptr);
  let results = (outs TT_PtrLike:$result);
  let builders = [
    OpBuilder<(ins "Value":$ptr), [{
      build($_builder, $_state, ptr.getType(), ptr);
    }]>
  ];
  let assemblyFormat = "$ptr attr-dict `:` type($ptr) `->` type($result)";
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

def TTI_ExperimentalStoreI32Op
    : TTI_Op<"experimental_store_i32",
             [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Store an i32 through a pointer with one elected thread per CTA.";
  let description = [{
    Emit a predicated global i32 store using one elected thread per CTA. This is
    used for ConSan cluster-scoped scratch updates where the generic scalar
    StoreOp lowering would otherwise elect only one thread for the whole
    cluster.
  }];
  let arguments = (ins TT_PtrLike:$ptr, I32:$value, Optional<I1>:$pred);
  let assemblyFormat = [{
    $ptr `,` $value (`,` $pred^)? attr-dict `:` type($ptr)
  }];
}

#endif // TRITONINSTRUMENT_OPS
