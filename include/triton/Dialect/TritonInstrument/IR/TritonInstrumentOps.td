#ifndef TRITONINSTRUMENT_OPS
#define TRITONINSTRUMENT_OPS

include "triton/Dialect/TritonInstrument/IR/TritonInstrumentDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
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
  let assemblyFormat = "$memdesc attr-dict `:` type($memdesc)";
}

def TTI_ExperimentalClusterCTAIdOp
    : TTI_Op<"experimental_cluster_cta_id", [Pure]> {
  let summary = "Get the CTA id within the current cluster";
  let description = [{
    Return the cluster-local CTA id used to index ConSan's multi-CTA scratch
    slabs. For single-CTA kernels this is always zero.
  }];
  let results = (outs I32:$result);
  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, $_builder.getI32Type());
    }]>
  ];
  let assemblyFormat = "attr-dict `:` type($result)";
}

def TTI_ExperimentalGSanInitOp
    : TTI_Op<"experimental_gsan_init"> {
  let summary = "Initialize GSan thread";
  let arguments = (ins);
}

def TTI_ExperimentalGSanTensorDescInfoOp
    : TTI_Op<"experimental_gsan_tensordesc_info", [Pure]> {
  let summary = "Decode GSan descriptor metadata from a native tensor descriptor";
  let description = [{
    Decodes a native tensor descriptor into the underlying
    base pointer, shape and stride values.
  }];
  let arguments = (ins TT_AnyTensorDescType:$desc);
  let results = (outs Variadic<AnyType>:$result);
  let assemblyFormat = [{
    $desc attr-dict `:` qualified(type($desc)) `->` type($result)
  }];
}

def TTI_ExperimentalGSanTensorAccessOp
    : TTI_Op<"experimental_gsan_tensor_access",
             [TypesMatchWith<"mask type matches ptr type", "ptr", "mask",
                             "getI1SameShape(getPointeeType($_self))",
                             "($_op.getOperands().size() <= 1) || std::equal_to<>()">]> {
  let summary = "Instrument a tensor load/store access for GSan";
  let description = [{
    Emits runtime instrumentation for a tensor pointer access. The pointer and
    optional mask are consumed by the GSan runtime.
  }];
  let arguments = (ins Arg<TT_PtrLike, "", [MemWrite<GlobalMemory>, MemRead<GlobalMemory>]>:$ptr,
                   Optional<TT_BoolLike>:$mask, BoolAttr:$isStore);
  let assemblyFormat = [{
    $ptr `,` $isStore (`,` $mask^)? attr-dict `:` type($ptr)
  }];
}

def TTI_ExperimentalGSanAtomicTensorAccessOp
    : TTI_Op<"experimental_gsan_atomic_tensor_access",
             [TypesMatchWith<"mask type matches ptr type", "ptr", "mask",
                             "getI1SameShape(getPointeeType($_self))",
                             "($_op.getOperands().size() <= 1) || std::equal_to<>()">]> {
  let summary = "Instrument a tensor atomic access for GSan";
  let description = [{
    Emits runtime instrumentation for a tensor pointer access whose individual
    elements are atomic read-modify-write operations.
  }];
  let arguments = (ins Arg<TT_PtrLike, "", [MemWrite<GlobalMemory>, MemRead<GlobalMemory>]>:$ptr,
                   Optional<TT_BoolLike>:$mask, TT_MemSemanticAttr:$sem, TT_MemSyncScopeAttr:$scope);
  let assemblyFormat = [{
    $sem `,` $scope `,` $ptr (`,` $mask^)? attr-dict `:` type($ptr)
  }];
}

def TTI_ExperimentalGSanAtomicRMWOp : TTI_Op<"experimental_gsan_atomic_rmw", [
  SameOperandsAndResultShape,
  SameOperandsAndResultEncoding,
  TypesMatchWith<"ptr type matches value type", "val", "ptr",
                 "getPointerTypeSameShape($_self)">,
  TypesMatchWith<"mask type matches value type",
                 "val", "mask", "getI1SameShape($_self)",
                 "($_op.getOperands().size() <= 2) || std::equal_to<>()">
]> {
  let summary = "Lower a GSan-instrumented atomic rmw";
  let arguments = (ins
    TT_AtomicRMWAttr:$atomic_rmw_op,
    Arg<TT_PtrLike, "", [MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>:$ptr,
    TT_Type:$val,
    Optional<TT_BoolLike>:$mask,
    TT_MemSemanticAttr:$sem,
    TT_MemSyncScopeAttr:$scope
  );
  let results = (outs TT_Type:$result);
  let assemblyFormat = [{
    $atomic_rmw_op `,` $sem `,` $scope `,` $ptr `,` $val (`,` $mask^)? attr-dict `:`
    functional-type(operands, $result)
  }];
}

def TTI_ExperimentalGSanAtomicCASOp : TTI_Op<"experimental_gsan_atomic_cas", [
  SameOperandsAndResultShape,
  SameOperandsAndResultEncoding,
  TypesMatchWith<"ptr type matches cmp type", "cmp", "ptr",
                 "getPointerTypeSameShape($_self)">,
  TypesMatchWith<"ptr type matches value type", "val", "ptr",
                 "getPointerTypeSameShape($_self)">
]> {
  let summary = "Lower a GSan-instrumented atomic cas";
  let arguments = (ins
    Arg<TT_PtrLike, "", [MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>:$ptr,
    TT_Type:$cmp,
    TT_Type:$val,
    TT_MemSemanticAttr:$sem,
    TT_MemSyncScopeAttr:$scope
  );
  let results = (outs TT_Type:$result);
  let assemblyFormat = [{
    $sem `,` $scope `,` $ptr `,` $cmp `,` $val attr-dict `:`
    functional-type(operands, $result)
  }];
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


// ===== FPSan ops =====


def TTI_ExperimentalFPSanEmbedOp : TTI_Op<"experimental_fpsan_embed", [
  Pure,
  Elementwise,
  SameOperandsAndResultShape,
  SameOperandsAndResultEncoding
]> {
  let summary = "Embed float value in the fpsan integer ring";
  let arguments = (ins TT_FloatLike:$val);
  let results = (outs TT_IntLike:$result);
  let assemblyFormat = [{
    $val attr-dict `:` functional-type(operands, $result)
  }];

  let hasCanonicalizer = true;
  let hasFolder = true;
}

def TTI_ExperimentalFPSanUnembedOp : TTI_Op<"experimental_fpsan_unembed", [
  Pure,
  Elementwise,
  SameOperandsAndResultShape,
  SameOperandsAndResultEncoding
]> {
  let summary = "Unembed fpsan integer payload as a float value";
  let arguments = (ins TT_IntLike:$val);
  let results = (outs TT_FloatLike:$result);
  let assemblyFormat = [{
    $val attr-dict `:` functional-type(operands, $result)
  }];

  let hasCanonicalizer = true;
  let hasFolder = true;
}

#endif // TRITONINSTRUMENT_OPS
