#ifndef TRITONINSTRUMENT_OPS
#define TRITONINSTRUMENT_OPS

include "triton/Dialect/TritonInstrument/IR/TritonInstrumentDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUMemoryEffects.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"
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

class TTI_Op<string mnemonic, list<Trait> traits = []>
    : Op<TritonInstrument_Dialect, mnemonic, traits> {
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

def TTI_ExperimentalMemoryOffsetToI32Op
    : TTI_Op<"experimental_memory_offset_to_i32", [Pure]> {
  let summary = "Materialize a memory-space offset as an i32 address";
  let description = [{
    Add a static byte or encoded tensor-memory offset to the current
    function's memory-space base and return the 32-bit address used by ConSan.
    This provides the runtime identity corresponding to a compile-time
    BufferRegion key without materializing a descriptor table.
  }];
  let arguments = (ins I32Attr:$offset, TT_MemTypeAttr:$memType);
  let results = (outs I32:$result);
  let assemblyFormat = "$offset `,` $memType attr-dict";
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

def TTI_ExperimentalLocalGatherOp
    : TTI_Op<"experimental_local_gather"> {
  let summary = "Gather elements from shared memory with logical base offsets";
  let description = [{
    Gather elements from a shared memory descriptor using an index tensor along
    one axis, after shifting the logical source coordinates by rank-sized scalar
    offsets. This is intentionally private to instrumentation passes.
  }];
  let arguments = (ins
    Arg<TTG_MemDescType, "", GenericSharedRead>:$src,
    TT_IntTensor:$indices,
    Variadic<I32>:$offsets,
    I32Attr:$axis
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $src `[` $indices `]` `offsets` `=` `[` $offsets `]`
    attr-dict `:` qualified(type($src)) `,` type($indices) `->` type($result)
  }];
  let hasVerifier = 1;
}

def TTI_ExperimentalGSanInitOp
    : TTI_Op<"experimental_gsan_init"> {
  let summary = "Initialize GSan thread";
  let arguments = (ins BoolAttr:$acquireStreamClock);
  let assemblyFormat = "$acquireStreamClock attr-dict";
}

def TTI_ExperimentalGSanKernelExitOp
    : TTI_Op<"experimental_gsan_kernel_exit", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Publish a completed GSan thread clock to its launch stream";
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

def TTI_ExperimentalGSanGridDependencyWaitOp
    : TTI_Op<"experimental_gsan_grid_dependency_wait", [MemoryEffects<[MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>]> {
  let summary = "Acquire completed predecessor clocks after a grid dependency wait";
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

def TTI_ExperimentalGSanClusterBarrierInitOp
    : TTI_Op<"experimental_gsan_cluster_barrier_init",
             [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Initialize a GSan cluster-barrier rendezvous";
  let description = [{
    Initializes cluster-shared profile scratch used to synchronize GSan vector
    clocks. This operation is internal to GSan instrumentation.
  }];
  let arguments = (ins TT_Ptr:$scratch);
  let assemblyFormat = "$scratch attr-dict `:` type($scratch)";
}

def TTI_ExperimentalGSanClusterBarrierSyncOp
    : TTI_Op<"experimental_gsan_cluster_barrier_sync",
             [MemoryEffects<[MemRead<GlobalMemory>,
                             MemWrite<GlobalMemory>]>]> {
  let summary = "Synchronize GSan clocks at a cluster barrier";
  let description = [{
    Joins the vector clocks of all CTAs participating in a non-relaxed cluster
    barrier. This operation is internal to GSan instrumentation.
  }];
  let arguments = (ins TT_Ptr:$scratch);
  let assemblyFormat = "$scratch attr-dict `:` type($scratch)";
}

def TTI_ExperimentalGSanMBarrierTableInitOp
    : TTI_Op<"experimental_gsan_mbarrier_table_init",
             [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Initialize GSan mbarrier release state";
  let description = [{
    Initializes GSan's internal mbarrier table, used to sequence memory accesses
    through cluster mbarriers. This operation is internal to GSan instrumentation.
  }];
  let arguments = (ins TT_Ptr:$scratch, I32Attr:$capacity);
  let assemblyFormat = "$scratch `,` $capacity attr-dict `:` type($scratch)";
}

def TTI_ExperimentalGSanMBarrierInitOp
    : TTI_Op<"experimental_gsan_mbarrier_init",
             [MemoryEffects<[MemRead<GlobalMemory>,
                             MemWrite<GlobalMemory>]>]> {
  let summary = "Initialize an mbarrier in GSan";
  let description = [{
    Associates an mbarrier with its expected arrival count.
    This operation is internal to GSan instrumentation.
  }];
  let arguments = (ins TT_Ptr:$scratch, TTG_MemDescType:$barrier,
                       I32Attr:$expectedCount);
  let assemblyFormat = [{
    $scratch `,` $barrier `,` $expectedCount attr-dict `:` type($scratch) `,`
    qualified(type($barrier))
  }];
}

def TTI_ExperimentalGSanMBarrierArriveOp
    : TTI_Op<"experimental_gsan_mbarrier_arrive",
             [MemoryEffects<[MemRead<GlobalMemory>,
                             MemWrite<GlobalMemory>]>]> {
  let summary = "Record an mbarrier arrival in GSan";
  let description = [{
    Records an arrival for every mbarrier targeted by the corresponding
    hardware operation. When `publishClock` is true, the issuing CTA's vector
    clock is also published without advancing its local epoch. Waiters acquire
    the issuing CTA's completed epochs and its imported clocks, but not its
    current epoch. A false value records only phase
    progress for asynchronous completion signals that provide no generic
    memory ordering. `multicastMasks` contains CTA-id broadcast masks whose
    recipient bitsets are unioned. When `multicast` is false, the arrival
    targets the leader selected by the barrier's CGA layout.
    This operation is internal to GSan instrumentation.
  }];
  let arguments = (ins TT_Ptr:$scratch, TTG_MemDescType:$barrier, I1:$pred,
                       I32Attr:$count, DenseI32ArrayAttr:$multicastMasks,
                       BoolAttr:$multicast,
                       I32Attr:$sourceBroadcastMask,
                       DefaultValuedAttr<BoolAttr, "true">:$publishClock);
  let assemblyFormat = [{
    $scratch `,` $barrier `,` $pred `,` $count attr-dict `:` type($scratch) `,`
    qualified(type($barrier))
  }];
}

def TTI_ExperimentalGSanMBarrierWaitOp
    : TTI_Op<"experimental_gsan_mbarrier_wait",
             [MemoryEffects<[MemRead<GlobalMemory>,
                             MemWrite<GlobalMemory>]>]> {
  let summary = "Acquire GSan clocks after an mbarrier wait";
  let description = [{
    Acquires the clocks published to the completed mbarrier phase, excluding
    each publisher's current epoch. Global-memory handoffs that rely solely on
    an mbarrier may therefore be conservatively reported as races.
    This operation is internal to GSan instrumentation.
  }];
  let arguments = (ins TT_Ptr:$scratch, TTG_MemDescType:$barrier,
                       I32:$phase, I1:$pred);
  let assemblyFormat = [{
    $scratch `,` $barrier `,` $phase `,` $pred attr-dict `:` type($scratch) `,`
    qualified(type($barrier))
  }];
}

def TTI_ExperimentalGSanTensorDescAccessOp
    : TTI_Op<"experimental_gsan_tensordesc_access",
             [AttrSizedOperandSegments]> {
  let summary = "Instrument a TMA tensor descriptor access for GSan";
  let description = [{
    Passes the native tensor descriptor and its coordinates directly to the
    GSan runtime, which decodes the descriptor and iterates over the tile.
  }];
  let arguments = (ins TT_AnyTensorDescType:$desc, Variadic<I32>:$coords,
                   Optional<I1>:$pred, BoolAttr:$isStore);
  let assemblyFormat = [{
    $desc `[` $coords `]` `,` $isStore (`,` $pred^)? attr-dict `:`
    qualified(type($desc))
  }];
}

def TTI_ExperimentalGSanIndexedTensorDescAccessOp
    : TTI_Op<"experimental_gsan_indexed_tensordesc_access"> {
  let summary = "Instrument an indexed TMA gather or scatter for GSan";
  let description = [{
    Passes a native tensor descriptor and its row indices directly to the GSan
    runtime, which checks row and column bounds before updating shadow memory.
  }];
  let arguments = (ins TT_AnyTensorDescType:$desc,
                   RankedTensorOf<[I32]>:$indices, I32:$offset,
                   Optional<I1>:$pred, BoolAttr:$isStore);
  let assemblyFormat = [{
    $desc `[` $indices `,` $offset `]` `,` $isStore (`,` $pred^)? attr-dict
    `:` qualified(type($desc)) `,` type($indices)
  }];
}

def TTI_ExperimentalGSanAtomicTensorDescAccessOp
    : TTI_Op<"experimental_gsan_atomic_tensordesc_access"> {
  let summary = "Instrument an atomic TMA descriptor access for GSan";
  let description = [{
    Passes a native tensor descriptor and coordinates to the GSan runtime for
    atomic instrumentation with the specified memory semantics and scope.
  }];
  let arguments = (ins TT_AnyTensorDescType:$desc, Variadic<I32>:$coords,
                   TT_MemSemanticAttr:$sem, TT_MemSyncScopeAttr:$scope);
  let assemblyFormat = [{
    $desc `[` $coords `]` `,` $sem `,` $scope attr-dict `:`
    qualified(type($desc))
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

def TTI_ExperimentalGSanAtomicPollOp
    : TTI_Op<"experimental_gsan_atomic_poll"> {
  let summary = "Instrument a completed atomic poll for GSan";
  let description = [{
    Records the successful read performed by an atomic poll after the hardware
    polling loop has completed.
  }];
  let arguments = (ins
    Arg<TT_Ptr, "", [MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>:$ptr,
    I1:$matched,
    TT_MemSemanticAttr:$sem,
    TT_MemSyncScopeAttr:$scope
  );
  let assemblyFormat = [{
    $sem `,` $scope `,` $ptr `,` $matched attr-dict `:` qualified(type($ptr))
  }];
}

def TTI_ExperimentalGSanAtomicRMWOp : TTI_Op<"experimental_gsan_atomic_rmw", [
  AtomicOpInterface,
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
  AtomicOpInterface,
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


def TTI_DotI8Op : TTI_Op<"dot_i8", [
  Pure,
  DeclareOpInterfaceMethods<DotOpInterface>,
  TypesMatchWith<"result's type matches accumulator's type",
                 "d", "c", "$_self">
]> {
  let summary = "non-saturating NVIDIA MMAv2 i8 dot";
  let description = [{
    Performs a wrapping i8 matrix multiplication into an i32 accumulator using
    NVIDIA MMAv2. The A and B operands have independent signedness.
  }];
  let arguments = (ins
    RankedTensorOf<[I8]>:$a,
    RankedTensorOf<[I8]>:$b,
    RankedTensorOf<[I32]>:$c,
    BoolAttr:$aSigned,
    BoolAttr:$bSigned
  );
  let results = (outs RankedTensorOf<[I32]>:$d);
  let assemblyFormat = [{
    $a `,` $b `,` $c `,` `aSigned` `=` $aSigned `,` `bSigned` `=` $bSigned
    attr-dict `:` type($a) `*` type($b) `->` type($d)
  }];
  let hasVerifier = 1;
}

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
