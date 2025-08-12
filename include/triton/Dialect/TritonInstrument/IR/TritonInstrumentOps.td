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
  let arguments = (ins I1Tensor:$condition, StrAttr:$message, BoolAttr:$check_any);
  let assemblyFormat = "$condition `,` $message attr-dict `:` type($condition)";
}


def TTI_ExperimentalBufferPointersOp : TTI_Op<"experimental_buffer_pointers", [Pure]> {
  let summary = "definte an array of pointers to shared memory buffers";
  let description = [{
    Create a tensor of pointers to shared memory buffers.
  }];
  let arguments = (ins DenseI32ArrayAttr:$offsets, TT_MemTypeAttr:$memType);
  let results = (outs TT_Tensor:$result);
  let assemblyFormat = [{
    $offsets `,` $memType attr-dict `:` type($result)
  }];
}


def TTI_ExperimentalCheckWriteStateOp : TTI_Op<"experimental_check_write_state", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "check if there are outstanding writes to a buffer guarded by a mbar";
  let description = [{
    Check if the writeState tensor has non-zero value associated with the buffer.

    `writeState` is a tensor of 8b bitfields, where:
    - bit 0: 1 if the buffer is being written to
    - bit 1: 1 if the write is *not* hwPipelined

    If hwPipelined is true, shift the bitfield by 1 to check the second bit - this
    means that the error won't be triggered if another pipelined write is outstanding.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    TT_Tensor:$buffers,
    TT_PtrLike:$writeBars,
    TypeAttr:$writeBarsType,
    TT_PtrLike:$writeState,
    TypeAttr:$writeStateType,
    I1Attr:$hwPipelined,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $buf `{` $buffers `,` $writeBars `(` $writeBarsType `)` `,` $writeState `(` $writeStateType `)` `}` (`,` $pred^)? `pipelined` $hwPipelined attr-dict `:` type($buf) `,` type($buffers) `,` type($writeBars) `,` type($writeState)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalCheckReadBarriersOp : TTI_Op<"experimental_check_read_barriers", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "check if there are outstanding reads from a buffer guarded by a mbar";
  let description = [{
    Check if there are outstanding reads from a buffer guarded by a mbar.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    TT_Tensor:$buffers,
    TT_PtrLike:$readBars,
    TypeAttr:$readBarsType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $buf `{` $buffers `,` $readBars `(` $readBarsType `)` `}` (`,` $pred^)? attr-dict `:` type($buf) `,` type($buffers) `,` type($readBars)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalSetWriteStateOp : TTI_Op<"experimental_set_write_state", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "mark a buffer as being written in writeState tensor";
  let description = [{
    Mark a buffer as being written to. It is not yet tracked by a barrier, until
    `commit_write_with_barrier` is called, at which point all the buffers being written
    to are marked as tracked by the barrier.

    `writeState` is a tensor of 8b bitfields, where:
    - bit 0: 1 if the buffer is being written to
    - bit 1: 1 if the write is *not* hwPipelined

    If hwPipelined is true, the write won't trigger an error if another pipelined
    write is executed later without waiting for the barrier.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    TT_Tensor:$buffers,
    TT_PtrLike:$writeState,
    TypeAttr:$writeStateType,
    I1Attr:$hwPipelined,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $buf `{` $buffers `,` $writeState `(` $writeStateType `)` `}` (`,` $pred^)? `pipelined` $hwPipelined attr-dict `:` type($buf) `,` type($buffers) `,` type($writeState)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalCommitWriteWithBarrierOp : TTI_Op<"experimental_commit_write_with_barrier", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Mark all buffers being currently written as tracked by the barrier.";
  let description = [{
    For all buffers currently marked in writeState tensor, mark them as tracked by the mbar in
    writeBars tensor.
  }];
  let arguments = (ins
    TTG_MemDescType:$mbar,
    TT_Tensor:$barriers,
    TT_PtrLike:$writeBars,
    TypeAttr:$writeBarsType,
    TT_PtrLike:$writeState,
    TypeAttr:$writeStateType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $mbar `{` $barriers `,` $writeBars `(` $writeBarsType `)` `,` $writeState `(` $writeStateType `)` `}` (`,` $pred^)? attr-dict `:` type($mbar) `,` type($barriers) `,` type($writeBars) `,` type($writeState)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalSetReadBarrierOp : TTI_Op<"experimental_set_read_barrier", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "mark a buffer as being read from using mbar as a guard";
  let description = [{
    Mark a buffer as being read from using mbar as a guard.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    TTG_MemDescType:$mbar,
    TT_Tensor:$buffers,
    TT_Tensor:$barriers,
    TT_PtrLike:$readBars,
    TypeAttr:$readBarsType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $buf `,` $mbar `{` $buffers `,` $barriers `,` $readBars `(` $readBarsType `)` `}` (`,` $pred^)? attr-dict `:` type($buf) `,` type($mbar) `,` type($buffers) `,` type($barriers) `,` type($readBars)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalClearWriteBarrierOp : TTI_Op<"experimental_clear_write_barrier", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "clear the write state for buffers being guarded by an mbar";
  let description = [{
    Clear the write state for buffers being guarded by an mbar.
  }];
  let arguments = (ins
    TTG_MemDescType:$mbar,
    TT_Tensor:$barriers,
    TT_PtrLike:$writeBars,
    TypeAttr:$writeBarsType,
    TT_PtrLike:$writeState,
    TypeAttr:$writeStateType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $mbar `{` $barriers `,` $writeBars `(` $writeBarsType `)` `,` $writeState `(` $writeStateType `)` `}` (`,` $pred^)? attr-dict `:` type($mbar) `,` type($barriers) `,` type($writeBars) `,` type($writeState)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalClearReadBarrierOp : TTI_Op<"experimental_clear_read_barrier", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "clear the read state for buffers being guarded by an mbar";
  let description = [{
    Clear the read state for buffers being guarded by an mbar.
  }];
  let arguments = (ins
    TTG_MemDescType:$mbar,
    TT_Tensor:$barriers,
    TT_PtrLike:$readBars,
    TypeAttr:$readBarsType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $mbar `{` $barriers `,` $readBars `(` $readBarsType `)` `}` (`,` $pred^)? attr-dict `:` type($mbar) `,` type($barriers) `,` type($readBars)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalCheckBarrierWritesClearedOp : TTI_Op<"experimental_check_barrier_writes_cleared", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "verify that the barrier is not used to track any writes";
  let description = [{
    Verify that the barrier is not used to track any writes.
  }];
  let arguments = (ins
    TTG_MemDescType:$mbar,
    TT_Tensor:$barriers,
    TT_PtrLike:$writeBars,
    TypeAttr:$writeBarsType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $mbar `{` $barriers `,` $writeBars `(` $writeBarsType `)` `}` (`,` $pred^)? attr-dict `:` type($mbar) `,` type($barriers) `,` type($writeBars)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalStageAccessForCommitOp : TTI_Op<"experimental_stage_access_for_commit", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "";
  let description = [{
    For operations that use `outstanding` to track the number of outstanding commits (rather than mbarriers),
    mark the buffer as being accessed, but not commited yet, by marking it with `-1`.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    TT_Tensor:$buffers,
    TT_PtrLike:$outstandingCommits,
    TypeAttr:$outstandingCommitsType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $buf `{` $buffers `,` $outstandingCommits `(` $outstandingCommitsType `)` `}` (`,` $pred^)? attr-dict `:` type($buf) `,` type($buffers) `,` type($outstandingCommits)
  }];
  let hasVerifier = 1;
}

def TTI_ExperimentalCommitAccessesOp : TTI_Op<"experimental_commit_accesses", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Commit all the staged accesses for all the buffers.";
  let description = [{
    Commit all the staged accesses for all the buffers.
  }];
  let arguments = (ins
    TT_PtrLike:$outstandingCommits,
    TypeAttr:$outstandingCommitsType,
    Optional<I1>:$pred);
  let assemblyFormat = [{
    `{` $outstandingCommits `(` $outstandingCommitsType `)` `}` (`,` $pred^)? attr-dict `:` type($outstandingCommits)
  }];
}

def TTI_ExperimentalClearOutstandingCommitsOp : TTI_Op<"experimental_clear_outstanding_commits", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Clear all the outstanding commits more distant than `outstandingNum.";
  let description = [{
    Clear all the outstanding commits more distant than `outstandingNum` from the current thread.
  }];
  let arguments = (ins
    TT_PtrLike:$outstandingCommits,
    TypeAttr:$outstandingCommitsType,
    I32Attr:$outstandingNum,
    Optional<I1>:$pred);
  let assemblyFormat = [{
    `{` $outstandingCommits `(` $outstandingCommitsType `)` `}` `,` $outstandingNum (`,` $pred^)? attr-dict `:` type($outstandingCommits)
  }];
}

def TTI_ExperimentalCheckOutstandingCommitsOp : TTI_Op<"experimental_check_outstanding_commits", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Check if the buffer has an outstanding commit.";
  let description = [{
    Check if the buffer has an outstanding commit.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    TT_Tensor:$buffers,
    TT_PtrLike:$outstandingCommits,
    TypeAttr:$outstandingCommitsType,
    StrAttr:$pendingAccessType,
    Optional<I1>:$pred);
  let assemblyFormat = [{
    $buf `{` $buffers `,` $outstandingCommits `(` $outstandingCommitsType `)` `}` `,` $pendingAccessType (`,` $pred^)? attr-dict `:` type($buf) `,` type($buffers) `,` type($outstandingCommits)
  }];
  let hasVerifier = 1;
}

#endif // TRITONINSTRUMENT_OPS
