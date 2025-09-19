#ifndef TRITONINSTRUMENT_OPS
#define TRITONINSTRUMENT_OPS

include "triton/Dialect/TritonInstrument/IR/TritonInstrumentDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton/Dialect/TritonInstrument/IR/TritonInstrumentAttrDefs.td"

// Concurrency Sanitizer data structures:
// ConSan keeps auxilary data requied for tracking memory accesses in tensors.
// These tensors are stored as a distributed tensor or in global scratch memory.
//
// Name            | Storage | Rank/Type       | Description
// ----------------|---------|-----------------|------------
// buffers         | tensor  | <B x i64>       | Base pointers of all (sub)buffers
// barriers        | tensor  | <K x i64>       | Pointers to all individual mbarriers
// writeVisibility | scratch | <B x i64>       | Per-buffer thread-visibility bitmask (bit i => thread i visible)
// readVisibility  | scratch | <B x T x i64>   | Per-buffer, per-thread visibility lanes (row-updated; values are bitmasks)
// writeTracking   | scratch | <B x K x i8>    | Map buffers -> barriers that track writes
// readTracking    | scratch | <B x K x i64>   | Map buffers -> barriers that track reads
// outstandingCommits
//   (async/wgmma) | scratch | <B x T x i8>    | Number of outstanding commits per buffer/thread (2D replaces prior 1D)

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


def TTI_ExperimentalSetWriteVisibilityOp : TTI_Op<"experimental_set_write_visibility", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "set the write visibility for a buffer";
  let description = [{
    Set the write visibility for a buffer. Marks the buffer as visible to the threads set in threadMask.
    Clears out any other threads from the visibility bitmask. We know
    this is safe because there cannot be outstanding writes to this buffer at this point.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    I64Attr:$threadMask,
    TT_Tensor:$buffers,
    TT_PtrLike:$writeVisibility,
    TypeAttr:$writeVisibilityType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $buf `,` $threadMask `{` $buffers `,` $writeVisibility `(` $writeVisibilityType `)` `}` (`,` $pred^)? attr-dict `:` type($buf) `,` type($buffers) `,` type($writeVisibility)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalClearWriteTrackingOp : TTI_Op<"experimental_clear_write_tracking", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "clear the write tracking for a buffer";
  let description = [{
    Clear all the information about threads writing to a buffer.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    TT_Tensor:$buffers,
    TT_PtrLike:$writeTracking,
    TypeAttr:$writeTrackingType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $buf `{` $buffers `,` $writeTracking `(` $writeTrackingType `)` `}` (`,` $pred^)? attr-dict `:` type($buf) `,` type($buffers) `,` type($writeTracking)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalClearReadVisibilityOp : TTI_Op<"experimental_clear_read_visibility", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "clear the read visibility for a buffer";
  let description = [{
    Clear the read visibility for a buffer.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    TT_Tensor:$buffers,
    TT_PtrLike:$readVisibility,
    TypeAttr:$readVisibilityType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $buf `{` $buffers `,` $readVisibility `(` $readVisibilityType `)` `}` (`,` $pred^)? attr-dict `:` type($buf) `,` type($buffers) `,` type($readVisibility)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalClearReadTrackingOp : TTI_Op<"experimental_clear_read_tracking", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "clear the read tracking for a buffer";
  let description = [{
    Clear the read tracking for a buffer.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    TT_Tensor:$buffers,
    TT_PtrLike:$readTracking,
    TypeAttr:$readTrackingType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $buf `{` $buffers `,` $readTracking `(` $readTrackingType `)` `}` (`,` $pred^)? attr-dict `:` type($buf) `,` type($buffers) `,` type($readTracking)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalSetReadVisibilityOp : TTI_Op<"experimental_set_read_visibility", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "set the read visibility for a buffer";
  let description = [{
    Add the threads set in threadMask to the read visibility bitmask for the buffer.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    I64Attr:$threadMask,
    TT_Tensor:$buffers,
    TT_PtrLike:$readVisibility,
    TypeAttr:$readVisibilityType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $buf `,` $threadMask `{` $buffers `,` $readVisibility `(` $readVisibilityType `)` `}` (`,` $pred^)? attr-dict `:` type($buf) `,` type($buffers) `,` type($readVisibility)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalTrackVisibleWritesOp : TTI_Op<"experimental_track_visible_writes", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "start tracking visible writes to a buffer";
  let description = [{
    For all buffers that are marked as visible to the thread, start tracking them with a barrier.
  }];
  let arguments = (ins
    TTG_MemDescType:$mbar,
    I32Attr:$thread,
    TT_Tensor:$barriers,
    TT_PtrLike:$writeVisibility,
    TypeAttr:$writeVisibilityType,
    TT_PtrLike:$writeTracking,
    TypeAttr:$writeTrackingType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $mbar `,` $thread `{` $barriers `,` $writeVisibility `(` $writeVisibilityType `)` `,` $writeTracking `(` $writeTrackingType `)` `}` (`,` $pred^)? attr-dict `:` type($mbar) `,` type($barriers) `,` type($writeVisibility) `,` type($writeTracking)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalTrackVisibleReadsOp : TTI_Op<"experimental_track_visible_reads", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "start tracking visible reads from a buffer";
  let description = [{
    For all buffers that are marked as visible to the thread, start tracking them with a barrier.
  }];
  let arguments = (ins
    TTG_MemDescType:$mbar,
    I32Attr:$thread,
    TT_Tensor:$barriers,
    TT_PtrLike:$readVisibility,
    TypeAttr:$readVisibilityType,
    TT_PtrLike:$readTracking,
    TypeAttr:$readTrackingType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $mbar `,` $thread `{` $barriers `,` $readVisibility `(` $readVisibilityType `)` `,` $readTracking `(` $readTrackingType `)` `}` (`,` $pred^)? attr-dict `:` type($mbar) `,` type($barriers) `,` type($readVisibility) `,` type($readTracking)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalTransferVisibleWritesOp : TTI_Op<"experimental_transfer_visible_writes", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "transfer all the writes tracked by a barrier to visible by thread";
  let description = [{
    Transfer all the writes tracked by a barrier to visible by threads set in threadMask.
  }];
  let arguments = (ins
    TTG_MemDescType:$mbar,
    I64Attr:$threadMask,
    TT_Tensor:$barriers,
    TT_PtrLike:$writeVisibility,
    TypeAttr:$writeVisibilityType,
    TT_PtrLike:$writeTracking,
    TypeAttr:$writeTrackingType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $mbar `,` $threadMask `{` $barriers `,` $writeVisibility `(` $writeVisibilityType `)` `,` $writeTracking `(` $writeTrackingType `)` `}` (`,` $pred^)? attr-dict `:` type($mbar) `,` type($barriers) `,` type($writeVisibility) `,` type($writeTracking)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalTransferVisibleReadsOp : TTI_Op<"experimental_transfer_visible_reads", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "transfer all the reads tracked by a barrier to visible by thread";
  let description = [{
    Transfer all the reads tracked by a barrier to visible by threads set in threadMask.
  }];
  let arguments = (ins
    TTG_MemDescType:$mbar,
    I64Attr:$threadMask,
    TT_Tensor:$barriers,
    TT_PtrLike:$readVisibility,
    TypeAttr:$readVisibilityType,
    TT_PtrLike:$readTracking,
    TypeAttr:$readTrackingType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $mbar `,` $threadMask `{` $barriers `,` $readVisibility `(` $readVisibilityType `)` `,` $readTracking `(` $readTrackingType `)` `}` (`,` $pred^)? attr-dict `:` type($mbar) `,` type($barriers) `,` type($readVisibility) `,` type($readTracking)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalVerifyWriteVisibilityOp : TTI_Op<"experimental_verify_write_visibility", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "verify that the buffer write is visible to the thread";
  let description = [{
    Verify that the buffer write is visible to the thread, or no threads are writing to the buffer.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    I32Attr:$thread,
    TT_Tensor:$buffers,
    TT_PtrLike:$writeVisibility,
    TypeAttr:$writeVisibilityType,
    StrAttr:$operandName,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $buf `,` $thread `{` $buffers `,` $writeVisibility `(` $writeVisibilityType `)` `}` `,` $operandName (`,` $pred^)? attr-dict `:` type($buf) `,` type($buffers) `,` type($writeVisibility)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalVerifyReadVisibilityOp : TTI_Op<"experimental_verify_read_visibility", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "verify that all the reads from a buffer are visible to the thread";
  let description = [{
    Verify that all the reads from a buffer are visible to the thread.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    I32Attr:$thread,
    TT_Tensor:$buffers,
    TT_PtrLike:$readVisibility,
    TypeAttr:$readVisibilityType,
    StrAttr:$operandName,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $buf `,` $thread `{` $buffers `,` $readVisibility `(` $readVisibilityType `)` `}` `,` $operandName (`,` $pred^)? attr-dict `:` type($buf) `,` type($buffers) `,` type($readVisibility)
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
    I32Attr:$thread,
    TT_Tensor:$buffers,
    TT_PtrLike:$outstandingCommits,
    TypeAttr:$outstandingCommitsType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $buf `,` $thread `{` $buffers `,` $outstandingCommits `(` $outstandingCommitsType `)` `}` (`,` $pred^)? attr-dict `:` type($buf) `,` type($buffers) `,` type($outstandingCommits)
  }];
  let hasVerifier = 1;
}

def TTI_ExperimentalCommitAccessesOp : TTI_Op<"experimental_commit_accesses", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Commit all the staged accesses for all the buffers.";
  let description = [{
    Increment the value in outstandingCommits tensor for each entry greater than 0.
    Change all the `-1` entries in outstandingCommits tensor to 1, signifying 1 outstanding commit.
  }];
  let arguments = (ins
    I32Attr:$thread,
    TT_PtrLike:$outstandingCommits,
    TypeAttr:$outstandingCommitsType,
    Optional<I1>:$pred);
  let assemblyFormat = [{
    $thread `{` $outstandingCommits `(` $outstandingCommitsType `)` `}` (`,` $pred^)? attr-dict `:` type($outstandingCommits)
  }];
}

def TTI_ExperimentalClearOutstandingCommitsTransferWritesOp : TTI_Op<"experimental_clear_outstanding_commits_transfer_writes", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Clear distant outstanding commits and set write visibility for current thread.";
  let description = [{
    For all rows where outstanding commits are more distant than `outstandingNum` from the current thread,
    clear those entries and mark the corresponding buffer as visible in the write visibility table for the threads set in threadMask.
  }];
  let arguments = (ins
    I32Attr:$thread,
    I64Attr:$transferThreadMask,
    I32Attr:$outstandingNum,
    TT_PtrLike:$outstandingCommits,
    TypeAttr:$outstandingCommitsType,
    TT_PtrLike:$writeVisibility,
    TypeAttr:$writeVisibilityType,
    Optional<I1>:$pred);
  let assemblyFormat = [{
    $thread `,` $transferThreadMask `{` $outstandingCommits `(` $outstandingCommitsType `)` `}` `,` $writeVisibility `(` $writeVisibilityType `)` (`,` $pred^)? attr-dict `:` type($outstandingCommits) `,` type($writeVisibility)
  }];
}

def TTI_ExperimentalClearOutstandingCommitsTransferReadsOp : TTI_Op<"experimental_clear_outstanding_commits_transfer_reads", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Clear distant outstanding commits and set read visibility for current thread.";
  let description = [{
    For all rows where outstanding commits are more distant than `outstandingNum` from the current thread,
    clear those entries and mark the corresponding buffer as visible in the read visibility table for the threads set in threadMask.
  }];
  let arguments = (ins
    I32Attr:$thread,
    I64Attr:$transferThreadMask,
    I32Attr:$outstandingNum,
    TT_PtrLike:$outstandingCommits,
    TypeAttr:$outstandingCommitsType,
    TT_PtrLike:$readVisibility,
    TypeAttr:$readVisibilityType,
    Optional<I1>:$pred);
  let assemblyFormat = [{
    $thread `,` $transferThreadMask `{` $outstandingCommits `(` $outstandingCommitsType `)` `}` `,` $readVisibility `(` $readVisibilityType `)` (`,` $pred^)? attr-dict `:` type($outstandingCommits) `,` type($readVisibility)
  }];
}

def TTI_ExperimentalCheckOutstandingCommitsOp : TTI_Op<"experimental_check_outstanding_commits", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Check if the buffer has an outstanding commit.";
  let description = [{
    Verify that the entry corresponding to the buffer in outstandingCommits tensor is 0.
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
