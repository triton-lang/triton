#ifndef TRITONINSTRUMENT_OPS
#define TRITONINSTRUMENT_OPS

include "triton/Dialect/TritonInstrument/IR/TritonInstrumentDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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


def TTI_ExperimentalSharedBufferPointersOp : TTI_Op<"experimental_shared_buffer_pointers", [Pure]> {
  let summary = "definte an array of pointers to shared memory buffers";
  let description = [{
    Create a tensor of pointers to shared memory buffers.
  }];
  let arguments = (ins DenseI32ArrayAttr:$offsets);
  let results = (outs TT_Tensor:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}


def TTI_ExperimentalCheckOutstandingWritesOp : TTI_Op<"experimental_check_outstanding_writes", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "check if there are outstanding writes to a buffer guarded by a mbar";
  let description = [{
    Check if there are outstanding writes to a buffer guarded by a mbar.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    TT_Tensor:$buffers,
    TT_PtrLike:$writeBars,
    TypeAttr:$writeBarsType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $buf `{` $buffers `,` $writeBars `(` $writeBarsType `)` `}` (`,` $pred^)? attr-dict `:` type($buf) `,` type($buffers) `,` type($writeBars)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalCheckOutstandingReadsOp : TTI_Op<"experimental_check_outstanding_reads", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
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


def TTI_ExperimentalMarkAsWriteOp : TTI_Op<"experimental_mark_as_write", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "mark a buffer as being written to using mbar as a guard";
  let description = [{
    Mark a buffer as being written to using mbar as a guard.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    TTG_MemDescType:$mbar,
    TT_Tensor:$buffers,
    TT_PtrLike:$writeBars,
    TypeAttr:$writeBarsType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $buf `,` $mbar `{` $buffers `,` $writeBars `(` $writeBarsType `)` `}` (`,` $pred^)? attr-dict `:` type($buf) `,` type($mbar) `,` type($buffers) `,` type($writeBars)
  }];
  let hasVerifier = 1;
}


def TTI_ExperimentalMarkAsReadOp : TTI_Op<"experimental_mark_as_read", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
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
    TT_PtrLike:$writeBars,
    TypeAttr:$writeBarsType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $mbar `{` $writeBars `(` $writeBarsType `)` `}` (`,` $pred^)? attr-dict `:` type($mbar) `,` type($writeBars)
  }];
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

def TTI_ExperimentalStageWriteForCommitOp : TTI_Op<"experimental_stage_write_for_commit", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Preapre to an async copy of a buffer. Staged until commit_group is called.";
  let description = [{
    Preapre to an async copy of a buffer. Staged until commit_group is called. The implementation will write `-1` to the
    `write_commits` tensor under the indices corresponding to the buffer.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    TT_Tensor:$buffers,
    TT_PtrLike:$writeCommits,
    TypeAttr:$writeCommitsType,
    Optional<I1>:$pred
  );
  let assemblyFormat = [{
    $buf `{` $buffers `,` $writeCommits `(` $writeCommitsType `)` `}` (`,` $pred^)? attr-dict `:` type($buf) `,` type($buffers) `,` type($writeCommits)
  }];
  // let hasVerifier = 1;
}

def TTI_ExperimentalCommitWritesOp : TTI_Op<"experimental_commit_writes", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Commit all the staged writes for all the buffers.";
  let description = [{
    Commit all the staged writes for all the buffers.
  }];
  let arguments = (ins
    TT_PtrLike:$writeCommits,
    TypeAttr:$writeCommitsType,
    Optional<I1>:$pred);
  let assemblyFormat = [{
    `{` $writeCommits `(` $writeCommitsType `)` `}` (`,` $pred^)? attr-dict `:` type($writeCommits)
  }];
  // let hasVerifier = 1;
}

def TTI_ExperimentalClearWriteCommitsOp : TTI_Op<"experimental_clear_write_commits", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Clear all the write commits more distant than `outstandingNum.";
  let description = [{
    Clear all the write commits more distant than `outstandingNum` from the current thread.
  }];
  let arguments = (ins
    TT_PtrLike:$writeCommits,
    TypeAttr:$writeCommitsType,
    I32Attr:$outstandingNum,
    Optional<I1>:$pred);
  let assemblyFormat = [{
    `{` $writeCommits `(` $writeCommitsType `)` `}` `,` $outstandingNum (`,` $pred^)? attr-dict `:` type($writeCommits)
  }];
  // let hasVerifier = 1;
}

def TTI_ExperimentalCheckWriteCommitOp : TTI_Op<"experimental_check_write_commit", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Check if the buffer has an outstanding write commit.";
  let description = [{
    Check if the buffer has an outstanding write commit.
  }];
  let arguments = (ins
    TTG_MemDescType:$buf,
    TT_Tensor:$buffers,
    TT_PtrLike:$writeCommits,
    TypeAttr:$writeCommitsType,
    Optional<I1>:$pred);
  let assemblyFormat = [{
    $buf `{` $buffers `,` $writeCommits `(` $writeCommitsType `)` `}` (`,` $pred^)? attr-dict `:` type($buf) `,` type($buffers) `,` type($writeCommits)
  }];
  // let hasVerifier = 1;
}

#endif // TRITONINSTRUMENT_OPS
