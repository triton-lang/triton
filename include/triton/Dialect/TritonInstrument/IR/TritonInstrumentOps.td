#ifndef TRITONINSTRUMENT_OPS
#define TRITONINSTRUMENT_OPS

include "triton/Dialect/TritonInstrument/IR/TritonInstrumentDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TTI_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonInstrument_Dialect, mnemonic, traits> {
}

def TTI_ExperimentalAssertInThreadOp : TTI_Op<"experimental_assert_in_thread", [Pure]> {
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


def TTI_ExperimentalCheckOutstandingWritesOp : TTI_Op<"experimental_check_outstanding_writes", [Pure]> {
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
}


def TTI_ExperimentalCheckOutstandingReadsOp : TTI_Op<"experimental_check_outstanding_reads", [Pure]> {
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
}


// TODO: This should not be pure I guess
// TODO: Add verifiers for the type of the scratch buffers be consistent with the dimensions of buffers/barriers
def TTI_ExperimentalMarkAsWriteOp : TTI_Op<"experimental_mark_as_write", [Pure]> {
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
}


def TTI_ExperimentalMarkAsReadOp : TTI_Op<"experimental_mark_as_read", [Pure]> {
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
}


def TTI_ExperimentalClearWriteBarrierOp : TTI_Op<"experimental_clear_write_barrier", [Pure]> {
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


def TTI_ExperimentalClearReadBarrierOp : TTI_Op<"experimental_clear_read_barrier", [Pure]> {
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
}

#endif // TRITONINSTRUMENT_OPS
