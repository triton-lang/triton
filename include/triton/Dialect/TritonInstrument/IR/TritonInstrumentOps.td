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

// Define an array of pointers to shared memory buffers
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


def TTI_ExperimentalCheckAsyncWriteWithMbarSharedOp : TTI_Op<"experimental_check_async_write_with_mbar_shared", [Pure]> {
  let summary = "Check if writing to a buffer guarded is valid";
  let description = [{
    Check if writing to a shared memory buffer guarded is valid.
    Update the buffer state and assert if the buffer is being read or written.
  }];
  let arguments = (ins
    TTG_MemDescType:$buffer,
    TTG_MemDescType:$mbar,
    TT_Tensor:$buffers,
    TT_Tensor:$states,
    TT_Tensor:$barriers
  );
  let results = (outs
    TT_Tensor:$outStates,
    TT_Tensor:$outBarriers
  );
  let assemblyFormat = [{
    $buffer `,` $mbar `{` $buffers `,` $states `,` $barriers `}` attr-dict `:` type($buffer) `,` type($mbar) `,` type($buffers) `,` type($states) `,` type($barriers) `->` type($outStates) `,` type($outBarriers)
  }];
  let builders = [
    OpBuilder<(ins "Value":$buffer, "Value":$mbar, "Value":$buffers, "Value":$states, "Value":$barriers),[{
      build($_builder, $_state, {states.getType(), barriers.getType()}, buffer, mbar, buffers, states, barriers);
    }]>
  ];
}


def TTI_ExperimentalCheckReadSharedOp : TTI_Op<"experimental_check_read_shared", [Pure]> {
  let summary = "Check if reading from a shared memory buffer is valid";
  let description = [{
    Check if reading from a shared memory buffer is valid.
  }];
  let arguments = (ins
    TTG_MemDescType:$buffer,
    TT_Tensor:$buffers,
    TT_Tensor:$states,
    TT_Tensor:$barriers
  );
  let results = (outs
    TT_Tensor:$outStates,
    TT_Tensor:$outBarriers
  );
  let assemblyFormat = [{
    $buffer `{` $buffers `,` $states `,` $barriers `}` attr-dict `:` type($buffer) `,` type($buffers) `,` type($states) `,` type($barriers) `->` type($outStates) `,` type($outBarriers)
  }];
  let builders = [
    OpBuilder<(ins "Value":$buffer, "Value":$buffers, "Value":$states, "Value":$barriers),
    [{
      build($_builder, $_state, {states.getType(), barriers.getType()}, buffer, buffers, states, barriers);
    }]>
  ];
}


def TTI_ExperimentalCheckWaitMbarOp : TTI_Op<"experimental_check_wait_mbar", [Pure]> {
  let summary = "check if waiting on a mbar is valid and update the barrier state";
  let description = [{
    Check if waiting on a mbar is valid and update the barrier state.
  }];
  let arguments = (ins
    TTG_MemDescType:$mbar,
    TT_Tensor:$barriers,
    TT_Tensor:$states
    );

  let results = (outs
    TT_Tensor:$outStates,
    TT_Tensor:$outBarriers);

  let assemblyFormat = [{
    $mbar `{` $states `,` $barriers `}` attr-dict `:` type($mbar) `,` type($states) `,` type($barriers) `->` type($outStates) `,` type($outBarriers)
  }];

  let builders = [
    OpBuilder<(ins "Value":$mbar, "Value":$barriers, "Value":$states),
    [{
      build($_builder, $_state, {states.getType(), barriers.getType()}, mbar, barriers, states);
    }]>];

}

#endif // TRITONINSTRUMENT_OPS
