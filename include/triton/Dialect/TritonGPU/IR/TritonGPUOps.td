#ifndef TRITONGPU_OPS
#define TRITONGPU_OPS

include "TritonGPUDialect.td"
// include ""

def TTG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonGPU_Dialect, mnemonic, traits>;

def TTG_ConvertLayoutOp : TTG_Op<"convert_layout",
                                 [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "convert layout";

  let arguments = (ins TT_Tensor:$src);

  let results = (outs TT_Tensor:$result);
}

def TTG_AsyncWaitOp : TTG_Op<"async_wait"> {
  let summary = "async wait";

  let arguments = (ins I32Attr:$num);
}

// def TTG_CopyAsyncOp : TTG_Op<"copy_async"> {}

#endif
