#ifndef PROTONGPU_TYPES
#define PROTONGPU_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "proton/dialect/include/Dialect/ProtonGPU/IR/ProtonGPUDialect.td"

class PTG_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<ProtonGPU_Dialect, name, traits> {
    let mnemonic = _mnemonic;
}

def PTG_SegmentBase : PTG_TypeDef<"SegmentBase", "seg", []> {
  let summary = "location of a segment in the profiling buffer";
  let description = [{
    `proton_gpu.segbase` is a type returned by `PTG_SegmentBaseOp`.
    It is used to represent the location of a segment in the profiling buffer
    that we store our profiling events into. It is lowered to `i32` type. Here,
    `-1` means the segment location is invalid.
  }];
}

#endif
