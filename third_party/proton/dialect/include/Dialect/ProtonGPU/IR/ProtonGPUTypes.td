#ifndef PROTONGPU_TYPES
#define PROTONGPU_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "proton/dialect/include/Dialect/ProtonGPU/IR/ProtonGPUDialect.td"
include "proton/dialect/include/Dialect/ProtonGPU/IR/ProtonGPUAttrDefs.td"

class PTG_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<ProtonGPU_Dialect, name, traits> {
    let mnemonic = _mnemonic;
}

def PTG_SegmentAlloc : PTG_TypeDef<"SegmentAlloc", "seg", []> {
  let summary = "location of a segment in the profiling buffer";
  let description = [{
    `proton_gpu.segbase` is a type returned by `PTG_SegmentAllocOp`.
    It is used to represent the location of a segment in the profiling buffer
    that we store our profiling events into. It is lowered to `i32` type. Here,
    `-1` means the segment location is invalid.
  }];
}

def PTG_SegmentType : PTG_TypeDef<"Segment", "segment", []> {
  let summary = "A segment in the internal buffer";

  let description = [{
    `proton_gpu.segment` is a type returned by `PTG_SegmentOp`.
  }];

  let parameters = (ins
    "int32_t":$size,
    "Attribute":$memorySpace,
    EnumParameter<GranularityAttr>:$granularity,
    ArrayRefParameter<"int32_t">:$selectIds
  );

  let assemblyFormat = [{
    `<` $size `,` $memorySpace `,` $granularity `,` `[` $selectIds `]` `>`
  }];
}

#endif
