#ifndef PROTONGPU_TYPES
#define PROTONGPU_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "proton/Dialect/include/Dialect/ProtonGPU/IR/ProtonGPUDialect.td"
include "proton/Dialect/include/Dialect/ProtonGPU/IR/ProtonGPUAttrDefs.td"

class PTG_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<ProtonGPU_Dialect, name, traits> {
    let mnemonic = _mnemonic;
}

def PTG_SegmentType : PTG_TypeDef<"Segment", "segment", []> {
  let summary = "A segment in the internal buffer";
  let description = [{
    The `proton_gpu.segment` type represents a segment returned by `PTG_SegmentOp`.

    Each segment is private to a profiling unit as defined by the `granularity` attribute.
    The selected segments, specified by the `selectIds` attribute, collectively total `nBytes` bytes.

    When lowered to LLVM, a segment becomes a struct containing:
    - `base`: pointer to the start of the internal buffer
    - `segmentBase`: pointer to each segment's start in the internal buffer
    - `indexPtr`: pointer to the current index within the segment

    The segment can reside in global memory or shared memory depending on the `memorySpace` attribute.
  }];

  let parameters = (ins
    "int32_t":$nBytes,
    "Attribute":$memorySpace,
    EnumParameter<GranularityAttr>:$granularity,
    OptionalArrayRefParameter<"int32_t">:$selectIds
  );

  let assemblyFormat = [{
    `<` $nBytes `,` $memorySpace `,` $granularity (`,` `[` $selectIds^ `]`)?  `>`
  }];
}

#endif
