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

def PTG_SegmentType : PTG_TypeDef<"Segment", "segment", []> {
  let summary = "A segment in the internal buffer";
  let description = [{ `proton_gpu.segment` is a type returned by `PTG_SegmentOp`. }];

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
