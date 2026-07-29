#ifndef PROTON_OPS
#define PROTON_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "proton/Dialect/include/Dialect/Proton/IR/ProtonDialect.td"
include "proton/Dialect/include/Dialect/Proton/IR/ProtonAttrDefs.td"

class PT_Op<string mnemonic, list<Trait> traits = []> :
  Op<Proton_Dialect, mnemonic, !listconcat(traits, [])> {
}

def PT_RecordOp : PT_Op<"record", [
  MemoryEffects<[MemRead<DefaultResource>, MemWrite<DefaultResource>]>
]> {
  let summary = "Record an event";

  let description = [{
    This operation annotates a region of IR where events are recorded.
    Events can be classified as hardware or software events.
    Hardware events are provided by the hardware performance counters obtained in later passes that convert Triton to target-specific IR.
    Software events are provided by the user or the compiler.

    Example:

    ```mlir
    proton.record start "name0"
    ...
    proton.record end "name0"
    ```

    Scope names cannot be reused within the same function.
  }];
  let arguments = (
    ins UnitAttr: $isStart,
    StrAttr: $name,
    Optional<AnyType>: $metric,
    OptionalAttr<StrAttr>: $metricName,
    DefaultValuedAttr<MetricValueTypeAttr, "MetricValueType::NONE">: $metricType
  );

  let assemblyFormat = [{
    (`start` $isStart^):(`end`)? $name
    (`metric` $metricName `=` $metric^ `:` type($metric))? attr-dict
  }];

  let hasVerifier = 1;
}

def PT_EventOp : PT_Op<"event", [
  MemoryEffects<[MemRead<DefaultResource>, MemWrite<DefaultResource>]>
]> {
  let summary = "Record an asynchronous event";
  let description = [{
    Record the timestamp at which an asynchronous transaction starts or ends.
    The event identifies the corresponding endpoints even when they execute in
    different regions.
  }];
  let arguments = (ins UnitAttr:$isStart, I32:$event);
  let assemblyFormat = [{
    (`start` $isStart^):(`end`)? $event attr-dict `:` type($event)
  }];
}

def PT_AllocateEventOp : PT_Op<"allocate_event", [
  MemoryEffects<[MemWrite<DefaultResource>]>
]> {
  let summary = "Allocate a named asynchronous event";
  let description = [{
    Allocate a static correlation identity for an asynchronous event without
    recording a timestamp. Each event operation remains distinct even when
    names are reused; its write effect prevents common-subexpression
    elimination from merging those identities.
  }];
  let arguments = (ins StrAttr:$name);
  let results = (outs I32:$event);
  let assemblyFormat = "$name attr-dict `:` type($event)";
}

def PT_MarkOp : PT_Op<"mark", [
  MemoryEffects<[MemRead<DefaultResource>, MemWrite<DefaultResource>]>
]> {
  let summary = "Record an instantaneous timestamp";
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "$name attr-dict";
}

#endif // PROTON_OPS
