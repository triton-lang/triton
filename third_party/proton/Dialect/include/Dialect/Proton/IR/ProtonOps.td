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
    StrAttr: $name
  );

  let assemblyFormat = "(`start` $isStart^):(`end`)? $name attr-dict";
}

def PT_AsyncRecordOp : PT_Op<"async_record", [
  MemoryEffects<[MemRead<DefaultResource>, MemWrite<DefaultResource>]>
]> {
  let summary = "Record an asynchronous event";
  let description = [{
    Record the timestamp at which an asynchronous transaction starts or ends.
    The token identifies the corresponding endpoints even when they execute in
    different regions.
  }];
  let arguments = (ins UnitAttr:$isStart, I32:$token);
  let assemblyFormat = [{
    (`start` $isStart^):(`end`)? $token attr-dict `:` type($token)
  }];
}

def PT_AllocateAsyncTokenOp : PT_Op<"allocate_async_token", [
  MemoryEffects<[MemWrite<DefaultResource>]>
]> {
  let summary = "Allocate a named asynchronous scope token";
  let description = [{
    Allocate a static correlation identity for an asynchronous scope without
    recording a timestamp. Each token operation remains distinct even when
    names are reused; its write effect prevents common-subexpression
    elimination from merging those identities.
  }];
  let arguments = (ins StrAttr:$name);
  let results = (outs I32:$token);
  let assemblyFormat = "$name attr-dict `:` type($token)";
}

#endif // PROTON_OPS
