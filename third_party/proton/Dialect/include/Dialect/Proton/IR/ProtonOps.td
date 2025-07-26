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

#endif // PROTON_OPS
