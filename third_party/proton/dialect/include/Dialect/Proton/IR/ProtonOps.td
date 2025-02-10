#ifndef PROTON_OPS
#define PROTON_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "proton/dialect/include/Dialect/Proton/IR/ProtonDialect.td"
include "proton/dialect/include/Dialect/Proton/IR/ProtonAttrDefs.td"

class PT_Op<string mnemonic, list<Trait> traits = []> :
  Op<Proton_Dialect, mnemonic, !listconcat(traits, [])> {
}

def PT_InitScopeOp : PT_Op<"init_scope", [Pure]> {
  let summary = "Initialize a scope";

  let description = [{
    This operation initializes a scope with the given name and returns a unique id for the scope.

    Example:

    ```mlir
    %scope0 = proton.init_scope "name0" : i32
    ```
  }];

  let arguments = (
    ins StrAttr: $scopeName
  );
  let results = (outs I32 : $scopeId);

  let assemblyFormat = "$scopeName attr-dict `:` type($scopeId) ";
  // hasVerifier = 1; verify that (1) each scope is used twice, and (2) the use ops are record like ops
}

def PT_RecordOp : PT_Op<"record", [
  MemoryEffects<[MemRead<DefaultResource>, MemWrite<DefaultResource>]>
]> {
  let summary = "Record an event";

  let description = [{
    This operation records events of a particular metric.

    Example:

    ```mlir
    proton.record start %scope0
    ...
    proton.record end %scope0
    ```
  }];
  let arguments = (
    ins UnitAttr: $isStart,
    I32: $scopeId
  );

  let assemblyFormat = "(`start` $isStart^):(`end`)? $scopeId attr-dict";
}

#endif // PROTON_OPS
