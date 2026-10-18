#ifndef TRITON_AMDGPU_TYPES
#define TRITON_AMDGPU_TYPES

include "mlir/IR/AttrTypeBase.td"
include "TritonAMDGPUDialect.td"

def TritonAMDGPU_BufferDescType : TypeDef<TritonAMDGPU_Dialect, "BufferDesc"> {
  let mnemonic = "buffer_desc";
  let summary = "AMD buffer resource descriptor";
  let description = [{
    A buffer resource descriptor (V#): the scalar, program-uniform value that
    buffer memory instructions read from four SGPRs. It holds the fields of
    the hardware descriptor that the program chooses (base address and
    num_records); the remaining configuration bits are filled in by the
    compiler for the target. `elementType` is the type the buffer ops access
    through it; offsets of those ops count elements of this type.
  }];
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

#endif
