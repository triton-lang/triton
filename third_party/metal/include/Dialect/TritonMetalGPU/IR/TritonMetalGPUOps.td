#ifndef TRITON_METALGPU_OPS
#define TRITON_METALGPU_OPS

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

include "TritonMetalGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"


class TT_MetalGPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<TritonMetalGPU_Dialect, mnemonic, !listconcat(traits, [])>;

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

def SimdgroupMMAOp : TT_MetalGPU_Op<"simdgroup_mma", [Pure]> {
  let summary = "simdgroup matrix multiply-accumulate";
  let description = [{
    This op represents a simdgroup MMA instruction.
  }];

  let arguments = (ins
    Arg<TT_Ptr, "Shared memory pointer", [MemRead<SharedMemory>, MemWrite<SharedMemory>]>:$a_ptr,
    Arg<TT_Ptr, "Shared memory pointer", [MemRead<SharedMemory>, MemWrite<SharedMemory>]>:$b_ptr,
    TT_FpIntTensor:$c,
    I32:$strideA,
    I32:$strideB
  );

  let results = (outs TT_FpIntTensor:$d);
  
  let assemblyFormat = [{
    $a_ptr `,` $b_ptr `,` $c `,` $strideA `,` $strideB
    attr-dict `:` functional-type(operands, $d)
  }];
}

def SimdgroupAsyncCopyOp : TT_MetalGPU_Op<"simdgroup_async_copy"> {
  let summary = "async copy from global memory to threadgroup memory";
  let description = [{
    This op represents an async copy instruction from global memory to threadgroup memory of the input block.
  }];

  let arguments = (ins
    Arg<TT_PtrTensor, "Global memory pointer tensor", [MemRead<GlobalMemory>]>:$src,
    Arg<TTG_MemDescType, "", [MemWrite<SharedMemory>]>:$dst
  );

  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` type($src) `->` qualified(type($dst))
  }];
}

def SimdgroupStoreOp : TT_MetalGPU_Op<"simdgroup_store", [
  TypesMatchWith<"value type matches ptr type", "ptr", "value",
                 "getPointeeType($_self)">,
  TypesMatchWith<"mask type matches ptr type", "ptr", "mask",
                 "getI1SameShape(getPointeeType($_self))",
                 "($_op.getOperands().size() <= 2) || std::equal_to<>()">
]> {
  let summary = "store simdgroup tensor in global memory";
  let description = [{
    This op represents an store instruction of a simdgroup tensor in global memory.
  }];

  let arguments = (ins
    Arg<TT_PtrLike, "", [MemWrite<GlobalMemory>]>:$ptr,
    TT_Type:$value,
    Optional<TT_BoolLike>:$mask
  );

  let builders = [
    // A tensor of pointers or a pointer to a scalar
    OpBuilder<(ins "Value":$ptr, "Value":$value)>
  ];
  
  let assemblyFormat = [{
    $ptr `,` $value (`,` $mask^)?
    attr-dict `:` type($ptr)
  }];
}

#endif // TRITON_METALGPU_OPS