#ifndef TRITON_METALGPU_OPS
#define TRITON_METALGPU_OPS

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

include "TritonMetalGPUDialect.td"
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

def SimdgroupAsyncCopy : TT_MetalGPU_Op<"simdgroup_async_copy"> {
  let summary = "async copy from global memory to threadgroup memory";
  let description = [{
    This op represents an async copy instruction from global memory to threadgroup memory of the input block.
  }];

  let arguments = (ins
    Arg<TT_Ptr, "Global memory pointer", [MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>:$base_ptr,
    I32:$stride,
    Arg<TT_Ptr, "Shared memory pointer", [MemRead<SharedMemory>, MemWrite<SharedMemory>]>:$dest_ptr
  );
  
  let assemblyFormat = [{
    $base_ptr `,` $stride `,` $dest_ptr
    attr-dict `:` type(operands)
  }];
}

#endif // TRITON_METALGPU_OPS