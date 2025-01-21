// Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TRITONNVIDIAGPU_OPS
#define TRITONNVIDIAGPU_OPS

include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUDialect.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypeInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

class TTNG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonNvidiaGPU_Dialect, mnemonic,
       !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

def TTNG_FenceAsyncSharedOp : TTNG_Op<"fence_async_shared"> {
  let arguments = (ins BoolAttr:$bCluster);

  let summary = "fence proxy async";

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 90;
    }
  }];
}

def TTNG_ClusterArriveOp : TTNG_Op<"cluster_arrive", []> {
  let arguments = (ins I1Attr:$relaxed);
  let assemblyFormat = "attr-dict";
}

def TTNG_ClusterWaitOp : TTNG_Op<"cluster_wait", []> {
  let assemblyFormat = "attr-dict";
}

//
// WarpGroupDot Op
//
def TTNG_WarpGroupDotOp : TTNG_Op<"warp_group_dot", [DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                                     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                                                     DotLike,
                                                     TypesMatchWith<"result's type matches accumulator's type",
                                                                     "d", "c", "$_self">]> {
    let summary = "warp group dot";

    let description = [{
        $d = matrix_multiply($a, $b) + $c. For docs on InputPrecisionAttr, see TT_DotOp
    }];

    let arguments = (ins TTG_TensorOrMemDesc:$a,
                         TTG_TensorOrMemDesc:$b,
                         TT_FpIntTensor:$c,
                         Optional<I1>:$useC,
                         DefaultValuedAttr<TT_InputPrecisionAttr, "::mlir::triton::InputPrecision::IEEE">:$inputPrecision,
                         DefaultValuedAttr<I32Attr, "0">:$maxNumImpreciseAcc,
                         DefaultValuedAttr<BoolAttr, "false">:$isAsync);

    let results = (outs TT_FpIntTensor:$d);

    let assemblyFormat = "$a`,` $b`,` $c (`,` $useC^)? attr-dict `:` type($a) `*` type($b) `->` type($d)";

    let extraClassDeclaration = [{
      bool needsPartialAccumulator();
    }];
}

def TTNG_WarpGroupDotWaitOp : TTNG_Op<"warp_group_dot_wait", [DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                                              AllTypesMatch<["inputs", "outputs"]>]> {
  let summary = "warp group dot wait";
  let arguments = (ins Variadic<TTG_TensorOrMemDesc>:$inputs, I32Attr:$pendings);
  let results = (outs Variadic<TTG_TensorOrMemDesc>:$outputs);
  let description = [{
    Waits until there are $pendings or fewer outstanding async dot operations.

    $inputs must be the tensors corresponding to the async dot ops that we're
    waiting on.  For example, if there are N pending async dot ops and we call
    `warp_group_dot_wait 1`, then $inputs must be the result of the first dot op.
  }];

  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
}

def TTNG_InitBarrierOp : TTNG_Op<"init_barrier", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "Initialize a barrier in the given shared memory allocation.";

    let description = [{
        Initializes a shared memory allocation with mbarrier information.
        `alloc` is a descriptor to the shared memory allocation. `count` is the
        number of arrives expected by the barrier.

        This lowers to PTX mbarrier.init.shared::cta.b64.
    }];

    let hasVerifier = 1;
    let arguments = (ins TTG_MemDescType:$alloc,
                         I32Attr:$count);
    let assemblyFormat = "$alloc `,` $count attr-dict `:` qualified(type($alloc))";
}

def TTNG_InvalBarrierOp : TTNG_Op<"inval_barrier", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "Invalidate a barrier allocation.";

    let description = [{
      Invalidate a barrier allocation so that it can be re-used. According to PTX
      spec this has to be done before any reuse of the memory used by mbarrier.

      https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval
    }];

    let hasVerifier = 1;
    let arguments = (ins TTG_MemDescType:$alloc);
    let assemblyFormat = "$alloc attr-dict `:` qualified(type($alloc))";
}

def TTNG_BarrierExpectOp : TTNG_Op<"barrier_expect", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Signal a barrier of an expected number of bytes to be copied.";

  let description = [{
    This signal the barrier that `size` bytes are expected to be copied. The
    associated barrier wait will block until the expected number of bytes are copied.
  }];

  let hasVerifier = 1;
  let arguments = (
    ins TTG_MemDescType:$alloc,
    I32Attr:$size,
    I1:$pred
  );

  let assemblyFormat = [{
    $alloc `,` $size attr-dict `,` $pred `:` qualified(type($alloc))
  }];
}

def TTNG_WaitBarrierOp : TTNG_Op<"wait_barrier", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "wait until the mbarrier phase completes.";

    let description = [{
      Blocks the program progress until the mbarrier object in `alloc` completes
      its current phase.

      This lowers a waitloop using PTX instruction
      mbarrier.try_wait.parity.shared.b64.

      The barrier behavior is described here:
      https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms
    }];

    let hasVerifier = 1;
    let arguments = (ins TTG_MemDescType:$alloc,
                         I32:$phase);
    let assemblyFormat = "$alloc `,` $phase attr-dict `:` qualified(type($alloc))";
}

def TTNG_TensorDescToTMAPtrOp : TTNG_Op<"tensor_desc_to_tma_ptr", [Pure]> {
  let summary = "Convert tensor descriptor to pointer to tma descriptor";

  let arguments = (ins TT_TensorDescType:$desc);
  let results = (outs TT_Ptr:$ptr);

  let assemblyFormat = [{
    $desc attr-dict `:` qualified(type($desc)) `to` qualified(type($ptr))
  }];

  let builders = [
    OpBuilder<(ins "Value":$desc), [{
      auto ptrTy = triton::PointerType::get($_builder.getI8Type(), 1);
      build($_builder, $_state, ptrTy, desc);
    }]>
  ];

  let hasCanonicalizeMethod = 1;
}


def TTNG_AsyncTMACopyGlobalToLocalOp : TTNG_Op<"async_tma_copy_global_to_local", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "copy data based on descriptor from global memory to local memory asynchronously";

  let description = [{
    This operation copies data from global memory to local memory
    asynchronously.  This is analogue to tt.load except the data are copied to
    local memory pointed by the memory descriptor instead of a distributed
    tensor. The data copied depends on the global memory descriptor pointed to
    by `desc_ptr`.
  }];

  let hasVerifier = 1;
  let arguments = (
    ins TT_PtrType:$desc_ptr,
    Variadic<I32>:$coord,
    TTG_MemDescType:$barrier,
    TTG_MemDescType:$result,
    I1:$pred,
    DefaultValuedAttr<TT_CacheModifierAttr, "triton::CacheModifier::NONE">:$cache,
    DefaultValuedAttr<TT_EvictionPolicyAttr, "triton::EvictionPolicy::NORMAL">:$evict,
    DefaultValuedAttr<BoolAttr, "false">:$isVolatile
  );

  let assemblyFormat = [{
    $desc_ptr `[` $coord `]` $result `,` $barrier `,` $pred
    oilist(`cacheModifier` `=` $cache | `evictionPolicy` `=` $evict)
    attr-dict `:` qualified(type($desc_ptr)) `,` qualified(type($barrier)) `->` qualified(type($result))
  }];
}

def TTNG_AsyncTMACopyLocalToGlobalOp : TTNG_Op<"async_tma_copy_local_to_global", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "copy data based on descriptor from local memory to global memory asynchronously";

  let description = [{
    This operation copies data from local memory to global memory
    asynchronously.  This is analogue to tt.store except the data are copied from
    local memory pointed by the memory descriptor instead of a distributed
    tensor. The data copied depends on the global memory descriptor pointed to
    by `desc_ptr`.
  }];

  let arguments = (
    ins TT_PtrType:$desc_ptr,
    Variadic<I32>:$coord,
    TTG_MemDescType:$src);

  let assemblyFormat = [{
    $desc_ptr `[` $coord `]` $src
    attr-dict `:` qualified(type($desc_ptr)) `,` qualified(type($src))
  }];
}

def TTNG_TMAStoreWait : TTNG_Op<"async_tma_store_wait"> {
  let summary = "wait until all the inputs are read.";
  let arguments = (ins I32Attr:$pendings);
  let description = [{
    Wait until all the read operations are done from the associated store operations.
    This is needed before the shared memory can be written to.
  }];

  let assemblyFormat = "attr-dict";
}

#endif
