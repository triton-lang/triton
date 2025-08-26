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
include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOpInterfaces.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypeInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;
def TensorMemory : Resource<"::mlir::triton::nvidia_gpu::TensorMemory">;

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
def TTNG_WarpGroupDotOp : TTNG_Op<"warp_group_dot", [
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DotOpInterface>,
  TypesMatchWith<"result's type matches accumulator's type", "d", "c", "$_self">
]> {
  let summary = "warp group dot";

  let description = [{
    $d = matrix_multiply($a, $b) + $c. For docs on InputPrecisionAttr, see TT_DotOp
  }];

  let arguments = (ins
    TTG_TensorOrMemDesc:$a,
    TTG_MemDescType:$b,
    TT_FpIntTensor:$c,
    Optional<I1>:$useC,
    DefaultValuedAttr<TT_InputPrecisionAttr, "::mlir::triton::InputPrecision::IEEE">:$inputPrecision,
    DefaultValuedAttr<I32Attr, "0">:$maxNumImpreciseAcc,
    DefaultValuedAttr<BoolAttr, "false">:$isAsync
  );

  let results = (outs TT_FpIntTensor:$d);

  let assemblyFormat = [{
    $a`,` $b`,` $c (`,` $useC^)? attr-dict
    `:` type($a) `*` qualified(type($b)) `->` type($d)
  }];

  let extraClassDeclaration = [{
    bool needsPartialAccumulator();
  }];

  let hasVerifier = 1;
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
  let hasVerifier = 1;
}

def TTNG_InitBarrierOp : TTNG_Op<"init_barrier"> {
  let summary = "Initialize a barrier in the given shared memory allocation.";

  let description = [{
      Initializes a shared memory allocation with mbarrier information.
      `alloc` is a descriptor to the shared memory allocation. `count` is the
      number of arrives expected by the barrier.

      This lowers to PTX mbarrier.init.shared::cta.b64.
  }];

  let arguments = (ins
    Arg<TTG_MemDescType, "", [MemWrite<SharedMemory>]>:$alloc,
    I32Attr:$count
  );
  let assemblyFormat = "$alloc `,` $count attr-dict `:` qualified(type($alloc))";
  let hasVerifier = 1;
}

def TTNG_InvalBarrierOp : TTNG_Op<"inval_barrier"> {
  let summary = "Invalidate a barrier allocation.";

  let description = [{
    Invalidate a barrier allocation so that it can be re-used. According to PTX
    spec this has to be done before any reuse of the memory used by mbarrier.

    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval
  }];

  let hasVerifier = 1;
  let arguments = (ins Arg<TTG_MemDescType, "", [MemWrite<SharedMemory>]>:$alloc);
  let assemblyFormat = "$alloc attr-dict `:` qualified(type($alloc))";
}

def TTNG_BarrierExpectOp : TTNG_Op<"barrier_expect"> {
  let summary = "Signal a barrier of an expected number of bytes to be copied.";

  let description = [{
    This signal the barrier that `size` bytes are expected to be copied. The
    associated barrier wait will block until the expected number of bytes are copied.
  }];

  let hasVerifier = 1;
  let arguments = (ins
    Arg<TTG_MemDescType, "", [MemWrite<SharedMemory>]>:$alloc,
    I32Attr:$size,
    I1:$pred
  );

  let assemblyFormat = [{
    $alloc `,` $size attr-dict `,` $pred `:` qualified(type($alloc))
  }];
}

def TTNG_WaitBarrierOp : TTNG_Op<"wait_barrier", [AttrSizedOperandSegments]> {
  let summary = "wait until the mbarrier phase completes.";

  let description = [{
    Blocks the program progress until the mbarrier object in `alloc` completes
    its current phase.

    This lowers a waitloop using PTX instruction
    mbarrier.try_wait.parity.shared.b64.

    Accepts optional list of memory. If present, it is assumed that any of the
    dependencies may be accessed until the barrier completes.

    The barrier behavior is described here:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms
  }];

  let arguments = (ins
    Arg<TTG_MemDescType, "", [MemRead<SharedMemory>, MemWrite<SharedMemory>]>:$alloc,
    I32:$phase,
    Optional<I1>:$pred,
    Variadic<TTG_MemDescType>:$deps
  );

  let builders = [
    OpBuilder<(ins "Value":$alloc, "Value":$phase),
    [{
    build($_builder, $_state, alloc, phase, /*pred=*/static_cast<mlir::Value>(nullptr), /*deps=*/{});
    }]>,
    OpBuilder<(ins "Value":$alloc, "Value":$phase, "Value":$pred),
    [{
    build($_builder, $_state, alloc, phase, pred, /*deps=*/{});
    }]>,
    OpBuilder<(ins "Value":$alloc, "Value":$phase, "ValueRange":$deps),
    [{
    build($_builder, $_state, alloc, phase, /*pred=*/static_cast<mlir::Value>(nullptr), deps);
    }]>,
  ];

  let assemblyFormat = [{
    $alloc `,` $phase (`,` $pred^)? (`deps` $deps^)?
    attr-dict `:` qualified(type($alloc)) (`,` type($deps)^)?
  }];
  let hasVerifier = 1;
}

def TTNG_ArriveBarrierOp : TTNG_Op<"arrive_barrier"> {
  let summary = "perform the arrive operation on an mbarrier";
  let description = [{
    The `ttng.arrive_barrier` operation performs the "arrive" operation on an
    mbarrier object in shared memory. The operation requires a `count` attribute
    of at least 1, and decreasing the pending arrival count of the mbarrier by
    the specific count.

    The operation accepts an optional predicate.

    Example:

    ```mlir
    ttng.arrive_barrier %barrier, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %barrier, 1, %pred : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ```
  }];

  let arguments = (ins
    Arg<TTG_MemDescType, "", [MemRead<SharedMemory>, MemWrite<SharedMemory>]>:$alloc,
    I32Attr:$count,
    Optional<I1>:$pred
  );

  let assemblyFormat = [{
    $alloc `,` $count (`,` $pred^)? attr-dict `:` qualified(type($alloc))
  }];

  let builders = [
    OpBuilder<(ins "Value":$alloc, "uint32_t":$count), [{
      return build($_builder, $_state, alloc, count, /*pred=*/Value());
    }]>
  ];

  let hasVerifier = 1;
}

def TTNG_AsyncCopyMbarrierArriveOp : TTNG_Op<"async_copy_mbarrier_arrive"> {
  let summary = "arrive on mbarrier once all previously issued copies are completed";
  let arguments = (ins
    Arg<TTG_MemDescType, "", [MemWrite<SharedMemory>]>:$barrier,
    UnitAttr:$noIncrement
  );
  let assemblyFormat = "$barrier attr-dict `:` qualified(type($barrier))";
}


def TTNG_AsyncTMACopyGlobalToLocalOp : TTNG_Op<"async_tma_copy_global_to_local"> {
  let summary = "copy data based on descriptor from global memory to local memory asynchronously";

  let description = [{
    This operation copies data from global memory to local memory
    asynchronously.  This is analogue to tt.load except the data are copied to
    local memory pointed by the memory descriptor instead of a distributed
    tensor. The data copied depends on the global memory descriptor pointed to
    by `desc`.
  }];

  let hasVerifier = 1;
  let arguments = (ins
    Arg<TT_TensorDescType, "", [MemRead<GlobalMemory>]>:$desc,
    Variadic<I32>:$coord,
    Arg<TTG_MemDescType, "", [MemWrite<SharedMemory>]>:$barrier,
    Arg<TTG_MemDescType, "", [MemWrite<SharedMemory>]>:$result,
    I1:$pred,
    DefaultValuedAttr<TT_CacheModifierAttr, "triton::CacheModifier::NONE">:$cache,
    DefaultValuedAttr<TT_EvictionPolicyAttr, "triton::EvictionPolicy::NORMAL">:$evict,
    DefaultValuedAttr<BoolAttr, "false">:$isVolatile
  );

  let assemblyFormat = [{
    $desc `[` $coord `]` $result `,` $barrier `,` $pred
    oilist(`cacheModifier` `=` $cache | `evictionPolicy` `=` $evict)
    attr-dict `:` qualified(type($desc)) `,` qualified(type($barrier)) `->` qualified(type($result))
  }];
}

def TTNG_AsyncTMACopyLocalToGlobalOp : TTNG_Op<"async_tma_copy_local_to_global"> {
  let summary = "copy data based on descriptor from local memory to global memory asynchronously";

  let description = [{
    This operation copies data from local memory to global memory
    asynchronously.  This is analogue to tt.store except the data are copied from
    local memory pointed by the memory descriptor instead of a distributed
    tensor. The data copied depends on the global memory descriptor pointed to
    by `desc`.
  }];

  let arguments = (ins
    Arg<TT_TensorDescType, "", [MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>:$desc,
    Variadic<I32>:$coord,
    Arg<TTG_MemDescType, "", [MemRead<SharedMemory>]>:$src
  );

  let assemblyFormat = [{
    $desc `[` $coord `]` $src
    attr-dict `:` qualified(type($desc)) `,` qualified(type($src))
  }];
}

def TTNG_AsyncTMAReduceOp : TTNG_Op<"async_tma_reduce", [MemoryEffects<[MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>]> {
  let summary = "reduce result in gmem based on a TMA descriptor";

  let description = [{
    This operation copies data from local memory to global memory
    asynchronously, and atomically performs the specified reduction kind.
    Atomicity is at the granularity of individual elements, and only relaxed
    semantics are implied.
  }];

  let arguments = (ins
    TT_DescriptorReduceKindAttr:$kind,
    Arg<TT_TensorDescType, "", [MemRead<GlobalMemory>]>:$desc,
    Variadic<I32>:$coord,
    Arg<TTG_MemDescType, "", [MemRead<SharedMemory>]>:$src
  );

  let assemblyFormat = [{
    $kind `,` $desc `[` $coord `]` $src
    attr-dict `:` qualified(type($desc)) `,` qualified(type($src))
  }];
}

def TTNG_AsyncTMAGatherOp : TTNG_Op<"async_tma_gather"> {
  let summary = "gather data based on descriptor from global memory to local memory asynchronously";

  let description = [{
    This operation gathers multiple rows of data from global memory matrix to
    local memory asynchronously.  This is similar to
    async_tma_copy_global_to_local except that each row is indexed independently.
  }];

  let arguments = (ins
    Arg<TT_TensorDescType, "", [MemRead<GlobalMemory>]>:$desc,
    RankedTensorOf<[I32]>:$x_offsets,
    I32:$y_offset,
    Arg<TTG_MemDescType, "", [MemWrite<SharedMemory>]>:$barrier,
    Arg<TTG_MemDescType, "", [MemWrite<SharedMemory>]>:$result,
    I1:$pred
  );

  let assemblyFormat = [{
    $desc `[` $x_offsets `,` $y_offset `]` $result `,` $barrier `,` $pred
    attr-dict `:` type(operands)
  }];

  let hasVerifier = 1;
}

def TTNG_AsyncTMAScatterOp : TTNG_Op<"async_tma_scatter"> {
  let summary = "scatter data from local memory into global memory based on a descriptor asynchronously";

  let description = [{
    The `ttng.async_tma_scatter` operation scatters multiple separately-indexed
    rows of data from local memory into global memory asynchronously. The
    operation scatters a 2D tensor in shared memory, laid out by core tensor
    tiles nvmma_shared layout into separately indexed rows in global
    memory at a given `y` offset.
  }];

  let arguments = (ins
    Arg<TT_TensorDescType, "", [MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>:$desc,
    RankedTensorOf<[I32]>:$x_offsets,
    I32:$y_offset,
    Arg<TTG_MemDescType, "", [MemRead<SharedMemory>]>:$src
  );

  let assemblyFormat = [{
    $desc `[` $x_offsets `,` $y_offset `]` $src
    attr-dict `:` type(operands)
  }];

  let hasVerifier = 1;
}

def TTNG_TMAStoreWaitOp : TTNG_Op<"async_tma_store_wait"> {
  let summary = "wait until all the inputs are read.";
  let arguments = (ins I32Attr:$pendings);
  let description = [{
    Wait until all the read operations are done from the associated store operations.
    This is needed before the shared memory can be written to.
  }];

  let assemblyFormat = "attr-dict";
}

def TTNG_TCGen5MMAOp : TTNG_Op<"tc_gen5_mma", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DotOpInterface>,
    DeclareOpInterfaceMethods<MMAv5OpInterface>,
    AttrSizedOperandSegments
]> {
  let summary = "block level op mapping to tensorcore gen5 mma";

  let description = [{
    $d += matrix_multiply($a, $b).
    if is_async is false, the op executes synchronously. The barrier operands must not be present in that case.
    Otherwise, if a barrier is given, the op will trigger a commit/arrive on it. The result will be safe to read after a barrier wait.
    If $two_ctas is set the op will execute a matmul across two contiguous CTAs, it will read the data distributed across the two CTAs.
    and syncronize both CTAs if the op is synchronous.

    This operation takes and produces an optional token to indicate TMEM read
    and write on its accumulator operand. When the tokens are present, they can
    be used to check aliasing and modref on the accumulator memory.
  }];

  let arguments = (ins
    TTG_MemDescType:$a,
    TTG_MemDescType:$b,
    TTG_MemDescType:$d,
    Optional<TTG_AsyncToken>:$acc_dep,
    I1:$useD,
    I1:$pred,
    Variadic<TTG_MemDescType>:$barriers,
    Variadic<I1>:$barrier_preds,
    UnitAttr:$is_async,
    UnitAttr:$two_ctas
  );
  let results = (outs Optional<TTG_AsyncToken>:$token);

  let builders = [
    OpBuilder<(ins "Type":$token,
      "Value":$a, "Value":$b, "Value":$d, "Value":$acc_dep, "Value":$useD,
      "Value":$pred, CArg<"bool", "false">:$two_ctas,
      CArg<"ValueRange", "{}">:$barriers,
      CArg<"ValueRange", "{}">:$barrier_preds,
      CArg<"bool", "false">:$is_async)>
  ];

  let assemblyFormat = [{
    $a `,` $b `,` $d `` custom<Token>($acc_dep, type($token)) `,` $useD`,`
    $pred `` custom<BarriersAndPreds>($barriers, $barrier_preds)
    attr-dict `:` qualified(type($a)) `,` qualified(type($b)) `,`
    qualified(type($d)) (`,` qualified(type($barriers))^)?
  }];

  let hasVerifier = 1;
}

def TTNG_TCGen5MMAScaledOp : TTNG_Op<"tc_gen5_mma_scaled", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DotOpInterface, ["verifyDims", "verifyOutputDims"]>,
    DeclareOpInterfaceMethods<MMAv5OpInterface>,
    AttrSizedOperandSegments
]> {
  let summary = "block level op mapping to tensorcore gen5 mma";

  let description = [{
    $d += matrix_multiply(scale($lhs, $lhs_scale), scale(rlhs, $rhs_scale))
    if is_async is false, the op executes synchronously. The barrier operands must not be present in that case.
    Otherwise, if a barrier is given, the op will trigger a commit/arrive on it.
    The result will be safe to read after a barrier wait.

    This operation takes and produces an optional token to indicate TMEM read
    and write on its accumulator operand. When the tokens are present, they can
    be used to check aliasing and modref on the accumulator memory.
  }];

  let arguments = (ins
    TTG_MemDescType:$a,
    TTG_MemDescType:$b,
    TTG_MemDescType:$d,
    Optional<TTG_AsyncToken>:$acc_dep,
    TTG_MemDescType:$a_scale,
    TTG_MemDescType:$b_scale,
    TT_ScaleDotElemTypeAttr:$a_type,
    TT_ScaleDotElemTypeAttr:$b_type,
    I1:$useD,
    I1:$pred,
    Variadic<TTG_MemDescType>:$barriers,
    Variadic<I1>:$barrier_preds,
    UnitAttr:$is_async
  );
  let results = (outs Optional<TTG_AsyncToken>:$token);

  let extraClassDeclaration = [{
    int64_t getBlockM();
    int64_t getBlockN();
    int64_t getBlockK();
  }];

  let builders = [
    // Namespaces need to be prefixed so ODS prefers our
    // custom builder signature over the default-generated one.
    OpBuilder<(ins "::mlir::Type":$token,
      "::mlir::Value":$a, "::mlir::Value":$b, "::mlir::Value":$d,
      "::mlir::Value":$acc_dep, "::mlir::Value":$a_scale,
      "::mlir::Value":$b_scale, "::mlir::triton::ScaleDotElemType":$a_type,
      "::mlir::triton::ScaleDotElemType":$b_type,
      "::mlir::Value":$useD, "::mlir::Value":$pred,
      CArg<"::mlir::ValueRange", "{}">:$barriers,
      CArg<"::mlir::ValueRange", "{}">:$barrier_preds,
      CArg<"bool", "false">:$is_async)>
  ];

  let assemblyFormat = [{
    $a `,` $b `,` $d `` custom<Token>($acc_dep, type($token)) `,` $a_scale `,`
    $b_scale `,` $useD `,` $pred `lhs` `=` $a_type `rhs` `=` $b_type
    `` custom<BarriersAndPreds>($barriers, $barrier_preds)
    attr-dict `:` qualified(type($a)) `,` qualified(type($b)) `,`
    qualified(type($d)) `,` qualified(type($a_scale)) `,`
    qualified(type($b_scale)) (`,` qualified(type($barriers))^)?
  }];

  let hasVerifier = 1;
}

def TTNG_TCGen5CommitOp : TTNG_Op<"tc_gen5_commit"> {
  let summary = "make an mbarrier track completion of all prior async tcgen5 ops";

  let description = [{
    The `ttng.tc_gen5_commit` is an asynchronous operation that makes the
    mbarrier object track the completion of all prior asynchronous tcgen5
    operations. Upon completion of all asynchronous operations, the mbarrier
    arrive operation is performed on the mbarrier with a count of 1.

    If `two_ctas` is set, then the mbarrier tracks all prior operations
    initiated with `two_ctas` set as well. Otherwise, it tracks all prior
    operations initiated without `two_ctas`.

    Note that the completion mechanisms are guaranteed to occur sequentially in
    the order the commit operations were issued. This means, for example:

    ```mlir
    ttng.tmem_copy
    ttng.tc_gen5_mma
    ttng.tc_gen5_commit %barrierA
    ttng.tc_gen5_commit %barrierB
    ```

    `%barrierA` tracks the completion of the previous TMEM copy and MMA
    operations, but since the commit groups are sequential, the arrive-on
    operation on `%barrierA` is guaranteed to be performed before the arrive-on
    operation on `%barrierB`, even though its commit group is empty.
  }];

  let arguments = (ins
    Arg<TTG_MemDescType, "", [MemWrite<SharedMemory>]>:$barrier,
    Optional<I1>:$pred,
    UnitAttr:$two_ctas
  );

  let assemblyFormat = [{
    $barrier (`,` $pred^)? attr-dict `:` qualified(type($barrier))
  }];

  let builders = [
    OpBuilder<(ins "Value":$barrier, CArg<"bool", "false">:$two_ctas), [{
      build($_builder, $_state, barrier, /*pred=*/Value(), two_ctas);
    }]>,
  ];
}

def TTNG_TMEMLoadOp : TTNG_Op<"tmem_load"> {
  let summary = "Load a buffer from tensor memory into a distributed tensor";

  let description = [{
    This is similar to ttg.local_load except the result layout is restricted to only few possibility.
    Therefore we cannot combine this op with any convert layout like local_load.

    This operation takes and produces an optional token to indicate TMEM read
    on its source operand. When the tokens are present, they can
    be used to check aliasing and modref on the TMEM buffer.
  }];
  let arguments = (ins
    Arg<TTG_MemDescType, "", [MemRead<TensorMemory>]>:$src,
    Optional<TTG_AsyncToken>:$dep
  );
  let results = (outs
    TT_Tensor:$result,
    Optional<TTG_AsyncToken>:$token
  );

  let assemblyFormat = [{
    $src `` custom<Token>($dep, type($token))
    attr-dict `:` qualified(type($src)) `->` type($result)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    RankedTensorType getType() { return getResult().getType(); }
    operator TypedValue<RankedTensorType>() { return getResult(); }
  }];
}

def TTNG_TMEMStoreOp : TTNG_Op<"tmem_store"> {
  let summary = "Store a distributed tensor into a buffer in tensor memory";

  let description = [{
    This is similar to ttg.local_store except the source layout is restricted to only few possibility.

    This operation takes and produces an optional token to indicate TMEM write
    on its source operand. When the tokens are present, they can
    be used to check aliasing and modref on the TMEM buffer.
  }];
  let arguments = (ins
    Arg<TTG_MemDescType, "", [MemWrite<TensorMemory>]>:$dst,
    Optional<TTG_AsyncToken>:$dep,
    TT_Tensor:$src,
    I1:$pred
  );
  let results = (outs Optional<TTG_AsyncToken>:$token);

  let builders = [
    OpBuilder<(ins "Value":$dst, "Value":$src, "Value":$pred), [{
      build($_builder, $_state, Type(), dst, Value(), src, pred);
    }]>
  ];

  let assemblyFormat = [{
    $src `,` $dst `` custom<Token>($dep, type($token)) `,` $pred
    attr-dict `:` type($src) `->` qualified(type($dst))
  }];
  let hasVerifier = 1;
}

def TTNG_TMEMAllocOp : TTNG_Op<"tmem_alloc", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "allocate tensor memory";
  let description = [{
    This operation allocates buffer in tensor memory and return a descriptor
    containing the address and a view of the buffer.
    This is similar to ttg.local_alloc except the buffer is allocated in tensor memory.

    Explicitly deallocating a buffer is optional; see local_dealloc.
  }];
  let arguments = (ins Optional<TT_Tensor>:$src);
  let results = (outs
    TTG_MemDescType:$result,
    Optional<TTG_AsyncToken>:$token
  );

  let assemblyFormat = [{
    ($src^)? attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    triton::gpu::MemDescType getType() { return getResult().getType(); }
    operator TypedValue<triton::gpu::MemDescType>() { return getResult(); }
  }];
}

def TTNG_TMEMSubSliceOp : TTNG_Op<"tmem_subslice", [Pure]> {
  let summary = "Take a subslice of a tensor memory allocation";
  let description = [{
    This operation takes a subslice of a tensor memory allocation and returns a new descriptor
    containing the address and a view of the subslice.
    This is similar to ttg.memdesc_subslice except we can only slice along the inner dimension
    of a 2D memdesc as this is the only one we can do for TMem.
  }];
  let arguments = (ins TTG_MemDescType:$src, I32Attr:$N);

  let assemblyFormat = [{
    $src attr-dict `:` qualified(type($src)) `->` qualified(type($result))
  }];

  let builders = [
      OpBuilder<(ins "Value":$alloc, "int":$offset, "int":$size)>,
    ];
  let results = (outs TTG_MemDescType:$result);
  let hasVerifier = 1;
}

def TTNG_TMEMCopyOp : TTNG_Op<"tmem_copy"> {
  let summary = "Initiate an asynchronous copy operation from shared memory to the Tensor Memory.";

  let description = [{
    2D blocks stored contiguously in SMEM are copied into TMEM as specified by the destination address.
    The completion of the copy can be observed by waiting on the optional barrier. If this op is used
    together with an MMA op, one barrier can be used to wait for both copy and MMA. We do not need to wait
    for the completion of the copy before MMA, since tcgen05.cp followed by tcgen05.mma is guaranteed to
    execute in that order.

    This op lowers to the PTX instruction tcgen05.cp. This supports writing either to scales tmem layout as well as default tmem layout.
    Currently the semantic is different when writing to tmem scale layout.

    In case of default layout the copy doesn't change the logical elements between the source and destination memdesc.

    In case of scale layout:
    Each 32x128b block in SMEM is duplicated over 4 warps and stored into 128 rows
    and 4 columns of TMEM. The primary use case of this op is to copy blocked scales from SMEM to TMEM.

    The shape of the input SMEM can be flexibily chosen depending on use cases. In the simplest case (e.g. unit test),
    the source SMEM can be of shape (32 x num_blocks, 16), and the destination TMEM should be of shape (128, 16 x num_blocks),
    for copying 8 bit values. For scaled GEMM, rep_m x rep_k copies of a 32x128b block need to be stored in SMEM, where
    rep_m = BLOCK_M / 128, rep_k = BLOCK_K / scale_vec_size / 4, and scale_vec_size = 32 for MXFP.
    Conceptually, the SMEM is organized in a high-dimensional layout, (rep_m, rep_k, 32, 4, 4B).
    Some of axes can be flattened into one, to reduce the rank of the load. For example, the following patterns are supported:
     * (rep_m, rep_k * 32 x 4 x 4B), 2D scale load with cp.async
     * (rep_m, rep_k, 32, 16B), 4D scale load with TMA
     * (rep_m, rep_k, 32, 4, 4B), 5D scale load with cp.async
    Since rep_m blocks are not contiguous in SMEM, this axis cannot be flattened into inner ones.

    In Triton, the TMEM memdesc for blocked scales must be of the following form:
    * Its shape must be (BLOCK_MN, BLOCK_K / scale_vec_size), representing the logical shape of blocked scales.
    * It must be attached with `tensor_memory_scales_encoding` to indicate the chunk-based layout and its duplication over 4 warps.

    In contrast, the src SMEM must be in the explicit chunk-based layout as described above. So the IR might look like this:

    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
    ttng.tmem_copy %1, %0 : (!ttg.memdesc<1x1x32x4x4xi8, #shared1, #smem>, !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>) -> ()

    We interpret the semantics of this copy operation as follows. The chunk-based layout in SMEM implies that
    the logical shape (BLOCK_MN, BLOCK_K / scale_vec_size) in TMEM is the result of certain reshape and transpose operations.
    In practice, to take an advantage of the native scale layout and the TMEM copy op,  users need to do
    `scales5D.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // scale_vec_size)` before feeding scales into dot_scaled.
    When we use tmem_copy in the IR, such reshape and transpose operations are removed. But the change in the logical shape they have caused on
    registers is now understood to be incorporated into tmem_copy itself. Ideally, we would lift reshape / transpose done on registers onto
    the SMEM memdesc, making tmem_copy a straightforward 2D copy operation: (BLOCK_MN, BLOCK_K / scale_vec_size) -> (BLOCK_MN, BLOCK_K / scale_vec_size).
    In the absence of such operations on memdesc, we resort to implicitly encoding the reshape/transpose semantics in tmem_copy.

  }];
  let arguments = (ins
    Arg<TTG_MemDescType, "", [MemRead<SharedMemory>]>:$src,
    Arg<TTG_MemDescType, "", [MemWrite<TensorMemory>]>:$dst,
    Optional<TTG_MemDescType>:$barrier
  );

  let assemblyFormat = [{$src `,` $dst (`,` $barrier^)? attr-dict `:` qualified(type(operands))}];
  let hasVerifier = 1;
}

def TTNG_ReinterpretTensorDescOp : TTNG_Op<"reinterpret_tensor_descriptor", [Pure]> {
  let summary = "Reinterpret a pointer as a tensor descriptor";

  let description = [{
     This Op exists to help the transition from untyped raw TMA objects to typed Tensor descriptor objects.
     Ideally, we can remove this once the APIs are fully fleshed out.
  }];

  let arguments = (ins TT_Ptr:$rawDesc);
  let results = (outs TT_TensorDescType:$result);

  let assemblyFormat = [{
    $rawDesc attr-dict `:` qualified(type($rawDesc))  `to` qualified(type($result))
  }];
}

def TTNG_TensormapCreateOp: TTNG_Op<
  "tensormap_create",
  [
    MemoryEffects<[MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>,
    AttrSizedOperandSegments,
  ]
> {
  let summary = "Create a new TMA descriptor on device";
  let arguments = (
      ins
      TT_PtrType:$desc_ptr,
      TT_PtrType:$global_address,
      Variadic<I32>:$box_dim,
      Variadic<I32>:$global_dim,
      Variadic<I64>:$global_stride,
      Variadic<I32>:$element_stride,
      ConfinedAttr<I32Attr, [IntNonNegative, IntMaxValue<15>]>:$elem_type,
      ConfinedAttr<I32Attr, [IntNonNegative, IntMaxValue<2>]>:$interleave_layout,
      ConfinedAttr<I32Attr, [IntNonNegative, IntMaxValue<3>]>:$swizzle_mode,
      ConfinedAttr<I32Attr, [IntNonNegative, IntMaxValue<1>]>:$fill_mode
  );
  let extraClassDeclaration = [{
      int32_t getRank() {
          return getBoxDim().size();
      }
  }];
  let assemblyFormat = [{
    $desc_ptr `,` $global_address `,`
    `[` $box_dim `]` `,`
    `[` $global_dim `]` `,`
    `[` $global_stride `]` `,`
    `[` $element_stride `]`
    attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def TTNG_TensormapFenceproxyAcquireOp: TTNG_Op<
  "tensormap_fenceproxy_acquire",
  [MemoryEffects<[MemWrite<GlobalMemory>]>]
> {
  let summary = "Acquire fence on a tensormap object";
  let arguments = (ins TT_PtrType:$desc_ptr);
  let assemblyFormat = [{
    $desc_ptr attr-dict `:` qualified(type($desc_ptr))
  }];
}

#endif
