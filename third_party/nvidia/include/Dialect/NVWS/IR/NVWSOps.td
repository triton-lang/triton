// Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
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

#ifndef NVWS_OPS
#define NVWS_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/ControlFlowInterfaces.td" // RegionBranchOpInterface
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"  // SameOperandsAndResultType
include "mlir/Interfaces/SideEffectInterfaces.td"  // Pure
include "mlir/Interfaces/ViewLikeInterface.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "NVWSDialect.td"
include "NVWSTypes.td"
include "NVWSAttrDefs.td"
include "NVWSOpInterfaces.td"

class NVWS_Op<string mnemonic, list<Trait> traits = []> :
    Op<NVWS_Dialect, mnemonic, traits>;

def NVWS_ArefCreateOp : NVWS_Op<"aref.create", [
    RangedTypesMatchWith<"input types match Aref output type",
                        "result", "buffers", "::llvm::cast<ArefType>($_self).getBaseType()">, Pure]> {
  let summary = "Create an asynchronous reference.";
  let description = [{
    Create an asynchronous reference.

    Takes as inputs a variadic number of buffers, and returns an ARef.
    The inputs are expected to be array-like (i.e., Tensor, MemDesc, etc)
    and the first axis of the shape should match between all inputs, representing
    multi-buffering of the values.
  }];
  let arguments = (ins Variadic<TTG_MemDescType>:$buffers);

  let results = (outs NVWS_ArefType:$result);

  let assemblyFormat = [{$buffers attr-dict `:` type($result)}];
  let hasVerifier = 1;
}

def NVWS_ArefGetEnterOp : NVWS_Op<"aref.get.enter"> {
  let summary = "Enter ArefGet region where the buffer can be used to read data";
  let description = [{ Enter a "region" where you can freely read from the buffer)
                      These ArefGet "regions" can span multiple iterations. }];

  let arguments = (ins NVWS_ArefType:$aref,
                       I32:$stage,
                       I32:$phase);
  let results = (outs Variadic<TTG_MemDescType>:$buffers,
                      TTG_AsyncToken:$token);
  let hasVerifier=1;
  let assemblyFormat = [{
    $aref `[` $stage `,` $phase `]` attr-dict
    `:` type($aref) `->` type(results)
  }];
}

def NVWS_ArefGetExitOp : NVWS_Op<"aref.get.exit"> {
  let summary = "Exit ArefGet region, where the buffer should no longer be used";
  let description = [{ Leave the region where you can freely read from the buffer).
                      These ArefGet "regions" can span multiple iterations. }];

  let arguments = (ins NVWS_ArefType:$aref,
                       TTG_AsyncToken:$token,
                       I32:$stage,
                       NVWS_AsyncOpArrayAttr:$async_ops);
  let assemblyFormat = [{
    $aref `[` $stage `]` `,` $token $async_ops attr-dict
    `:` type($aref) `,` type($token)
 }];
}

def NVWS_ArefPutEnterOp : NVWS_Op<"aref.put.enter"> {
  let summary = "Enter ArefPut region where the buffer can be used to read data";
  let description = [{ Enter a "region" where you can freely write to the buffer)
                      These ArefPut "regions" can span multiple iterations. }];

  let arguments = (ins NVWS_ArefType:$aref,
                       I32:$stage,
                       I32:$phase);
  let results = (outs Variadic<TTG_MemDescType>:$buffers,
                      TTG_AsyncToken:$token);
  let hasVerifier=1;
  let assemblyFormat = [{
    $aref `[` $stage `,` $phase `]` attr-dict
    `:` type($aref) `->` type(results)
  }];
}

def NVWS_ArefPutExitOp : NVWS_Op<"aref.put.exit"> {
  let summary = "Exit ArefPut region, where the buffer should no longer be used";
  let description = [{ Leave the region where you can freely write to the buffer).
                      These ArefPut "regions" can span multiple iterations. }];

  let arguments = (ins NVWS_ArefType:$aref,
                       TTG_AsyncToken:$token,
                       I32:$stage,
                       NVWS_AsyncOpArrayAttr:$async_ops);
  let assemblyFormat = [{
    $aref`[` $stage `]` `,` $token  $async_ops attr-dict
    `:` type($aref) `,` type($token)
 }];
}

def NVWS_WarpGroupOp : NVWS_Op<"warp_group", [
  RecursiveMemoryEffects, RecursivelySpeculatable,
]> {
  let summary = "Container Op for Warp Specialization";
  let description = [{
    Higher level container for Warp Specialization Analysis.

    Contains a variadic number warp groups, with
    the number of warps in each group, plus a region to hold the
    computation for that warp group.

    The results of this op, if any, are those of the first region, as returned by
    nvws.warp_group.yield op.

    nvws.warp_group should be lowered to ttg.warp_specialize
    before execution.
  }];

  let arguments = (ins DenseI32ArrayAttr:$numWarps);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<MinSizedRegion<1>>:$partitionRegions);
  let hasVerifier=1;
  let hasCustomAssemblyFormat = 1;
}

def NVWS_WarpGroupYieldOp : NVWS_Op<"warp_group.yield", [
  Pure, Terminator, ReturnLike, HasParent<"WarpGroupOp">,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>
]> {
  let summary = "yield from the first region of `nvws.warp_group`";
  let description = [{
    This op is equivalent to ttg.warp_yield op for ttg.warp_specialize op.

    TODO: Decide if we should move nvws.warp_group to TritonGPU, or continue to
    have TritonGPU depend on NVWS. In the former case, this op can be removed.
    The latter one involves a circular dependency between TritonGPU and NVWS.
  }];

  let arguments = (ins Variadic<AnyType>:$values);

  let assemblyFormat = "($values^)? attr-dict (`:` type($values)^)?";
}

def NVWS_WarpGroupReturnOp : NVWS_Op<"warp_group.return", [
  Pure, Terminator, HasParent<"WarpGroupOp">
]> {
  let summary = "Terminator for a warp group region";
  let description = [{
    Warp groups are expected to return values via referential modification
    of their inputs. Thus, the warp_group.return op takes no values to
    return from the warp group.
  }];

  let assemblyFormat = "attr-dict";
}

def NVWS_CreateTokenOp : NVWS_Op<"create_token"> {
  let summary = "Create a token to be used for synchronizations in communication channels";
  let description = [{ A token will be used by the producer and consumer to synchronize.
    The producer will acquire and hold the token, until it has filled the buffers,
    and signal the waiting consumer.
    The consumer will hold the token until it has consumed the buffers,
    and will signal the waiting producer trying to acquire the token.
  }];

  let results = (outs TensorOf<[NVWS_TokenType]>:$result);

  let arguments = (ins I32Attr:$numBuffers, NVWS_TokenLoadTypeAttr:$loadType);

  let builders = [OpBuilder<(ins "uint32_t":$numBuffers, "triton::nvws::TokenLoadType":$loadType)>];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def NVWS_ProducerAcquireOp : NVWS_Op<"producer_acquire"> {
  let summary = "Producer acquires a token to fill buffers";
  let description = [{ The producer will try to acquire the token prior to filling
    the buffers. If the buffers are not ready to be filled, the producer will wait to be
    signalled by the consumer which finishes consuming the buffers and
    releases the token.
  }];

  let arguments = (ins TensorOf<[NVWS_TokenType]>:$token, I32:$idx, I1:$phase);

  let assemblyFormat = "$token `,` $idx `,` $phase attr-dict `:` type(operands)";
}

def NVWS_ProducerCommitOp : NVWS_Op<"producer_commit"> {
  let summary = "Producer commits the buffer changes";
  let description = [{ The producer will release the token and signal the consumer
    that the buffers are ready to be consumed.
  }];

  let arguments = (ins TensorOf<[NVWS_TokenType]>:$token, I32:$idx);

  let assemblyFormat = "$token `,` $idx attr-dict `:` type(operands)";
}

def NVWS_ConsumerWaitOp : NVWS_Op<"consumer_wait"> {
  let summary = "Consumer awaits buffer readiness";
  let description = [{ The consumer will wait for the buffer to be ready
    to be consumed. If the buffers are not ready, the consumer will wait to be
    signalled by the producer which finishes filling the buffers and
    releases the token.
  }];

  let arguments = (ins TensorOf<[NVWS_TokenType]>:$token, I32:$idx, I1: $phase);

  let assemblyFormat = "$token `,` $idx `,` $phase attr-dict `:` type(operands)";
}

def NVWS_ConsumerReleaseOp : NVWS_Op<"consumer_release"> {
  let summary = "Consumer releases the token";
  let description = [{ The consumer will release the token and signal the producer
    that the buffers are ready to be filled.
  }];

  let arguments = (ins TensorOf<[NVWS_TokenType]>:$token, I32:$idx);

  let assemblyFormat = "$token `,` $idx attr-dict `:` type(operands)";
}

def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

def NVWS_DescriptorLoadOp : NVWS_Op<"descriptor_load", [NVWS_DescriptorLoadOpInterface]> {
  let summary = "Load from descriptor and store into shared memory";
  let description = [{
    This op behaves exactly like the op with the same name in Triton Dialect, but the result of the load is stored into shared memory.
    The execution is still synchronous.
  }];
  let arguments = (ins
    Arg<TT_TensorDescType, "", [MemRead<GlobalMemory>]>:$desc,
    Variadic<I32>:$indices,
    I32Attr:$txCount,
    Arg<TTG_MemDescType, "", [MemWrite<SharedMemory>]>:$result,
    DefaultValuedAttr<TT_CacheModifierAttr, "::mlir::triton::CacheModifier::NONE">:$cache,
    DefaultValuedAttr<TT_EvictionPolicyAttr, "::mlir::triton::EvictionPolicy::NORMAL">:$evict
  );

  let assemblyFormat = [{
    $desc `[` $indices `]` $txCount $result
    oilist(
      `cacheModifier` `=` $cache |
      `evictionPolicy` `=` $evict
    )
    attr-dict `:` type(operands)
  }];
}

def NVWS_DescriptorGatherOp : NVWS_Op<"descriptor_gather", [NVWS_DescriptorLoadOpInterface]> {
  let summary = "gather multiple rows from a descriptor into shared memory";
  let description = [{
    This op behaves exactly like the op with the same name in Triton Dialect, but the result of the load is stored into shared memory.
    The execution is still synchronous.
  }];

  let arguments = (ins
    Arg<TT_TensorDescType, "", [MemRead<GlobalMemory>]>:$desc,
    RankedTensorOf<[I32]>:$x_offsets,
    I32:$y_offset,
    I32Attr:$txCount,
    Arg<TTG_MemDescType, "", [MemWrite<SharedMemory>]>:$result
  );

  let assemblyFormat = [{
    $desc `[` $x_offsets `,` $y_offset `]` $txCount $result
    attr-dict `:` type(operands)
  }];
}

#endif
