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
include "NVWSDialect.td"
include "NVWSTypes.td"
include "NVWSAttrDefs.td"

class NVWS_Op<string mnemonic, list<Trait> traits = []> :
    Op<NVWS_Dialect, mnemonic, traits>;

def NVWS_ArefCreateOp : NVWS_Op<"aref.create", [
    RangedTypesMatchWith<"input types match Aref output type",
                        "result", "operands", "::llvm::cast<ArefType>($_self).getBaseType()">]> {
  let summary = "Create an asynchronous reference.";
  let description = [{
    Create an asynchronous reference.

    Takes as inputs a variadic number of operands, and returns an ARef.
    The inputs can be of any type, but if the numBatchAxes attribute is
    set on the result type, the inputs are expected to be array-like
    (i.e., Tensor, MemDesc, etc) and have matching values for the first
    numBatchAxes of the shapes.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);

  let results = (outs NVWS_ArefType:$result);

  let assemblyFormat = [{$operands attr-dict `:` type($result)}];
}

def NVWS_ArefGetOp : NVWS_Op<"aref.get", []> {
  let summary = "Get value(s) from an asynchronous reference";
  let description = [{
    Get values from the asynchronous reference for reading.

    If indexes is empty, the op will return the entire underlying value(s).
    If the indexes is size 1, the op will ensure that the underlying
    value is Tensor-Like (i.e, has a shape parameter) with rank at least 1,
    and index into the first dimension.
    Similarly, if indexes is size 2, the op will index into the first 2 dimensions
    etc.

    These values are passed as arguments to the region.
    At the start of the region, the values are guaranteed to be “full”,
    and operations are free to read from them. When the compute in the region
    completes, the buffers have been read from, and the aref is marked as “empty”
    so that a producer group can write to the aref.

  }];
  let arguments = (
    ins NVWS_ArefType:$operand,
    Variadic<TT_IntLike>:$indexes
  );

  let regions = (region SizedRegion<1>:$region);

  let results = (outs Variadic<AnyType>:$results);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def NVWS_ArefPutOp : NVWS_Op<"aref.put", [SameVariadicOperandSize]> {
  let summary = "Put value(s) into an asynchronous reference";
  let description = [{
    Get values from the asynchronous reference for writing.

    If indexes is empty, the op will return the entire underlying value(s).
    If the indexes is size 1, the op will ensure that the underlying
    value is Tensor-Like (i.e, has a shape parameter) with rank at least 1,
    and index into the first dimension.
    Similarly, if indexes is size 2, the op will index into the first 2 dimensions
    etc.

    These values are passed as arguments to the region.
    At the start of the region, the values are guaranteed to be “empty”,
    and operations are free to write to them. When the compute in the region
    completes, the buffers have been written to, and the aref is marked as “full”
    so that a consumer can read from them.

    If the Aref contains in-register types (i.e., not ttg.MemDesc), the region should
    return the updated values. This represents a semantic copy at the aref level,
    but should be optimized away during lowering.

  }];
  let arguments = (
    ins NVWS_ArefType:$operand,
    Variadic<TT_IntLike>:$indexes
  );

  let regions = (region SizedRegion<1>:$region);

  let results = (outs Variadic<AnyType>:$results);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def NVWS_ArefReturnOp: NVWS_Op<"aref.return",
                            [Pure, ReturnLike, Terminator]> {
  let summary = "terminator for aref_put/get operator";
  let arguments = (ins Variadic<AnyType>:$srcs);
  let assemblyFormat = "attr-dict ($srcs^ `:` type($srcs))?";
  let hasVerifier = 1;
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

    Regions are not Isolated from Above to aid in analysis,
    and take inputs purely by reference.

    nvws.warp_group should be lowered to ttg.warp_specialize
    before execution.
  }];

  let arguments = (ins DenseI32ArrayAttr:$numWarps);

  let regions = (region VariadicRegion<MinSizedRegion<1>>:$partitionRegions);
  let hasVerifier=1;
  let hasCustomAssemblyFormat = 1;
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

#endif
