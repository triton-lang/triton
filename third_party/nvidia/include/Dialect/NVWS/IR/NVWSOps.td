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
                        "result", "operands", "::llvm::cast<ArefType>($_self).getBaseType()">, Pure]> {
  let summary = "Create an asynchronous reference.";
  let description = [{
    Create an asynchronous reference.

    Takes as inputs a variadic number of operands, and returns an ARef.
    The inputs are expected to be array-like (i.e., Tensor, MemDesc, etc)
    and the first axis of the shape should match between all inputs, representing
    multi-buffering of the values.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);

  let results = (outs NVWS_ArefType:$result);

  let assemblyFormat = [{$operands attr-dict `:` type($result)}];
  let hasVerifier = 1;
}

def NVWS_ArefGetEnterOp : NVWS_Op<"aref.get.enter"> {
  let summary = "Enter ArefGet region where the buffer can be used to read data";
  let description = [{ Enter a "region" where you can freely read from the buffer)
                      These ArefGet "regions" can span multiple iterations. }];

  let arguments = (ins NVWS_ArefType:$aref,
                       I32:$index,
                       I32:$phase);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier=1;
  let assemblyFormat = [{
    $aref `[` $index `,` $phase `]` attr-dict
    `:` type($aref) `->` type($results)
  }];
}

def NVWS_ArefGetExitOp : NVWS_Op<"aref.get.exit"> {
  let summary = "Exit ArefGet region, where the buffer should no longer be used";
  let description = [{ Leave the region where you can freely read from the buffer).
                      These ArefGet "regions" can span multiple iterations. }];

  let arguments = (ins NVWS_ArefType:$aref,
                       I32:$index);
  let assemblyFormat = [{
    $aref `[` $index `]` attr-dict
    `:` type($aref)
 }];
}

def NVWS_ArefPutEnterOp : NVWS_Op<"aref.put.enter"> {
  let summary = "Enter ArefPut region where the buffer can be used to read data";
  let description = [{ Enter a "region" where you can freely write to the buffer)
                      These ArefPut "regions" can span multiple iterations. }];

  let arguments = (ins NVWS_ArefType:$aref,
                       I32:$index,
                       I32:$phase);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier=1;
  let assemblyFormat = [{
    $aref `[` $index `,` $phase `]` attr-dict
    `:` type($aref) `->` type($results)
  }];
}

def NVWS_ArefPutExitOp : NVWS_Op<"aref.put.exit"> {
  let summary = "Exit ArefPut region, where the buffer should no longer be used";
  let description = [{ Leave the region where you can freely write to the buffer).
                      These ArefPut "regions" can span multiple iterations. }];

  let arguments = (ins NVWS_ArefType:$aref,
                       I32:$index);
  let assemblyFormat = [{
    $aref `[` $index `]` attr-dict
    `:` type($aref)
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
