// Copyright (c) 2024 Meta Platforms Corporation & Affiliates. All rights reserved.
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


#ifndef PROTON_OPS
#define PROTON_OPS

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "mlir/IR/EnumAttr.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "ProtonDialect.td"
include "ProtonAttrDefs.td"

class TT_Proton_Op<string mnemonic, list<Trait> traits = []> :
    Op<Proton_Dialect, mnemonic, !listconcat(traits, [])> {
}

// Proton profiling metric.
def MetricAttr : I32EnumAttr<
    "Metric", "",
    [
        I32EnumAttrCase<"CYCLE", 0, "cycle">,
        I32EnumAttrCase<"INVALID", 1, "invalid">,
    ]> {
    let cppNamespace = "::mlir::triton::proton";
}

// Proton profiling granularity.
def GranularityAttr : I32EnumAttr<
    "Granularity", "",
    [
        I32EnumAttrCase<"WARPGROUP", 0, "warpgroup">,
        I32EnumAttrCase<"WARP", 1, "warp">,
    ]> {
    let cppNamespace = "::mlir::triton::proton";
}

def TT_RecordOp : TT_Proton_Op<"record", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Record a GPU hardware event";

  let description = [{
    The operator records GPU events from performance counters.
    Currently only cycle counter is supported.
  }];
  let arguments = (
    ins BoolAttr: $isStart,
    I32Attr: $regionId,
    DefaultValuedAttr<MetricAttr, "proton::Metric::CYCLE">:$metric,
    DefaultValuedAttr<GranularityAttr, "proton::Granularity::WARPGROUP">:$granularity
  );

  let hasCustomAssemblyFormat = 0;
  let hasVerifier = 1;
}

#endif // PROTON_OPS
