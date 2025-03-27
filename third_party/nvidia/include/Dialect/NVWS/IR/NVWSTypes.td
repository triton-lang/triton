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

#ifndef NWVS_TYPES
#define NWVS_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "NVWSDialect.td"

class NVWS_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<NVWS_Dialect, name, traits> {
    let mnemonic = _mnemonic;
}

def NVWS_ArefType : NVWS_TypeDef<"Aref", "aref"> {
  let summary = "Asynchronous Reference";
  let description = [{
        A meta-type that holds an asynchronous reference to an underlying Type.

        Can wrap multiple underlying values simultaneously.

        Useful for syncing asynchronous operations while doing transformations such
        as pipelining and warp specialization. Lowers to the underlying type, and
        operations that use this should insert appropriate barriers during lowering.
    }];
  let parameters = (ins "TypeArrayAttr":$baseType,
                         DefaultValuedParameter<"std::optional<int>", "0">:$numBatchAxes);
  let assemblyFormat = "`<` $baseType (`,` $numBatchAxes^)? `>`";
}

#endif // NVWS_TYPES
