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

#ifndef NVWS_ATTRDEFS
#define NVWS_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "NVWSDialect.td"

class NVWS_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<NVWS_Dialect, name, traits, baseCppClass> {
}

def NVWS_TypeArray : ArrayOfAttr<NVWS_Dialect, "TypeArray", "type_array", "Type"> {}
def NVWS_IntArray : ArrayOfAttr<NVWS_Dialect, "IntArray", "int_array", "int"> {}

// Type for synchronization tokens.
def NVWS_TokenLoadTypeAttr : I32EnumAttr<
    "TokenLoadType", "",
    [
      I32EnumAttrCase<"None", 0, "none">,
      I32EnumAttrCase<"AsyncLoadOp", 1, "asyncLoadOp">,
      I32EnumAttrCase<"TMALoadOp", 2, "tmaLoadOp">,
      I32EnumAttrCase<"LocalStoreOp", 3, "localStoreOp">,
      I32EnumAttrCase<"TmemLoadOp", 4, "TmemLoadOp">,
    ]>{
  let cppNamespace = "::mlir::triton::nvws";
}

def NVWS_AsyncOpAttr: I32EnumAttr<
  "AsyncOp", "",
  [
    I32EnumAttrCase<"NONE", 0, "none">,
    I32EnumAttrCase<"TMALoad", 1, "tma_load">,
    I32EnumAttrCase<"TC5MMA", 2, "tc5mma">,
    I32EnumAttrCase<"TMEMCopy", 3, "tmem_copy">,
    I32EnumAttrCase<"CpAsync", 4, "cp_async">,
    I32EnumAttrCase<"WGMMA", 5, "wgmma">,
  ]> {
  let cppNamespace = "::mlir::triton::nvws";
  let genSpecializedAttr = 0;
}

def NVWS_AsyncOpEnum : EnumAttr<NVWS_Dialect, NVWS_AsyncOpAttr, "async_op"> {
  let assemblyFormat = "`<` $value `>`";
}

def NVWS_AsyncOpArrayAttr : TypedArrayAttrBase<NVWS_AsyncOpEnum, "array of async op attributes">;

#endif
