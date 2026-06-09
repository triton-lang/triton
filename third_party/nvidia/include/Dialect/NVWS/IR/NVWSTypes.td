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

def NVWS_SemaphoreType : NVWS_TypeDef<"Semaphore", "semaphore"> {
  let summary = "Semaphore with embedded buffer type";
  let description = [{
    A semaphore guards access to a multi-buffered allocation. It embeds the
    buffer types, and infers the number of pipeline stages from their leading
    dimension. Multiple semaphores can share the same buffer and will use a
    shared stage counter.
  }];
  let parameters = (ins "TypeArrayAttr":$baseType);
  let assemblyFormat = "`<` $baseType `>`";
  let extraClassDeclaration = [{
    int getNumStages() const;
  }];
  let genVerifyDecl = 1;
}

def NVWS_TokenType : NVWS_TypeDef<"Token", "token">;

#endif // NVWS_TYPES
