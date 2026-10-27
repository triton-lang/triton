/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_AMDGPU_ATTRDEFS
#define TRITON_AMDGPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "TritonAMDGPUDialect.td"

class TritonAMDGPU_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<TritonAMDGPU_Dialect, name, traits, baseCppClass> {
}

def SetFP8Clamping : TritonAMDGPU_Attr<"SetFP8Clamping"> {
  let mnemonic = "amdgcn.set.fp8.clamping";
}

def TritonAMDGPU_CachePolicyGfx9Attr : TritonAMDGPU_Attr<"CachePolicyGfx9"> {
  let mnemonic = "cache_policy_gfx9";
  let description = [{
    Cache hints for gfx9 (CDNA3/CDNA4) buffer memory accesses.

    Only the hint bits are held here. The coherence scope bits (SC0/SC1) come
    from the op's `scope` attribute. This attribute is a performance hint and
    has no memory-ordering or coherence semantics.

    - `nontemporal`: sets NT, i.e. the data is not expected to be reused.
  }];
  let parameters = (ins
    DefaultValuedParameter<"bool", "false">:$nontemporal
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif
