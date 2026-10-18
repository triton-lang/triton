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

#ifndef TRITON_AMDGPU_TYPES
#define TRITON_AMDGPU_TYPES

include "mlir/IR/AttrTypeBase.td"
include "TritonAMDGPUDialect.td"

def TritonAMDGPU_BufferDescType : TypeDef<TritonAMDGPU_Dialect, "BufferDesc"> {
  let mnemonic = "buffer_desc";
  let summary = "AMD buffer resource descriptor";
  let description = [{
    A buffer resource descriptor (V#): the scalar, program-uniform value that
    buffer memory instructions read from four SGPRs. It holds the fields of
    the hardware descriptor that the program chooses (base address and
    num_records); the remaining configuration bits are filled in by the
    compiler for the target. `elementType` is the type the buffer ops access
    through it; offsets of those ops count elements of this type.
  }];
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

#endif
