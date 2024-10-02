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


#ifndef TRITON_AMDGPU_OPS
#define TRITON_AMDGPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "TritonAMDGPUDialect.td"
include "TritonAMDGPUAttrDefs.td"

class TT_AMDGPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonAMDGPU_Dialect, mnemonic, !listconcat(traits, [])> {
}

def InstructionSchedHint : TT_AMDGPU_Op<"instruction_sched_hint", []> {
  let summary = "A placeholder op for instruction scheduling hints within a basic block";
  let description = [{
    A placeholder op for instruction scheduling hints applied to instructions within
    a basic block where the placeholder op is located. This op is primarily intended
    to be used to adjust instruction scheduling inside the resulting main loop
    of a `tt.dot` operation. It's easier to identify dot ops at a high level and, thus,
    to mark intended scheduling regions. The hint ops are eventually lowered
    into LLVM AMDGPU instruction scheduling primitives, which are meant to control
    how different kinds of instructions (valu/mfma, global/shared memory, etc.) should
    interleave for better instruction level parallelism.
  }];

  let assemblyFormat = [{attr-dict}];
}

#endif
