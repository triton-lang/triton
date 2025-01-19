#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonAMDGPUStreamPipeline : Pass<"tritonamdgpu-stream-pipeline", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    Pipeline global loads through registers to shared memory while computing on previous
    tile
  }];

  let constructor = "mlir::createTritonAMDGPUStreamPipelinePass()";

  let dependentDialects = ["mlir::triton::amdgpu::TritonAMDGPUDialect"];

  let options = [
    Option<"numStages", "num_stages",
           "int32_t", /*default*/"2",
           "Number of Pipeline stages">,
    Option<"prefetch", "prefetch",
           "int32_t", /*default*/"0",
           "Enable prefetch from shared memory">
  ];
}

def TritonAMDGPUAccelerateMatmul : Pass<"tritonamdgpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., AMD matrix cores)
  }];

  let constructor = "mlir::createTritonAMDGPUAccelerateMatmulPass()";

  let dependentDialects = [];

  let options = [
    Option<"archGenerationName", "arch-generation-name",
           "std::string", /*default=*/"std::string{}",
           "GFX generation name of target device.">,
    Option<"matrixInstructionSize", "matrix-instruction-size",
           "int32_t", /*default*/"0",
           "enforce matrix instruction MN size">,
    Option<"kPack", "kPack",
           "int32_t", /*default*/"1",
           "KWidth / kBase">
  ];
}

def TritonAMDGPUOptimizeEpilogue : Pass<"tritonamdgpu-optimize-epilogue", "mlir::ModuleOp"> {
  let summary = "Optimize epilogue: (1) Store accumulators directly without going thorough SMEM in epilogue.";

  let description = [{
  }];

  let constructor = "mlir::createTritonAMDGPUOptimizeEpiloguePass()";

  let dependentDialects = [];

}

def TritonAMDGPUCanonicalizePointers : Pass<"tritonamdgpu-canonicalize-pointers", "mlir::ModuleOp"> {
  let summary = "Canonicalize pointers: rewrite pointers passed to load/store operation as a `<basePtr, offset>` pair.";

  let description = [{
  This pass pushes all the constant pointer arithmetic on a scalar basePtr, while all the vector
  pointer arithmetic to a vector offset. I.e., if we consider the following IR:
  ```
    %v_ptr = tt.splat %s_ptr
    %c_offset = tt.splat %s_offset
    %v_offset0 = tt.make_range
    %v_offset1 = tt.make_range
    %v_ptr0 = tt.addptr %v_ptr, %c_offset
    %v_ptr1 = tt.addptr %v_ptr0, %v_offset0
    %v_ptr2 = tt.addptr %v_ptr0, %v_offset1
    %data = tt.load(%v_ptr2)
  ```
  We transform this into:
  ```
    %s_ptr0 = tt.addptr %s_ptr, %s_offset
    %v_offset = %zero
    %v_offset = arith.addi %v_offset, %v_offset0
    %v_offset = arith.addi %v_offset, %v_offset1
    %c_ptr = tt.splat %s_ptr0
    %v_ptr = tt.addptr %c_ptr, %v_offset
    %data = tt.load(%v_ptr)
  ```
  In the above IR:
  -  `v_` means "variable vector across the program"
  -  `c_` means "constant vector across the program"
  -  `s_` means "scalar"
  So we transform the IR such that the constant updates become scalar updates, and the variable updates happen on the offset. Note that
  when we have to load the data, we splat the scalar pointer, add the "variable" offset and then issue the load.
  }];

  let constructor = "mlir::createTritonAMDGPUCanonicalizePointersPass()";

  let dependentDialects = [];
}

def TritonAMDGPUBypassLDSForDotOperand: Pass<"tritonamdgpu-bypass-lds-for-dot-operand", "mlir::ModuleOp"> {
  let summary = "Bypass moving data trough LDS for dot operand when possible.";

  let description = [{
  Under certain conditions, the dot layout of one of the operands allows direct
  loading from HBM to VGPRs in the MFMA dot layout, without losing of vectorization of global loads
  or increasing the number of global loads due to shared data between threads.
  The required conditions are:
  1. K-Major Tensor Layout:
     The operand we want to bypass LDS for must be K-major (i.e., row-major for
     operand 0 or column-major for operand 1). This supports vectorized global
     load instructions, as MFMA instructions require each thread to hold B
     operand elements along the K dimension.
  2. kWidth * sizeof(dataType) == 128:
     Using the maximum kWidth for a specific data type ensures optimal global
     load vectorization (e.g., using global_load_dwordx4 instructions).
  3. Single Warp per CTA Dimension:
     Either warpsPerCTA[ndim] == 1 for operand A bypass or warpsPerCTA[mDim] ==
     1 for operand B bypass. This guarantees that each tensor element is
     handled by exactly one thread, maintaining the same number of global loads
     as in the blocked layout (i.e., each element is loaded only once).
  }];

  let constructor = "mlir::createTritonAMDGPUBypassLDSForDotOperand()";

  let dependentDialects = [];
}

def TritonAMDGPUReorderInstructions: Pass<"tritonamdgpu-reorder-instructions", "mlir::ModuleOp"> {
  let summary = "Reorder instructions";

  let description = "This pass reorder instructions so as to (1) decrease register pressure (e.g., by moving "
                    "conversions from shared memory before their first use) and (2) promote LLVM instruction "
                    "order more friendly to `ptxas`.";

  let constructor = "mlir::createTritonAMDGPUReorderInstructionsPass()";

  let dependentDialects = [];
}

def TritonAMDGPUConvertToBufferOps : Pass<"tritonamdgpu-convert-buffer-ops", "mlir::ModuleOp"> {
  let summary = "Convert memory operations to buffer operations";

  let description = "This pass converts memory and atomic operations (e.g., tt.load/tt.store/tt.atomic_rmw) to  amdgpu buffer operations, if possible";

  let constructor = "mlir::createTritonAMDGPUConvertToBufferOpsPass()";

  let dependentDialects = ["mlir::triton::amdgpu::TritonAMDGPUDialect"];

  let options = [
    Option<"archGenerationName", "arch-generation-name",
           "std::string", /*default=*/"std::string{}",
           "GFX generation name of target device.">,
  ];
}

def TritonAMDGPUBlockPingpong: Pass<"tritonamdgpu-block-pingpong", "mlir::ModuleOp"> {
  let summary = "Interleaving instructions from two warps on the same SIMD to better utilize matrix core";

  let description = [{
    This pass reorder instructions to interleave instructions from two warps on the same SIMD unit.
    We call this a ping-pong scheduling pattern, where two warps run concurrently in the synchronized fashion
    This block ping-pong pattern could be beneficial under few conditions including
    occupancy and number of warps.
  }];

  let constructor = "mlir::createTritonAMDGPUBlockPingpongPass()";

  let dependentDialects = ["mlir::ROCDL::ROCDLDialect, mlir::triton::amdgpu::TritonAMDGPUDialect"];
}

#endif
