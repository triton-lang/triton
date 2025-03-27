// RUN: triton-opt %s -split-input-file -triton-amdgpu-insert-instruction-sched-hints="variant=attention" | FileCheck %s -check-prefix=INSTR_HINT
// RUN: triton-opt %s -split-input-file -triton-amdgpu-insert-instruction-sched-hints="variant=attention" -triton-amdgpu-lower-insert-instruction-sched-hints -verify-diagnostics | FileCheck %s -check-prefix=LOWER_HINT

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>
#dot_op_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dot_op_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
// INSTR_HINT-LABEL: @insert_schedule_hint
// LOWER_HINT-LABEL: @insert_schedule_hint
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @insert_schedule_hint(
    %lb : index, %ub : index, %step : index,
    %arg0: tensor<128x128xf32, #dot_op_a>,
    %arg1: tensor<128x128xf32, #dot_op_b>,
    %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>
  ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    // INSTR_HINT: scf.for
    // INSTR_HINT-NEXT: amdgpu.instruction_sched_hint
    // INSTR_HINT-SAME: variant = #amdgpu.SchedHintVariant<attention>

    // LOWER_HINT: scf.for
    // LOWER_HINT-NEXT: rocdl.sched.barrier 0
    // LOWER_HINT-COUNT-2: tt.dot
    // LOWER_HINT: rocdl.iglp.opt 2
    // LOWER_HINT-NEXT: rocdl.sched.barrier 0
    // LOWER_HINT-NEXT: scf.yield
    %loop = scf.for %iv = %lb to %ub step %step iter_args(%c = %cst) -> (tensor<128x128xf32, #mma>) {
      %4 = tt.dot %arg0, %arg1, %c : tensor<128x128xf32, #dot_op_a> * tensor<128x128xf32, #dot_op_b> -> tensor<128x128xf32, #mma>
      %5 = math.exp2 %4 : tensor<128x128xf32, #mma>
      %6 = ttg.convert_layout %5 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #dot_op_a>
      %7 = tt.dot %6, %arg1, %c : tensor<128x128xf32, #dot_op_a> * tensor<128x128xf32, #dot_op_b> -> tensor<128x128xf32, #mma>
      scf.yield %7 : tensor<128x128xf32, #mma>
    }
    %8 = ttg.convert_layout %loop : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %8 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
