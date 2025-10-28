// RUN: triton-opt %s --pass-pipeline="builtin.module(tritongpu-coalesce)" -o /dev/null

module attributes {"triton_gpu.num_ctas" = 1 : i32,"triton_gpu.num_warps" = 4 : i32,"triton_gpu.target" = "cuda"} {

tt.func public @attn_fwd() {
  %c64_i32   = arith.constant 64 : i32
  %c0_i64    = arith.constant 0  : i64
  %c0_i32    = arith.constant 0  : i32
  %c1_i32    = arith.constant 1  : i32
  %c64_i64   = arith.constant 64 : i64

  %cmp = arith.cmpi slt, %c0_i32, %c1_i32 : i32

  %if_res:2 = scf.if %cmp -> (i32, i32) {
    scf.yield %c0_i32, %c64_i32 : i32, i32
  } else {
    scf.yield %c1_i32, %c64_i32 : i32, i32
  }

  %for_res = scf.for %arg0 = %if_res#0 to %if_res#1 step %c64_i32
             iter_args(%arg1 = %c0_i64) -> (i64):i32 {
    %next = arith.addi %arg1, %c64_i64 : i64
    scf.yield %next : i64
  }

  tt.return
}
}
