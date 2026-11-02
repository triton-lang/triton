// RUN: triton-opt %s -convert-warp-pipeline="gfx-arch=gfx942" -verify-diagnostics=only-expected | FileCheck %s

// The mfma-valu co-execution model is validated on gfx950 only: elsewhere the
// hint is ignored with a warning and no marker is emitted.

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-LABEL: tt.func @hinted_dot_stage_gfx942
// CHECK-NOT: rocdl.sched.group.barrier
tt.func @hinted_dot_stage_gfx942(%a: tensor<64x16xf16, #dot_a>, %b: tensor<16x64xf16, #dot_b>, %n: index, %ptr: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
  %v0 = arith.constant 0.0 : f32
  %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero, %q = %zero) -> (tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>) {
    // expected-warning @below {{schedule_hint="mfma-valu" is not supported on gfx942 and is ignored}}
    %d, %p = scf.execute_region -> (tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>) {
      %dot = tt.dot %a, %b, %acc : tensor<64x16xf16, #dot_a> * tensor<16x64xf16, #dot_b> -> tensor<64x64xf32, #mma>
      %e = math.exp2 %q : tensor<64x64xf32, #mma>
      scf.yield %dot, %e : tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>
    } {triton.warp_pipeline.stage = "dot", triton.warp_pipeline.schedule_hint = "mfma-valu"}
    scf.execute_region {
      tt.store %ptr, %v0 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "mem"}
    scf.yield %d, %p : tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>
  } {triton.warp_pipeline.pipelined_for}
  tt.return
}

}
