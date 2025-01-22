// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm='arch=gfx942' | FileCheck %s

// CHECK-LABEL: llvm.func @fp32_to_bf16
// CHECK: llvm.inline_asm {{.*}} "v_cmp_u_f32 $0, $1, $2", "=s,v,v"
// CHECK: llvm.inline_asm {{.*}} "v_bfe_u32 $0, $1, $2, $3", "=v,v,v,v"
// CHECK: llvm.inline_asm {{.*}} "v_add3_u32 $0, $1, $2, $3", "=v,v,v,v"
// CHECK: llvm.inline_asm {{.*}} "v_cndmask_b32 $0, $1, $2, $3", "=v,v,v,s"

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fp32_to_bf16(
    %arg: tensor<256xf32, #blocked>) {
    %8 = arith.truncf %arg : tensor<256xf32, #blocked> to tensor<256xbf16, #blocked>
    tt.return
  }
}
