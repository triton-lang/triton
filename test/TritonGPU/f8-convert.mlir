// RUN: triton-reduce --convert-triton-gpu-to-llvm %s | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {
  "triton_gpu.num-ctas" = 1 : i32,
  "triton_gpu.num-warps" = 4 : i32,
  triton_gpu.target = "cuda:90",
  "triton_gpu.threads-per-warp" = 32 : i32
} {
  tt.func public @triton_(%arg0: !tt.ptr<f8E5M2FNUZ> {tt.divisibility = 16 : i32},
                          %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                          %arg5: i32 {tt.divisibility = 16 : i32}
                         ) attributes {noinline = false} {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked>
    %5 = tt.splat %arg5 : i32 -> tensor<256xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f8E5M2FNUZ> -> tensor<256x!tt.ptr<f8E5M2FNUZ>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f8E5M2FNUZ>, #blocked>, tensor<256xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<256x!tt.ptr<f8E5M2FNUZ>, #blocked>
    // CHECK: cvt.rn.f16x2.e5m2x2
    %19 = tt.fp_to_fp %9 : tensor<256xf8E5M2FNUZ, #blocked> -> tensor<256xf32, #blocked>
    %26 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %27 = tt.addptr %26, %4 : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %27, %19, %6 : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
