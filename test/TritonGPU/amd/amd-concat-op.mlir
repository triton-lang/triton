// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm='arch=gfx942' | FileCheck %s

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @concat_blocked(
    %arg0: tensor<32x64xf32, #blocked1>,
    %arg1: tensor<32x64xf32, #blocked1>,
    %arg2: tensor<32x64xf32, #blocked1>,
    %arg3: tensor<32x64xf32, #blocked1>,
    %arg4: tensor<32x64xf32, #blocked1>,
    %arg5: tensor<32x64xf32, #blocked1>,
    %arg6: tensor<32x64xf32, #blocked1>,
    %arg7: tensor<32x64xf32, #blocked1>) {
    // CHECK: llvm.func @concat_blocked

    // CHECK-COUNT-8: %{{.*}} = llvm.extractvalue %arg0[{{.*}}] : !llvm.struct
    // CHECK-COUNT-8: %{{.*}} = llvm.extractvalue %arg1[{{.*}}] : !llvm.struct
    // CHECK-COUNT-8: %{{.*}} = llvm.extractvalue %arg2[{{.*}}] : !llvm.struct
    // CHECK-COUNT-8: %{{.*}} = llvm.extractvalue %arg3[{{.*}}] : !llvm.struct
    // CHECK-COUNT-8: %{{.*}} = llvm.extractvalue %arg4[{{.*}}] : !llvm.struct
    // CHECK-COUNT-8: %{{.*}} = llvm.extractvalue %arg5[{{.*}}] : !llvm.struct
    // CHECK-COUNT-8: %{{.*}} = llvm.extractvalue %arg6[{{.*}}] : !llvm.struct
    // CHECK-COUNT-8: %{{.*}} = llvm.extractvalue %arg7[{{.*}}] : !llvm.struct

    // CHECK-COUNT-64: %{{[0-9]*}} = llvm.insertvalue %{{.*}} : !llvm.struct

    %1 = amdgpu.concat %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7:
    tensor<32x64xf32, #blocked1>,tensor<32x64xf32, #blocked1>, tensor<32x64xf32, #blocked1>, tensor<32x64xf32, #blocked1>, tensor<32x64xf32, #blocked1>, tensor<32x64xf32, #blocked1>, tensor<32x64xf32, #blocked1>, tensor<32x64xf32, #blocked1> -> tensor<128x128xf32, #blocked1>
    tt.return
  }
}

// -----

#src_layout = #ttg.linear<{register=[[0, 1], [0, 2], [0, 8], [0, 16], [0, 64], [64, 0]], lane=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp=[[0, 32], [32, 0]], block=[]}>
#dst_layout = #ttg.linear<{register=[[0, 1], [0, 2], [0, 8], [0, 16], [0, 64], [0, 128], [64, 0], [128, 0]], lane=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp=[[0, 32], [32, 0]], block=[]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @concat_ll_2d_1(
    %arg0: tensor<128x128xf32, #src_layout>,
    %arg1: tensor<128x128xf32, #src_layout>,
    %arg2: tensor<128x128xf32, #src_layout>,
    %arg3: tensor<128x128xf32, #src_layout>){
    // CHECK: llvm.func @concat_ll_2d_1

    // CHECK-COUNT-64: %{{.*}} = llvm.extractvalue %arg0[{{.*}}] : !llvm.struct
    // CHECK-COUNT-64: %{{.*}} = llvm.extractvalue %arg1[{{.*}}] : !llvm.struct
    // CHECK-COUNT-64: %{{.*}} = llvm.extractvalue %arg2[{{.*}}] : !llvm.struct
    // CHECK-COUNT-64: %{{.*}} = llvm.extractvalue %arg3[{{.*}}] : !llvm.struct
    // CHECK-COUNT-256: %{{.*}} = llvm.insertvalue %{{.*}} : !llvm.struct

    %1 = amdgpu.concat %arg0, %arg1, %arg2, %arg3:
    tensor<128x128xf32, #src_layout>, tensor<128x128xf32, #src_layout>, tensor<128x128xf32, #src_layout>, tensor<128x128xf32, #src_layout> -> tensor<256x256xf32, #dst_layout>
    tt.return
  }
}

// -----

#src_layout = #ttg.linear<{register=[[1, 0], [2, 0], [4, 0]], lane=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0], [16, 0]], warp=[[0, 16]], block=[]}>
#dst_layout = #ttg.linear<{register=[[1, 0], [2, 0], [4, 0], [32, 0], [0, 32]], lane=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0], [16, 0]], warp=[[0, 16]], block=[]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @concat_ll_2d_2(
    %arg0: tensor<32x32xf32, #src_layout>,
    %arg1: tensor<32x32xf32, #src_layout>,
    %arg2: tensor<32x32xf32, #src_layout>,
    %arg3: tensor<32x32xf32, #src_layout>){
    // CHECK: llvm.func @concat_ll_2d_2

    // CHECK-COUNT-8: %{{.*}} = llvm.extractvalue %arg0[{{.*}}] : !llvm.struct
    // CHECK-COUNT-8: %{{.*}} = llvm.extractvalue %arg1[{{.*}}] : !llvm.struct
    // CHECK-COUNT-8: %{{.*}} = llvm.extractvalue %arg2[{{.*}}] : !llvm.struct
    // CHECK-COUNT-8: %{{.*}} = llvm.extractvalue %arg3[{{.*}}] : !llvm.struct
    // CHECK-COUNT-32: %{{.*}} = llvm.insertvalue %{{.*}} : !llvm.struct

    %1 = amdgpu.concat %arg0, %arg1, %arg2, %arg3:
    tensor<32x32xf32, #src_layout>, tensor<32x32xf32, #src_layout>, tensor<32x32xf32, #src_layout>, tensor<32x32xf32, #src_layout> -> tensor<64x64xf32, #dst_layout>
    tt.return
  }
}

// -----

#src_layout = #ttg.linear<{register=[[1]], lane=[[2], [4], [8], [16], [32], [64]], warp=[[128]], block=[]}>
#dst_layout = #ttg.linear<{register=[[1], [256], [512]], lane=[[2], [4], [8], [16], [32], [64]], warp=[[128]], block=[]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @concat_ll_1d(
    %arg0: tensor<256xf32, #src_layout>,
    %arg1: tensor<256xf32, #src_layout>,
    %arg2: tensor<256xf32, #src_layout>,
    %arg3: tensor<256xf32, #src_layout>){
    // CHECK: llvm.func @concat_ll_1d

    // CHECK-COUNT-2: %{{.*}} = llvm.extractvalue %arg0[{{.*}}] : !llvm.struct
    // CHECK-COUNT-2: %{{.*}} = llvm.extractvalue %arg1[{{.*}}] : !llvm.struct
    // CHECK-COUNT-2: %{{.*}} = llvm.extractvalue %arg2[{{.*}}] : !llvm.struct
    // CHECK-COUNT-2: %{{.*}} = llvm.extractvalue %arg3[{{.*}}] : !llvm.struct
    // CHECK-COUNT-8: %{{.*}} = llvm.insertvalue %{{.*}} : !llvm.struct

    %1 = amdgpu.concat %arg0, %arg1, %arg2, %arg3:
    tensor<256xf32, #src_layout>, tensor<256xf32, #src_layout>, tensor<256xf32, #src_layout>, tensor<256xf32, #src_layout> -> tensor<1024xf32, #dst_layout>
    tt.return
  }
}

// -----

// Each input tensor broadcasts 4 registers along dimension 1, resulting in total 16 values per input.
// Output tensor do not have redundancy in registers and holds 8 values.
// Check that concat copies only 4 values from each input tensor, 8 in total.
#src_layout = #ttg.linear<{register=[[0, 0], [0, 0], [1, 0], [2, 0]], lane=[[0, 0], [0, 0], [0, 0], [4, 0], [8, 0], [16, 0]], warp=[[0, 0], [32, 0], [64, 0]], block=[]}>
#dst_layout = #ttg.linear<{register=[                [1, 0], [2, 0]], lane=[[0, 0], [0, 0], [0, 0], [4, 0], [8, 0], [16, 0]], warp=[[0, 0], [32, 0], [64, 0]], block=[]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @concat_from_broadcasted_tensor(%arg0: tensor<128x1xi32, #src_layout>, %arg1: tensor<128x1xi32, #src_layout> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: llvm.func @concat_from_broadcasted_tensor
    // CHECK-COUNT-16: %{{.*}} = llvm.extractvalue %arg0[{{.*}}] : !llvm.struct
    // CHECK-COUNT-16: %{{.*}} = llvm.extractvalue %arg1[{{.*}}] : !llvm.struct
    // CHECK-COUNT-8: %{{.*}} = llvm.insertvalue %{{.*}} : !llvm.struct
    %1 = amdgpu.concat %arg0, %arg1: tensor<128x1xi32, #src_layout>, tensor<128x1xi32, #src_layout> -> tensor<256x1xi32, #dst_layout>
    tt.return
  }
}

// -----

// Input tensors do not have redundancy in register and hold 4 values each.
// Output tensor broadcasts 4 registers along dimension 1, resulting in total 32 values.
// Check that concat duplicates 4 values from each input 4 times, resulting in total 32 values.
#src_layout = #ttg.linear<{register=[                [1, 0], [2, 0]], lane=[[0, 0], [0, 0], [0, 0], [4, 0], [8, 0], [16, 0]], warp=[[0, 0], [32, 0], [64, 0]], block=[]}>
#dst_layout = #ttg.linear<{register=[[0, 0], [0, 0], [1, 0], [2, 0]], lane=[[0, 0], [0, 0], [0, 0], [4, 0], [8, 0], [16, 0]], warp=[[0, 0], [32, 0], [64, 0]], block=[]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @concat_to_broadcasted_tensor(%arg0: tensor<128x1xi32, #src_layout>, %arg1: tensor<128x1xi32, #src_layout> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: llvm.func @concat_to_broadcasted_tensor
    // CHECK-COUNT-4: %{{.*}} = llvm.extractvalue %arg0[{{.*}}] : !llvm.struct
    // CHECK-COUNT-4: %{{.*}} = llvm.extractvalue %arg1[{{.*}}] : !llvm.struct
    // CHECK-COUNT-32: %{{.*}} = llvm.insertvalue %{{.*}} : !llvm.struct
    %1 = amdgpu.concat %arg0, %arg1: tensor<128x1xi32, #src_layout>, tensor<128x1xi32, #src_layout> -> tensor<256x1xi32, #dst_layout>
    tt.return
  }
}
