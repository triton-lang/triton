// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm='arch=gfx942' | FileCheck %s

#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @basic_concat(
    %arg0: tensor<32x64xf32, #blocked1>,
    %arg1: tensor<32x64xf32, #blocked1>,
    %arg2: tensor<32x64xf32, #blocked1>,
    %arg3: tensor<32x64xf32, #blocked1>,
    %arg4: tensor<32x64xf32, #blocked1>,
    %arg5: tensor<32x64xf32, #blocked1>,
    %arg6: tensor<32x64xf32, #blocked1>,
    %arg7: tensor<32x64xf32, #blocked1>) {
    // CHECK: llvm.func @basic_concat

    // CHECK-COUNT-8: %{{[0-9]*}} = llvm.extractvalue %arg0[{{[0-9]*}}] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
    // CHECK-COUNT-8: %{{[0-9]*}} = llvm.extractvalue %arg1[{{[0-9]*}}] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
    // CHECK-COUNT-8: %{{[0-9]*}} = llvm.extractvalue %arg2[{{[0-9]*}}] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
    // CHECK-COUNT-8: %{{[0-9]*}} = llvm.extractvalue %arg3[{{[0-9]*}}] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
    // CHECK-COUNT-8: %{{[0-9]*}} = llvm.extractvalue %arg4[{{[0-9]*}}] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
    // CHECK-COUNT-8: %{{[0-9]*}} = llvm.extractvalue %arg5[{{[0-9]*}}] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
    // CHECK-COUNT-8: %{{[0-9]*}} = llvm.extractvalue %arg6[{{[0-9]*}}] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
    // CHECK-COUNT-8: %{{[0-9]*}} = llvm.extractvalue %arg7[{{[0-9]*}}] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>

    // CHECK: %64 = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>

    // CHECK-COUNT-64: %{{[0-9]*}} = llvm.insertvalue %{{[0-9]*}}, %{{[0-9]*}}[{{[0-9]*}}] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>

    %1 = amdgpu.concat %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 [4, 2] :
    tensor<32x64xf32, #blocked1>,tensor<32x64xf32, #blocked1>, tensor<32x64xf32, #blocked1>, tensor<32x64xf32, #blocked1>, tensor<32x64xf32, #blocked1>, tensor<32x64xf32, #blocked1>, tensor<32x64xf32, #blocked1>, tensor<32x64xf32, #blocked1> -> tensor<128x128xf32, #blocked1>
    tt.return
  }
}
