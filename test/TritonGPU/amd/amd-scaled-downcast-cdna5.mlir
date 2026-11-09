// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1250" --canonicalize --cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#half_in = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#half_out = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#half_scale = #ttg.linear<{register = [], lane = [[0, 0], [0, 1], [0, 2], [0, 4], [0, 8]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scaled_downcast_fp4(
      %input: tensor<1x256xbf16, #unpacked>,
      %scale: tensor<1x32xi8, #scale>,
      %output: tensor<1x128x!tt.ptr<i8>, #blocked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp4
    // CHECK: llvm.insertelement {{.*}} : vector<8xbf16>
    // CHECK: %[[PACKED:.+]] = rocdl.cvt.scalef32.pk8.fp4.bf16
    // CHECK: llvm.lshr %[[PACKED]]
    %result = amdg.scaled_downcast_fp4 %input scale %scale {axis = 1 : i32} : tensor<1x256xbf16, #unpacked>, tensor<1x32xi8, #scale> -> tensor<1x128xi8, #blocked>
    tt.store %output, %result : tensor<1x128x!tt.ptr<i8>, #blocked>
    tt.return
  }

  tt.func public @scaled_downcast_fp16(
      %input: tensor<1x256xf16, #unpacked>,
      %scale: tensor<1x32xi8, #scale>,
      %output: tensor<1x128x!tt.ptr<i8>, #blocked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp16
    // CHECK: rocdl.cvt.scalef32.pk8.fp4.f16
    %result = amdg.scaled_downcast_fp4 %input scale %scale {axis = 1 : i32} : tensor<1x256xf16, #unpacked>, tensor<1x32xi8, #scale> -> tensor<1x128xi8, #blocked>
    tt.store %output, %result : tensor<1x128x!tt.ptr<i8>, #blocked>
    tt.return
  }

  tt.func public @scaled_downcast_fp32(
      %input: tensor<1x256xf32, #unpacked>,
      %scale: tensor<1x32xi8, #scale>,
      %output: tensor<1x128x!tt.ptr<i8>, #blocked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp32
    // CHECK: rocdl.cvt.scalef32.pk8.fp4.f32
    %result = amdg.scaled_downcast_fp4 %input scale %scale {axis = 1 : i32} : tensor<1x256xf32, #unpacked>, tensor<1x32xi8, #scale> -> tensor<1x128xi8, #blocked>
    tt.store %output, %result : tensor<1x128x!tt.ptr<i8>, #blocked>
    tt.return
  }

  tt.func public @scaled_downcast_fp8_e4m3_bf16(
      %input: tensor<1x256xbf16, #unpacked>,
      %scale: tensor<1x32xi8, #scale>,
      %output: tensor<1x256x!tt.ptr<f8E4M3FN>, #unpacked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp8_e4m3_bf16
    // CHECK: llvm.insertelement {{.*}} : vector<8xbf16>
    // CHECK: rocdl.cvt.scalef32.pk8.fp8.bf16
    %result = amdg.scaled_downcast_fp8 %input scale %scale {axis = 1 : i32} : tensor<1x256xbf16, #unpacked>, tensor<1x32xi8, #scale> -> tensor<1x256xf8E4M3FN, #unpacked>
    tt.store %output, %result : tensor<1x256x!tt.ptr<f8E4M3FN>, #unpacked>
    tt.return
  }

  tt.func public @scaled_downcast_fp8_e5m2_f16(
      %input: tensor<1x256xf16, #unpacked>,
      %scale: tensor<1x32xi8, #scale>,
      %output: tensor<1x256x!tt.ptr<f8E5M2>, #unpacked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp8_e5m2_f16
    // CHECK: rocdl.cvt.scalef32.pk8.bf8.f16
    %result = amdg.scaled_downcast_fp8 %input scale %scale {axis = 1 : i32} : tensor<1x256xf16, #unpacked>, tensor<1x32xi8, #scale> -> tensor<1x256xf8E5M2, #unpacked>
    tt.store %output, %result : tensor<1x256x!tt.ptr<f8E5M2>, #unpacked>
    tt.return
  }

  tt.func public @scaled_downcast_fp8_e4m3_f32(
      %input: tensor<1x256xf32, #unpacked>,
      %scale: tensor<1x32xi8, #scale>,
      %output: tensor<1x256x!tt.ptr<f8E4M3FN>, #unpacked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp8_e4m3_f32
    // CHECK: rocdl.cvt.scalef32.pk8.fp8.f32
    %result = amdg.scaled_downcast_fp8 %input scale %scale {axis = 1 : i32} : tensor<1x256xf32, #unpacked>, tensor<1x32xi8, #scale> -> tensor<1x256xf8E4M3FN, #unpacked>
    tt.store %output, %result : tensor<1x256x!tt.ptr<f8E4M3FN>, #unpacked>
    tt.return
  }

  tt.func public @scaled_downcast_fp4_partial_group(
      %input: tensor<1x128xbf16, #half_in>,
      %scale: tensor<1x16xi8, #half_scale>,
      %output: tensor<1x64x!tt.ptr<i8>, #half_out>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp4_partial_group
    // CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-DAG: %[[C4:.+]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG: %[[C5:.+]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK-DAG: %[[C6:.+]] = llvm.mlir.constant(6 : i32) : i32
    // CHECK-DAG: %[[C7:.+]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK-DAG: %[[ZERO_BITS:.+]] = llvm.mlir.constant(0 : i16) : i16
    // CHECK: %[[X3:.+]] = llvm.extractvalue %arg0[3]
    // CHECK: %[[ZERO:.+]] = llvm.bitcast %[[ZERO_BITS]] : i16 to bf16
    // CHECK: %[[V3:.+]] = llvm.insertelement %[[X3]], %{{.+}}[%[[C3]] : i32] : vector<8xbf16>
    // CHECK-NEXT: %[[V4:.+]] = llvm.insertelement %[[ZERO]], %[[V3]][%[[C4]] : i32] : vector<8xbf16>
    // CHECK-NEXT: %[[V5:.+]] = llvm.insertelement %[[ZERO]], %[[V4]][%[[C5]] : i32] : vector<8xbf16>
    // CHECK-NEXT: %[[V6:.+]] = llvm.insertelement %[[ZERO]], %[[V5]][%[[C6]] : i32] : vector<8xbf16>
    // CHECK-NEXT: %[[SRC:.+]] = llvm.insertelement %[[ZERO]], %[[V6]][%[[C7]] : i32] : vector<8xbf16>
    // CHECK: %[[PACKED:.+]] = rocdl.cvt.scalef32.pk8.fp4.bf16 %[[SRC]]
    // CHECK-NEXT: llvm.trunc %[[PACKED]] : i32 to i8
    // CHECK-NEXT: %[[HI:.+]] = llvm.lshr %[[PACKED]]
    // CHECK-NEXT: llvm.trunc %[[HI]] : i32 to i8
    // CHECK-NOT: llvm.lshr
    // CHECK: llvm.return
    %result = amdg.scaled_downcast_fp4 %input scale %scale {axis = 1 : i32} : tensor<1x128xbf16, #half_in>, tensor<1x16xi8, #half_scale> -> tensor<1x64xi8, #half_out>
    tt.store %output, %result : tensor<1x64x!tt.ptr<i8>, #half_out>
    tt.return
  }

  tt.func public @scaled_downcast_fp8_partial_group(
      %input: tensor<1x128xbf16, #half_in>,
      %scale: tensor<1x16xi8, #half_scale>,
      %output: tensor<1x128x!tt.ptr<f8E4M3FN>, #half_in>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp8_partial_group
    // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-DAG: %[[C4:.+]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG: %[[C5:.+]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK-DAG: %[[C6:.+]] = llvm.mlir.constant(6 : i32) : i32
    // CHECK-DAG: %[[C7:.+]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK-DAG: %[[ZERO_BITS:.+]] = llvm.mlir.constant(0 : i16) : i16
    // CHECK: %[[X3:.+]] = llvm.extractvalue %arg0[3]
    // CHECK: %[[ZERO:.+]] = llvm.bitcast %[[ZERO_BITS]] : i16 to bf16
    // CHECK: %[[V3:.+]] = llvm.insertelement %[[X3]], %{{.+}}[%[[C3]] : i32] : vector<8xbf16>
    // CHECK-NEXT: %[[V4:.+]] = llvm.insertelement %[[ZERO]], %[[V3]][%[[C4]] : i32] : vector<8xbf16>
    // CHECK-NEXT: %[[V5:.+]] = llvm.insertelement %[[ZERO]], %[[V4]][%[[C5]] : i32] : vector<8xbf16>
    // CHECK-NEXT: %[[V6:.+]] = llvm.insertelement %[[ZERO]], %[[V5]][%[[C6]] : i32] : vector<8xbf16>
    // CHECK-NEXT: %[[SRC:.+]] = llvm.insertelement %[[ZERO]], %[[V6]][%[[C7]] : i32] : vector<8xbf16>
    // CHECK: %[[PACKED:.+]] = rocdl.cvt.scalef32.pk8.fp8.bf16 %[[SRC]]
    // CHECK-NEXT: %[[LO:.+]] = llvm.extractelement %[[PACKED]][%[[C0]] : i32] : vector<2xi32>
    // CHECK-COUNT-4: llvm.trunc {{.*}} : i32 to i8
    // CHECK-NOT: llvm.extractelement %[[PACKED]]
    // CHECK: llvm.return
    %result = amdg.scaled_downcast_fp8 %input scale %scale {axis = 1 : i32} : tensor<1x128xbf16, #half_in>, tensor<1x16xi8, #half_scale> -> tensor<1x128xf8E4M3FN, #half_in>
    tt.store %output, %result : tensor<1x128x!tt.ptr<f8E4M3FN>, #half_in>
    tt.return
  }
}
