// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --canonicalize --cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scaled_downcast_fp4(
      %input: tensor<1x512xbf16, #unpacked>,
      %scale: tensor<1x64xi8, #scale>,
      %output: tensor<1x256x!tt.ptr<i8>, #blocked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp4
    // CHECK: %[[CVT0:.+]] = rocdl.cvt.scalef32.pk.fp4.bf16 {{.*}} -> {{.*}}[0] : i32
    // CHECK: %[[CVT1:.+]] = rocdl.cvt.scalef32.pk.fp4.bf16 {{.*}} -> %[[CVT0]][1] : i32
    // CHECK: %[[CVT2:.+]] = rocdl.cvt.scalef32.pk.fp4.bf16 {{.*}} -> %[[CVT1]][2] : i32
    // CHECK: %[[CVT3:.+]] = rocdl.cvt.scalef32.pk.fp4.bf16 {{.*}} -> %[[CVT2]][3] : i32
    // CHECK: llvm.lshr %[[CVT3]]
    %result = amdg.scaled_downcast_fp4 %input scale %scale {axis = 1 : i32} : tensor<1x512xbf16, #unpacked>, tensor<1x64xi8, #scale> -> tensor<1x256xi8, #blocked>
    tt.store %output, %result : tensor<1x256x!tt.ptr<i8>, #blocked>
    tt.return
  }

  tt.func public @scaled_downcast_fp16(
      %input: tensor<1x512xf16, #unpacked>,
      %scale: tensor<1x64xi8, #scale>,
      %output: tensor<1x256x!tt.ptr<i8>, #blocked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp16
    // CHECK: rocdl.cvt.scalef32.pk.fp4.f16
    %result = amdg.scaled_downcast_fp4 %input scale %scale {axis = 1 : i32} : tensor<1x512xf16, #unpacked>, tensor<1x64xi8, #scale> -> tensor<1x256xi8, #blocked>
    tt.store %output, %result : tensor<1x256x!tt.ptr<i8>, #blocked>
    tt.return
  }

  tt.func public @scaled_downcast_fp32(
      %input: tensor<1x512xf32, #unpacked>,
      %scale: tensor<1x64xi8, #scale>,
      %output: tensor<1x256x!tt.ptr<i8>, #blocked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp32
    // CHECK: rocdl.cvt.scalef32.pk.fp4.f32
    %result = amdg.scaled_downcast_fp4 %input scale %scale {axis = 1 : i32} : tensor<1x512xf32, #unpacked>, tensor<1x64xi8, #scale> -> tensor<1x256xi8, #blocked>
    tt.store %output, %result : tensor<1x256x!tt.ptr<i8>, #blocked>
    tt.return
  }

  tt.func public @scaled_downcast_fp8_e4m3_bf16(
      %input: tensor<1x512xbf16, #unpacked>,
      %scale: tensor<1x64xi8, #scale>,
      %output: tensor<1x512x!tt.ptr<f8E4M3FN>, #unpacked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp8_e4m3_bf16
    // CHECK: %[[CVT0:.+]] = rocdl.cvt.scalef32.pk.fp8.bf16 {{.*}} -> {{.*}}[false] : vector<2xi16>
    // CHECK: %[[CVT1:.+]] = rocdl.cvt.scalef32.pk.fp8.bf16 {{.*}} -> %[[CVT0]][true] : vector<2xi16>
    %result = amdg.scaled_downcast_fp8 %input scale %scale {axis = 1 : i32} : tensor<1x512xbf16, #unpacked>, tensor<1x64xi8, #scale> -> tensor<1x512xf8E4M3FN, #unpacked>
    tt.store %output, %result : tensor<1x512x!tt.ptr<f8E4M3FN>, #unpacked>
    tt.return
  }

  tt.func public @scaled_downcast_fp8_e5m2_f16(
      %input: tensor<1x512xf16, #unpacked>,
      %scale: tensor<1x64xi8, #scale>,
      %output: tensor<1x512x!tt.ptr<f8E5M2>, #unpacked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp8_e5m2_f16
    // CHECK: rocdl.cvt.scalef32.pk.bf8.f16
    %result = amdg.scaled_downcast_fp8 %input scale %scale {axis = 1 : i32} : tensor<1x512xf16, #unpacked>, tensor<1x64xi8, #scale> -> tensor<1x512xf8E5M2, #unpacked>
    tt.store %output, %result : tensor<1x512x!tt.ptr<f8E5M2>, #unpacked>
    tt.return
  }

  tt.func public @scaled_downcast_fp8_e4m3_f32(
      %input: tensor<1x512xf32, #unpacked>,
      %scale: tensor<1x64xi8, #scale>,
      %output: tensor<1x512x!tt.ptr<f8E4M3FN>, #unpacked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp8_e4m3_f32
    // CHECK: rocdl.cvt.scalef32.pk.fp8.f32
    %result = amdg.scaled_downcast_fp8 %input scale %scale {axis = 1 : i32} : tensor<1x512xf32, #unpacked>, tensor<1x64xi8, #scale> -> tensor<1x512xf8E4M3FN, #unpacked>
    tt.store %output, %result : tensor<1x512x!tt.ptr<f8E4M3FN>, #unpacked>
    tt.return
  }
}
