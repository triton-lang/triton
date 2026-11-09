// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --canonicalize --cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
// Register 1 repeats the element of register 0, so a thread's unique values are
// columns 0, 1, 8 and 9. With 8 values per scale, the pair (8, 9) is in the
// second scale block.
#bcast = #ttg.linear<{register = [[0, 1], [0, 0], [0, 8]], lane = [[0, 2], [0, 4], [0, 16], [0, 32], [0, 64], [0, 128]], warp = [], block = []}>
#bcast_scale = #ttg.linear<{register = [[0, 1]], lane = [[0, 0], [0, 0], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [], block = []}>
#one_scale = #ttg.linear<{register = [], lane = [[0, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [], block = []}>
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

  tt.func public @scaled_downcast_fp4_one_scale_per_byte(
      %input: tensor<1x512xf32, #unpacked>,
      %scale: tensor<1x256xi8, #blocked>,
      %output: tensor<1x256x!tt.ptr<i8>, #blocked>) {
    // Each packed output byte has its own scale, including the four bytes
    // written into the same destination register by separate conversions.
    // CHECK-LABEL: llvm.func @scaled_downcast_fp4_one_scale_per_byte
    // CHECK: %[[S0:.+]] = llvm.bitcast {{.*}} : i32 to f32
    // CHECK: %[[C0:.+]] = rocdl.cvt.scalef32.pk.fp4.f32 {{.*}}, %[[S0]] -> {{.*}}[0] : i32
    // CHECK: %[[S1:.+]] = llvm.bitcast {{.*}} : i32 to f32
    // CHECK: %[[C1:.+]] = rocdl.cvt.scalef32.pk.fp4.f32 {{.*}}, %[[S1]] -> %[[C0]][1] : i32
    // CHECK: %[[S2:.+]] = llvm.bitcast {{.*}} : i32 to f32
    // CHECK: %[[C2:.+]] = rocdl.cvt.scalef32.pk.fp4.f32 {{.*}}, %[[S2]] -> %[[C1]][2] : i32
    // CHECK: %[[S3:.+]] = llvm.bitcast {{.*}} : i32 to f32
    // CHECK: rocdl.cvt.scalef32.pk.fp4.f32 {{.*}}, %[[S3]] -> %[[C2]][3] : i32
    // CHECK-NOT: rocdl.cvt.scalef32.pk.fp4.f32
    %result = amdg.scaled_downcast_fp4 %input scale %scale {axis = 1 : i32} : tensor<1x512xf32, #unpacked>, tensor<1x256xi8, #blocked> -> tensor<1x256xi8, #blocked>
    tt.store %output, %result : tensor<1x256x!tt.ptr<i8>, #blocked>
    tt.return
  }

  tt.func public @scaled_downcast_fp8_broadcast_register(
      %input: tensor<1x256xf32, #bcast>,
      %scale: tensor<1x32xi8, #bcast_scale>,
      %output: tensor<1x256x!tt.ptr<f8E4M3FN>, #bcast>) {
    // Index unique values, not registers, so each pair gets its own scale.
    // CHECK-LABEL: llvm.func @scaled_downcast_fp8_broadcast_register
    // CHECK-DAG: %[[SCALE0:.+]] = llvm.extractvalue %{{.+}}[0] : !llvm.struct<(i8, i8)>
    // CHECK-DAG: %[[SCALE1:.+]] = llvm.extractvalue %{{.+}}[1] : !llvm.struct<(i8, i8)>
    // CHECK: llvm.zext %[[SCALE0]] : i8 to i32
    // CHECK: %[[S0:.+]] = llvm.bitcast {{.*}} : i32 to f32
    // CHECK: %[[C0:.+]] = rocdl.cvt.scalef32.pk.fp8.f32 {{.*}}, %[[S0]] -> {{.*}}[false] : vector<2xi16>
    // CHECK: llvm.zext %[[SCALE1]] : i8 to i32
    // CHECK: %[[S1:.+]] = llvm.bitcast {{.*}} : i32 to f32
    // CHECK: rocdl.cvt.scalef32.pk.fp8.f32 {{.*}}, %[[S1]] -> %[[C0]][true] : vector<2xi16>
    // CHECK-NOT: rocdl.cvt.scalef32.pk.fp8.f32
    %result = amdg.scaled_downcast_fp8 %input scale %scale {axis = 1 : i32} : tensor<1x256xf32, #bcast>, tensor<1x32xi8, #bcast_scale> -> tensor<1x256xf8E4M3FN, #bcast>
    tt.store %output, %result : tensor<1x256x!tt.ptr<f8E4M3FN>, #bcast>
    tt.return
  }

  tt.func public @scaled_downcast_fp8_one_value_per_thread(
      %input: tensor<1x64xf32, #scale>,
      %scale: tensor<1x32xi8, #one_scale>,
      %output: tensor<1x64x!tt.ptr<f8E4M3FN>, #scale>) {
    // A thread's single value is converted with a zero second element.
    // CHECK-LABEL: llvm.func @scaled_downcast_fp8_one_value_per_thread
    // CHECK: %[[ZERO:.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
    // CHECK: rocdl.cvt.scalef32.pk.fp8.f32 %{{.+}}, %[[ZERO]], %{{.+}} -> %{{.+}}[false] : vector<2xi16>
    // CHECK-NOT: rocdl.cvt.scalef32.pk.fp8.f32
    %result = amdg.scaled_downcast_fp8 %input scale %scale {axis = 1 : i32} : tensor<1x64xf32, #scale>, tensor<1x32xi8, #one_scale> -> tensor<1x64xf8E4M3FN, #scale>
    tt.store %output, %result : tensor<1x64x!tt.ptr<f8E4M3FN>, #scale>
    tt.return
  }
}
