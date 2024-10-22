// RUN: triton-opt %s -split-input-file -tritonamdgpu-bypass-lds-for-dot-operand -tritonamdgpu-stream-pipeline-v2=num_stages=2 -tritongpu-remove-layout-conversions | FileCheck %s

// For Bypass LDS optimization to be efficient we need collaboration of 3 passes:
//     1) Pipelining pass: This is because pipelining in registers is necessary.
//     2) Bypass LDS pass: To convert load from blocked->dot layout.
//     3) Remove layout conversion pass: To remove blocked->dot layout by changing layout of all ops that form tensor of pointers to dot layout.
// Check that all of the optimizations were done properly to create efficient IR.

// CHECK-LABEL: bypass_lds
//       CHECK-NOT: triton_gpu.convert_layout %{{.*}} : tensor<{{.*}}, #blocked2> -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
//       CHECK: %[[DOT_LOAD_1:.+]] = tt.load %{{.*}} : tensor<64x256x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
//       CHECK: %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %[[DOT_LOAD_1]], %{{.*}} = %{{.*}}) -> (tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked>, i32, !tt.memdesc<256x64xf16, #shared, #triton_gpu.shared_memory, mutable>, tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, tensor<64x256x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>)
//       CHECK: %[[DOT_LOAD_2:.+]] = tt.load %{{.*}} : tensor<64x256x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
//       CHECK: scf.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[DOT_LOAD_2:.+]], %{{.*}} : tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked>, i32, !tt.memdesc<256x64xf16, #shared, #triton_gpu.shared_memory, mutable>, tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, tensor<64x256x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>

#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#sliced_blocked1 = #triton_gpu.slice<{parent=#blocked1, dim=0}>
#sliced_blocked2 = #triton_gpu.slice<{parent=#blocked2, dim=0}>
#mfma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 16], isTransposed = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx90a", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @bypass_lds(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<256x64xi32, #blocked1>
    %cst_0 = arith.constant dense<64> : tensor<64x256xi32, #blocked2>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c63_i32 = arith.constant 63 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mfma>
    %a_ptr_splat = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #blocked1>
    %a_tmp0 = tt.make_range {end = 64: i32, start = 0: i32} : tensor<64xi32, #sliced_blocked1>
    %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : tensor<64xi32, #sliced_blocked1> -> tensor<1x64xi32, #blocked1>
    %a_offs = tt.broadcast %a_tmp1 : tensor<1x64xi32, #blocked1> -> tensor<256x64xi32, #blocked1>
    %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
    %b_ptr_splat = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked2>
    %b_tmp0 = tt.make_range {end = 256: i32, start = 0: i32} : tensor<256xi32, #sliced_blocked2>
    %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : tensor<256xi32, #sliced_blocked2> -> tensor<1x256xi32, #blocked2>
    %b_offs = tt.broadcast %b_tmp1 : tensor<1x256xi32, #blocked2> -> tensor<64x256xi32, #blocked2>
    %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<64x256x!tt.ptr<f16>, #blocked2>, tensor<64x256xi32, #blocked2>
    %56:3 = scf.for %arg10 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg11 = %cst_1, %arg12 = %a_ptr_init, %arg13 = %b_ptr_init) -> (tensor<256x256xf32, #mfma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked2>)  : i32 {
      %74 = tt.load %arg12 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %75 = tt.load %arg13 : tensor<64x256x!tt.ptr<f16>, #blocked2>
      %76 = triton_gpu.convert_layout %74 : tensor<256x64xf16, #blocked1> -> tensor<256x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
      %77 = triton_gpu.convert_layout %75 : tensor<64x256xf16, #blocked2> -> tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
      %78 = tt.dot %76, %77, %arg11 : tensor<256x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<256x256xf32, #mfma>
      %79 = tt.addptr %arg12, %cst : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %80 = tt.addptr %arg13, %cst_0 : tensor<64x256x!tt.ptr<f16>, #blocked2>, tensor<64x256xi32, #blocked2>
      scf.yield %78, %79, %80 : tensor<256x256xf32, #mfma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked2>
    }
    %addr_res = tt.splat %arg2 : !tt.ptr<f16> -> tensor<256x256x!tt.ptr<f16>, #blocked3>
    %57 = arith.truncf %56#0 : tensor<256x256xf32, #mfma> to tensor<256x256xf16, #mfma>
    %72 = triton_gpu.convert_layout %addr_res : tensor<256x256x!tt.ptr<f16>, #blocked3> -> tensor<256x256x!tt.ptr<f16>, #mfma>
    tt.store %72, %57 : tensor<256x256x!tt.ptr<f16>, #mfma>
    tt.return
  }
}

// -----

// Check that bypass LDS optimization is not done because warpsPerCTA condition is not satisfied.

// CHECK-LABEL: no_bypass_lds_warps_per_cta
//       CHECK-NOT: tt.load %{{.*}} : tensor<64x256x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#sliced_blocked1 = #triton_gpu.slice<{parent=#blocked1, dim=0}>
#sliced_blocked2 = #triton_gpu.slice<{parent=#blocked2, dim=0}>
#mfma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 16], isTransposed = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx90a", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @no_bypass_lds_warps_per_cta(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<256x64xi32, #blocked1>
    %cst_0 = arith.constant dense<64> : tensor<64x256xi32, #blocked2>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c63_i32 = arith.constant 63 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mfma>
    %a_ptr_splat = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #blocked1>
    %a_tmp0 = tt.make_range {end = 64: i32, start = 0: i32} : tensor<64xi32, #sliced_blocked1>
    %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : tensor<64xi32, #sliced_blocked1> -> tensor<1x64xi32, #blocked1>
    %a_offs = tt.broadcast %a_tmp1 : tensor<1x64xi32, #blocked1> -> tensor<256x64xi32, #blocked1>
    %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
    %b_ptr_splat = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked2>
    %b_tmp0 = tt.make_range {end = 256: i32, start = 0: i32} : tensor<256xi32, #sliced_blocked2>
    %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : tensor<256xi32, #sliced_blocked2> -> tensor<1x256xi32, #blocked2>
    %b_offs = tt.broadcast %b_tmp1 : tensor<1x256xi32, #blocked2> -> tensor<64x256xi32, #blocked2>
    %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<64x256x!tt.ptr<f16>, #blocked2>, tensor<64x256xi32, #blocked2>
    %56:3 = scf.for %arg10 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg11 = %cst_1, %arg12 = %a_ptr_init, %arg13 = %b_ptr_init) -> (tensor<256x256xf32, #mfma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked2>)  : i32 {
      %74 = tt.load %arg12 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %75 = tt.load %arg13 : tensor<64x256x!tt.ptr<f16>, #blocked2>
      %76 = triton_gpu.convert_layout %74 : tensor<256x64xf16, #blocked1> -> tensor<256x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
      %77 = triton_gpu.convert_layout %75 : tensor<64x256xf16, #blocked2> -> tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
      %78 = tt.dot %76, %77, %arg11 : tensor<256x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<256x256xf32, #mfma>
      %79 = tt.addptr %arg12, %cst : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %80 = tt.addptr %arg13, %cst_0 : tensor<64x256x!tt.ptr<f16>, #blocked2>, tensor<64x256xi32, #blocked2>
      scf.yield %78, %79, %80 : tensor<256x256xf32, #mfma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked2>
    }
    %addr_res = tt.splat %arg2 : !tt.ptr<f16> -> tensor<256x256x!tt.ptr<f16>, #blocked3>
    %57 = arith.truncf %56#0 : tensor<256x256xf32, #mfma> to tensor<256x256xf16, #mfma>
    %72 = triton_gpu.convert_layout %addr_res : tensor<256x256x!tt.ptr<f16>, #blocked3> -> tensor<256x256x!tt.ptr<f16>, #mfma>
    tt.store %72, %57 : tensor<256x256x!tt.ptr<f16>, #mfma>
    tt.return
  }
}

// -----

// Check that bypass LDS optimization is not done because kWidth condition is not satisfied.

// CHECK-LABEL: no_bypass_lds_kWidth
//       CHECK-NOT: tt.load %{{.*}} : tensor<64x256x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#sliced_blocked1 = #triton_gpu.slice<{parent=#blocked1, dim=0}>
#sliced_blocked2 = #triton_gpu.slice<{parent=#blocked2, dim=0}>
#mfma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 16], isTransposed = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx90a", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @no_bypass_lds_kWidth(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<256x64xi32, #blocked1>
    %cst_0 = arith.constant dense<64> : tensor<64x256xi32, #blocked2>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c63_i32 = arith.constant 63 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mfma>
    %a_ptr_splat = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #blocked1>
    %a_tmp0 = tt.make_range {end = 64: i32, start = 0: i32} : tensor<64xi32, #sliced_blocked1>
    %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : tensor<64xi32, #sliced_blocked1> -> tensor<1x64xi32, #blocked1>
    %a_offs = tt.broadcast %a_tmp1 : tensor<1x64xi32, #blocked1> -> tensor<256x64xi32, #blocked1>
    %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
    %b_ptr_splat = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked2>
    %b_tmp0 = tt.make_range {end = 256: i32, start = 0: i32} : tensor<256xi32, #sliced_blocked2>
    %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : tensor<256xi32, #sliced_blocked2> -> tensor<1x256xi32, #blocked2>
    %b_offs = tt.broadcast %b_tmp1 : tensor<1x256xi32, #blocked2> -> tensor<64x256xi32, #blocked2>
    %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<64x256x!tt.ptr<f16>, #blocked2>, tensor<64x256xi32, #blocked2>
    %56:3 = scf.for %arg10 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg11 = %cst_1, %arg12 = %a_ptr_init, %arg13 = %b_ptr_init) -> (tensor<256x256xf32, #mfma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked2>)  : i32 {
      %74 = tt.load %arg12 : tensor<256x64x!tt.ptr<f16>, #blocked1>
      %75 = tt.load %arg13 : tensor<64x256x!tt.ptr<f16>, #blocked2>
      %76 = triton_gpu.convert_layout %74 : tensor<256x64xf16, #blocked1> -> tensor<256x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %77 = triton_gpu.convert_layout %75 : tensor<64x256xf16, #blocked2> -> tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %78 = tt.dot %76, %77, %arg11 : tensor<256x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<256x256xf32, #mfma>
      %79 = tt.addptr %arg12, %cst : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
      %80 = tt.addptr %arg13, %cst_0 : tensor<64x256x!tt.ptr<f16>, #blocked2>, tensor<64x256xi32, #blocked2>
      scf.yield %78, %79, %80 : tensor<256x256xf32, #mfma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked2>
    }
    %addr_res = tt.splat %arg2 : !tt.ptr<f16> -> tensor<256x256x!tt.ptr<f16>, #blocked3>
    %57 = arith.truncf %56#0 : tensor<256x256xf32, #mfma> to tensor<256x256xf16, #mfma>
    %72 = triton_gpu.convert_layout %addr_res : tensor<256x256x!tt.ptr<f16>, #blocked3> -> tensor<256x256x!tt.ptr<f16>, #mfma>
    tt.store %72, %57 : tensor<256x256x!tt.ptr<f16>, #mfma>
    tt.return
  }
}
