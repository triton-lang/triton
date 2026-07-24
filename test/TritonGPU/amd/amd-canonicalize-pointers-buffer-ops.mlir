// RUN: triton-opt %s --tritonamdgpu-canonicalize-pointers --split-input-file | FileCheck %s

// Regression test: when an AMD buffer op (amdg.buffer_load / amdg.buffer_store)
// has a loop-carried base pointer, canonicalize-pointers must reconstitute the
// scalar base from the split base+offset fat pointer instead of leaving behind
// a dangling zero-operand builtin.unrealized_conversion_cast. The leftover cast
// used to crash the downstream tritonamdgpu-convert-buffer-ops pass.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @v1_buffer_load
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // Each loop-carried base pointer is split into a (base, i32 offset) pair.
  // CHECK: scf.for {{.*}} iter_args(%[[A_BASE_ARG:.*]] = %{{.*}}, %[[A_OFF:.*]] = %{{.*}}, %[[B_BASE_ARG:.*]] = %{{.*}}, %[[B_OFF:.*]] = %{{.*}}, {{.*}}) -> (!tt.ptr<f16>, i32, !tt.ptr<f16>, i32, tensor<256x256xf32, #mma>)
  // The buffer base is rematerialized as a scalar tt.addptr from the carried offset.
  // CHECK: %[[A_BASE:.*]] = tt.addptr %[[A_BASE_ARG]], %[[A_OFF]] : !tt.ptr<f16>, i32
  // CHECK: amdg.buffer_load %[[A_BASE]]
  // CHECK: %[[B_BASE:.*]] = tt.addptr %[[B_BASE_ARG]], %[[B_OFF]] : !tt.ptr<f16>, i32
  // CHECK: amdg.buffer_load %[[B_BASE]]
  // CHECK: tt.addptr %{{.*}} : !tt.ptr<f16>, i32
  // CHECK: amdg.buffer_store
  // CHECK: tt.return
  tt.func public @v1_buffer_load(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %stride_am: i32 {tt.divisibility = 16 : i32}, %stride_bk: i32 {tt.divisibility = 16 : i32}, %stride_cm: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %num_pid_n = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %pid = tt.get_program_id x : i32
    %num_pid_n_0 = arith.addi %N, %num_pid_n : i32
    %num_pid_n_1 = arith.divsi %num_pid_n_0, %c256_i32 : i32
    %pid_m = arith.divsi %pid, %num_pid_n_1 : i32
    %pid_n = arith.remsi %pid, %num_pid_n_1 : i32
    %offs_am = arith.muli %pid_m, %c256_i32 : i32
    %offs_am_2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_am_3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_am_4 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_am_5 = tt.splat %offs_am : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_am_6 = tt.splat %offs_am : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_am_7 = arith.addi %offs_am_5, %offs_am_2 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_am_8 = arith.addi %offs_am_6, %offs_am_3 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_bn = arith.muli %pid_n, %c256_i32 : i32
    %offs_bn_9 = tt.splat %offs_bn : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_bn_10 = arith.addi %offs_bn_9, %offs_am_4 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %a_offsets = tt.expand_dims %offs_am_7 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %a_offsets_11 = tt.expand_dims %offs_am_8 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %a_offsets_12 = tt.splat %stride_am : i32 -> tensor<256x1xi32, #blocked>
    %a_offsets_13 = arith.muli %a_offsets, %a_offsets_12 : tensor<256x1xi32, #blocked>
    %a_offsets_14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %a_offsets_15 = tt.expand_dims %a_offsets_14 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %a_offsets_16 = tt.broadcast %a_offsets_13 : tensor<256x1xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %a_offsets_17 = tt.broadcast %a_offsets_15 : tensor<1x64xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %a_offsets_18 = arith.addi %a_offsets_16, %a_offsets_17 : tensor<256x64xi32, #blocked>
    %b_offsets = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %b_offsets_19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %b_offsets_20 = tt.expand_dims %b_offsets {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %b_offsets_21 = tt.expand_dims %b_offsets_19 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %b_offsets_22 = tt.splat %stride_bk : i32 -> tensor<64x1xi32, #blocked1>
    %b_offsets_23 = arith.muli %b_offsets_20, %b_offsets_22 : tensor<64x1xi32, #blocked1>
    %b_offsets_24 = tt.expand_dims %offs_bn_10 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %b_offsets_25 = tt.broadcast %b_offsets_23 : tensor<64x1xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
    %b_offsets_26 = tt.broadcast %b_offsets_24 : tensor<1x256xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
    %b_offsets_27 = arith.addi %b_offsets_25, %b_offsets_26 : tensor<64x256xi32, #blocked1>
    %0 = arith.addi %K, %c63_i32 : i32
    %1 = arith.divsi %0, %c64_i32 : i32
    %b_ptr_28 = arith.muli %stride_bk, %c64_i32 : i32
    %acc_29:3 = scf.for %k = %c0_i32 to %1 step %c1_i32 iter_args(%a_ptr_40 = %a_ptr, %b_ptr_41 = %b_ptr, %acc_42 = %acc) -> (!tt.ptr<f16>, !tt.ptr<f16>, tensor<256x256xf32, #mma>)  : i32 {
      %a = arith.muli %k, %c64_i32 : i32
      %a_43 = arith.subi %K, %a : i32
      %a_44 = tt.splat %a_43 : i32 -> tensor<1x64xi32, #blocked>
      %a_45 = arith.cmpi slt, %a_offsets_15, %a_44 : tensor<1x64xi32, #blocked>
      %a_46 = tt.broadcast %a_45 : tensor<1x64xi1, #blocked> -> tensor<256x64xi1, #blocked>
      %a_47 = amdg.buffer_load %a_ptr_40[%a_offsets_18], %a_46 : tensor<256x64xf16, #blocked>
      %b = tt.splat %a_43 : i32 -> tensor<64x1xi32, #blocked1>
      %b_48 = arith.cmpi slt, %b_offsets_21, %b : tensor<64x1xi32, #blocked1>
      %b_49 = tt.broadcast %b_48 : tensor<64x1xi1, #blocked1> -> tensor<64x256xi1, #blocked1>
      %b_50 = amdg.buffer_load %b_ptr_41[%b_offsets_27], %b_49 : tensor<64x256xf16, #blocked1>
      %a_51 = ttg.local_alloc %a_47 : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #shared, #smem>
      %a_52 = ttg.local_load %a_51 : !ttg.memdesc<256x64xf16, #shared, #smem> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_53 = ttg.local_alloc %b_50 : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared1, #smem>
      %b_54 = ttg.local_load %b_53 : !ttg.memdesc<64x256xf16, #shared1, #smem> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %acc_55 = tt.dot %a_52, %b_54, %acc_42 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x256xf32, #mma>
      %a_ptr_56 = tt.addptr %a_ptr_40, %c64_i32 : !tt.ptr<f16>, i32
      %b_ptr_57 = tt.addptr %b_ptr_41, %b_ptr_28 : !tt.ptr<f16>, i32
      scf.yield %a_ptr_56, %b_ptr_57, %acc_55 : !tt.ptr<f16>, !tt.ptr<f16>, tensor<256x256xf32, #mma>
    } {tt.num_stages = 0 : i32}
    %c = arith.truncf %acc_29#2 : tensor<256x256xf32, #mma> to tensor<256x256xf16, #mma>
    %c_offsets = tt.splat %stride_cm : i32 -> tensor<256x1xi32, #blocked1>
    %c_offsets_30 = arith.muli %c_offsets, %a_offsets_11 : tensor<256x1xi32, #blocked1>
    %c_offsets_31 = tt.broadcast %c_offsets_30 : tensor<256x1xi32, #blocked1> -> tensor<256x256xi32, #blocked1>
    %c_offsets_32 = tt.broadcast %b_offsets_24 : tensor<1x256xi32, #blocked1> -> tensor<256x256xi32, #blocked1>
    %c_offsets_33 = arith.addi %c_offsets_31, %c_offsets_32 : tensor<256x256xi32, #blocked1>
    %c_mask = tt.splat %M : i32 -> tensor<256x1xi32, #blocked1>
    %c_mask_34 = arith.cmpi slt, %a_offsets_11, %c_mask : tensor<256x1xi32, #blocked1>
    %c_mask_35 = tt.splat %N : i32 -> tensor<1x256xi32, #blocked1>
    %c_mask_36 = arith.cmpi slt, %b_offsets_24, %c_mask_35 : tensor<1x256xi32, #blocked1>
    %c_mask_37 = tt.broadcast %c_mask_34 : tensor<256x1xi1, #blocked1> -> tensor<256x256xi1, #blocked1>
    %c_mask_38 = tt.broadcast %c_mask_36 : tensor<1x256xi1, #blocked1> -> tensor<256x256xi1, #blocked1>
    %c_mask_39 = arith.andi %c_mask_37, %c_mask_38 : tensor<256x256xi1, #blocked1>
    %2 = ttg.convert_layout %c : tensor<256x256xf16, #mma> -> tensor<256x256xf16, #blocked1>
    amdg.buffer_store %2, %c_ptr[%c_offsets_33], %c_mask_39 : tensor<256x256xf16, #blocked1>
    tt.return
  }
}
