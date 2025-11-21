// RUN: triton-opt %s --split-input-file --tritongpu-hoist-tmem-alloc --tritongpu-partition-scheduling -allow-unregistered-dialect | FileCheck %s
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @mxfp_matmul(%a_desc: !tt.tensordesc<tensor<128x128xf8E5M2, #shared>>, %b_desc: !tt.tensordesc<tensor<256x128xf8E5M2, #shared>>, %output_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %a_scale_ptr_35: tensor<128x4x!tt.ptr<i8>, #blocked>, %b_scale: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %stride_cm: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %accumulator = arith.constant false
    %a_scale_ptr = arith.constant dense<64> : tensor<128x1xi32, #blocked>
    %b_scale_ptr = arith.constant dense<64> : tensor<256x1xi32, #blocked>
    %cst = arith.constant dense<4> : tensor<128x4xi32, #blocked>
    %cst_8 = arith.constant dense<4> : tensor<256x4xi32, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %accumulator_9 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #linear>
    %pid = tt.get_program_id x : i32
    %num_pid_m = arith.addi %M, %c127_i32 : i32
    %num_pid_m_10 = arith.divsi %num_pid_m, %c128_i32 : i32
    %pid_m = arith.remsi %pid, %num_pid_m_10 : i32
    %pid_n = arith.divsi %pid, %num_pid_m_10 : i32
    %offs_am_scale = arith.muli %pid_m, %c128_i32 : i32
    %offs_am_scale_11 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_am_scale_12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_am_scale_13 = tt.splat %offs_am_scale : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_am_scale_14 = tt.splat %offs_am_scale : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_am_scale_15 = arith.addi %offs_am_scale_13, %offs_am_scale_11 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_am_scale_16 = arith.addi %offs_am_scale_14, %offs_am_scale_12 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_am_scale_17 = tt.splat %M : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_am_scale_18 = arith.remsi %offs_am_scale_15, %offs_am_scale_17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_bn_scale = arith.muli %pid_n, %c256_i32 : i32
    %offs_bn_scale_19 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_bn_scale_20 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_bn_scale_21 = tt.splat %offs_bn_scale : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_bn_scale_22 = tt.splat %offs_bn_scale : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_bn_scale_23 = arith.addi %offs_bn_scale_21, %offs_bn_scale_19 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_bn_scale_24 = arith.addi %offs_bn_scale_22, %offs_bn_scale_20 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_bn_scale_25 = tt.splat %N : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_bn_scale_26 = arith.remsi %offs_bn_scale_23, %offs_bn_scale_25 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %a_scale_ptr_27 = tt.expand_dims %offs_am_scale_18 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %a_scale_ptr_28 = arith.muli %a_scale_ptr_27, %a_scale_ptr : tensor<128x1xi32, #blocked>
    %a_scale_ptr_31 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %a_scale_ptr_32 = tt.expand_dims %a_scale_ptr_31 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x4xi32, #blocked>
    %b_scale_ptr_36 = tt.expand_dims %offs_bn_scale_26 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %b_scale_ptr_37 = arith.muli %b_scale_ptr_36, %b_scale_ptr : tensor<256x1xi32, #blocked>
    %b_scale_ptr_38 = tt.splat %b_scale : !tt.ptr<i8> -> tensor<256x1x!tt.ptr<i8>, #blocked>
    %b_scale_ptr_39 = tt.addptr %b_scale_ptr_38, %b_scale_ptr_37 : tensor<256x1x!tt.ptr<i8>, #blocked>, tensor<256x1xi32, #blocked>
    %b_scale_ptr_40 = tt.broadcast %b_scale_ptr_39 : tensor<256x1x!tt.ptr<i8>, #blocked> -> tensor<256x4x!tt.ptr<i8>, #blocked>
    %b_scale_ptr_41 = tt.broadcast %a_scale_ptr_32 : tensor<1x4xi32, #blocked> -> tensor<256x4xi32, #blocked>
    %b_scale_ptr_42 = tt.addptr %b_scale_ptr_40, %b_scale_ptr_41 : tensor<256x4x!tt.ptr<i8>, #blocked>, tensor<256x4xi32, #blocked>
    %0 = arith.addi %K, %c127_i32 : i32
    %1 = arith.divsi %0, %c128_i32 : i32
    %accumulator_43, %accumulator_44 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %accumulator_45 = ttng.tmem_store %accumulator_9, %accumulator_43[%accumulator_44], %true : tensor<128x256xf32, #linear> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %accumulator_46:5 = scf.for %accumulator_63 = %c0_i32 to %1 step %c1_i32 iter_args(%arg18 = %c0_i32, %a_scale_ptr_64 = %a_scale_ptr_35, %b_scale_ptr_65 = %b_scale_ptr_42, %accumulator_66 = %accumulator, %accumulator_67 = %accumulator_45) -> (i32, tensor<128x4x!tt.ptr<i8>, #blocked>, tensor<256x4x!tt.ptr<i8>, #blocked>, i1, !ttg.async.token)  : i32 {
      %a = tt.descriptor_load %a_desc[%offs_am_scale, %arg18] {loop.cluster = 3 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> -> tensor<128x128xf8E5M2, #blocked2>
      %a_68 = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf8E5M2, #blocked2>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem>
      %b = tt.descriptor_load %b_desc[%offs_bn_scale, %arg18] {loop.cluster = 3 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<256x128xf8E5M2, #shared>> -> tensor<256x128xf8E5M2, #blocked2>
      %scale_a = tt.load %a_scale_ptr_64 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<128x4x!tt.ptr<i8>, #blocked>
      %scale_a_69 = ttg.local_alloc %scale_a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x4xi8, #blocked>) -> !ttg.memdesc<128x4xi8, #shared1, #smem>
      %accumulator_70 = ttg.local_load %scale_a_69 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x4xi8, #shared1, #smem> -> tensor<128x4xi8, #linear1>
      %scale_b = tt.load %b_scale_ptr_65 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<256x4x!tt.ptr<i8>, #blocked>
      %scale_b_71 = ttg.local_alloc %scale_b {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<256x4xi8, #blocked>) -> !ttg.memdesc<256x4xi8, #shared1, #smem>
      %accumulator_72 = ttg.local_load %scale_b_71 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<256x4xi8, #shared1, #smem> -> tensor<256x4xi8, #linear2>
      %accumulator_73 = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<256x128xf8E5M2, #blocked2>) -> !ttg.memdesc<256x128xf8E5M2, #shared, #smem>
      %accumulator_74 = ttg.memdesc_trans %accumulator_73 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<256x128xf8E5M2, #shared, #smem> -> !ttg.memdesc<128x256xf8E5M2, #shared2, #smem>
      %accumulator_75 = ttng.tmem_alloc %accumulator_70 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x4xi8, #linear1>) -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
      %accumulator_76 = ttng.tmem_alloc %accumulator_72 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<256x4xi8, #linear2>) -> !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>
      %accumulator_77 = ttng.tc_gen5_mma_scaled %a_68, %accumulator_74, %accumulator_43[%accumulator_67], %accumulator_75, %accumulator_76, %accumulator_66, %true lhs = e5m2 rhs = e5m2 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf8E5M2, #shared, #smem>, !ttg.memdesc<128x256xf8E5M2, #shared2, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>
      %offs_k = arith.addi %arg18, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : i32
      %a_scale_ptr_78 = tt.addptr %a_scale_ptr_64, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x4x!tt.ptr<i8>, #blocked>, tensor<128x4xi32, #blocked>
      %b_scale_ptr_79 = tt.addptr %b_scale_ptr_65, %cst_8 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<256x4x!tt.ptr<i8>, #blocked>, tensor<256x4xi32, #blocked>
      scf.yield %offs_k, %a_scale_ptr_78, %b_scale_ptr_79, %true, %accumulator_77 : i32, tensor<128x4x!tt.ptr<i8>, #blocked>, tensor<256x4x!tt.ptr<i8>, #blocked>, i1, !ttg.async.token
    } {tt.num_stages = 3 : i32, tt.scheduled_max_stage = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
