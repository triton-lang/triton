// RUN: triton-opt -split-input-file --nvws-aref-canonicalize %s | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  ttg.target = "cuda:100",
  "ttg.threads-per-warp" = 32 : i32,
  nvws.group.gr1 = {num_warps = 4 : i32, start_warp = 0 : i32},
  nvws.group.gr2 = {num_warps = 4 : i32, start_warp = 4 : i32},
  nvws.group.gr3 = {num_warps = 4 : i32, start_warp = 8 : i32}
  } {

    // CHECK-LABEL: @loop_token1
    tt.func public @loop_token1(
      %c0 : i32,
      %ub : index,
      %lb : index,
      %step : index,
      %ptr0: !tt.ptr<f16>
      ) {
      %desc = tt.reinterpret_tensor_descriptor %ptr0 { groups = [@nvws.group.gr1] }: !tt.ptr<f16> to !tt.tensordesc<tensor<128x64xf16, #shared>>
      %valA = tt.descriptor_load %desc[%c0, %c0] { groups = [@nvws.group.gr1] }: !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %opndA = ttg.local_alloc %valA { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
      %opndB = ttg.memdesc_trans %opndA {groups = [@nvws.group.gr1], order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>
      %acc, %tok = ttng.tmem_alloc { groups = [@nvws.group.gr1, @nvws.group.gr2] }: () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %true = arith.constant { groups = [@nvws.group.gr1, @nvws.group.gr2] } true
      %zero = arith.constant { groups = [@nvws.group.gr1, @nvws.group.gr2] } dense<0.000000e+00> : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %tok1 = ttng.tmem_store %zero, %acc[%tok], %true { groups = [@nvws.group.gr1] } : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %tok3 = scf.for %i = %lb to %ub step %step iter_args(%tok2 = %tok1) -> !ttg.async.token {
        %tok3 = ttng.tmem_store %zero, %acc[%tok2], %true { groups = [@nvws.group.gr2] } : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %tok4 = ttng.tc_gen5_mma %opndA, %opndB, %acc[%tok3], %true, %true { groups = [@nvws.group.gr1] } : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %val, %tok5 = ttng.tmem_load %acc[%tok4] { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
        // CHECK: %[[TOK:.*]] = ttng.tc_gen5_mma
        // CHECK: ttng.tmem_load {{.*}}[%[[TOK]]]
        // CHECK: scf.yield %[[TOK]]
        scf.yield %tok5 : !ttg.async.token
      } {groups = [@nvws.group.gr1, @nvws.group.gr2], groups.0 = [@nvws.group.gr1,@nvws.groups.gr2]}
      // CHECK: groups.0 = [@nvws.group.gr1]

      tt.return
    }

    // CHECK-LABEL: @loop_token2
    tt.func public @loop_token2(
      %c0 : i32,
      %ub : index,
      %lb : index,
      %step : index,
      %ptr0: !tt.ptr<f16>
      ) {
      %desc = tt.reinterpret_tensor_descriptor %ptr0 { groups = [@nvws.group.gr1] }: !tt.ptr<f16> to !tt.tensordesc<tensor<128x64xf16, #shared>>
      %valA = tt.descriptor_load %desc[%c0, %c0] { groups = [@nvws.group.gr1] }: !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %opndA = ttg.local_alloc %valA { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
      %opndB = ttg.memdesc_trans %opndA {groups = [@nvws.group.gr1], order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>
      %acc, %tok = ttng.tmem_alloc { groups = [@nvws.group.gr1, @nvws.group.gr2] }: () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %true = arith.constant { groups = [@nvws.group.gr1, @nvws.group.gr2] } true
      %zero = arith.constant { groups = [@nvws.group.gr1, @nvws.group.gr2] } dense<0.000000e+00> : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %tok1 = ttng.tmem_store %zero, %acc[%tok], %true { groups = [@nvws.group.gr1] } : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %tok3 = scf.for %i = %lb to %ub step %step iter_args(%tok2 = %tok1) -> !ttg.async.token {
        %tok3 = ttng.tmem_store %zero, %acc[%tok2], %true { groups = [@nvws.group.gr2] } : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %tok4 = ttng.tc_gen5_mma %opndA, %opndB, %acc[%tok3], %true, %true { groups = [@nvws.group.gr1] } : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %val, %tok5 = ttng.tmem_load %acc[%tok4] { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
        %tok6 = ttng.tmem_store %zero, %acc[%tok5], %true { groups = [@nvws.group.gr3] } : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %val1, %tok7 = ttng.tmem_load %acc[%tok6] { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
        // CHECK: ttng.tc_gen5_mma
        // CHECK: ttng.tmem_load
        // CHECK: %[[TOK:.*]] = ttng.tmem_store
        // CHECK: ttng.tmem_load {{.*}}[%[[TOK]]]
        // CHECK: scf.yield %[[TOK]]
        scf.yield %tok7 : !ttg.async.token
      } {groups = [@nvws.group.gr1, @nvws.group.gr2], groups.0 = [@nvws.group.gr1,@nvws.groups.gr2]}
      // CHECK: groups.0 = [@nvws.group.gr1]

      tt.return
    }

    // CHECK-LABEL: @tmem_alloc_no_source
    tt.func public @tmem_alloc_no_source() {
      //CHECK:     ttng.tmem_alloc {groups = [@nvws.group.gr1, @nvws.group.gr2]}
      %acc, %tok = ttng.tmem_alloc { groups = [@nvws.group.gr1] }: () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %zero = arith.constant { groups = [@nvws.group.gr1] } dense<0.000000e+00> : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %true = arith.constant { groups = [@nvws.group.gr1] } true
      %tok1 = ttng.tmem_store %zero, %acc[%tok], %true { groups = [@nvws.group.gr1] } : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %val, %tok2 = ttng.tmem_load %acc[%tok1] { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %tok3 = ttng.tmem_store %zero, %acc[%tok2], %true { groups = [@nvws.group.gr2] } : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      tt.return
    }

    // CHECK-LABEL: @patch_for_op_results(
    tt.func public @patch_for_op_results(
      %ub : index,
      %lb : index,
      %step : index,
      %ptr: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      ) {
      %valA = tt.load %ptr { groups = [@nvws.group.gr2] }: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %valB = tt.load %ptr { groups = [@nvws.group.gr1] }: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %tok3:2 = scf.for %i = %lb to %ub step %step iter_args(%val1 = %valA, %val2 = %valB) -> (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) {
        tt.store %ptr, %val1 { groups = [@nvws.group.gr2] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
        %val3 = arith.addf %val1, %val2 { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
        %val4 = arith.addf %val2, %val1 { groups = [@nvws.group.gr2] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
        scf.yield %val3, %val4 : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      } {groups = [@nvws.group.gr1, @nvws.group.gr2], groups.0 = [@nvws.group.gr2, @nvws.group.gr3], groups.1 = [@nvws.group.gr1, @nvws.group.gr3]}
      // CHECK: groups.0 = [@nvws.group.gr1], groups.1 = [@nvws.group.gr2]

      tt.return
    }

  }
