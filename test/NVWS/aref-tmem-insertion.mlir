// RUN: triton-opt -split-input-file --nvws-aref-tmem-insertion %s | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
module attributes {
  "ttg.num-ctas" = 1 : i32, 
  "ttg.num-warps" = 4 : i32,
  ttg.target = "cuda:100",
  "ttg.threads-per-warp" = 32 : i32,
  nvws.group.gr1 = {num_warps = 4 : i32, start_warp = 0 : i32},
  nvws.group.gr2 = {num_warps = 4 : i32, start_warp = 4 : i32}
  } {

    // CHECK-LABEL: load_tmem_immutable
    tt.func public @load_tmem_immutable(
      %c1 : f16, 
      %ptr: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      ) {
      // CHECK: %[[AREF_BUF:.*]] = ttng.tmem_alloc {aref_tmem_buffer} 
      // CHECK: %[[AREF:.*]] = ttng.aref_create %[[AREF_BUF]]
      %a = tt.load %ptr { groups = [@nvws.group.gr1] }: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %b = tt.splat %c1 { groups = [@nvws.group.gr1] }: f16 -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %c = arith.addf %a, %b { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %buf2 = ttng.tmem_alloc %c { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory>
      %val = ttng.tmem_load %buf2 { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      // CHECK: %[[BUF:.*]], %[[TOK:.*]] = ttng.aref_put.enter %[[AREF]]{{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: ttng.tmem_store {{.*}}, %[[BUF]][%[[TOK]]], {{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: ttng.aref_put.exit %[[AREF]]{{.*}}, producers = [#ttng.aref_producer<sttm>] {groups = [@nvws.group.gr1]}
      // CHECK: %[[BUF2:.*]], %[[TOK2:.*]] = ttng.aref_get.enter %[[AREF]]{{.*}} {groups = [@nvws.group.gr2]}
      // CHECK: %[[RET:.*]] = ttng.tmem_load %[[BUF2]] {groups = [@nvws.group.gr2]}
      // CHECK: ttng.aref_get.exit %[[AREF]]{{.*}}, consumers = [#ttng.aref_consumer<ldtm>] {groups = [@nvws.group.gr2]}
      tt.store %ptr, %val { groups = [@nvws.group.gr2] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.return
    }

    // CHECK-LABEL: load_tmem_mutable
    tt.func public @load_tmem_mutable(
      %c1 : f16, 
      %ptr: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      ) {
      // CHECK: %[[AREF_BUF:.*]] = ttng.tmem_alloc {aref_tmem_buffer} 
      // CHECK: %[[AREF:.*]] = ttng.aref_create %[[AREF_BUF]]
      %a = tt.load %ptr { groups = [@nvws.group.gr1] }: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %b = tt.splat %c1 { groups = [@nvws.group.gr1] }: f16 -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %c = arith.addf %a, %b { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %buf2, %tok = ttng.tmem_alloc { groups = [@nvws.group.gr1] }: () -> (!ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %true = arith.constant { groups = [@nvws.group.gr1] } true
      %tok1 = ttng.tmem_store %c, %buf2[%tok], %true { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable>
      %val, %tok3 = ttng.tmem_load %buf2[%tok1] { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      // CHECK: %[[BUF:.*]], %[[TOK:.*]] = ttng.aref_put.enter %[[AREF]]{{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: ttng.tmem_store {{.*}}, %[[BUF]][%[[TOK]]], {{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: ttng.aref_put.exit %[[AREF]]{{.*}}, producers = [#ttng.aref_producer<sttm>] {groups = [@nvws.group.gr1]}
      // CHECK: %[[BUF2:.*]], %[[TOK2:.*]] = ttng.aref_get.enter %[[AREF]]{{.*}} {groups = [@nvws.group.gr2]}
      // CHECK: %[[RET:.*]], %[[TOK3:.*]] = ttng.tmem_load %[[BUF2]][%[[TOK2]]] {groups = [@nvws.group.gr2]}
      // CHECK: ttng.aref_get.exit %[[AREF]]{{.*}}, consumers = [#ttng.aref_consumer<ldtm>] {groups = [@nvws.group.gr2]}

      tt.store %ptr, %val { groups = [@nvws.group.gr2] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.return
    }

    // CHECK-LABEL: @loop_token1
    tt.func public @loop_token1(
      %c0 : i32,
      %ub : index,
      %lb : index,
      %step : index,
      %zero : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>,
      %ptr0: !tt.ptr<f16>
      ) {
      // CHECK: %[[AREF_BUF:.*]] = ttng.tmem_alloc {aref_tmem_buffer} 
      // CHECK: %[[AREF:.*]] = ttng.aref_create %[[AREF_BUF]]
      %desc = tt.reinterpret_tensor_descriptor %ptr0 { groups = [@nvws.group.gr1] }: !tt.ptr<f16> to !tt.tensordesc<tensor<128x64xf16, #shared>>
      %valA = tt.descriptor_load %desc[%c0, %c0] { groups = [@nvws.group.gr1] }: !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %opndA = ttg.local_alloc %valA { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
      %opndB = ttg.memdesc_trans %opndA {groups = [@nvws.group.gr1], order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>
      %acc, %tok = ttng.tmem_alloc { groups = [@nvws.group.gr1, @nvws.group.gr2] }: () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %true = arith.constant { groups = [@nvws.group.gr1] } true
      %tok1 = ttng.tmem_store %zero, %acc[%tok], %true { groups = [@nvws.group.gr1] } : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: %[[BUF:.*]], %[[TOK:.*]] = ttng.aref_put.enter %[[AREF]]{{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: %[[TOK1:.*]] = ttng.tmem_store {{.*}}, %[[BUF]][%[[TOK]]], {{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: iter_args(%[[TOK2:.*]] = %[[TOK1]], %[[BUF2:.*]] = %[[BUF]]) 
      %tok3 = scf.for %i = %lb to %ub step %step iter_args(%tok2 = %tok1) -> !ttg.async.token {
        %tok3 = ttng.tc_gen5_mma %opndA, %opndB, %acc[%tok2], %true, %true { groups = [@nvws.group.gr1] } : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: put.exit %[[AREF]]{{.*}}, producers = [#ttng.aref_producer<umma>] {groups = [@nvws.group.gr1]}
        %val, %tok4 = ttng.tmem_load %acc[%tok3] { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
        // CHECK: %[[BUF2:.*]], %[[TOK2:.*]] = ttng.aref_get.enter %[[AREF]]{{.*}} {groups = [@nvws.group.gr2]}
        // CHECK: %[[RET:.*]] = ttng.tmem_load %[[BUF2]][%[[TOK2]]] {groups = [@nvws.group.gr2]}
        // CHECK: ttng.aref_get.exit %[[AREF]]{{.*}}, consumers = [#ttng.aref_consumer<ldtm>] {groups = [@nvws.group.gr2]}
        // CHECK: %[[BUF2:.*]], %[[TOK2:.*]] = ttng.aref_put.enter %[[AREF]]{{.*}} {groups = [@nvws.group.gr1]}
        // CHECK: scf.yield %[[TOK2]], %[[BUF2]]
        scf.yield %tok4 : !ttg.async.token
        // CHECK: ttng.aref_put.exit %[[AREF]]{{.*}}, producers = [#ttng.aref_producer<umma>] {groups = [@nvws.group.gr1]}
        // CHECK: %[[BUF2:.*]], %[[TOK2:.*]] = ttng.aref_get.enter %[[AREF]]{{.*}} {groups = [@nvws.group.gr2]}
        // CHECK: ttng.aref_get.exit %[[AREF]]{{.*}}, consumers = [#ttng.aref_consumer<ldtm>] {groups = [@nvws.group.gr2]}
      } {groups = [@nvws.group.gr1, @nvws.group.gr2], groups.0 = [@nvws.group.gr1]}

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
      // CHECK: %[[AREF_BUF:.*]] = ttng.tmem_alloc {aref_tmem_buffer} 
      // CHECK-NEXT: %[[AREF0:.*]] = ttng.aref_create %[[AREF_BUF]]
      // CHECK-NEXT: %[[AREF1:.*]] = ttng.aref_create %[[AREF_BUF]]
      // CHECK-NEXT: %[[AREF2:.*]] = ttng.aref_create %[[AREF_BUF]]
      %desc = tt.reinterpret_tensor_descriptor %ptr0 { groups = [@nvws.group.gr1] }: !tt.ptr<f16> to !tt.tensordesc<tensor<128x64xf16, #shared>>
      %valA = tt.descriptor_load %desc[%c0, %c0] { groups = [@nvws.group.gr1] }: !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %opndA = ttg.local_alloc %valA { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
      %opndB = ttg.memdesc_trans %opndA {groups = [@nvws.group.gr1], order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>
      %acc, %tok = ttng.tmem_alloc { groups = [@nvws.group.gr1, @nvws.group.gr2] }: () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %true = arith.constant { groups = [@nvws.group.gr1, @nvws.group.gr2] } true
      %zero = arith.constant { groups = [@nvws.group.gr1, @nvws.group.gr2] } dense<0.000000e+00> : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      // CHECK: %[[BUF:.*]], %[[TOK:.*]] = ttng.aref_put.enter %[[AREF2]]{{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: ttng.tmem_store {{.*}}, %[[BUF]][%[[TOK]]], {{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: ttng.aref_put.exit %[[AREF2]]{{.*}}, producers = [#ttng.aref_producer<sttm>] {groups = [@nvws.group.gr1]}
      %tok1 = ttng.tmem_store %zero, %acc[%tok], %true { groups = [@nvws.group.gr1] } : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: %[[BUF:.*]], %[[TOK:.*]] = ttng.aref_get.enter %[[AREF2]]{{.*}} {groups = [@nvws.group.gr2]}
      // CHECK: iter_args(%[[TOK2:.*]] = %[[TOK]], %[[BUF2:.*]] = %[[BUF]]) 
      %tok3 = scf.for %i = %lb to %ub step %step iter_args(%tok2 = %tok1) -> !ttg.async.token {
        // CHECK: ttng.aref_get.exit %[[AREF2]]{{.*}}, consumers = [#ttng.aref_consumer<umma>] {groups = [@nvws.group.gr2]}
        // CHECK: ttng.aref_put.enter %[[AREF2]]{{.*}} {groups = [@nvws.group.gr1]}
        %tok3 = ttng.tmem_store %zero, %acc[%tok2], %true { groups = [@nvws.group.gr1] } : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: ttng.aref_put.exit %[[AREF2]]{{.*}}
        // CHECK: ttng.aref_get.enter %[[AREF2]]{{.*}} {groups = [@nvws.group.gr2]}
        %tok4 = ttng.tc_gen5_mma %opndA, %opndB, %acc[%tok3], %true, %true { groups = [@nvws.group.gr2] } : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: ttng.aref_get.exit %[[AREF2]]{{.*}}, consumers = [#ttng.aref_consumer<umma>] {groups = [@nvws.group.gr2]}
        // CHECK: ttng.aref_get.enter %[[AREF1]]{{.*}} {groups = [@nvws.group.gr2]}
        // CHECK: ttng.aref_get.exit %[[AREF1]]{{.*}}, consumers = [#ttng.aref_consumer<umma>] {groups = [@nvws.group.gr2]}
        // CHECK: ttng.aref_put.enter %[[AREF1]]{{.*}} {groups = [@nvws.group.gr3]}
        %val, %tok5 = ttng.tmem_load %acc[%tok4] { groups = [@nvws.group.gr3] }: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
        // CHECK: ttng.aref_put.exit %[[AREF1]]{{.*}}, producers = [#ttng.aref_producer<sttm>] {groups = [@nvws.group.gr3]}
        // CHECK: ttng.aref_put.enter %[[AREF0]]{{.*}} {groups = [@nvws.group.gr3]}
        // CHECK: ttng.aref_put.exit %[[AREF0]]{{.*}}, producers = [#ttng.aref_producer<sttm>] {groups = [@nvws.group.gr3]}
        // CHECK: ttng.aref_get.enter %[[AREF0]]{{.*}} {groups = [@nvws.group.gr1]}
        // CHECK: ttng.aref_get.exit %[[AREF0]]{{.*}}, consumers = [#ttng.aref_consumer<ldtm>] {groups = [@nvws.group.gr1]}
        // CHECK: ttng.aref_put.enter %[[AREF2]]{{.*}} {groups = [@nvws.group.gr1]}
        // CHECK: ttng.aref_put.exit %[[AREF2]]{{.*}}, producers = [#ttng.aref_producer<sttm>] {groups = [@nvws.group.gr1]}
        // CHECK: %[[BUF2:.*]], %[[TOK2:.*]] = ttng.aref_get.enter %[[AREF2]]{{.*}} {groups = [@nvws.group.gr2]}
        // CHECK: scf.yield %[[TOK2]], %[[BUF2]]
        scf.yield %tok5 : !ttg.async.token
      } {groups = [@nvws.group.gr1, @nvws.group.gr2], groups.0 = [@nvws.group.gr2]}
      // CHECK: ttng.aref_get.exit %[[AREF2]]{{.*}}, consumers = [#ttng.aref_consumer<umma>] {groups = [@nvws.group.gr2]}

      tt.return
    }

    // CHECK-LABEL: mma_tmem_immutable
    tt.func public @mma_tmem_immutable(
      %c0 : i32,
      %c1 : f16, 
      %ptr0 : !tt.ptr<f16>,
      %cnst : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>,
      %ptr: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      ) {

      // CHECK-DAG: %[[AREF_BUF1:.*]] = ttng.tmem_alloc {aref_tmem_buffer} : () -> !ttg.memdesc<1x128x64xf16
      // CHECK-DAG: %[[AREF0:.*]] = ttng.aref_create %[[AREF_BUF1]] 
      // CHECK-DAG: %[[AREF_BUF2:.*]] = ttng.tmem_alloc {aref_tmem_buffer} : () -> !ttg.memdesc<1x128x128xf32
      // CHECK-DAG: %[[AREF1:.*]] = ttng.aref_create %[[AREF_BUF2]] 
      
      %a = tt.load %ptr { groups = [@nvws.group.gr1] }: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %b = tt.splat %c1 { groups = [@nvws.group.gr1] }: f16 -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %c = arith.addf %a, %b { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %desc = tt.reinterpret_tensor_descriptor %ptr0 { groups = [@nvws.group.gr1] }: !tt.ptr<f16> to !tt.tensordesc<tensor<128x64xf16, #shared>>
      %valA = tt.descriptor_load %desc[%c0, %c0] { groups = [@nvws.group.gr1] }: !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %opndA = ttg.local_alloc %valA { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
      %opndB = ttg.memdesc_trans %opndA {groups = [@nvws.group.gr1], order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>

      %true = arith.constant { groups = [@nvws.group.gr2] } true
      %false = arith.constant { groups = [@nvws.group.gr2] } false
      %acc, %tok = ttng.tmem_alloc { groups = [@nvws.group.gr3, @nvws.group.gr2] }: () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK: %[[BUF1:.*]], %[[TOK1:.*]] = ttng.aref_put.enter %[[AREF1]]{{.*}} {groups = [@nvws.group.gr2]}

      %buf2 = ttng.tmem_alloc %c { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory>
      // CHECK: %[[BUF0:.*]], %[[TOK0:.*]] = ttng.aref_put.enter %[[AREF0]]{{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: ttng.tmem_store {{.*}}, %[[BUF0]][%[[TOK0]]], {{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: ttng.aref_put.exit %[[AREF0]]{{.*}}, producers = [#ttng.aref_producer<sttm>] {groups = [@nvws.group.gr1]}

      %tok2 = ttng.tmem_store %cnst, %acc[%tok], %true { groups = [@nvws.group.gr2] } : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_store {{.*}}, %[[BUF1]][%[[TOK1]]], {{.*}} {groups = [@nvws.group.gr2]}
      // CHECK: ttng.aref_put.exit %[[AREF1]]{{.*}}, producers = [#ttng.aref_producer<sttm>] {groups = [@nvws.group.gr2]}

      // CHECK: %[[BUF1:.*]], %[[TOK1:.*]] = ttng.aref_get.enter %[[AREF1]]{{.*}} {groups = [@nvws.group.gr3]}
      // CHECK: %[[BUF0:.*]], %[[TOK0:.*]] = ttng.aref_get.enter %[[AREF0]]{{.*}} {groups = [@nvws.group.gr3]}
      %tok3 = ttng.tc_gen5_mma %buf2, %opndB, %acc[%tok2], %true, %true { groups = [@nvws.group.gr3] } : !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma %[[BUF0]], {{.*}}, %[[BUF1]][%[[TOK1]]], {{.*}}, {{.*}} {groups = [@nvws.group.gr3]}
      // CHECK: ttng.aref_get.exit %[[AREF0]]{{.*}}, consumers = [#ttng.aref_consumer<umma>] {groups = [@nvws.group.gr3]}
      // CHECK: ttng.aref_get.exit %[[AREF1]]{{.*}}, consumers = [#ttng.aref_consumer<umma>] {groups = [@nvws.group.gr3]}

      tt.return
    }

    // CHECK-LABEL: tmem_sequenced_3groups
    tt.func public @tmem_sequenced_3groups(
      %c0 : i32,
      %c1 : f16, 
      %ptr0 : !tt.ptr<f16>,
      %cnst : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>,
      %ptr: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      ) {

      // CHECK: %[[AREF_BUF0:.*]] = ttng.tmem_alloc {aref_tmem_buffer}
      // CHECK-NEXT: %[[AREF0:.*]] = ttng.aref_create %[[AREF_BUF0]] {first_get}
      // CHECK-NEXT: %[[AREF1:.*]] = ttng.aref_create %[[AREF_BUF0]] 
      // CHECK-NEXT: %[[AREF2:.*]] = ttng.aref_create %[[AREF_BUF0]] 
      
      %a = tt.load %ptr { groups = [@nvws.group.gr1] }: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %b = tt.splat %c1 { groups = [@nvws.group.gr1] }: f16 -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %c = arith.addf %a, %b { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %desc = tt.reinterpret_tensor_descriptor %ptr0 { groups = [@nvws.group.gr1] }: !tt.ptr<f16> to !tt.tensordesc<tensor<128x64xf16, #shared>>
      %valA = tt.descriptor_load %desc[%c0, %c0] { groups = [@nvws.group.gr1] }: !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %opndA = ttg.local_alloc %valA { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
      %opndB = ttg.memdesc_trans %opndA {groups = [@nvws.group.gr1], order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>

      %true = arith.constant { groups = [@nvws.group.gr2] } true
      %false = arith.constant { groups = [@nvws.group.gr2] } false
      %acc, %tok = ttng.tmem_alloc { groups = [@nvws.group.gr1, @nvws.group.gr3, @nvws.group.gr2] }: () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      %buf2 = ttng.tmem_alloc %c { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory>
      %buf3 = ttng.tmem_alloc %c { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory>
 
      // CHECK: ttng.aref_put.enter %[[AREF2]]{{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: ttng.tc_gen5_mma
      %tok3 = ttng.tc_gen5_mma %buf2, %opndB, %acc[%tok], %false, %true { groups = [@nvws.group.gr1] } : !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.aref_put.exit %[[AREF2]]{{.*}}, producers = [#ttng.aref_producer<umma>] {groups = [@nvws.group.gr1]}

      // CHECK: ttng.aref_get.enter %[[AREF2]]{{.*}} {groups = [@nvws.group.gr2]}
      // CHECK: ttng.tmem_load
      %val1, %tok4 = ttng.tmem_load %acc[%tok3] { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      // CHECK: ttng.aref_get.exit %[[AREF2]]{{.*}}, consumers = [#ttng.aref_consumer<ldtm>] {groups = [@nvws.group.gr2]}
      
      // CHECK: ttng.aref_put.enter %[[AREF2]]{{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: ttng.tc_gen5_mma
      %tok5 = ttng.tc_gen5_mma %buf3, %opndB, %acc[%tok4], %false, %true { groups = [@nvws.group.gr1] } : !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.aref_put.exit %[[AREF2]]{{.*}}, producers = [#ttng.aref_producer<umma>] {groups = [@nvws.group.gr1]}

      // CHECK: ttng.aref_put.enter %[[AREF1]]{{.*}} {groups = [@nvws.group.gr1]}
      // CHECK-NEXT: arith.constant
      // CHECK-NEXT: ttng.aref_put.exit %[[AREF1]]{{.*}}, producers = [#ttng.aref_producer<umma>] {groups = [@nvws.group.gr1]}

      // CHECK: ttng.aref_get.enter %[[AREF1]]{{.*}} {groups = [@nvws.group.gr3]}
      // CHECK: ttng.tmem_load
      %val2, %tok6 = ttng.tmem_load %acc[%tok5] { groups = [@nvws.group.gr3] }: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      // CHECK: ttng.aref_get.exit %[[AREF1]]{{.*}}, consumers = [#ttng.aref_consumer<ldtm>] {groups = [@nvws.group.gr3]}

      // CHECK: ttng.aref_get.enter %[[AREF0]]{{.*}} {groups = [@nvws.group.gr3]}
      // CHECK-NEXT: arith.constant
      // CHECK-NEXT: ttng.aref_get.exit %[[AREF0]]{{.*}}, consumers = [#ttng.aref_consumer<ldtm>] {groups = [@nvws.group.gr3]}

      // CHECK: ttng.aref_get.enter %[[AREF2]]{{.*}} {groups = [@nvws.group.gr2]}
      // CHECK-NEXT: arith.constant
      // CHECK-NEXT: ttng.aref_get.exit %[[AREF2]]{{.*}}, consumers = [#ttng.aref_consumer<ldtm>] {groups = [@nvws.group.gr2]}

      tt.return
    }

  }
