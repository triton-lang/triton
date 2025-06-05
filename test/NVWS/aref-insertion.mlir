// RUN: triton-opt -split-input-file --nvws-aref-insertion %s | FileCheck %s

// CHECK: #[[$SHARED0:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
// CHECK: #[[$SHARED1:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
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
    // CHECK-LABEL: tma_load
    tt.func public @tma_load(%ptr0: !tt.ptr<f16>, %c0: i32) {
      // CHECK: %[[AREF_BUF:.*]] = ttg.local_alloc {aref_buffer} {{.*}} #[[$SHARED0]]
      // CHECK: %[[AREF:.*]] = ttng.aref_create %[[AREF_BUF]] 
      %desc = tt.reinterpret_tensor_descriptor %ptr0 { groups = [@nvws.group.gr1] }: !tt.ptr<f16> to !tt.tensordesc<tensor<128x64xf16, #shared>>
      %a = tt.descriptor_load %desc[%c0, %c0] { groups = [@nvws.group.gr1] }: !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %b = ttg.local_alloc %a { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
      // CHECK: %[[SRC:.*]] = ttg.local_alloc {{.*}} #shared
      // CHECK: %[[DST:.*]], %[[DST_TOK:.*]] = ttng.aref_put.enter %[[AREF]][{{.*}}] {aref_tag = "aref_0", groups = [@nvws.group.gr1]} 
      // CHECK: ttng.aref_copy %[[SRC]], %[[DST]] {groups = [@nvws.group.gr1]} 
      // CHECK: ttng.aref_put.exit %[[AREF]][{{.*}}], producers = [#ttng.aref_producer<none>] {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
      // CHECK: %[[SRC1:.*]], %[[SRC1_TOK:.*]] = ttng.aref_get.enter %[[AREF]][{{.*}}] {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      // CHECK: %[[DST1:.*]] = ttng.aref_clone %[[SRC1]] {groups = [@nvws.group.gr2]} {{.*}} #[[$SHARED0]]
      // CHECK: ttng.aref_get.exit %[[AREF]][{{.*}}], consumers = [#ttng.aref_consumer<none>] {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      // CHECK: ttg.local_load %[[DST1]] {groups = [@nvws.group.gr2]} {{.*}}
      %c = ttg.local_load %b { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %ptr1 = tt.splat %ptr0 { groups = [@nvws.group.gr2] }: !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.store %ptr1, %c { groups = [@nvws.group.gr2] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.return
    }

    // CHECK-LABEL: tma_load2
    tt.func public @tma_load2(%ptr0: !tt.ptr<f16>, %c0: i32) {
      // CHECK: %[[AREF_BUF:.*]] = ttg.local_alloc {aref_buffer} {{.*}} #[[$SHARED0]]
      // CHECK: %[[AREF:.*]] = ttng.aref_create %[[AREF_BUF]]
      %desc = tt.reinterpret_tensor_descriptor %ptr0 { groups = [@nvws.group.gr1] }: !tt.ptr<f16> to !tt.tensordesc<tensor<128x64xf16, #shared>>
      %a = tt.descriptor_load %desc[%c0, %c0] { groups = [@nvws.group.gr1] }: !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      // CHECK: %[[DST:.*]], %[[DST_TOK:.*]] = ttng.aref_put.enter %[[AREF]][{{.*}}] {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
      // CHECK: ttg.local_store {{.*}}, %[[DST]] {groups = [@nvws.group.gr1]}
      // CHECK: ttng.aref_put.exit %[[AREF]][{{.*}}], producers = [#ttng.aref_producer<tmaldg>] {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
      // CHECK: %[[SRC:.*]], %[[SRC_TOK:.*]] = ttng.aref_get.enter %[[AREF]][{{.*}}] {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      // CHECK: ttg.local_load %[[SRC]] {groups = [@nvws.group.gr2]}
      // CHECK: ttng.aref_get.exit %[[AREF]][{{.*}}], consumers = [#ttng.aref_consumer<lds>] {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      %b = ttg.local_alloc %a { groups = [@nvws.group.gr2] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
      %c = ttg.local_load %b { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %ptr1 = tt.splat %ptr0 { groups = [@nvws.group.gr2] }: !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.store %ptr1, %c { groups = [@nvws.group.gr2] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.return
    }

    // CHECK-LABEL: tma_load_scale
    tt.func public @tma_load_scale(%ptr0: !tt.ptr<f16>, %c0: i32, %c1: f16) {
      // CHECK: %[[AREF_BUF:.*]] = ttg.local_alloc {aref_buffer} {{.*}} #[[$SHARED0]]
      // CHECK: %[[AREF:.*]] = ttng.aref_create %[[AREF_BUF]]
      %desc = tt.reinterpret_tensor_descriptor %ptr0 { groups = [@nvws.group.gr1] }: !tt.ptr<f16> to !tt.tensordesc<tensor<128x64xf16, #shared>>
      %a = tt.descriptor_load %desc[%c0, %c0] { groups = [@nvws.group.gr1] }: !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      // CHECK: %[[DST:.*]], %[[DST_TOK:.*]] = ttng.aref_put.enter %[[AREF]][{{.*}}] {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
      // CHECK: ttg.local_store {{.*}}, %[[DST]] {groups = [@nvws.group.gr1]}
      // CHECK: ttng.aref_put.exit %[[AREF]][{{.*}}], producers = [#ttng.aref_producer<tmaldg>] {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
      // CHECK: %[[SRC:.*]], %[[SRC_TOK:.*]] = ttng.aref_get.enter %[[AREF]][{{.*}}] {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      // CHECK: ttg.local_load %[[SRC]] {groups = [@nvws.group.gr2]}
      // CHECK: ttng.aref_get.exit %[[AREF]][{{.*}}], consumers = [#ttng.aref_consumer<lds>] {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      %c1splat = tt.splat %c1 { groups = [@nvws.group.gr2] }: f16 -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %a1 = arith.addf %a, %c1splat { groups = [@nvws.group.gr2] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %b = ttg.local_alloc %a1 { groups = [@nvws.group.gr2] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
      %c = ttg.local_load %b { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %ptr1 = tt.splat %ptr0 { groups = [@nvws.group.gr2] }: !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.store %ptr1, %c { groups = [@nvws.group.gr2] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.return
    }

    // CHECK-LABEL: tma_load_scale2
    tt.func public @tma_load_scale2(%ptr0: !tt.ptr<f16>, %c0: i32, %c1: f16) {
      // CHECK: %[[AREF_BUF:.*]] = ttg.local_alloc {aref_buffer} : () -> !ttg.memdesc<1x128x64xf16, #[[$SHARED0]], #smem, mutable>
      // CHECK: %[[AREF:.*]] = ttng.aref_create %[[AREF_BUF]]
      %desc = tt.reinterpret_tensor_descriptor %ptr0 { groups = [@nvws.group.gr1] }: !tt.ptr<f16> to !tt.tensordesc<tensor<128x64xf16, #shared>>
      %a = tt.descriptor_load %desc[%c0, %c0] { groups = [@nvws.group.gr1] }: !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %c1splat = tt.splat %c1 { groups = [@nvws.group.gr1] }: f16 -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %a1 = arith.addf %a, %c1splat { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      // CHECK: %[[DST:.*]], %[[DST_TOK:.*]] = ttng.aref_put.enter %[[AREF]][{{.*}}] {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
      // CHECK: ttg.local_store {{.*}}, %[[DST]] {groups = [@nvws.group.gr1]}
      // CHECK: ttng.aref_put.exit %[[AREF]][{{.*}}], producers = [#ttng.aref_producer<sts>] {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
      // CHECK: %[[SRC:.*]], %[[SRC_TOK:.*]] = ttng.aref_get.enter %[[AREF]][{{.*}}] {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      // CHECK: ttg.local_load %[[SRC]] {groups = [@nvws.group.gr2]}
      // CHECK: ttng.aref_get.exit %[[AREF]][{{.*}}], consumers = [#ttng.aref_consumer<lds>] {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      %b = ttg.local_alloc %a1 { groups = [@nvws.group.gr2] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
      %c = ttg.local_load %b { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %ptr1 = tt.splat %ptr0 { groups = [@nvws.group.gr2] }: !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.store %ptr1, %c { groups = [@nvws.group.gr2] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.return
    }

    // CHECK-LABEL: load1
    tt.func public @load1(%ptr: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) {
    // CHECK: %[[AREF_BUF:.*]] = ttg.local_alloc {aref_buffer} {{.*}} #[[$SHARED0]]
    // CHECK: %[[AREF:.*]] = ttng.aref_create %[[AREF_BUF]]
      %a = tt.load %ptr { groups = [@nvws.group.gr1] }: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    // CHECK: %[[DST:.*]], %[[DST_TOK:.*]] = ttng.aref_put.enter %[[AREF]][{{.*}}] {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
    // CHECK: ttg.local_store {{.*}}, %[[DST]] {groups = [@nvws.group.gr1]}
    // CHECK: ttng.aref_put.exit %[[AREF]][{{.*}}], producers = [#ttng.aref_producer<ldgsts>] {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
    // CHECK: %[[SRC:.*]], %[[SRC_TOK:.*]] = ttng.aref_get.enter %[[AREF]][{{.*}}] {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
    // CHECK: ttg.local_load %[[SRC]] {groups = [@nvws.group.gr2]}
    // CHECK: ttng.aref_get.exit %[[AREF]][{{.*}}], consumers = [#ttng.aref_consumer<lds>] {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      tt.store %ptr, %a { groups = [@nvws.group.gr2] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.return
    }

    // CHECK-LABEL: load_scale1
    tt.func public @load_scale1(%c1 : f16, %ptr: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) {
      // CHECK: %[[AREF_BUF:.*]] = ttg.local_alloc {aref_buffer} {{.*}} #[[$SHARED1]]
      // CHECK: %[[AREF:.*]] = ttng.aref_create %[[AREF_BUF]]
      %a = tt.load %ptr { groups = [@nvws.group.gr1] }: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      // CHECK: %[[DST:.*]], %[[DST_TOK:.*]] = ttng.aref_put.enter %[[AREF]][{{.*}}] {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
      // CHECK: ttg.local_store {{.*}}, %[[DST]] {groups = [@nvws.group.gr1]}
      // CHECK: ttng.aref_put.exit %[[AREF]][{{.*}}], producers = [#ttng.aref_producer<ldgsts>] {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
      // CHECK: %[[SRC:.*]], %[[SRC_TOK:.*]] = ttng.aref_get.enter %[[AREF]][{{.*}}] {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      // CHECK: ttg.local_load %[[SRC]] {groups = [@nvws.group.gr2]}
      // CHECK: ttng.aref_get.exit %[[AREF]][{{.*}}], consumers = [#ttng.aref_consumer<lds>] {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      %b = tt.splat %c1 { groups = [@nvws.group.gr2] }: f16 -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %c = arith.addf %a, %b { groups = [@nvws.group.gr2] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.store %ptr, %c { groups = [@nvws.group.gr2] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.return
    }

    // CHECK-LABEL: load_scale2
    tt.func public @load_scale2(%c1 : f16, %ptr: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) {
      // CHECK: %[[AREF_BUF:.*]] = ttg.local_alloc {aref_buffer} {{.*}} #[[$SHARED1]]
      // CHECK: %[[AREF:.*]] = ttng.aref_create %[[AREF_BUF]]
      %a = tt.load %ptr { groups = [@nvws.group.gr1] }: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %b = tt.splat %c1 { groups = [@nvws.group.gr1] }: f16 -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %c = arith.addf %a, %b { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      // CHECK: %[[DST:.*]], %[[DST_TOK:.*]] = ttng.aref_put.enter %[[AREF]][{{.*}}] {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
      // CHECK: ttg.local_store {{.*}}, %[[DST]] {groups = [@nvws.group.gr1]}
      // CHECK: ttng.aref_put.exit %[[AREF]][{{.*}}], producers = [#ttng.aref_producer<sts>] {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
      // CHECK: %[[SRC:.*]], %[[SRC_TOK:.*]] = ttng.aref_get.enter %[[AREF]][{{.*}}] {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      // CHECK: ttg.local_load %[[SRC]] {groups = [@nvws.group.gr2]}
      // CHECK: ttng.aref_get.exit %[[AREF]][{{.*}}], consumers = [#ttng.aref_consumer<lds>] {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      tt.store %ptr, %c { groups = [@nvws.group.gr2] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.return
    }

    // CHECK-LABEL: load_tmem_immutable
    tt.func public @load_tmem_immutable(
      %c1 : f16, 
      %ptr: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      ) {
      // CHECK: %[[AREF_BUF:.*]] = ttng.tmem_alloc {aref_buffer} 
      // CHECK: %[[AREF:.*]] = ttng.aref_create %[[AREF_BUF]]
      %a = tt.load %ptr { groups = [@nvws.group.gr1] }: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %b = tt.splat %c1 { groups = [@nvws.group.gr1] }: f16 -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %c = arith.addf %a, %b { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %buf2 = ttng.tmem_alloc %c { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory>
      // CHECK: %[[BUF:.*]] = ttng.tmem_alloc {{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: %[[DST:.*]], %[[TOK:.*]] = ttng.aref_put.enter %[[AREF]] 
      // CHECK: ttng.aref_copy %[[BUF]], %[[DST]] {groups = [@nvws.group.gr1]}
      // CHECK: ttng.aref_put.exit %[[AREF]][{{.*}}], producers = [#ttng.aref_producer<none>]
      // CHECK: %[[SRC:.*]], %[[SRC_TOK:.*]] = ttng.aref_get.enter 
      // CHECK: %[[BUF:.*]] = ttng.aref_clone %[[SRC]] {groups = [@nvws.group.gr2]} 
      // CHECK: ttng.aref_get.exit 
      // CHECK: ttng.tmem_load %[[BUF]] {groups = [@nvws.group.gr2]}

      %val = ttng.tmem_load %buf2 { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.store %ptr, %val { groups = [@nvws.group.gr2] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.return
    }

    // CHECK-LABEL: load_tmem_mutable
    tt.func public @load_tmem_mutable(
      %c1 : f16, 
      %ptr: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      ) {
      // CHECK: %[[AREF_BUF:.*]] = ttng.tmem_alloc {aref_buffer} 
      // CHECK: %[[AREF:.*]] = ttng.aref_create %[[AREF_BUF]]
      %a = tt.load %ptr { groups = [@nvws.group.gr1] }: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %b = tt.splat %c1 { groups = [@nvws.group.gr1] }: f16 -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %c = arith.addf %a, %b { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %buf2, %tok = ttng.tmem_alloc { groups = [@nvws.group.gr1] }: () -> (!ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK: %[[BUF:.*]], %[[TOK:.*]] = ttng.tmem_alloc {groups = [@nvws.group.gr1]}
      %true = arith.constant { groups = [@nvws.group.gr1] } true
      %tok1 = ttng.tmem_store %c, %buf2[%tok], %true { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: %[[TOK:.*]] = ttng.tmem_store {{.*}} {{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: %[[DST:.*]], %[[DST_TOK:.*]] = ttng.aref_put.enter %[[AREF]] 
      // CHECK: ttng.aref_copy %[[BUF]][%[[TOK]]], %[[DST]][%[[DST_TOK]]] {groups = [@nvws.group.gr1]}
      // CHECK: ttng.aref_put.exit %[[AREF]][{{.*}}], producers = [#ttng.aref_producer<none>]
      // CHECK: %[[SRC:.*]], %[[SRC_TOK:.*]] = ttng.aref_get.enter 
      // CHECK: %[[TOK1:.*]] = ttng.aref_copy %[[SRC]][%[[SRC_TOK]]], %[[BUF]] {groups = [@nvws.group.gr2]} 
      // CHECK: ttng.aref_get.exit 
      // CHECK: ttng.tmem_load %[[BUF]][%[[TOK1]]] {groups = [@nvws.group.gr2]}
      %val, %tok3 = ttng.tmem_load %buf2[%tok1] { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.store %ptr, %val { groups = [@nvws.group.gr2] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.return
    }

    // CHECK-LABEL: arefphi
    tt.func public @arefphi(
      %c1 : f16, 
      %ptr: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>,
      %ptr1: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      ) {
      %a = tt.load %ptr { groups = [@nvws.group.gr1] }: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %b = tt.splat %c1 { groups = [@nvws.group.gr1] }: f16 -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %c = arith.addf %a, %b { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %buf2 = ttng.tmem_alloc %c { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory>
      // CHECK: %[[ALLOC:.*]] = ttng.tmem_alloc %[[SRCVAL:.*]]
      %val = ttng.tmem_load %buf2 { groups = [@nvws.group.gr1, @nvws.group.gr2] }: !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      // CHECK: %[[SRC:.*]], %[[SRCTOK:.*]] = ttng.aref_get.enter 
      // CHECK: %[[BUF:.*]] = ttng.aref_clone %[[SRC]]
      // CHECK: %[[RET:.*]] = ttng.aref_phi %[[ALLOC]], %[[BUF]]
      // CHECK: ttng.tmem_load %[[RET]]
      tt.store %ptr, %val { groups = [@nvws.group.gr2] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.store %ptr1, %val { groups = [@nvws.group.gr1] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.return
    }

    // CHECK-LABEL: arefphi_token
    tt.func public @arefphi_token(
      %c1 : f16, 
      %ptr: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>,
      %ptr1: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      ) {
      %a = tt.load %ptr { groups = [@nvws.group.gr1] }: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %b = tt.splat %c1 { groups = [@nvws.group.gr1] }: f16 -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %c = arith.addf %a, %b { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %buf, %tok = ttng.tmem_alloc { groups = [@nvws.group.gr1] }: () -> (!ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %true = arith.constant { groups = [@nvws.group.gr1] } true
      %tok1 = ttng.tmem_store %c, %buf[%tok], %true { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.aref_put.enter  {{.*}} {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
      // CHECK: %[[TOK1:.*]] = ttng.aref_copy {{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: ttng.aref_put.exit {{.*}} {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
      // CHECK: ttng.aref_get.enter {{.*}} {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      // CHECK: %[[TOK2:.*]] = ttng.aref_copy  {{.*}} {groups = [@nvws.group.gr2]}
      // CHECK: ttng.aref_get.exit {{.*}} {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      // CHECK: %[[TOK:.*]] = ttng.aref_phi %[[TOK1]], %[[TOK2]]
      // CHECK: ttng.tmem_load {{.*}}[%[[TOK]]] {groups = [@nvws.group.gr1, @nvws.group.gr2]}

      %val, %tok3 = ttng.tmem_load %buf[%tok1] { groups = [@nvws.group.gr1, @nvws.group.gr2] }: !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.store %ptr, %val { groups = [@nvws.group.gr2] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.store %ptr1, %val { groups = [@nvws.group.gr1] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.return
    }

    // CHECK-LABEL: arefphi_token2
    tt.func public @arefphi_token2(
      %c1 : f16, 
      %ptr: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>,
      %ptr1: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      ) {
      %a = tt.load %ptr { groups = [@nvws.group.gr1] }: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %b = tt.splat %c1 { groups = [@nvws.group.gr1] }: f16 -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %c = arith.addf %a, %b { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %buf, %tok = ttng.tmem_alloc { groups = [@nvws.group.gr1] }: () -> (!ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %true = arith.constant { groups = [@nvws.group.gr1] } true
      %tok1 = ttng.tmem_store %c, %buf[%tok], %true { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.aref_put.enter  {{.*}} {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
      // CHECK: %[[TOK1:.*]] = ttng.aref_copy {{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: ttng.aref_put.exit {{.*}} {aref_tag = "aref_0", groups = [@nvws.group.gr1]}
      // CHECK: ttng.aref_get.enter {{.*}} {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      // CHECK: %[[TOK2:.*]] = ttng.aref_copy  {{.*}} {groups = [@nvws.group.gr2]}
      // CHECK: ttng.aref_get.exit {{.*}} {aref_tag = "aref_0", groups = [@nvws.group.gr2]}
      // CHECK: %[[TOK:.*]] = ttng.aref_phi %[[TOK1]], %[[TOK2]]
      // CHECK: ttng.tmem_load {{.*}}[%[[TOK]]] {groups = [@nvws.group.gr2]}
      // CHECK: %[[TOK3:.*]] = ttng.tmem_store {{.*}}, {{.*}}[%[[TOK]]], {{.*}} {groups = [@nvws.group.gr1]}

      %val, %tok3 = ttng.tmem_load %buf[%tok1] { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %tok4 = ttng.tmem_store %c, %buf[%tok1], %true { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, mutable>
      tt.store %ptr1, %val { groups = [@nvws.group.gr1] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
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
      %desc = tt.reinterpret_tensor_descriptor %ptr0 { groups = [@nvws.group.gr1] }: !tt.ptr<f16> to !tt.tensordesc<tensor<128x64xf16, #shared>>
      %valA = tt.descriptor_load %desc[%c0, %c0] { groups = [@nvws.group.gr1] }: !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %opndA = ttg.local_alloc %valA { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
      %opndB = ttg.memdesc_trans %opndA {groups = [@nvws.group.gr1], order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>
      %acc, %tok = ttng.tmem_alloc { groups = [@nvws.group.gr1, @nvws.group.gr2] }: () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %true = arith.constant { groups = [@nvws.group.gr1] } true
      %tok1 = ttng.tmem_store %zero, %acc[%tok], %true { groups = [@nvws.group.gr1] } : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %tok3 = scf.for %i = %lb to %ub step %step iter_args(%tok2 = %tok1) -> !ttg.async.token {
        %tok3 = ttng.tc_gen5_mma %opndA, %opndB, %acc[%tok2], %true, %true { groups = [@nvws.group.gr1] } : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: aref_put.enter
        // CHECK: aref_copy
        // CHECK: aref_put.exit
        // CHECK: aref_get.enter
        // CHECK: aref_copy
        // CHECK: aref_get.exit
        %val, %tok4 = ttng.tmem_load %acc[%tok3] { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
        // CHECK: ttng.tmem_load
        // CHECK-NOT: aref_put.enter
        // CHECK-NOT: aref_get.enter
        // CHECK: scf.yield
        scf.yield %tok3 : !ttg.async.token
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
      %desc = tt.reinterpret_tensor_descriptor %ptr0 { groups = [@nvws.group.gr1] }: !tt.ptr<f16> to !tt.tensordesc<tensor<128x64xf16, #shared>>
      %valA = tt.descriptor_load %desc[%c0, %c0] { groups = [@nvws.group.gr1] }: !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %opndA = ttg.local_alloc %valA { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
      %opndB = ttg.memdesc_trans %opndA {groups = [@nvws.group.gr1], order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>
      %acc, %tok = ttng.tmem_alloc { groups = [@nvws.group.gr1, @nvws.group.gr2] }: () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %true = arith.constant { groups = [@nvws.group.gr1, @nvws.group.gr2] } true
      %zero = arith.constant { groups = [@nvws.group.gr1, @nvws.group.gr2] } dense<0.000000e+00> : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %tok1 = ttng.tmem_store %zero, %acc[%tok], %true { groups = [@nvws.group.gr1] } : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %tok3 = scf.for %i = %lb to %ub step %step iter_args(%tok2 = %tok1) -> !ttg.async.token {
        // CHECK: aref_put.enter
        // CHECK: aref_copy
        // CHECK: aref_put.exit
        // CHECK: aref_get.enter
        // CHECK: aref_copy
        // CHECK: aref_get.exit
        %tok3 = ttng.tmem_store %zero, %acc[%tok2], %true { groups = [@nvws.group.gr2] } : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: aref_put.enter
        // CHECK: aref_copy
        // CHECK: aref_put.exit
        // CHECK: aref_get.enter
        // CHECK: aref_copy
        // CHECK: aref_get.exit
        %tok4 = ttng.tc_gen5_mma %opndA, %opndB, %acc[%tok3], %true, %true { groups = [@nvws.group.gr1] } : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: aref_put.enter
        // CHECK: aref_copy
        // CHECK: aref_put.exit
        // CHECK: aref_get.enter
        // CHECK: aref_copy
        // CHECK: aref_get.exit
        %val, %tok5 = ttng.tmem_load %acc[%tok4] { groups = [@nvws.group.gr2] }: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
        // CHECK: ttng.tmem_load
        // CHECK-NOT: aref_put.enter
        // CHECK-NOT: aref_get.enter
        // CHECK: scf.yield
        scf.yield %tok4 : !ttg.async.token
      } {groups = [@nvws.group.gr1, @nvws.group.gr2], groups.0 = [@nvws.group.gr1]}

      tt.return
    }

    // CHECK-LABEL: @loop1
    tt.func public @loop1(
      %c0 : i32,
      %ub : index,
      %lb : index,
      %step : index,
      %ptr: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      ) {
      %valA = tt.load %ptr { groups = [@nvws.group.gr2] }: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %tok3 = scf.for %i = %lb to %ub step %step iter_args(%val = %valA) -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> {
        // CHECK: aref_put.enter
        // CHECK: ttg.local_store
        // CHECK: aref_put.exit
        // CHECK: aref_get.enter
        // CHECK: ttg.local_load
        // CHECK: aref_get.exit
        tt.store %ptr, %val { groups = [@nvws.group.gr2] } : tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
        %val1 = arith.addf %val, %val { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
        // CHECK: arith.addf
        // CHECK-NOT: aref_put.enter
        // CHECK-NOT: aref_get.enter
        // CHECK: scf.yield
        scf.yield %val1 : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      } {groups = [@nvws.group.gr1, @nvws.group.gr2], groups.0 = [@nvws.group.gr1]}

      tt.return
    }

    // CHECK-LABEL: @patch_for_op_results
    tt.func public @patch_for_op_results(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x64x!tt.ptr<f16>, #blocked>) {
      %0 = tt.load %arg3 {groups = [@nvws.group.gr2]} : tensor<128x64x!tt.ptr<f16>, #blocked>
      %1 = tt.load %arg3 {groups = [@nvws.group.gr1]} : tensor<128x64x!tt.ptr<f16>, #blocked>
      // CHECK-NOT: aref_put.enter
      // CHECK-NOT: aref_get.enter
      // CHECK: iter_args(%[[ARG0:.*]] = {{.*}}, %[[ARG1:.*]] = {{.*}})
      %2:2 = scf.for %arg4 = %arg1 to %arg0 step %arg2 iter_args(%arg5 = %0, %arg6 = %1) -> (tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>) {
        // CHECK: %[[BUF:.*]], %[[TOK:.*]] = ttng.aref_put.enter %[[AREF1:.*]][{{.*}}] {{.*}} groups = [@nvws.group.gr1]
        // CHECK: ttg.local_store %[[ARG1]], %[[BUF]] {groups = [@nvws.group.gr1]}
        // CHECK: %[[BUF:.*]], %[[TOK:.*]] = ttng.aref_get.enter %[[AREF1]][{{.*}}] {{.*}} groups = [@nvws.group.gr2]
        // CHECK: %[[ARG1:.*]] = ttg.local_load %[[BUF]] {groups = [@nvws.group.gr2]}

        // CHECK: %[[BUF:.*]], %[[TOK:.*]] = ttng.aref_put.enter %[[AREF2:.*]][{{.*}}] {{.*}} groups = [@nvws.group.gr2]
        // CHECK: ttg.local_store %[[ARG0]], %[[BUF]] {groups = [@nvws.group.gr2]}
        // CHECK: %[[BUF:.*]], %[[TOK:.*]] = ttng.aref_get.enter %[[AREF2]][{{.*}}] {{.*}} groups = [@nvws.group.gr1]
        // CHECK: %[[ARG0:.*]] = ttg.local_load %[[BUF]] {groups = [@nvws.group.gr1]}

        %3 = arith.addf %arg5, %arg5 {groups = [@nvws.group.gr1]} : tensor<128x64xf16, #blocked>
        // CHECK: %[[RET0:.*]] = arith.addf %[[ARG0]], %[[ARG0]] {groups = [@nvws.group.gr1]}
        // CHECK: %[[BUF:.*]], %[[TOK:.*]] = ttng.aref_put.enter %[[AREF3:.*]][{{.*}}] {{.*}} groups = [@nvws.group.gr1]
        // CHECK: ttg.local_store %[[RET0]], %[[BUF]] {groups = [@nvws.group.gr1]}
        // CHECK: %[[BUF:.*]], %[[TOK:.*]] = ttng.aref_get.enter %[[AREF3]][{{.*}}] {{.*}} groups = [@nvws.group.gr2]
        // CHECK: %[[RET0:.*]] = ttg.local_load %[[BUF]] {groups = [@nvws.group.gr2]}

        %4 = arith.addf %arg6, %arg6 {groups = [@nvws.group.gr2]} : tensor<128x64xf16, #blocked>
        // CHECK: %[[RET1:.*]] = arith.addf %[[ARG1]], %[[ARG1]] {groups = [@nvws.group.gr2]}
        // CHECK: %[[BUF:.*]], %[[TOK:.*]] = ttng.aref_put.enter %[[AREF4:.*]][{{.*}}] {{.*}} groups = [@nvws.group.gr2]
        // CHECK: ttg.local_store %[[RET1]], %[[BUF]] {groups = [@nvws.group.gr2]}
        // CHECK: %[[BUF:.*]], %[[TOK:.*]] = ttng.aref_get.enter %[[AREF4]][{{.*}}] {{.*}} groups = [@nvws.group.gr1]
        // CHECK: %[[RET1:.*]] = ttg.local_load %[[BUF]] {groups = [@nvws.group.gr1]}

        scf.yield %3, %4 : tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>
        // CHECK: scf.yield %[[RET0]], %[[RET1]]
      } {groups = [@nvws.group.gr1, @nvws.group.gr2], groups.0 = [@nvws.group.gr2], groups.1 = [@nvws.group.gr1]}
      tt.return
    }

    // CHECK-LABEL: mma_tmem_immutable
    tt.func public @mma_tmem_immutable(
      %c0 : i32,
      %c1 : f16, 
      %ptr0 : !tt.ptr<f16>,
      %ptr: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      ) {

      // CHECK: %[[AREF_BUF:.*]] = ttng.tmem_alloc {aref_buffer} {{.*}}
      // CHECK: %[[AREF:.*]] = ttng.aref_create %[[AREF_BUF]] 
      
      %a = tt.load %ptr { groups = [@nvws.group.gr1] }: tensor<128x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %b = tt.splat %c1 { groups = [@nvws.group.gr1] }: f16 -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %c = arith.addf %a, %b { groups = [@nvws.group.gr1] } : tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %desc = tt.reinterpret_tensor_descriptor %ptr0 { groups = [@nvws.group.gr1] }: !tt.ptr<f16> to !tt.tensordesc<tensor<128x64xf16, #shared>>
      %valA = tt.descriptor_load %desc[%c0, %c0] { groups = [@nvws.group.gr1] }: !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %opndA = ttg.local_alloc %valA { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
      %opndB = ttg.memdesc_trans %opndA {groups = [@nvws.group.gr1], order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>

      %true = arith.constant { groups = [@nvws.group.gr2] } true
      %false = arith.constant { groups = [@nvws.group.gr2] } false
      %acc, %tok = ttng.tmem_alloc { groups = [@nvws.group.gr2] }: () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      %buf2 = ttng.tmem_alloc %c { groups = [@nvws.group.gr1] }: (tensor<128x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory>
      // CHECK: %[[BUF2:.*]] = ttng.tmem_alloc {{.*}} {groups = [@nvws.group.gr1]}
      // CHECK: %[[DST:.*]], {{.*}} = ttng.aref_put.enter %[[AREF]][{{.*}}] {{.*}} groups = [@nvws.group.gr1]
      // CHECK: ttng.aref_copy %[[BUF2]], %[[DST]] {groups = [@nvws.group.gr1]}
      // CHECK: ttng.aref_put.exit %[[AREF]][{{.*}}] {{.*}} groups = [@nvws.group.gr1]

      // CHECK: %[[SRC:.*]], {{.*}} = ttng.aref_get.enter %[[AREF]][{{.*}}] {{.*}} groups = [@nvws.group.gr2]
      // CHECK: %[[BUF2:.*]] = ttng.aref_clone %[[SRC]] {groups = [@nvws.group.gr2]}
      // CHECK: ttng.aref_get.exit %[[AREF]][{{.*}}] {{.*}} groups = [@nvws.group.gr2]
      // CHECK: ttng.tc_gen5_mma %[[BUF2]], {{.*}} {groups = [@nvws.group.gr2]}

      %tok2 = ttng.tc_gen5_mma %buf2, %opndB, %acc[%tok], %false, %true { groups = [@nvws.group.gr2] } : !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<64x128xf16, #shared2, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      tt.return
    }
  }
