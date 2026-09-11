// RUN: triton-opt %s -split-input-file --triton-nvidia-tma-lowering --convert-scf-to-cf --allocate-shared-memory -test-print-membar | FileCheck %s
// RUN: triton-opt %s -split-input-file --triton-nvidia-tma-lowering --convert-scf-to-cf --allocate-shared-memory --triton-nvidia-gpu-membar='compute-capability=90 ptx-version=80' | FileCheck %s --check-prefix=PREP
// RUN: triton-opt %s -split-input-file --triton-nvidia-tma-lowering --convert-scf-to-cf --allocate-shared-memory --triton-nvidia-gpu-membar='compute-capability=90 ptx-version=80' --triton-nvidia-gpu-tmem-barrier-insertion --triton-nvidia-gpu-optimize-mbarrier-arrivals -triton-nvidia-gpu-optimize-synchronization | FileCheck %s --check-prefixes=FOLD,CLEANUP
// RUN: triton-opt %s -split-input-file -triton-nvidia-gpu-optimize-synchronization | FileCheck %s --check-prefix=WARP

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: init_barrier
	// CHECK: local_alloc
	// CHECK-NEXT: ttg.barrier local
	// CHECK-NEXT: init_barrier
  tt.func @init_barrier() {
  	%cst = arith.constant dense<0> : tensor<1xi64, #blocked0>
  	%alloc = ttg.local_alloc %cst : (tensor<1xi64, #blocked0>) -> !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    ttng.init_barrier %alloc, 1 : !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    tt.return
  }

  // Tensor-map creation only synchronizes a warp; other shared writes stay pending.
  // CHECK-LABEL: tensormap_create_before_shared_load
  tt.func @tensormap_create_before_shared_load(%desc: !tt.ptr<i8>, %src: !tt.ptr<i16>, %size: i32, %data: tensor<128xi32, #blocked0>) -> tensor<128xi32, #blocked0> {
    %c256 = arith.constant 256 : i32
    %c1 = arith.constant 1 : i32
    // CHECK: ttg.local_alloc
    // CHECK-NEXT: ttng.tensormap_create
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttg.local_load
    %mem = ttg.local_alloc %data : (tensor<128xi32, #blocked0>) -> !ttg.memdesc<128xi32, #shared0, #smem>
    ttng.tensormap_create %desc, %src, [%c256], [%size], [], [%c1] {elem_type = 3 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 2 : i32} : (!tt.ptr<i8>, !tt.ptr<i16>, i32, i32, i32) -> ()
    %loaded = ttg.local_load %mem : !ttg.memdesc<128xi32, #shared0, #smem> -> tensor<128xi32, #blocked0>
    tt.return %loaded : tensor<128xi32, #blocked0>
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: inval_barrier
	// CHECK: local_alloc
	// CHECK-NEXT: ttg.barrier local
	// CHECK-NEXT: init_barrier
	// CHECK-NEXT: inval_barrier
  tt.func @inval_barrier() {
  	%cst = arith.constant dense<0> : tensor<1xi64, #blocked0>
  	%alloc = ttg.local_alloc %cst : (tensor<1xi64, #blocked0>) -> !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    ttng.init_barrier %alloc, 1 : !ttg.memdesc<1xi64, #shared0, #smem, mutable>
		ttng.inval_barrier %alloc : !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: barrier_expect
	// CHECK: local_alloc
	// CHECK-NEXT: ttg.barrier local
	// CHECK-NEXT: init_barrier
	// CHECK-NEXT: ttg.barrier local
	// CHECK-NEXT: barrier_expect
  tt.func @barrier_expect(%pred : i1) {
  	%cst = arith.constant dense<0> : tensor<1xi64, #blocked0>
  	%alloc = ttg.local_alloc %cst : (tensor<1xi64, #blocked0>) -> !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    ttng.init_barrier %alloc, 1 : !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    ttng.barrier_expect %alloc, 16384, %pred : !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: wait_barrier
	// CHECK: local_alloc
	// CHECK-NEXT: ttg.barrier local
	// CHECK-NEXT: init_barrier
	// CHECK-NEXT: ttg.barrier local
	// CHECK-NEXT: wait_barrier
  tt.func @wait_barrier(%phase : i32) {
  	%cst = arith.constant dense<0> : tensor<1xi64, #blocked0>
  	%alloc = ttg.local_alloc %cst : (tensor<1xi64, #blocked0>) -> !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    ttng.init_barrier %alloc, 1 : !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    ttng.wait_barrier %alloc, %phase : !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    tt.return
  }
}

// -----



#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_load(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: i32) -> tensor<128x64xf16, #blocked0> {
		// CHECK-LABEL: tma_load
		// CHECK: local_dealloc
		// CHECK-NEXT: local_alloc
		// CHECK-NEXT: local_alloc
    // CHECK-NEXT: ttg.barrier local
		// CHECK-NEXT: init_barrier
  	%cst = arith.constant dense<0> : tensor<128x64xi64, #blocked0>
  	%alloc = ttg.local_alloc %cst : (tensor<128x64xi64, #blocked0>) -> !ttg.memdesc<128x64xi64, #shared1, #smem, mutable>
  	ttg.local_dealloc %alloc : !ttg.memdesc<128x64xi64, #shared1, #smem, mutable>
    %l = tt.descriptor_load %arg0[%arg1, %arg1] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked0>
    tt.return %l : tensor<128x64xf16, #blocked0>
  }
}


// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#nvmma32 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 32}>
#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tma_store
//       CHECK: ttg.local_alloc
//       CHECK-NEXT: ttg.local_dealloc
//       CHECK-NEXT: ttg.barrier local
//       CHECK-NEXT: ttg.local_alloc
  tt.func public @tma_store(%arg0: !tt.tensordesc<128x256xf32, #nvmma32>, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: tensor<128x256xf32, #blocked0>) {
    %cst = arith.constant dense<0> : tensor<128x64xi64, #blocked0>
    %alloc = ttg.local_alloc %cst : (tensor<128x64xi64, #blocked0>) -> !ttg.memdesc<128x64xi64, #shared0, #smem, mutable>
    ttg.local_dealloc %alloc : !ttg.memdesc<128x64xi64, #shared0, #smem, mutable>
    tt.descriptor_store %arg0[%arg1, %arg1], %arg2 : !tt.tensordesc<128x256xf32, #nvmma32>, tensor<128x256xf32, #blocked0>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {

// CHECK-LABEL: @wait_after_mma
tt.func @wait_after_mma(
  %a: !ttg.memdesc<128x128xf16, #shared, #smem>,
  %b: !ttg.memdesc<128x128xf16, #shared1, #smem>,
  %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
  %useAcc: i1,
  %pred: i1,
  %barrierPred: i1
) {
  %phase = arith.constant 0 : i32
  %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
  // The scheduled barrier synchronizes both the wait and the fused MMA.
  // CHECK: ttg.async_wait
  ttg.async_wait {num = 0 : i32}
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tc_gen5_mma
  ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %barrier[%barrierPred] {is_async} :
     !ttg.memdesc<128x128xf16, #shared, #smem>,
     !ttg.memdesc<128x128xf16, #shared1, #smem>,
     !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
     !ttg.memdesc<1xi64, #shared2, #smem, mutable>
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.wait_barrier
  ttng.wait_barrier %barrier, %phase : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @inline_asm_no_memdesc_effects
  tt.func @inline_asm_no_memdesc_effects(%data: tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked> {
    // CHECK: ttg.local_alloc
    %mem = ttg.local_alloc %data : (tensor<128xi32, #blocked>) -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    // CHECK-NEXT: ttg.inline_asm
    ttg.inline_asm "// access descriptor" {constraints = "r", pure = false} %mem : (!ttg.memdesc<128xi32, #shared, #smem, mutable>) -> ()
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: {{.*}}ttg.local_load
    %value = ttg.local_load %mem : !ttg.memdesc<128xi32, #shared, #smem, mutable> -> tensor<128xi32, #blocked>
    tt.return %value : tensor<128xi32, #blocked>
  }

  // CHECK-LABEL: @elementwise_inline_asm_memdesc_effects
  tt.func @elementwise_inline_asm_memdesc_effects(%data: tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked> {
    %mem = ttg.local_alloc %data : (tensor<128xi32, #blocked>) -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    // CHECK: ttg.barrier local
    // CHECK-NEXT: {{.*}}tt.elementwise_inline_asm
    %unused = tt.elementwise_inline_asm "mov.u32 $0, 0;" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %mem : !ttg.memdesc<128xi32, #shared, #smem, mutable> -> i32
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: {{.*}}ttg.local_load
    %value = ttg.local_load %mem : !ttg.memdesc<128xi32, #shared, #smem, mutable> -> tensor<128xi32, #blocked>
    tt.return %value : tensor<128xi32, #blocked>
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#store = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#load = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // Distributed publication leaves the cross-warp payload dependency pending.
  // PREP-LABEL: @distributed_arrival_keeps_payload_hazard
  // PREP: ttng.init_barrier %[[BAR:.*]], 1 :
  // PREP: ttng.arrive_barrier %[[BAR]], 1 {per_warp}
  // FOLD-LABEL: @distributed_arrival_keeps_payload_hazard
  // FOLD: ttng.init_barrier {{.*}}, 4 :
  // FOLD: ttg.local_store
  // FOLD-NEXT: ttg.barrier warp local
  // FOLD-NEXT: ttng.arrive_barrier {{.*}}, 4 {per_warp}
  // FOLD-NEXT: ttg.barrier local
  // FOLD-NEXT: %{{.*}} = ttg.local_load
  tt.func @distributed_arrival_keeps_payload_hazard(%data: tensor<512xi32, #store>) -> tensor<512xi32, #load> {
    %phase = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %mem = ttg.local_alloc : () -> !ttg.memdesc<512xi32, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.barrier local
    ttg.local_store %data, %mem : tensor<512xi32, #store> -> !ttg.memdesc<512xi32, #shared, #smem, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %loaded = ttg.local_load %mem : !ttg.memdesc<512xi32, #shared, #smem, mutable> -> tensor<512xi32, #load>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return %loaded : tensor<512xi32, #load>
  }

  // A following CTA barrier covers both warp barriers across pure arithmetic.
  // WARP-LABEL: @warp_barrier_before_cta
  // WARP-NEXT: %[[SUM:.*]] = arith.addi
  // WARP-NEXT: ttg.barrier local
  // WARP-NEXT: tt.return %[[SUM]]
  tt.func @warp_barrier_before_cta(%data: tensor<512xi32, #store>) -> tensor<512xi32, #store> {
    ttg.barrier warp local
    %sum = arith.addi %data, %data : tensor<512xi32, #store>
    ttg.barrier warp local
    ttg.barrier local
    tt.return %sum : tensor<512xi32, #store>
  }

  // Explicit memory effects and implicit shared scratch stop subsumption.
  // WARP-LABEL: @warp_barrier_before_cta_with_effects
  // WARP-NEXT: ttg.barrier warp local
  // WARP-NEXT: tt.store
  // WARP-NEXT: ttg.barrier local
  // WARP-NEXT: tt.store
  // WARP-NEXT: ttg.barrier warp local
  // WARP-NEXT: %[[CONVERTED:.*]] = ttg.convert_layout
  // WARP-NEXT: ttg.barrier local
  // WARP-NEXT: tt.return %[[CONVERTED]]
  tt.func @warp_barrier_before_cta_with_effects(%out: !tt.ptr<i32>, %value: i32, %data: tensor<512xi32, #store>) -> tensor<512xi32, #load> {
    ttg.barrier warp local
    tt.store %out, %value : !tt.ptr<i32>
    ttg.barrier local
    tt.store %out, %value : !tt.ptr<i32>
    ttg.barrier warp local
    %converted = ttg.convert_layout %data : tensor<512xi32, #store> -> tensor<512xi32, #load>
    ttg.barrier local
    tt.return %converted : tensor<512xi32, #load>
  }

  // A CTA rendezvous without local-memory ordering cannot cover warp local.
  // WARP-LABEL: @warp_barrier_before_cta_none
  // WARP-NEXT: ttg.barrier warp local
  // WARP-NEXT: ttg.barrier none
  // WARP-NEXT: tt.return
  tt.func @warp_barrier_before_cta_none() {
    ttg.barrier warp local
    ttg.barrier none
    tt.return
  }

  // Initialized allocations still write shared memory before publication.
  // Counts divisible by the warp count need no scaling.
  // CLEANUP-LABEL: @arrival_after_initialized_shared_alloc
  // CLEANUP: ttng.init_barrier {{.*}}, 4 :
  // CLEANUP: ttg.barrier local
  // CLEANUP-NEXT: %[[MEM:.*]] = ttg.local_alloc
  // CLEANUP-NEXT: ttg.barrier warp local
  // CLEANUP-NEXT: ttng.arrive_barrier {{.*}}, 4 {per_warp}
  tt.func @arrival_after_initialized_shared_alloc(%data: tensor<512xi32, #store>) -> tensor<512xi32, #store> {
    %phase = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 4 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.barrier local
    %mem = ttg.local_alloc %data : (tensor<512xi32, #store>) -> !ttg.memdesc<512xi32, #shared, #smem, mutable>
    ttng.arrive_barrier %bar, 4 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %result = ttg.local_load %mem : !ttg.memdesc<512xi32, #shared, #smem, mutable> -> tensor<512xi32, #store>
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return %result : tensor<512xi32, #store>
  }

  // A pure layout conversion can access shared scratch after the CTA barrier.
  // CLEANUP-LABEL: @arrival_after_scratch_conversion
  // CLEANUP: ttg.barrier local
  // CLEANUP-NEXT: %[[CONVERTED:.*]] = ttg.convert_layout
  // CLEANUP-NEXT: ttg.barrier warp local
  // CLEANUP-NEXT: ttng.arrive_barrier {{.*}}, 4 {per_warp}
  // CLEANUP-NEXT: ttng.wait_barrier
  // CLEANUP: ttng.inval_barrier
  // CLEANUP-NEXT: tt.return %[[CONVERTED]]
  tt.func @arrival_after_scratch_conversion(%data: tensor<512xi32, #store>) -> tensor<512xi32, #load> {
    %phase = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 4 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.barrier local
    %converted = ttg.convert_layout %data : tensor<512xi32, #store> -> tensor<512xi32, #load>
    ttng.arrive_barrier %bar, 4 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return %converted : tensor<512xi32, #load>
  }

  tt.func private @store_before_arrival(%out: !tt.ptr<i32>, %value: i32) attributes {noinline = true} {
    tt.store %out, %value : !tt.ptr<i32>
    tt.return
  }

  // The callee's store remains pending after the caller's earlier CTA barrier.
  // CLEANUP-LABEL: @arrival_after_call
  // CLEANUP: ttng.init_barrier {{.*}}, 4 :
  // CLEANUP: ttg.barrier local
  // CLEANUP-NEXT: tt.call @store_before_arrival
  // CLEANUP-NEXT: ttg.barrier warp local
  // CLEANUP-NEXT: ttng.arrive_barrier {{.*}}, 4 {per_warp}
  tt.func @arrival_after_call(%out: !tt.ptr<i32>, %value: i32) {
    %phase = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.barrier local
    tt.call @store_before_arrival(%out, %value) : (!tt.ptr<i32>, i32) -> ()
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // The store ends warp coverage on one path through the join.
  // FOLD-LABEL: @distributed_arrival_cfg_join
  // FOLD: ttng.init_barrier {{.*}}, 4 :
  // FOLD: cf.cond_br %{{.*}}, ^{{bb[0-9]+}}, ^[[JOIN:bb[0-9]+]]
  // FOLD: ^[[JOIN]]:
  // FOLD-NEXT: ttg.barrier warp local
  // FOLD-NEXT: ttng.arrive_barrier {{.*}}, 4 {per_warp}
  tt.func @distributed_arrival_cfg_join(%condition: i1, %out: !tt.ptr<i32>, %value: i32) {
    %phase = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.barrier local
    scf.if %condition {
      tt.store %out, %value : !tt.ptr<i32>
    }
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // Barrier-state dependencies keep both arrivals scalar.
  // FOLD-LABEL: @distributed_arrival_keeps_barrier_hazards
  // FOLD: ttng.init_barrier {{.*}}, 2 :
  // FOLD-NEXT: ttg.barrier local
  // FOLD-NEXT: ttng.arrive_barrier {{.*}}, 1 :
  // FOLD-NEXT: ttg.barrier local
  // FOLD-NEXT: ttng.arrive_barrier {{.*}}, 1 :
  // FOLD-NEXT: ttng.wait_barrier
  tt.func @distributed_arrival_keeps_barrier_hazards() {
    %phase = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // A transaction-count arrival cannot share the software-only count scale.
  // FOLD-LABEL: @arrival_with_expect_tx
  // FOLD: ttng.init_barrier {{.*}}, 2 :
  // FOLD: ttng.barrier_expect
  // FOLD: ttng.arrive_barrier {{.*}}, 1 :
  tt.func @arrival_with_expect_tx(%out: !tt.ptr<i32>, %value: i32) {
    %phase = arith.constant 0 : i32
    %true = arith.constant true
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.barrier local
    tt.store %out, %value : !tt.ptr<i32>
    ttng.barrier_expect %bar, 0, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }


  // Descriptor joins are rejected as a whole, including their direct users.
  // FOLD-LABEL: @arrival_with_selected_barrier
  // FOLD: ttng.init_barrier {{.*}}, 1 :
  // FOLD: ttng.init_barrier {{.*}}, 1 :
  // FOLD: ttng.arrive_barrier {{.*}}, 1 :
  // FOLD: ttng.arrive_barrier {{.*}}, 1 :
  // FOLD: arith.select
  // FOLD: ttng.arrive_barrier {{.*}}, 1 :
  tt.func @arrival_with_selected_barrier(%condition: i1, %out: !tt.ptr<i32>, %value: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %first = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %second = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %first, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %second, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.barrier local
    tt.store %out, %value : !tt.ptr<i32>
    ttng.arrive_barrier %first, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %second, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %first, %c0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %second, %c0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %selected = arith.select %condition, %first, %second : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %selected, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %selected, %c1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#per_cta = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#broadcast = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107"} {
  // Each barrier receives one local, one routed, and two multicast contributions.
  // FOLD-LABEL: @distributed_arrival_cta_routes
  // FOLD: ttng.init_barrier {{.*}}, 4 :
  // FOLD: ttng.arrive_barrier {{.*}}, 1 :
  // FOLD: ttng.arrive_barrier {{.*}}, 1 {fromCTA = 0 : i32} :
  // FOLD: ttng.arrive_barrier {{.*}}, 1 {multicastCTA = 1 : i32} :
  // FOLD-NEXT: ttng.wait_barrier
  tt.func @distributed_arrival_cta_routes() {
    %phase = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #per_cta, #smem, mutable>
    ttng.init_barrier %bar, 4 : !ttg.memdesc<2xi64, #per_cta, #smem, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<2xi64, #per_cta, #smem, mutable>
    ttng.arrive_barrier %bar, 1 {fromCTA = 0 : i32} : !ttg.memdesc<2xi64, #per_cta, #smem, mutable>
    ttng.arrive_barrier %bar, 1 {multicastCTA = 1 : i32} : !ttg.memdesc<2xi64, #per_cta, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<2xi64, #per_cta, #smem, mutable>
    tt.return
  }

  // Both CTAs contribute their warps to the same physical barrier.
  // FOLD-LABEL: @distributed_arrival_broadcast
  // FOLD: ttng.init_barrier {{.*}}, 4 :
  // FOLD: ttng.arrive_barrier {{.*}}, 4 {per_warp} :
  // FOLD-NEXT: ttng.wait_barrier
  tt.func @distributed_arrival_broadcast(%out: !tt.ptr<i32>, %value: i32) {
    %phase = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    ttg.barrier local
    tt.store %out, %value : !tt.ptr<i32>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    tt.return
  }

  // Scaling fits the logical count but overflows the physical count: 131072 * 4 * 2.
  // PREP-LABEL: @distributed_arrival_physical_count_overflow
  // PREP: ttng.init_barrier {{.*}}, 131072 :
  // PREP: ttng.arrive_barrier {{.*}}, 131071 :
  // PREP: ttng.arrive_barrier {{.*}}, 1 :
  tt.func @distributed_arrival_physical_count_overflow() {
    %phase = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    ttng.init_barrier %bar, 131072 : !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    ttng.arrive_barrier %bar, 131071 : !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    tt.return
  }

  tt.func private @pure_before_arrival(%value: i32) -> i32 attributes {noinline = true, "ttg.num-warps" = 2 : i32} {
    tt.return %value : i32
  }

  // The CTA barrier covers the pure call, so the arrival keeps its full count.
  // CLEANUP-LABEL: @outlined_arrival_uses_full_count
  // CLEANUP: ttng.init_barrier {{.*}}, 1 :
  // CLEANUP: ttg.barrier local
  // CLEANUP-NEXT: %[[VALUE:.*]] = tt.call @pure_before_arrival
  // CLEANUP-NEXT: ttng.arrive_barrier {{.*}}, 1 :
  // CLEANUP: tt.return %[[VALUE]]
  tt.func private @outlined_arrival_uses_full_count(%value: i32) -> i32 attributes {noinline = true, "ttg.num-warps" = 2 : i32} {
    %phase = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #per_cta, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #per_cta, #smem, mutable>
    ttg.barrier local
    %result = tt.call @pure_before_arrival(%value) : (i32) -> i32
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<2xi64, #per_cta, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<2xi64, #per_cta, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<2xi64, #per_cta, #smem, mutable>
    tt.return %result : i32
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {



  // Descriptor forwarding preserves the common software count scale.
  // FOLD-LABEL: @software_arrival_same_allocation_join
  // FOLD: ttng.init_barrier {{.*}}, 4 :
  // FOLD: ttng.init_barrier {{.*}}, 4 :
  // FOLD: ttng.arrive_barrier {{.*}}, 4 {per_warp}
  // FOLD: ttng.wait_barrier {{.*}} deps {{.*}}
  tt.func @software_arrival_same_allocation_join(%condition: i1, %out: !tt.ptr<i32>, %value: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
    %first = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %second = ttg.memdesc_index %bars[%c1] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %first, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %second, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.barrier local
    tt.store %out, %value : !tt.ptr<i32>
    %joined = scf.if %condition -> (!ttg.memdesc<1xi64, #shared, #smem, mutable>) {
      scf.yield %first : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    } else {
      scf.yield %second : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    }
    %selected = arith.select %condition, %joined, %first : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %selected, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %selected, %c0 deps %bars : !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
    tt.return
  }

}
