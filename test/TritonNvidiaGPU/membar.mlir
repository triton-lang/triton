// RUN: triton-opt %s -split-input-file --triton-nvidia-tma-lowering --convert-scf-to-cf --allocate-shared-memory -test-print-membar | FileCheck %s
// RUN: triton-opt %s -split-input-file --triton-nvidia-tma-lowering -triton-nvidia-gpu-optimize-mbarrier-arrivals -triton-nvidia-gpu-optimize-mbarrier-arrivals --convert-scf-to-cf --allocate-shared-memory -test-print-membar | FileCheck %s --check-prefixes=ARRIVAL,FLOW
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf -triton-nvidia-gpu-optimize-mbarrier-arrivals -triton-nvidia-gpu-optimize-mbarrier-arrivals | FileCheck %s --check-prefix=FLOW

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
  // ARRIVAL-LABEL: @distributed_arrival_keeps_payload_hazard
  // ARRIVAL: ttng.init_barrier {{.*}}, 4 :
  // ARRIVAL: ttg.local_store
  // ARRIVAL-NEXT: ttng.arrive_barrier {{.*}}, 4 {per_warp}
  // ARRIVAL-NEXT: ttg.barrier local
  // ARRIVAL-NEXT: %{{.*}} = ttg.local_load
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

  // Counts already divisible by the warp count need no scaling.
  // ARRIVAL-LABEL: @distributed_arrival_divisible_count
  // ARRIVAL: ttng.init_barrier {{.*}}, 8 :
  // ARRIVAL: ttng.arrive_barrier {{.*}}, 8 {per_warp}
  tt.func @distributed_arrival_divisible_count() {
    %phase = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 8 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %bar, 8 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // Barrier-state dependencies remain ordered between partial contributions.
  // ARRIVAL-LABEL: @distributed_arrival_keeps_barrier_hazards
  // ARRIVAL: ttng.init_barrier {{.*}}, 8 :
  // ARRIVAL-NEXT: ttg.barrier local
  // ARRIVAL-NEXT: ttng.arrive_barrier {{.*}}, 4 {per_warp}
  // ARRIVAL-NEXT: ttg.barrier local
  // ARRIVAL-NEXT: ttng.arrive_barrier {{.*}}, 4 {per_warp}
  // ARRIVAL-NEXT: ttng.wait_barrier
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
  // ARRIVAL-LABEL: @arrival_with_expect_tx
  // ARRIVAL: ttng.init_barrier {{.*}}, 2 :
  // ARRIVAL: ttng.barrier_expect
  // ARRIVAL: ttng.arrive_barrier {{.*}}, 1 :
  tt.func @arrival_with_expect_tx() {
    %phase = arith.constant 0 : i32
    %true = arith.constant true
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.barrier_expect %bar, 0, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // Both logical expectations contribute once per warp to the same phase.
  // ARRIVAL-LABEL: @multiple_expectations_per_phase
  // ARRIVAL: ttng.init_barrier {{.*}}, 8 :
  // ARRIVAL: ttng.barrier_expect {{.*}}, 0 {per_warp},
  // ARRIVAL: ttng.barrier_expect {{.*}}, 0 {per_warp},
  // ARRIVAL: tt.return
  tt.func @multiple_expectations_per_phase() {
    %phase = arith.constant 0 : i32
    %true = arith.constant true
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.barrier_expect %bar, 0, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.barrier_expect %bar, 0, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // A disabled expectation does not change the transaction balance.
  // FLOW-LABEL: @disabled_large_expectation
  // FLOW: ttng.init_barrier {{.*}}, 1 :
  // FLOW: ttng.barrier_expect {{.*}}, 1048576,
  // FLOW: ttng.inval_barrier
  tt.func @disabled_large_expectation() {
    %false = arith.constant false
    %true = arith.constant true
    %phase = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.barrier_expect %bar, 1048576, %false : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.barrier_expect %bar, 0, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // Bound both physical arrivals and total bytes across a phase. These
  // barriers have no asynchronous users and can be invalidated while pending.
  // FLOW-LABEL: @expectation_count_limits
  // FLOW: ttng.init_barrier {{.*}}, 1048572 :
  // FLOW: ttng.barrier_expect {{.*}}, 0 {per_warp},
  // FLOW: ttng.init_barrier {{.*}}, 262144 :
  // FLOW: ttng.barrier_expect {{.*}}, 0,
  // FLOW: ttng.init_barrier {{.*}}, 8 :
  // FLOW: ttng.barrier_expect {{.*}}, 524284 {per_warp},
  // FLOW: ttng.init_barrier {{.*}}, 2 :
  // FLOW: ttng.barrier_expect {{.*}}, 524288,
  tt.func @expectation_count_limits(%pred: i1) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %a, 262143 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.barrier_expect %a, 0, %pred : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %a : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %b, 262144 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.barrier_expect %b, 0, %pred : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %b : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %c = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %c, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.barrier_expect %c, 524284, %pred : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %c : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %d = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %d, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.barrier_expect %d, 524288, %pred : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %d : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // Descriptor joins are rejected as a whole, including their direct users.
  // ARRIVAL-LABEL: @arrival_with_selected_barrier
  // ARRIVAL: ttng.init_barrier {{.*}}, 1 :
  // ARRIVAL: ttng.init_barrier {{.*}}, 1 :
  // ARRIVAL: ttng.arrive_barrier {{.*}}, 1 :
  // ARRIVAL: ttng.arrive_barrier {{.*}}, 1 :
  // ARRIVAL: arith.select
  // ARRIVAL: ttng.arrive_barrier {{.*}}, 1 :
  tt.func @arrival_with_selected_barrier(%condition: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %first = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %second = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %first, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %second, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
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
  // ARRIVAL-LABEL: @distributed_arrival_cta_routes
  // ARRIVAL: ttng.init_barrier {{.*}}, 16 :
  // ARRIVAL: ttng.arrive_barrier {{.*}}, 4 {per_warp} :
  // ARRIVAL: ttng.arrive_barrier {{.*}}, 4 {fromCTA = 0 : i32, per_warp} :
  // ARRIVAL: ttng.arrive_barrier {{.*}}, 4 {multicastCTA = 1 : i32, per_warp} :
  // ARRIVAL-NEXT: ttng.wait_barrier
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
  // ARRIVAL-LABEL: @distributed_arrival_broadcast
  // ARRIVAL: ttng.init_barrier {{.*}}, 4 :
  // ARRIVAL: ttng.arrive_barrier {{.*}}, 4 {per_warp} :
  // ARRIVAL-NEXT: ttng.wait_barrier
  tt.func @distributed_arrival_broadcast() {
    %phase = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    tt.return
  }

  // Scaling fits the logical count but overflows the physical count: 131072 * 4 * 2.
  // ARRIVAL-LABEL: @distributed_arrival_physical_count_overflow
  // ARRIVAL: ttng.init_barrier {{.*}}, 131072 :
  // ARRIVAL: ttng.arrive_barrier {{.*}}, 131071 :
  // ARRIVAL: ttng.arrive_barrier {{.*}}, 1 :
  tt.func @distributed_arrival_physical_count_overflow() {
    %phase = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    ttng.init_barrier %bar, 131072 : !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    ttng.arrive_barrier %bar, 131071 : !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #broadcast, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // Expectations follow captured and loop-carried views of one allocation.
  // FLOW-LABEL: @distributed_expect_loop_carried_view
  // FLOW: ttng.init_barrier {{.*}}, 4 :
  // FLOW: ttng.barrier_expect {{.*}}, 0 {per_warp},
  // FLOW: ttng.inval_barrier
  tt.func @distributed_expect_loop_carried_view(%n: i32, %index: i32) {
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
    %bar = ttg.memdesc_index %bars[%index] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.warp_specialize(%n, %index, %bars, %bar)
    default {
      ttg.warp_yield
    }
    partition0(%limit: i32, %idx: i32, %storage: !ttg.memdesc<2x1xi64, #shared, #smem, mutable>, %initial: !ttg.memdesc<1xi64, #shared, #smem, mutable>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %true = arith.constant true
      %result:2 = scf.for %i = %c0 to %limit step %c1 iter_args(%iter = %initial, %phase = %c0) -> (!ttg.memdesc<1xi64, #shared, #smem, mutable>, i32) : i32 {
        ttng.barrier_expect %iter, 0, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.wait_barrier %iter, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %next = ttg.memdesc_index %storage[%idx] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %next_phase = arith.xori %phase, %c1 : i32
        scf.yield %next, %next_phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>, i32
      }
      ttng.inval_barrier %result#0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.warp_return
    } : (i32, i32, !ttg.memdesc<2x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }

  // Different-width producers can own successive phases of the same barrier.
  // FLOW-LABEL: @expectations_with_different_warp_counts
  // FLOW: ttng.init_barrier {{.*}}, 1 :
  // FLOW: ttng.barrier_expect {{.*}}, 0,
  // FLOW: ttng.barrier_expect {{.*}}, 0,
  tt.func @expectations_with_different_warp_counts() {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.barrier_expect %bar, 0, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %c0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.warp_specialize(%bar)
    default {
      ttg.warp_yield
    }
    partition0(%captured: !ttg.memdesc<1xi64, #shared, #smem, mutable>) num_warps(2) {
      %c1 = arith.constant 1 : i32
      %pred = arith.constant true
      ttng.barrier_expect %captured, 0, %pred : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttng.wait_barrier %captured, %c1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #shared, #smem, mutable>) -> ()
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // Descriptor forwarding preserves the common software count scale.
  // FLOW-LABEL: @software_arrival_same_allocation_join
  // FLOW: ttng.init_barrier {{.*}}, 4 :
  // FLOW: ttng.init_barrier {{.*}}, 4 :
  // FLOW: ttng.arrive_barrier {{.*}}, 4 {per_warp}
  // FLOW: ttng.wait_barrier {{.*}} deps {{.*}}
  tt.func @software_arrival_same_allocation_join(%condition: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
    %first = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %second = ttg.memdesc_index %bars[%c1] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %first, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %second, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
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

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Per-warp publication does not order another warp's old payload read.
  // ARRIVAL-LABEL: @distributed_expect_keeps_payload_war
  // ARRIVAL: ttg.local_load
  // ARRIVAL-NEXT: ttng.barrier_expect {{.*}}, 2048 {per_warp},
  // ARRIVAL-NEXT: ttg.barrier local
  // ARRIVAL-NEXT: ttng.async_tma_copy_global_to_local
  // ARRIVAL-NEXT: ttng.wait_barrier
  // ARRIVAL-NEXT: ttg.barrier local
  // ARRIVAL-NEXT: ttng.inval_barrier
  tt.func @distributed_expect_keeps_payload_war(%desc: !tt.tensordesc<16x64xf16, #shared>, %data: tensor<16x64xf16, #blocked>) -> tensor<16x64xf16, #blocked> {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %mem = ttg.local_alloc %data : (tensor<16x64xf16, #blocked>) -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable>
    ttg.barrier local
    %old = ttg.local_load %mem : !ttg.memdesc<16x64xf16, #shared, #smem, mutable> -> tensor<16x64xf16, #blocked>
    ttng.barrier_expect %bar, 2048, %true : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %mem, %bar, %true : !tt.tensordesc<16x64xf16, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %bar, %c0 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.return %old : tensor<16x64xf16, #blocked>
  }
}
