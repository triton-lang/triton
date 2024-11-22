// RUN: triton-opt %s -split-input-file --triton-nvidia-tma-lowering --allocate-shared-memory -test-print-membar | FileCheck %s

#shared0 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: init_barrier
	// CHECK: local_alloc
	// CHECK-NEXT: gpu.barrier
	// CHECK-NEXT: init_barrier
  tt.func @init_barrier() {
  	%cst = arith.constant dense<0> : tensor<1xi64, #blocked0>
  	%alloc = triton_gpu.local_alloc %cst : (tensor<1xi64, #blocked0>) -> !triton_gpu.memdesc<1xi64, #shared0, #triton_gpu.shared_memory, mutable>
    triton_nvidia_gpu.init_barrier %alloc, 1 : !triton_gpu.memdesc<1xi64, #shared0, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared0 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: inval_barrier
	// CHECK: local_alloc
	// CHECK-NEXT: gpu.barrier
	// CHECK-NEXT: init_barrier
	// CHECK-NEXT: gpu.barrier
	// CHECK-NEXT: inval_barrier
  tt.func @inval_barrier() {
  	%cst = arith.constant dense<0> : tensor<1xi64, #blocked0>
  	%alloc = triton_gpu.local_alloc %cst : (tensor<1xi64, #blocked0>) -> !triton_gpu.memdesc<1xi64, #shared0, #triton_gpu.shared_memory, mutable>
    triton_nvidia_gpu.init_barrier %alloc, 1 : !triton_gpu.memdesc<1xi64, #shared0, #triton_gpu.shared_memory, mutable>
		triton_nvidia_gpu.inval_barrier %alloc : !triton_gpu.memdesc<1xi64, #shared0, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared0 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: barrier_expect
	// CHECK: local_alloc
	// CHECK-NEXT: gpu.barrier
	// CHECK-NEXT: init_barrier
	// CHECK-NEXT: gpu.barrier
	// CHECK-NEXT: barrier_expect
  tt.func @barrier_expect(%pred : i1) {
  	%cst = arith.constant dense<0> : tensor<1xi64, #blocked0>
  	%alloc = triton_gpu.local_alloc %cst : (tensor<1xi64, #blocked0>) -> !triton_gpu.memdesc<1xi64, #shared0, #triton_gpu.shared_memory, mutable>
    triton_nvidia_gpu.init_barrier %alloc, 1 : !triton_gpu.memdesc<1xi64, #shared0, #triton_gpu.shared_memory, mutable>
    triton_nvidia_gpu.barrier_expect %alloc, 16384, %pred : <1xi64, #shared0, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared0 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: wait_barrier
	// CHECK: local_alloc
	// CHECK-NEXT: gpu.barrier
	// CHECK-NEXT: init_barrier
	// CHECK-NEXT: gpu.barrier
	// CHECK-NEXT: wait_barrier
  tt.func @wait_barrier(%phase : i32) {
  	%cst = arith.constant dense<0> : tensor<1xi64, #blocked0>
  	%alloc = triton_gpu.local_alloc %cst : (tensor<1xi64, #blocked0>) -> !triton_gpu.memdesc<1xi64, #shared0, #triton_gpu.shared_memory, mutable>
    triton_nvidia_gpu.init_barrier %alloc, 1 : !triton_gpu.memdesc<1xi64, #shared0, #triton_gpu.shared_memory, mutable>
    triton_nvidia_gpu.wait_barrier %alloc, %phase : <1xi64, #shared0, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----


#shared0 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @tma_load(%arg0: !tt.tensordesc<tensor<128x64xf16>>, %arg1: i32) -> tensor<128x64xf16, #blocked0> {
		// CHECK-LABEL: tma_load
		// CHECK: local_dealloc
		// CHECK-NEXT: local_alloc
		// CHECK-NEXT: local_alloc
		// CHECK-NEXT: gpu.barrier
		// CHECK-NEXT: init_barrier
  	%cst = arith.constant dense<0> : tensor<128x64xi64, #blocked0>
  	%alloc = triton_gpu.local_alloc %cst : (tensor<128x64xi64, #blocked0>) -> !triton_gpu.memdesc<128x64xi64, #shared0, #triton_gpu.shared_memory, mutable>
  	triton_gpu.local_dealloc %alloc : !triton_gpu.memdesc<128x64xi64, #shared0, #triton_gpu.shared_memory, mutable>
    %l = tt.experimental_descriptor_load %arg0[%arg1, %arg1] : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked0>
    tt.return %l : tensor<128x64xf16, #blocked0>
  }
}

// -----

#shared0 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tma_store
//       CHECK: triton_gpu.local_alloc
//       CHECK-NEXT: triton_gpu.local_dealloc
//       CHECK-NEXT: gpu.barrier
//       CHECK-NEXT: triton_gpu.local_alloc
  tt.func public @tma_store(%arg0: !tt.tensordesc<tensor<128x256xf32>>, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: tensor<128x256xf32, #blocked0>) {
  	%cst = arith.constant dense<0> : tensor<128x64xi64, #blocked0>
  	%alloc = triton_gpu.local_alloc %cst : (tensor<128x64xi64, #blocked0>) -> !triton_gpu.memdesc<128x64xi64, #shared0, #triton_gpu.shared_memory, mutable>
  	triton_gpu.local_dealloc %alloc : !triton_gpu.memdesc<128x64xi64, #shared0, #triton_gpu.shared_memory, mutable>
    tt.experimental_descriptor_store %arg0[%arg1, %arg1], %arg2 : !tt.tensordesc<tensor<128x256xf32>>, tensor<128x256xf32, #blocked0>
    tt.return
  }
}
