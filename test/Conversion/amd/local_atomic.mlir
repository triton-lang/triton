// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck %s
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: test_local_atomic_scatter_add
  // CHECK: llvm.atomicrmw add {{.*}} syncscope("workgroup") monotonic
  tt.func public @test_local_atomic_scatter_add(%arg0: tensor<1xi32, #blocked>, %arg1: tensor<1xi32, #blocked>) {
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %1 = ttg.local_atomic_scatter_rmw add, %0[%arg0], %arg1 {axis = 0 : i32} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, tensor<1xi32, #blocked>, tensor<1xi32, #blocked>) -> tensor<1xi32, #blocked>
    %2 = ttg.local_alloc {allocation.offset = 4 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    ttg.local_store %1, %2 : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: test_local_atomic_add_masked
  // CHECK: llvm.atomicrmw add {{.*}} syncscope("workgroup") monotonic
  tt.func public @test_local_atomic_add_masked(%arg0: tensor<1xi32, #blocked>, %mask: tensor<1xi1, #blocked>) {
    %c1 = arith.constant dense<1> : tensor<1xi32, #blocked>
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %1 = ttg.local_atomic_scatter_rmw add, %0[%arg0], %c1, %mask {axis = 0 : i32} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, tensor<1xi32, #blocked>, tensor<1xi32, #blocked>, tensor<1xi1, #blocked>) -> tensor<1xi32, #blocked>
    %2 = ttg.local_alloc {allocation.offset = 4 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    ttg.local_store %1, %2 : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: test_local_atomic_scatter_rmw_max_i32
  // CHECK: llvm.atomicrmw max {{.*}} syncscope("workgroup") monotonic
  tt.func public @test_local_atomic_scatter_rmw_max_i32(%arg0: tensor<1xi32, #blocked>, %arg1: tensor<1xi32, #blocked>, %mask: tensor<1xi1, #blocked>) {
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %1 = ttg.local_atomic_scatter_rmw max, %0[%arg0], %arg1, %mask {axis = 0 : i32} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, tensor<1xi32, #blocked>, tensor<1xi32, #blocked>, tensor<1xi1, #blocked>) -> tensor<1xi32, #blocked>
    %2 = ttg.local_alloc {allocation.offset = 4 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    ttg.local_store %1, %2 : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: test_local_atomic_scatter_add_f32_masked
  // CHECK: llvm.atomicrmw fadd {{.*}} syncscope("workgroup") monotonic
  tt.func public @test_local_atomic_scatter_add_f32_masked(%arg0: tensor<1xi32, #blocked>, %arg1: tensor<1xf32, #blocked>, %mask: tensor<1xi1, #blocked>) {
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1xf32, #shared, #smem, mutable>
    %1 = ttg.local_atomic_scatter_rmw fadd, %0[%arg0], %arg1, %mask {axis = 0 : i32} : (!ttg.memdesc<1xf32, #shared, #smem, mutable>, tensor<1xf32, #blocked>, tensor<1xi32, #blocked>, tensor<1xi1, #blocked>) -> tensor<1xf32, #blocked>
    %2 = ttg.local_alloc {allocation.offset = 4 : i32} : () -> !ttg.memdesc<1xf32, #shared, #smem, mutable>
    ttg.local_store %1, %2 : tensor<1xf32, #blocked> -> !ttg.memdesc<1xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: test_local_atomic_add_dead
  // CHECK: llvm.atomicrmw add {{.*}} syncscope("workgroup") monotonic
  tt.func public @test_local_atomic_add_dead(%arg0: tensor<1xi32, #blocked>) {
    %c1 = arith.constant dense<1> : tensor<1xi32, #blocked>
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %1 = ttg.local_atomic_scatter_rmw add, %0[%arg0], %c1 {axis = 0 : i32} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, tensor<1xi32, #blocked>, tensor<1xi32, #blocked>) -> tensor<1xi32, #blocked>
    tt.return
  }
}
