// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="arch-generation-name=gfx950" --tritonamdgpu-optimize-buffer-op-ptr| FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: add_after_load
// CHECK-DAG: [[X_OFFSET_CST:%.*]] = arith.constant dense<123>
// CHECK-DAG: [[Y_OFFSET_CST:%.*]] = arith.constant dense<321>
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, [[X_BASE:%.*]] = {{.*}}, [[Y_BASE:%.*]] = {{.*}})
// CHECK:   amdg.buffer_load [[X_BASE]]{{\[}}[[X_OFFSET_CST]]{{\]}} :
// CHECK:   amdg.buffer_load [[Y_BASE]]{{\[}}[[Y_OFFSET_CST]]{{\]}} cacheModifier = cg :
// CHECK:   [[NEXT_X_BASE:%.*]] = tt.addptr [[X_BASE]], %c64_i32
// CHECK:   [[NEXT_Y_BASE:%.*]] = tt.addptr [[Y_BASE]]
// CHECK:   scf.yield {{.*}}, [[NEXT_X_BASE]], [[NEXT_Y_BASE]]

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @add_after_load(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %Y: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride: i32) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<16x64xi32, #blocked>
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c64_i32 = arith.constant 64 : i32
    %c1 = arith.constant 1 : index

    %min_stride = arith.constant 1 : i32
    %max_stride = arith.constant 1024 : i32
    %0 = arith.cmpi sge, %stride, %min_stride : i32
    llvm.intr.assume %0 : i1
    %1 = arith.cmpi sle, %stride, %max_stride : i32
    llvm.intr.assume %1 : i1

    %Xoffset_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>
    %Yoffset_init = arith.constant dense<321> : tensor<64x32xi32, #blocked>

    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
    %y_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable, 64x32>

    %tmp = arith.muli %stride, %c64_i32 : i32
    %step = tt.splat %tmp : i32 -> tensor<64x32xi32, #blocked>
    %for:2 = scf.for %idx = %c0 to %c128 step %c1 iter_args(%Xoffset = %Xoffset_init, %Yoffset = %Yoffset_init) -> (tensor<16x64xi32, #blocked>, tensor<64x32xi32, #blocked>) {
      %x = amdg.buffer_load %X[%Xoffset] : tensor<16x64xf16, #blocked>
      %y = amdg.buffer_load %Y[%Yoffset] cacheModifier = cg : tensor<64x32xf16, #blocked>

      ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      ttg.local_store %y, %y_dummy_buffer : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable, 64x32>

      %Xoffset_next = arith.addi %Xoffset, %cst : tensor<16x64xi32, #blocked>
      %Yoffset_next = arith.addi %Yoffset, %step : tensor<64x32xi32, #blocked>
      scf.yield %Xoffset_next, %Yoffset_next : tensor<16x64xi32, #blocked>, tensor<64x32xi32, #blocked>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: buffer_load_to_local
// CHECK-DAG: [[X_OFFSET_CST:%.*]] = arith.constant dense<123>
// CHECK: scf.for {{.*}} iter_args({{.*}}, [[X_BASE:%.*]] = {{.*}}
// CHECK:   amdg.buffer_load_to_local [[X_BASE]]{{\[}}[[X_OFFSET_CST]]{{\]}}
// CHECK:   [[NEXT_X_BASE:%.*]] = tt.addptr [[X_BASE]], %c64_i32
// CHECK:   scf.yield {{.*}}, [[NEXT_X_BASE]]

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @buffer_load_to_local(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<16x64xi32, #blocked>
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index

    %Xoffset_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>

    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>

    %for = scf.for %idx = %c0 to %c128 step %c1 iter_args(%Xoffset = %Xoffset_init) -> (tensor<16x64xi32, #blocked>) {
      %x = amdg.buffer_load_to_local %X[%Xoffset] into %x_dummy_buffer : <f16>[tensor<16x64xi32, #blocked>] -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>

      %Xoffset_next = arith.addi %Xoffset, %cst : tensor<16x64xi32, #blocked>
      scf.yield %Xoffset_next : tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: add_before_load
// CHECK-DAG: [[X_OFFSET_CST:%.*]] = arith.constant dense<123>
// CHECK: scf.for {{.*}} iter_args({{.*}}, [[X_BASE:%.*]] = {{.*}})
// CHECK:   [[NEXT_X_BASE:%.*]] = tt.addptr [[X_BASE]], %c64_i32
// CHECK:   amdg.buffer_load [[NEXT_X_BASE]]{{\[}}[[X_OFFSET_CST]]{{\]}}
// CHECK:   scf.yield {{.*}}, [[NEXT_X_BASE]]

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @add_before_load(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %Y: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<16x64xi32, #blocked>
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %Xoffset_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>
    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
    %for = scf.for %idx = %c0 to %c128 step %c1 iter_args(%Xoffset = %Xoffset_init) -> (tensor<16x64xi32, #blocked>) {
      %Xoffset_next = arith.addi %Xoffset, %cst : tensor<16x64xi32, #blocked>
      %x = amdg.buffer_load %X[%Xoffset_next] : tensor<16x64xf16, #blocked>
      ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      scf.yield %Xoffset_next : tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: isolated_pattern_nested_loop1
// CHECK: [[X_OFFSET_CST:%.*]] = arith.constant dense<123>
// CHECK: scf.for
// CHECK:   scf.for {{.*}} iter_args({{.*}}, [[X_BASE:%.*]] = {{.*}})
// CHECK:     amdg.buffer_load [[X_BASE]]{{\[}}[[X_OFFSET_CST]]{{\]}}
// CHECK:     [[NEXT_X_BASE:%.*]] = tt.addptr [[X_BASE]]
// CHECK:     scf.yield {{.*}}, [[NEXT_X_BASE]]

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @isolated_pattern_nested_loop1(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<16x64xi32, #blocked>
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index

    %Xoffset_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>
    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
    scf.for %idx_outer = %c0 to %c32 step %c1 iter_args() -> () {
      %for_inner = scf.for %idx_innter = %c0 to %c32 step %c1 iter_args(%Xoffset = %Xoffset_init) -> (tensor<16x64xi32, #blocked>) {
        %x = amdg.buffer_load %X[%Xoffset] : tensor<16x64xf16, #blocked>
        ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
        %Xoffset_next = arith.addi %Xoffset, %cst : tensor<16x64xi32, #blocked>
        scf.yield %Xoffset_next : tensor<16x64xi32, #blocked>
      }
      scf.yield
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: isolated_pattern_nested_loop2
// CHECK: [[X_OFFSET_CST:%.*]] = arith.constant dense<123>
// CHECK: scf.for {{.*}} iter_args({{.*}}, [[X_BASE:%.*]] = {{.*}})
// CHECK:   scf.for
// CHECK:     amdg.buffer_load [[X_BASE]]{{\[}}[[X_OFFSET_CST]]{{\]}}
// CHECK:   [[NEXT_X_BASE:%.*]] = tt.addptr [[X_BASE]]
// CHECK:   scf.yield {{.*}}, [[NEXT_X_BASE]]

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @isolated_pattern_nested_loop2(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<16x64xi32, #blocked>
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index

    %Xoffset_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>
    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
    %for_outer = scf.for %idx_outer = %c0 to %c128 step %c1 iter_args(%Xoffset = %Xoffset_init) -> (tensor<16x64xi32, #blocked>) {
      scf.for %idx_inner = %c0 to %c128 step %c1 iter_args() -> () {
        %x = amdg.buffer_load %X[%Xoffset] : tensor<16x64xf16, #blocked>
        ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
        scf.yield
      }
      %Xoffset_next = arith.addi %Xoffset, %cst : tensor<16x64xi32, #blocked>
      scf.yield %Xoffset_next : tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: convert_with_base_ptr_optimization
// CHECK: [[X_OFFSET_CST:%.*]] = arith.constant dense<123>
// CHECK: scf.for {{.*}} iter_args({{.*}}, [[X_BASE:%.*]] = {{.*}})
// CHECK:   amdg.buffer_load [[X_BASE]]{{\[}}[[X_OFFSET_CST]]{{\]}}
// CHECK:   [[NEXT_X_BASE:%.*]] = tt.addptr [[X_BASE]]
// CHECK:   scf.yield {{.*}}, [[NEXT_X_BASE]]

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @convert_with_base_ptr_optimization(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %step = arith.constant dense<64> : tensor<16x64xi32, #blocked>
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %x_base = tt.splat %X : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>, #blocked>
    %offsets_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>
    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
    %for = scf.for %idx_outer = %c0 to %c128 step %c1 iter_args(%offsets = %offsets_init) -> (tensor<16x64xi32, #blocked>) {
      %X_ptr = tt.addptr %x_base, %offsets : tensor<16x64x!tt.ptr<f16>, #blocked>, tensor<16x64xi32, #blocked>
      %x = tt.load %X_ptr : tensor<16x64x!tt.ptr<f16>, #blocked>
      ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      %offsets_next = arith.addi %offsets, %step : tensor<16x64xi32, #blocked>
      scf.yield %offsets_next : tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: dynamic_base_negative
// CHECK:   [[X_BASE:%.*]] = tt.addptr
// CHECK:   amdg.buffer_load [[X_BASE]]
// CHECK-NOT: tt.addptr

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @dynamic_base_negative(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<16x64xi32, #blocked>
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %c1 = arith.constant 1 : i32

    %Xoffset_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>
    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
    %for = scf.for %idx = %c0 to %c128 step %c1 iter_args(%Xoffset = %Xoffset_init) -> (tensor<16x64xi32, #blocked>) : i32 {
      %x_base = tt.addptr %X, %idx : !tt.ptr<f16>, i32
      %x = amdg.buffer_load %x_base[%Xoffset] : tensor<16x64xf16, #blocked>
      ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      %Xoffset_next = arith.addi %Xoffset, %cst : tensor<16x64xi32, #blocked>
      scf.yield %Xoffset_next : tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: non_uniform_step_negative
// CHECK-NOT: tt.addptr

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @non_uniform_step_negative(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %step : tensor<16x64xi32, #blocked>) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %c1 = arith.constant 1 : i32

    %Xoffset_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>
    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
    %for = scf.for %idx = %c0 to %c128 step %c1 iter_args(%Xoffset = %Xoffset_init) -> (tensor<16x64xi32, #blocked>) : i32 {
      %x = amdg.buffer_load %X[%Xoffset] : tensor<16x64xf16, #blocked>
      ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      %Xoffset_next = arith.addi %Xoffset, %step : tensor<16x64xi32, #blocked>
      scf.yield %Xoffset_next : tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: offsets_possible_overflow_negative
// CHECK-NOT: tt.addptr

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @offsets_possible_overflow_negative(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %step_scalar : i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %c1 = arith.constant 1 : i32

    %Xoffset_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>
    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
    %step = tt.splat %step_scalar: i32 -> tensor<16x64xi32, #blocked>
    %for = scf.for %idx = %c0 to %c128 step %c1 iter_args(%Xoffset = %Xoffset_init) -> (tensor<16x64xi32, #blocked>) : i32 {
      %x = amdg.buffer_load %X[%Xoffset] : tensor<16x64xf16, #blocked>
      ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      %Xoffset_next = arith.addi %Xoffset, %step : tensor<16x64xi32, #blocked>
      scf.yield %Xoffset_next : tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: two_parallel_addi_negative
// CHECK-NOT: tt.addptr

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @two_parallel_addi_negative(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %c1 = arith.constant 1 : i32

    %Xoffset_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>
    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
    %step = arith.constant dense<64> : tensor<16x64xi32, #blocked>
    %for:2 = scf.for %idx = %c0 to %c128 step %c1 iter_args(%Xoffset = %Xoffset_init, %Xoffset_dummy = %Xoffset_init) -> (tensor<16x64xi32, #blocked>, tensor<16x64xi32, #blocked>) : i32 {
      %Xoffset_decoy = arith.addi %Xoffset, %step : tensor<16x64xi32, #blocked>
      %x = amdg.buffer_load %X[%Xoffset_decoy] : tensor<16x64xf16, #blocked>
      ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      %Xoffset_next = arith.addi %Xoffset, %step : tensor<16x64xi32, #blocked>
      scf.yield %Xoffset_next, %Xoffset_decoy : tensor<16x64xi32, #blocked>, tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}

// -----

// Check case with three buffer ops in a loop, first and third are optimized, second is rejected because step is not uniform
// CHECK-LABEL: mixed_optimized_rejected_optimized
// CHECK-DAG: [[X_OFFSET:%.*]] = arith.constant dense<123> : tensor<16x64xi32, #blocked>
// CHECK-DAG: [[Y_OFFSET_INIT:%.*]] = arith.constant dense<456> : tensor<64x32xi32, #blocked>
// CHECK-DAG: [[Z_OFFSET:%.*]] = arith.constant dense<789> : tensor<32x32xi32, #blocked>
// CHECK: scf.for {{.*}} iter_args({{%.*}}[[Y_OFFSET:%.*]] = [[Y_OFFSET_INIT]]
// CHECK-DAG: amdg.buffer_load {{%.*\[}}[[X_OFFSET]]{{\]}} : tensor<16x64xf16, #blocked>
// CHECK-DAG: amdg.buffer_load {{%.*\[}}[[Y_OFFSET]]{{\]}} cacheModifier = cg : tensor<64x32xf16, #blocked>
// CHECK-DAG: amdg.buffer_load {{%.*\[}}[[Z_OFFSET]]{{\]}} cacheModifier = cg : tensor<32x32xf16, #blocked>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mixed_optimized_rejected_optimized(
        %X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
        %Y: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
        %Z: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}
      ) attributes {noinline = false} {
    %Xstep = arith.constant dense<64> : tensor<16x64xi32, #blocked>
    %Ystep_slice = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %Ystep_2d = tt.expand_dims %Ystep_slice {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %Ystep = tt.broadcast %Ystep_2d : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %Zstep = arith.constant dense<256> : tensor<32x32xi32, #blocked>
    %iter_first = arith.constant 0 : index
    %iter_last = arith.constant 128 : index
    %iter_step = arith.constant 1 : index

    %Xoffset_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>
    %Yoffset_init = arith.constant dense<456> : tensor<64x32xi32, #blocked>
    %Zoffset_init = arith.constant dense<789> : tensor<32x32xi32, #blocked>

    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
    %y_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable, 64x32>
    %z_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable, 32x32>

    %for:3 = scf.for %iter = %iter_first to %iter_last step %iter_step iter_args(%Xoffset = %Xoffset_init, %Yoffset = %Yoffset_init, %Zoffset = %Zoffset_init) -> (tensor<16x64xi32, #blocked>, tensor<64x32xi32, #blocked>, tensor<32x32xi32, #blocked>) {
      %x = amdg.buffer_load %X[%Xoffset] : tensor<16x64xf16, #blocked>
      %y = amdg.buffer_load %Y[%Yoffset] cacheModifier = cg : tensor<64x32xf16, #blocked>
      %z = amdg.buffer_load %Z[%Zoffset] cacheModifier = cg : tensor<32x32xf16, #blocked>

      ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      ttg.local_store %y, %y_dummy_buffer : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable, 64x32>
      ttg.local_store %z, %z_dummy_buffer : tensor<32x32xf16, #blocked> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable, 32x32>

      %Xoffset_next = arith.addi %Xoffset, %Xstep : tensor<16x64xi32, #blocked>
      %Yoffset_next = arith.addi %Yoffset, %Ystep : tensor<64x32xi32, #blocked>
      %Zoffset_next = arith.addi %Zoffset, %Zstep : tensor<32x32xi32, #blocked>
      scf.yield %Xoffset_next, %Yoffset_next, %Zoffset_next : tensor<16x64xi32, #blocked>, tensor<64x32xi32, #blocked>, tensor<32x32xi32, #blocked>
    }
    tt.return
  }
}

// -----

// Check case with three buffer ops in a loop, only second one is optimized
// CHECK-LABEL: mixed_rejected_optimized_rejected
// CHECK-DAG: [[X_OFFSET_INIT:%.*]] = arith.constant dense<123> : tensor<16x64xi32, #blocked>
// CHECK-DAG: [[Y_OFFSET:%.*]] = arith.constant dense<456> : tensor<64x32xi32, #blocked>
// CHECK-DAG: [[Z_OFFSET_INIT:%.*]] = arith.constant dense<789> : tensor<32x32xi32, #blocked>
// CHECK: scf.for {{.*}} iter_args([[X_OFFSET:%.*]] = [[X_OFFSET_INIT]], {{.*}}[[Z_OFFSET:%.*]] = [[Z_OFFSET_INIT]]
// CHECK-DAG: amdg.buffer_load {{%.*\[}}[[X_OFFSET]]{{\]}} : tensor<16x64xf16, #blocked>
// CHECK-DAG: amdg.buffer_load {{%.*\[}}[[Y_OFFSET]]{{\]}} cacheModifier = cg : tensor<64x32xf16, #blocked>
// CHECK-DAG: amdg.buffer_load {{%.*\[}}[[Z_OFFSET]]{{\]}} cacheModifier = cg : tensor<32x32xf16, #blocked>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mixed_rejected_optimized_rejected(
        %X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
        %Y: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
        %Z: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
        %Xstride: i32,
        %Zstride: i32
      ) attributes {noinline = false} {
    %Xstep = tt.splat %Xstride: i32 -> tensor<16x64xi32, #blocked>
    %Ystep = arith.constant dense<128> : tensor<64x32xi32, #blocked>
    %Zstep = tt.splat %Zstride : i32 -> tensor<32x32xi32, #blocked>
    %iter_first = arith.constant 0 : index
    %iter_last = arith.constant 128 : index
    %iter_step = arith.constant 1 : index

    %Xoffset_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>
    %Yoffset_init = arith.constant dense<456> : tensor<64x32xi32, #blocked>
    %Zoffset_init = arith.constant dense<789> : tensor<32x32xi32, #blocked>

    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
    %y_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable, 64x32>
    %z_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable, 32x32>

    %for:3 = scf.for %iter = %iter_first to %iter_last step %iter_step iter_args(%Xoffset = %Xoffset_init, %Yoffset = %Yoffset_init, %Zoffset = %Zoffset_init) -> (tensor<16x64xi32, #blocked>, tensor<64x32xi32, #blocked>, tensor<32x32xi32, #blocked>) {
      %x = amdg.buffer_load %X[%Xoffset] : tensor<16x64xf16, #blocked>
      %y = amdg.buffer_load %Y[%Yoffset] cacheModifier = cg : tensor<64x32xf16, #blocked>
      %z = amdg.buffer_load %Z[%Zoffset] cacheModifier = cg : tensor<32x32xf16, #blocked>

      ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      ttg.local_store %y, %y_dummy_buffer : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable, 64x32>
      ttg.local_store %z, %z_dummy_buffer : tensor<32x32xf16, #blocked> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable, 32x32>

      %Xoffset_next = arith.addi %Xoffset, %Xstep : tensor<16x64xi32, #blocked>
      %Yoffset_next = arith.addi %Yoffset, %Ystep : tensor<64x32xi32, #blocked>
      %Zoffset_next = arith.addi %Zoffset, %Zstep : tensor<32x32xi32, #blocked>
      scf.yield %Xoffset_next, %Yoffset_next, %Zoffset_next : tensor<16x64xi32, #blocked>, tensor<64x32xi32, #blocked>, tensor<32x32xi32, #blocked>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: multiple_offset_uses
// CHECK: tt.addptr

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @multiple_offset_uses(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %Y: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %offset_step = arith.constant dense<64> : tensor<16x64xi32, #blocked>
    %iter_first = arith.constant 0 : index
    %iter_last = arith.constant 128 : index
    %iter_step = arith.constant 1 : index
    %Xoffset_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>
    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
    %offset_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xi32, #shared, #smem, mutable, 16x64>
    %for = scf.for %idx = %iter_first to %iter_last step %iter_step iter_args(%Xoffset = %Xoffset_init) -> (tensor<16x64xi32, #blocked>) {
      ttg.local_store %Xoffset, %offset_buffer : tensor<16x64xi32, #blocked> -> !ttg.memdesc<16x64xi32, #shared, #smem, mutable, 16x64>
      %Xoffset_next = arith.addi %Xoffset, %offset_step : tensor<16x64xi32, #blocked>
      %x = amdg.buffer_load %X[%Xoffset_next] : tensor<16x64xf16, #blocked>
      ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      ttg.local_store %Xoffset_next, %offset_buffer : tensor<16x64xi32, #blocked> -> !ttg.memdesc<16x64xi32, #shared, #smem, mutable, 16x64>
      scf.yield %Xoffset_next : tensor<16x64xi32, #blocked>
    }
    ttg.local_store %for, %offset_buffer : tensor<16x64xi32, #blocked> -> !ttg.memdesc<16x64xi32, #shared, #smem, mutable, 16x64>
    tt.return
  }
}

// -----

// CHECK-LABEL: multiple_offset_uses_over_multiple_loops
// CHECK: scf.for
// CHECK:   tt.addptr
// CHECK:   scf.for
// CHECK:     tt.addptr

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @multiple_offset_uses_over_multiple_loops(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %Y: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %offset_step = arith.constant dense<64> : tensor<16x64xi32, #blocked>
    %iter_first = arith.constant 0 : index
    %iter_last = arith.constant 32 : index
    %iter_step = arith.constant 1 : index
    %Xoffset_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>
    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
    %offset_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xi32, #shared, #smem, mutable, 16x64>
    %for1 = scf.for %idx = %iter_first to %iter_last step %iter_step iter_args(%Xoffset1 = %Xoffset_init) -> (tensor<16x64xi32, #blocked>) {
      %Xoffset_next1 = arith.addi %Xoffset1, %offset_step : tensor<16x64xi32, #blocked>

      %for2 = scf.for %idx2 = %iter_first to %iter_last step %iter_step iter_args(%Xoffset2 = %Xoffset_next1) -> (tensor<16x64xi32, #blocked>) {
        %Xoffset_next2 = arith.addi %Xoffset2, %offset_step : tensor<16x64xi32, #blocked>
        %x2 = amdg.buffer_load %X[%Xoffset_next2] : tensor<16x64xf16, #blocked>
        ttg.local_store %x2, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
        scf.yield %Xoffset_next2 : tensor<16x64xi32, #blocked>
      }

      %x1 = amdg.buffer_load %X[%Xoffset_next1] : tensor<16x64xf16, #blocked>
      ttg.local_store %x1, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      scf.yield %Xoffset_next1 : tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: multiple_addi_in_sequence_negative
// CHECK-NOT:  tt.addptr

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @multiple_addi_in_sequence_negative(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %cst1 = arith.constant dense<64> : tensor<16x64xi32, #blocked>
    %cst2 = arith.constant dense<128> : tensor<16x64xi32, #blocked>
    %iter_first = arith.constant 0 : index
    %iter_last = arith.constant 128 : index
    %iter_step = arith.constant 1 : index

    %Xoffset_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>

    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>

    %for = scf.for %idx = %iter_first to %iter_last step %iter_step iter_args(%Xoffset = %Xoffset_init) -> (tensor<16x64xi32, #blocked>) {
      %x = amdg.buffer_load_to_local %X[%Xoffset] into %x_dummy_buffer : <f16>[tensor<16x64xi32, #blocked>] -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>

      %Xoffset_tmp = arith.addi %Xoffset, %cst1 : tensor<16x64xi32, #blocked>
      %Xoffset_next = arith.addi %Xoffset_tmp, %cst2 : tensor<16x64xi32, #blocked>
      scf.yield %Xoffset_next : tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: multiple_buffer_loads_on_one_addi
// CHECK-COUNT-2: tt.addptr

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @multiple_buffer_loads_on_one_addi(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<16x64xi32, #blocked>
    %iter_first = arith.constant 0 : index
    %iter_last = arith.constant 128 : index
    %iter_step = arith.constant 1 : index

    %Xoffset_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>

    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>

    %for = scf.for %idx = %iter_first to %iter_last step %iter_step iter_args(%Xoffset = %Xoffset_init) -> (tensor<16x64xi32, #blocked>) {
      %x1 = amdg.buffer_load_to_local %X[%Xoffset] into %x_dummy_buffer : <f16>[tensor<16x64xi32, #blocked>] -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      %Xoffset_next = arith.addi %Xoffset, %cst : tensor<16x64xi32, #blocked>
      %x2 = amdg.buffer_load_to_local %X[%Xoffset_next] into %x_dummy_buffer : <f16>[tensor<16x64xi32, #blocked>] -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      scf.yield %Xoffset_next : tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: messed_add_chain_negative
// CHECK-NOT:  tt.addptr

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @messed_add_chain_negative(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) -> tensor<256xi32, #blocked> attributes {noinline = false} {
    %zero = arith.constant dense<0> : tensor<256xi32, #blocked>
    %iter_first = arith.constant 0 : i32
    %iter_last = arith.constant 15 : i32
    %iter_step = arith.constant 1 : i32

    %for:2 = scf.for %iter = %iter_first to %iter_last step %iter_step iter_args(%arg1 = %zero, %arg2 = %zero) -> (tensor<256xi32, #blocked>, tensor<256xi32, #blocked>) : i32 {
      %data = amdg.buffer_load %arg0[%arg1] : tensor<256xi32, #blocked>
      scf.yield %arg2, %data : tensor<256xi32, #blocked>, tensor<256xi32, #blocked>
    }
    tt.return %for#1 : tensor<256xi32, #blocked>
  }
}
