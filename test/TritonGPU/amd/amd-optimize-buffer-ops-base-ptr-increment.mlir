// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="arch-generation-name=gfx950" --tritonamdgpu-optimize-buffer-op-ptr| FileCheck %s --check-prefixes=COMMON

// COMMON-LABEL: add_after_load
// COMMON-DAG: [[X_OFFSET_CST:%.*]] = arith.constant dense<123>
// COMMON-DAG: [[Y_OFFSET_CST:%.*]] = arith.constant dense<321>
// COMMON: scf.for {{.*}} iter_args({{.*}}, {{.*}}, [[X_BASE:%.*]] = {{.*}}, [[Y_BASE:%.*]] = {{.*}})
// COMMON:   amdgpu.buffer_load [[X_BASE]]{{\[}}[[X_OFFSET_CST]]{{\]}} :
// COMMON:   amdgpu.buffer_load [[Y_BASE]]{{\[}}[[Y_OFFSET_CST]]{{\]}} cacheModifier = cg :
// COMMON:   [[NEXT_X_BASE:%.*]] = tt.addptr [[X_BASE]], %c64_i32
// COMMON:   [[NEXT_Y_BASE:%.*]] = tt.addptr [[Y_BASE]]
// COMMON:   scf.yield {{.*}}, [[NEXT_X_BASE]], [[NEXT_Y_BASE]]

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
      %x = amdgpu.buffer_load %X[%Xoffset] : tensor<16x64xf16, #blocked>
      %y = amdgpu.buffer_load %Y[%Yoffset] cacheModifier = cg : tensor<64x32xf16, #blocked>

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

// COMMON-LABEL: buffer_load_to_local
// COMMON-DAG: [[X_OFFSET_CST:%.*]] = arith.constant dense<123>
// COMMON: scf.for {{.*}} iter_args({{.*}}, [[X_BASE:%.*]] = {{.*}}
// COMMON:   amdgpu.buffer_load_to_local [[X_BASE]]{{\[}}[[X_OFFSET_CST]]{{\]}}
// COMMON:   [[NEXT_X_BASE:%.*]] = tt.addptr [[X_BASE]], %c64_i32
// COMMON:   scf.yield {{.*}}, [[NEXT_X_BASE]]

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
      %x = amdgpu.buffer_load_to_local %X[%Xoffset] into %x_dummy_buffer : <f16>[tensor<16x64xi32, #blocked>] -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>

      %Xoffset_next = arith.addi %Xoffset, %cst : tensor<16x64xi32, #blocked>
      scf.yield %Xoffset_next : tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}

// -----

// COMMON-LABEL: add_before_load
// COMMON-DAG: [[X_OFFSET_CST:%.*]] = arith.constant dense<123>
// COMMON: scf.for {{.*}} iter_args({{.*}}, [[X_BASE:%.*]] = {{.*}})
// COMMON:   [[NEXT_X_BASE:%.*]] = tt.addptr [[X_BASE]], %c64_i32
// COMMON:   amdgpu.buffer_load [[NEXT_X_BASE]]{{\[}}[[X_OFFSET_CST]]{{\]}}
// COMMON:   scf.yield {{.*}}, [[NEXT_X_BASE]]

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
      %x = amdgpu.buffer_load %X[%Xoffset_next] : tensor<16x64xf16, #blocked>
      ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      scf.yield %Xoffset_next : tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}

// -----

// COMMON-LABEL: isolated_pattern_nested_loop1
// COMMON: [[X_OFFSET_CST:%.*]] = arith.constant dense<123>
// COMMON: scf.for
// COMMON:   scf.for {{.*}} iter_args({{.*}}, [[X_BASE:%.*]] = {{.*}})
// COMMON:     amdgpu.buffer_load [[X_BASE]]{{\[}}[[X_OFFSET_CST]]{{\]}}
// COMMON:     [[NEXT_X_BASE:%.*]] = tt.addptr [[X_BASE]]
// COMMON:     scf.yield {{.*}}, [[NEXT_X_BASE]]

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
        %x = amdgpu.buffer_load %X[%Xoffset] : tensor<16x64xf16, #blocked>
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

// COMMON-LABEL: isolated_pattern_nested_loop2
// COMMON: [[X_OFFSET_CST:%.*]] = arith.constant dense<123>
// COMMON: scf.for {{.*}} iter_args({{.*}}, [[X_BASE:%.*]] = {{.*}})
// COMMON:   scf.for
// COMMON:     amdgpu.buffer_load [[X_BASE]]{{\[}}[[X_OFFSET_CST]]{{\]}}
// COMMON:   [[NEXT_X_BASE:%.*]] = tt.addptr [[X_BASE]]
// COMMON:   scf.yield {{.*}}, [[NEXT_X_BASE]]

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
        %x = amdgpu.buffer_load %X[%Xoffset] : tensor<16x64xf16, #blocked>
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

// COMMON-LABEL: convert_with_base_ptr_optimization
// COMMON: [[X_OFFSET_CST:%.*]] = arith.constant dense<123>
// COMMON: scf.for {{.*}} iter_args({{.*}}, [[X_BASE:%.*]] = {{.*}})
// COMMON:   amdgpu.buffer_load [[X_BASE]]{{\[}}[[X_OFFSET_CST]]{{\]}}
// COMMON:   [[NEXT_X_BASE:%.*]] = tt.addptr [[X_BASE]]
// COMMON:   scf.yield {{.*}}, [[NEXT_X_BASE]]

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

// COMMON-LABEL: dynamic_base_negative
// COMMON:   [[X_BASE:%.*]] = tt.addptr
// COMMON:   amdgpu.buffer_load [[X_BASE]]
// COMMON-NOT: tt.addptr

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
      %x = amdgpu.buffer_load %x_base[%Xoffset] : tensor<16x64xf16, #blocked>
      ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      %Xoffset_next = arith.addi %Xoffset, %cst : tensor<16x64xi32, #blocked>
      scf.yield %Xoffset_next : tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}

// -----

// COMMON-LABEL: non_uniform_step_negative
// COMMON-NOT: tt.addptr

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
      %x = amdgpu.buffer_load %X[%Xoffset] : tensor<16x64xf16, #blocked>
      ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      %Xoffset_next = arith.addi %Xoffset, %step : tensor<16x64xi32, #blocked>
      scf.yield %Xoffset_next : tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}

// -----

// COMMON-LABEL: offsets_possible_overflow_negative
// COMMON-NOT: tt.addptr

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
      %x = amdgpu.buffer_load %X[%Xoffset] : tensor<16x64xf16, #blocked>
      ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      %Xoffset_next = arith.addi %Xoffset, %step : tensor<16x64xi32, #blocked>
      scf.yield %Xoffset_next : tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}

// -----

// COMMON-LABEL: two_addi_negative
// COMMON-NOT: tt.addptr

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @two_addi_negative(%X: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %c1 = arith.constant 1 : i32

    %Xoffset_init = arith.constant dense<123> : tensor<16x64xi32, #blocked>
    %x_dummy_buffer = ttg.local_alloc : () -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
    %step = arith.constant dense<64> : tensor<16x64xi32, #blocked>
    %for = scf.for %idx = %c0 to %c128 step %c1 iter_args(%Xoffset = %Xoffset_init) -> (tensor<16x64xi32, #blocked>) : i32 {
      %Xoffset_decoy = arith.addi %Xoffset, %step : tensor<16x64xi32, #blocked>
      %x = amdgpu.buffer_load %X[%Xoffset_decoy] : tensor<16x64xf16, #blocked>
      ttg.local_store %x, %x_dummy_buffer : tensor<16x64xf16, #blocked> -> !ttg.memdesc<16x64xf16, #shared, #smem, mutable, 16x64>
      %Xoffset_next = arith.addi %Xoffset, %step : tensor<16x64xi32, #blocked>
      scf.yield %Xoffset_next : tensor<16x64xi32, #blocked>
    }
    tt.return
  }
}
