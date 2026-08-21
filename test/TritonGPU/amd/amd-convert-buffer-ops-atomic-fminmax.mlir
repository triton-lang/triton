// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx942" | FileCheck %s --check-prefixes=CHECK,NO-F32,BUF-F64
// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx950" | FileCheck %s --check-prefixes=CHECK,NO-F32,BUF-F64
// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1100" | FileCheck %s --check-prefixes=CHECK,NO-F32,NO-F64
// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1170" | FileCheck %s --check-prefixes=CHECK,NO-F32,NO-F64
// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1200" | FileCheck %s --check-prefixes=CHECK,BUF-F32,NO-F64
// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1250" | FileCheck %s --check-prefixes=CHECK,BUF-F32,BUF-F64

// Buffer atomic min/max is only emitted for target/type combinations that have a
// native instruction; everything else must stay on the generic atomic path.
//
// A target uses the BUF-F32 / BUF-F64 prefix when it has a native buffer atomic
// fmin/fmax for that type, and the NO-F32 / NO-F64 prefix when it does not, either
// because the ISA family lacks the instruction or because the target has no buffer
// atomic RMW at all. The unprefixed checks hold everywhere: no target has a buffer
// atomic fmin/fmax for f16/bf16, and signed integer min/max always stays generic.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: atomic_max_f16
  tt.func public @atomic_max_f16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %values: tensor<1024xf16, #blocked>) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f16>, i32
    %4 = tt.splat %3 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK-NOT: amdg.buffer_atomic_rmw
    // CHECK: tt.atomic_rmw max
    %6 = tt.atomic_rmw max, acq_rel, gpu, %5, %values : (tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xf16, #blocked>) -> tensor<1024xf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: atomic_min_bf16
  tt.func public @atomic_min_bf16(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %values: tensor<1024xbf16, #blocked>) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<bf16>, i32
    %4 = tt.splat %3 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK-NOT: amdg.buffer_atomic_rmw
    // CHECK: tt.atomic_rmw min
    %6 = tt.atomic_rmw min, acq_rel, gpu, %5, %values : (tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xbf16, #blocked>) -> tensor<1024xbf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // BUF-F32-LABEL: atomic_max_f32
  // NO-F32-LABEL: atomic_max_f32
  tt.func public @atomic_max_f32(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %values: tensor<1024xf32, #blocked>) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.splat %3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // BUF-F32: amdg.buffer_atomic_rmw max
    // NO-F32-NOT: amdg.buffer_atomic_rmw
    // NO-F32: tt.atomic_rmw max
    %6 = tt.atomic_rmw max, acq_rel, gpu, %5, %values : (tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // BUF-F64-LABEL: atomic_min_f64
  // NO-F64-LABEL: atomic_min_f64
  tt.func public @atomic_min_f64(%arg0: !tt.ptr<f64> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %values: tensor<1024xf64, #blocked>) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f64>, i32
    %4 = tt.splat %3 : !tt.ptr<f64> -> tensor<1024x!tt.ptr<f64>, #blocked>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<f64>, #blocked>, tensor<1024xi32, #blocked>
    // BUF-F64: amdg.buffer_atomic_rmw min
    // NO-F64-NOT: amdg.buffer_atomic_rmw
    // NO-F64: tt.atomic_rmw min
    %6 = tt.atomic_rmw min, acq_rel, gpu, %5, %values : (tensor<1024x!tt.ptr<f64>, #blocked>, tensor<1024xf64, #blocked>) -> tensor<1024xf64, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: atomic_max_i32
  tt.func public @atomic_max_i32(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %values: tensor<1024xi32, #blocked>) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<i32>, i32
    %4 = tt.splat %3 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK-NOT: amdg.buffer_atomic_rmw
    // CHECK: tt.atomic_rmw max
    %6 = tt.atomic_rmw max, acq_rel, gpu, %5, %values : (tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    tt.return
  }
}
