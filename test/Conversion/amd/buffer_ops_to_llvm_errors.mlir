// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 --verify-diagnostics

// A tensor soffset goes in the scalar offset register, so lowering fails when
// it may differ across the lanes of a wave.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_lane_varying_soffset(%arg0: !tt.ptr<f32>, %offsets: tensor<64xi32, #blocked>, %s: tensor<64xi32, #blocked>, %n: i32) {
    %desc = amdg.make_buffer_desc %arg0, %n : !tt.ptr<f32> -> !amdg.buffer_desc<f32>
    // expected-error@+1 {{can't prove the soffset tensor is the same across the lanes of a wave}}
    %0 = amdg.buffer_load %desc[%offsets] soffset = %s : tensor<64xi32, #blocked> : !amdg.buffer_desc<f32> -> tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_store_lane_varying_soffset(%arg0: !tt.ptr<f32>, %offsets: tensor<64xi32, #blocked>, %s: tensor<64xi32, #blocked>, %v: tensor<64xf32, #blocked>, %n: i32) {
    %desc = amdg.make_buffer_desc %arg0, %n : !tt.ptr<f32> -> !amdg.buffer_desc<f32>
    // expected-error@+1 {{can't prove the soffset tensor is the same across the lanes of a wave}}
    amdg.buffer_store %v, %desc[%offsets] soffset = %s : tensor<64xi32, #blocked> : !amdg.buffer_desc<f32> -> tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_atomic_load_lane_varying_soffset(%arg0: !tt.ptr<i32>, %offsets: tensor<64xi32, #blocked>, %s: tensor<64xi32, #blocked>, %n: i32) {
    %desc = amdg.make_buffer_desc %arg0, %n : !tt.ptr<i32> -> !amdg.buffer_desc<i32>
    // expected-error@+1 {{can't prove the soffset tensor is the same across the lanes of a wave}}
    %0 = amdg.buffer_atomic_load relaxed, gpu, %desc[%offsets] soffset = %s : tensor<64xi32, #blocked> : !amdg.buffer_desc<i32> -> tensor<64xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_atomic_store_lane_varying_soffset(%arg0: !tt.ptr<i32>, %offsets: tensor<64xi32, #blocked>, %s: tensor<64xi32, #blocked>, %v: tensor<64xi32, #blocked>, %n: i32) {
    %desc = amdg.make_buffer_desc %arg0, %n : !tt.ptr<i32> -> !amdg.buffer_desc<i32>
    // expected-error@+1 {{can't prove the soffset tensor is the same across the lanes of a wave}}
    amdg.buffer_atomic_store relaxed, gpu, %v, %desc[%offsets] soffset = %s : tensor<64xi32, #blocked> : !amdg.buffer_desc<i32> -> tensor<64xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_to_local_lane_varying_soffset(%arg0: !tt.ptr<f32>, %offsets: tensor<64xi32, #blocked>, %s: tensor<64xi32, #blocked>, %n: i32, %dst: !ttg.memdesc<64xf32, #shared, #smem, mutable>) {
    %desc = amdg.make_buffer_desc %arg0, %n : !tt.ptr<f32> -> !amdg.buffer_desc<f32>
    // expected-error@+2 {{can't prove the soffset tensor is the same across the lanes of a wave}}
    // expected-error@+1 {{failed to legalize operation 'amdg.buffer_load_to_local'}}
    %0 = amdg.buffer_load_to_local %desc[%offsets] soffset = %s : tensor<64xi32, #blocked> into %dst : !amdg.buffer_desc<f32>[tensor<64xi32, #blocked>] -> <64xf32, #shared, #smem, mutable>
    tt.return
  }
}
