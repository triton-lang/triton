// RUN: triton-opt %s -split-input-file -verify-diagnostics -tritongpu-gather-to-local-gather -cse | FileCheck %s --implicit-check-not=ttg.local_alloc

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @warp_local_gather
  // CHECK-NOT: ttg.local_alloc
  // CHECK: tt.gather
  // CHECK-NEXT: tt.return
  tt.func @warp_local_gather(%src: tensor<32xf32, #blocked>, %idx: tensor<32xi32, #blocked>) -> tensor<32xf32, #blocked> {
    %result = tt.gather %src[%idx] {axis = 0 : i32} : (tensor<32xf32, #blocked>, tensor<32xi32, #blocked>) -> tensor<32xf32, #blocked>
    tt.return %result : tensor<32xf32, #blocked>
  }

  // CHECK-LABEL: @shared_source_gathers
  // CHECK: %[[BUF:.*]] = ttg.local_alloc %arg0
  // CHECK-SAME: !ttg.memdesc<256xf32, #{{.*}}, #{{.*}}>
  // CHECK-NEXT: %[[FIRST:.*]] = ttg.local_gather %[[BUF]][%arg1] {axis = 0 : i32}
  // CHECK-NEXT: %[[SECOND:.*]] = ttg.local_gather %[[BUF]][%arg2] {axis = 0 : i32}
  // CHECK-NEXT: tt.return %[[FIRST]], %[[SECOND]]
  tt.func @shared_source_gathers(%src: tensor<256xf32, #blocked>, %idx0: tensor<64xi16, #blocked>, %idx1: tensor<128xi64, #blocked>) -> (tensor<64xf32, #blocked>, tensor<128xf32, #blocked>) {
    %first = tt.gather %src[%idx0] {axis = 0 : i32} : (tensor<256xf32, #blocked>, tensor<64xi16, #blocked>) -> tensor<64xf32, #blocked>
    %second = tt.gather %src[%idx1] {axis = 0 : i32} : (tensor<256xf32, #blocked>, tensor<128xi64, #blocked>) -> tensor<128xf32, #blocked>
    tt.return %first, %second : tensor<64xf32, #blocked>, tensor<128xf32, #blocked>
  }

  // CHECK-LABEL: @boolean_gather
  // CHECK: %[[BYTES:.*]] = arith.extui %arg0 : tensor<256xi1, #{{.*}}> to tensor<256xi8, #{{.*}}>
  // CHECK-NEXT: %[[BUF:.*]] = ttg.local_alloc %[[BYTES]]
  // CHECK-SAME: !ttg.memdesc<256xi8, #{{.*}}, #{{.*}}>
  // CHECK-NEXT: %[[RESULT:.*]] = ttg.local_gather %[[BUF]][%arg1]
  // CHECK-NEXT: %[[BOOL:.*]] = arith.trunci %[[RESULT]] : tensor<64xi8, #{{.*}}> to tensor<64xi1, #{{.*}}>
  // CHECK-NEXT: tt.return %[[BOOL]]
  tt.func @boolean_gather(%src: tensor<256xi1, #blocked>, %idx: tensor<64xi32, #blocked>) -> tensor<64xi1, #blocked> {
    %result = tt.gather %src[%idx] {axis = 0 : i32} : (tensor<256xi1, #blocked>, tensor<64xi32, #blocked>) -> tensor<64xi1, #blocked>
    tt.return %result : tensor<64xi1, #blocked>
  }

  // CHECK-LABEL: @pointer_gather
  // CHECK: %[[BUF:.*]] = ttg.local_alloc %arg0
  // CHECK-SAME: !ttg.memdesc<256x!tt.ptr<f32>, #{{.*}}, #{{.*}}>
  // CHECK-NEXT: %[[RESULT:.*]] = ttg.local_gather %[[BUF]][%arg1]
  // CHECK-NEXT: tt.return %[[RESULT]]
  tt.func @pointer_gather(%src: tensor<256x!tt.ptr<f32>, #blocked>, %idx: tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<f32>, #blocked> {
    %result = tt.gather %src[%idx] {axis = 0 : i32} : (tensor<256x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<f32>, #blocked>
    tt.return %result : tensor<64x!tt.ptr<f32>, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @gather_second_axis
  // CHECK: %[[BUF:.*]] = ttg.local_alloc
  // CHECK-SAME: !ttg.memdesc<32x64xf32, #{{.*}}, #{{.*}}>
  // CHECK-NEXT: ttg.local_gather %[[BUF]][%arg1] {axis = 1 : i32}
  // CHECK-SAME: -> tensor<32x8xf32, #{{.*}}>
  tt.func @gather_second_axis(%src: tensor<32x64xf32, #blocked>, %idx: tensor<32x8xi64, #blocked>) -> tensor<32x8xf32, #blocked> {
    %result = tt.gather %src[%idx] {axis = 1 : i32} : (tensor<32x64xf32, #blocked>, tensor<32x8xi64, #blocked>) -> tensor<32x8xf32, #blocked>
    tt.return %result : tensor<32x8xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[1]]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @cross_cta_gather(%src: tensor<256xi32, #blocked>, %idx: tensor<256xi32, #blocked>) -> tensor<256xi32, #blocked> {
    // expected-error @+1 {{cross-CTA gathers are not supported}}
    %result = tt.gather %src[%idx] {axis = 0 : i32} : (tensor<256xi32, #blocked>, tensor<256xi32, #blocked>) -> tensor<256xi32, #blocked>
    tt.return %result : tensor<256xi32, #blocked>
  }
}

// -----

#rows = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
#cols = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 1]]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @nonindexed_cross_cta_gather(%src: tensor<32x64xf32, #rows>, %idx: tensor<32x64xi32, #cols>) -> tensor<32x64xf32, #cols> {
    // expected-error @+1 {{cross-CTA gathers are not supported}}
    %result = tt.gather %src[%idx] {axis = 1 : i32} : (tensor<32x64xf32, #rows>, tensor<32x64xi32, #cols>) -> tensor<32x64xf32, #cols>
    tt.return %result : tensor<32x64xf32, #cols>
  }
}

// -----

#rows = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @cta_local_gather_in_cluster
  // CHECK: %[[BUF:.*]] = ttg.local_alloc %arg0
  // CHECK-NEXT: ttg.local_gather %[[BUF]][%arg1]
  tt.func @cta_local_gather_in_cluster(%src: tensor<32x64xf32, #rows>, %idx: tensor<32x64xi32, #rows>) -> tensor<32x64xf32, #rows> {
    %result = tt.gather %src[%idx] {axis = 1 : i32} : (tensor<32x64xf32, #rows>, tensor<32x64xi32, #rows>) -> tensor<32x64xf32, #rows>
    tt.return %result : tensor<32x64xf32, #rows>
  }
}

// -----

#broadcast = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[0]]}>
#split = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[1]]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @replicated_source_gather
  // CHECK: %[[BUF:.*]] = ttg.local_alloc %arg0
  // CHECK-NEXT: ttg.local_gather %[[BUF]][%arg1]
  tt.func @replicated_source_gather(%src: tensor<256xi32, #broadcast>, %idx: tensor<512xi32, #split>) -> tensor<512xi32, #split> {
    %result = tt.gather %src[%idx] {axis = 0 : i32} : (tensor<256xi32, #broadcast>, tensor<512xi32, #split>) -> tensor<512xi32, #split>
    tt.return %result : tensor<512xi32, #split>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1], instrShape = [8, 8]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @gather_dot_operand_source
  // CHECK: %[[BUF:.*]] = ttg.local_alloc %arg0
  // CHECK-NEXT: ttg.local_gather %[[BUF]][%arg1]
  tt.func @gather_dot_operand_source(%src: tensor<8x4xf32, #dot>, %idx: tensor<16x4xi32, #blocked>) -> tensor<16x4xf32, #blocked> {
    %result = tt.gather %src[%idx] {axis = 0 : i32} : (tensor<8x4xf32, #dot>, tensor<16x4xi32, #blocked>) -> tensor<16x4xf32, #blocked>
    tt.return %result : tensor<16x4xf32, #blocked>
  }
}
