// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx1150 | FileCheck %s

// Verify that RDNA3 (and 3.5) cache modifier qualifiers emit the correct cachepolicy
// aux values in rocdl.raw.ptr.buffer.load/store operations.
//
// DLC (bit 2) = non-temporal hint for MALL. DLC=1 means skip MALL allocation.
//
//   Load  .cg -> 1   (GLC only: bypass GL1)
//   Load  .cs -> 7   (GLC|SLC|DLC: non-temporal everywhere)
//   Load  .cv -> 7   (GLC|SLC|DLC: non-temporal everywhere)
//   Store .cs -> 7   (GLC|SLC|DLC: non-temporal everywhere)
//   Store .wt -> 7   (GLC|SLC|DLC: non-temporal everywhere)

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_cg
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_load_cg(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    // .cg load on RDNA3.5: aux = 1 (GLC)
    // CHECK: %[[aux:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, %[[aux]]
    %ret = amdg.buffer_load %arg0[%offset] cacheModifier = cg : tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_cs
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_load_cs(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    // .cs load on RDNA3.5: aux = 7 (GLC|SLC|DLC)
    // CHECK: %[[aux:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, %[[aux]]
    %ret = amdg.buffer_load %arg0[%offset] cacheModifier = cs : tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_load_cv
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_load_cv(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    // .cv load on RDNA3.5: aux = 7 (GLC|SLC|DLC)
    // CHECK: %[[aux:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, %[[aux]]
    %ret = amdg.buffer_load %arg0[%offset] cacheModifier = cv : tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_store_cs
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_store_cs(%value: tensor<128xf32, #blocked>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    // .cs store on RDNA3.5: aux = 7 (GLC|SLC|DLC)
    // CHECK: %[[aux:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[aux]]
    amdg.buffer_store %value, %arg0[%offset] cacheModifier = cs stride = %c256_i32 : tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: buffer_store_wt
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @buffer_store_wt(%value: tensor<128xf32, #blocked>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<128xi32, #blocked> {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    // .wt store on RDNA3.5: aux = 7 (GLC|SLC|DLC)
    // CHECK: %[[aux:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[aux]]
    amdg.buffer_store %value, %arg0[%offset] cacheModifier = wt stride = %c256_i32 : tensor<128xf32, #blocked>
    tt.return
  }
}
