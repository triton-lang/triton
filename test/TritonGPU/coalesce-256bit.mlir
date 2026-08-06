// RUN: triton-opt %s -split-input-file -tritongpu-coalesce='max-vec-bits=256' | FileCheck --check-prefix=VEC256 %s
// RUN: triton-opt %s -split-input-file -tritongpu-coalesce='max-vec-bits=128' | FileCheck --check-prefix=VEC128 %s

// Test that max-vec-bits=256 allows sizePerThread=8 for f32 loads with sufficient divisibility.
// With divisibility=32 and f32 (4 bytes): alignment = min(32/4, 1024) = 8
// With max-vec-bits=128: min(8, 128/32) = 4 -> sizePerThread=4
// With max-vec-bits=256: min(8, 256/32) = 8 -> sizePerThread=8

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

// VEC256: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// VEC128: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
tt.func public @coalesce_f32_load(%arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}, %arg1: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // VEC256: tt.load {{.*}} : tensor<1024x!tt.ptr<f32>, [[LAYOUT]]>
    // VEC128: tt.load {{.*}} : tensor<1024x!tt.ptr<f32>, [[LAYOUT]]>
    %7 = tt.load %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
}

}

// -----

// Test that max-vec-bits=256 allows sizePerThread=4 for f64 loads with sufficient divisibility.
// With divisibility=32 and f64 (8 bytes): alignment = min(32/8, 1024) = 4
// With max-vec-bits=128: min(4, 128/64) = 2 -> sizePerThread=2
// With max-vec-bits=256: min(4, 256/64) = 4 -> sizePerThread=4

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

// VEC256: [[LAYOUT64:#.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// VEC128: [[LAYOUT64:#.*]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
tt.func public @coalesce_f64_load(%arg0: !tt.ptr<f64> {tt.divisibility = 32 : i32}, %arg1: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked1>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked1>
    %5 = tt.splat %arg0 : !tt.ptr<f64> -> tensor<1024x!tt.ptr<f64>, #blocked1>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f64>, #blocked1>, tensor<1024xi32, #blocked1>
    // VEC256: tt.load {{.*}} : tensor<1024x!tt.ptr<f64>, [[LAYOUT64]]>
    // VEC128: tt.load {{.*}} : tensor<1024x!tt.ptr<f64>, [[LAYOUT64]]>
    %7 = tt.load %6 : tensor<1024x!tt.ptr<f64>, #blocked1>
    tt.return
}

}

// -----

// Test that max-vec-bits=256 also works for stores.
// With divisibility=32 and f32 (4 bytes): alignment = min(32/4, 1024) = 8
// With max-vec-bits=128: store cap min(8, 4) = 4 -> sizePerThread=4
// With max-vec-bits=256: store cap min(8, 8) = 8 -> sizePerThread=8

#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

// VEC256: [[SLAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// VEC128: [[SLAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
tt.func public @coalesce_f32_store(%arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}, %arg1: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %cst = arith.constant dense<1.0> : tensor<1024xf32, #blocked2>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked2>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked2>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked2>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked2>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>, #blocked2>, tensor<1024xi32, #blocked2>
    // VEC256: tt.store {{.*}} : tensor<1024x!tt.ptr<f32>, [[SLAYOUT]]>
    // VEC128: tt.store {{.*}} : tensor<1024x!tt.ptr<f32>, [[SLAYOUT]]>
    tt.store %6, %cst : tensor<1024x!tt.ptr<f32>, #blocked2>
    tt.return
}

}
