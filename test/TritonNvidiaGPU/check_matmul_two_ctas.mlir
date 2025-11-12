// RUN: triton-opt %s -split-input-file --triton-nvidia-check-matmul-two-cta | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEST MATRIX: CTA Mode × Copy Count × MMA Count
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Total: 2 CTA modes × 3 copy counts × 3 MMA counts = 18
// Minus 1 (0ops case CTA-independent) = 17 unique tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test 1: 1 CTA, 0 Copy, 0 MMA (CTA mode irrelevant)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// CHECK: module attributes {{.*}}"ttng.two-ctas" = false
// CHECK-LABEL: test_1cta_0copy_0mma
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_1cta_0copy_0mma() {
    // No tcgen05 ops - defaults to false
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test 2: 1 CTA, 0 Copy, 1 MMA
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// CHECK: module attributes {{.*}}"ttng.two-ctas" = false
// CHECK-LABEL: test_1cta_0copy_1mma
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_1cta_0copy_1mma(%arg0: !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>,
                                        %arg1: !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>,
                                        %arg2: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>,
                                        %arg3: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) {
    %true = arith.constant true
    // CHECK: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %arg0, %arg1, %arg2, %true, %true, %arg3[%true] {is_async} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test 3: 1 CTA, 0 Copy, 2 MMA
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// CHECK: module attributes {{.*}}"ttng.two-ctas" = false
// CHECK-LABEL: test_1cta_0copy_2mma
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_1cta_0copy_2mma(%arg0: !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>,
                                        %arg1: !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>,
                                        %arg2: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>,
                                        %arg3: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) {
    %true = arith.constant true
    // CHECK: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %arg0, %arg1, %arg2, %true, %true, %arg3[%true] {is_async} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    // CHECK: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %arg0, %arg1, %arg2, %true, %true, %arg3[%true] {is_async} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test 4: 1 CTA, 1 Copy, 0 MMA
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// CHECK: module attributes {{.*}}"ttng.two-ctas" = false
// CHECK-LABEL: test_1cta_1copy_0mma
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_1cta_1copy_0mma(%smem: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>,
                                        %tmem: !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>) {
    // CHECK: ttng.tmem_copy
    ttng.tmem_copy %smem, %tmem : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test 5: 1 CTA, 1 Copy, 1 MMA
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// CHECK: module attributes {{.*}}"ttng.two-ctas" = false
// CHECK-LABEL: test_1cta_1copy_1mma
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_1cta_1copy_1mma(%arg0: !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>,
                                        %arg1: !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>,
                                        %arg2: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>,
                                        %arg3: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                                        %arg4: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>,
                                        %arg5: !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    // CHECK: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %arg0, %arg1, %arg2, %true, %true, %arg3[%true] {is_async} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    // CHECK: ttng.tmem_copy
    ttng.tmem_copy %arg4, %arg5 : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test 6: 1 CTA, 1 Copy, 2 MMA
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// CHECK: module attributes {{.*}}"ttng.two-ctas" = false
// CHECK-LABEL: test_1cta_1copy_2mma
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_1cta_1copy_2mma(%arg0: !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>,
                                        %arg1: !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>,
                                        %arg2: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>,
                                        %arg3: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                                        %arg4: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>,
                                        %arg5: !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    // CHECK: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %arg0, %arg1, %arg2, %true, %true, %arg3[%true] {is_async} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    // CHECK: ttng.tmem_copy
    ttng.tmem_copy %arg4, %arg5 : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %arg0, %arg1, %arg2, %true, %true, %arg3[%true] {is_async} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test 7: 1 CTA, 2 Copy, 0 MMA
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// CHECK: module attributes {{.*}}"ttng.two-ctas" = false
// CHECK-LABEL: test_1cta_2copy_0mma
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_1cta_2copy_0mma(%smem: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>,
                                        %tmem: !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>) {
    // CHECK: ttng.tmem_copy
    ttng.tmem_copy %smem, %tmem : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_copy
    ttng.tmem_copy %smem, %tmem : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test 8: 1 CTA, 2 Copy, 1 MMA
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// CHECK: module attributes {{.*}}"ttng.two-ctas" = false
// CHECK-LABEL: test_1cta_2copy_1mma
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_1cta_2copy_1mma(%arg0: !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>,
                                        %arg1: !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>,
                                        %arg2: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>,
                                        %arg3: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                                        %arg4: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>,
                                        %arg5: !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    // CHECK: ttng.tmem_copy
    ttng.tmem_copy %arg4, %arg5 : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %arg0, %arg1, %arg2, %true, %true, %arg3[%true] {is_async} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    // CHECK: ttng.tmem_copy
    ttng.tmem_copy %arg4, %arg5 : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test 9: 1 CTA, 2 Copy, 2 MMA
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// CHECK: module attributes {{.*}}"ttng.two-ctas" = false
// CHECK-LABEL: test_1cta_2copy_2mma
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_1cta_2copy_2mma(%arg0: !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>,
                                        %arg1: !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>,
                                        %arg2: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>,
                                        %arg3: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                                        %arg4: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>,
                                        %arg5: !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    // CHECK: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %arg0, %arg1, %arg2, %true, %true, %arg3[%true] {is_async} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    // CHECK: ttng.tmem_copy
    ttng.tmem_copy %arg4, %arg5 : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %arg0, %arg1, %arg2, %true, %true, %arg3[%true] {is_async} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    // CHECK: ttng.tmem_copy
    ttng.tmem_copy %arg4, %arg5 : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 2 CTA TESTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Note: 2 CTA MMA tests require complex TMEM layout setup with
// CTASplitM/N = [2,1], which is beyond simple LIT test scope.
// These are tested in E2E Python tests instead.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test 10: 2 CTA, 0 Copy, 1 MMA (Not testable - needs complex layout)
// Test 11: 2 CTA, 0 Copy, 2 MMA (Not testable - needs complex layout)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test 12: 2 CTA, 1 Copy, 0 MMA
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// CHECK: module attributes {{.*}}"ttng.two-ctas" = true
// CHECK-LABEL: test_2cta_1copy_0mma
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_2cta_1copy_0mma(%smem: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>,
                                        %tmem: !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>) {
    // CHECK: ttng.tmem_copy
    ttng.tmem_copy %smem, %tmem : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test 13: 2 CTA, 1 Copy, 1 MMA (Not testable - needs complex layout)
// Test 14: 2 CTA, 1 Copy, 2 MMA (Not testable - needs complex layout)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test 15: 2 CTA, 2 Copy, 0 MMA
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// CHECK: module attributes {{.*}}"ttng.two-ctas" = true
// CHECK-LABEL: test_2cta_2copy_0mma
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_2cta_2copy_0mma(%smem: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>,
                                        %tmem: !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>) {
    // CHECK: ttng.tmem_copy
    ttng.tmem_copy %smem, %tmem : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_copy
    ttng.tmem_copy %smem, %tmem : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test 16: 2 CTA, 2 Copy, 1 MMA (Not testable - needs complex layout)
// Test 17: 2 CTA, 2 Copy, 2 MMA (Not testable - needs complex layout)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SUMMARY: 17 Total Test Cases in Matrix
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tested in LIT: 11 cases (all 1 CTA + 2 CTA copy-only)
// Not testable: 6 cases (2 CTA with MMA - complex layout)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
