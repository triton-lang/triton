// RUN: triton-opt %s -split-input-file --triton-nvidia-check-matmul-two-cta -verify-diagnostics

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// NEGATIVE TEST MATRIX: CTA Mode Inconsistencies
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests verify that CheckMatmulTwoCTAs pass catches
// inconsistent CTA modes between tcgen05 operations.
//
// Test Categories (2 ops each):
// 1. 2 CTA Copy vs. 1 CTA MMA (testable)
// 2. 1 CTA Copy vs. 2 CTA MMA (not testable - complex layout)
// 3. 1 CTA MMA vs. 2 CTA MMA   (not testable - complex layout)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Category 1: 2 CTA Copy vs. 1 CTA MMA (TESTABLE)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Copy infers 2 CTA mode from num_ctas=2
// MMA uses 1 CTA mode (no two_ctas attribute)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Test 1a: MMA (1 CTA) first, then Copy (2 CTA)
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_1cta_mma_then_2cta_copy(%arg0: !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>,
                                                %arg1: !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>,
                                                %arg2: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>,
                                                %arg3: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                                                %arg4: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>,
                                                %arg5: !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    // First op: MMA without two_ctas = 1 CTA mode
    // expected-note @+1 {{but first tcgen05 op uses 1 CTA mode}}
    ttng.tc_gen5_mma %arg0, %arg1, %arg2, %true, %true, %arg3[%true] {is_async} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    
    // Second op: Copy infers 2 CTA from num_ctas=2 - inconsistent!
    // expected-error @+1 {{inconsistent CTA mode between tcgen05 operations; this op uses 2 CTA mode}}
    ttng.tmem_copy %arg4, %arg5 : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// Test 1b: Copy (2 CTA) first, then MMA (1 CTA) - reverse order
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_2cta_copy_then_1cta_mma(%arg0: !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>,
                                                %arg1: !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>,
                                                %arg2: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>,
                                                %arg3: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                                                %arg4: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>,
                                                %arg5: !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    // First op: Copy infers 2 CTA from num_ctas=2
    // expected-note @+1 {{but first tcgen05 op uses 2 CTA mode}}
    ttng.tmem_copy %arg4, %arg5 : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    
    // Second op: MMA without two_ctas = 1 CTA mode - inconsistent!
    // expected-error @+1 {{inconsistent CTA mode between tcgen05 operations; this op uses 1 CTA mode}}
    ttng.tc_gen5_mma %arg0, %arg1, %arg2, %true, %true, %arg3[%true] {is_async} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Category 2: 1 CTA Copy vs. 2 CTA MMA (NOT TESTABLE)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Reason: 2 CTA MMA operations require TMEM layouts with
//         CTASplitM/N = [2,1] parameters, which is beyond
//         the scope of simple LIT tests. These scenarios
//         are validated through E2E Python tests.
//
// Would test:
// - Copy infers 1 CTA mode from num_ctas=1
// - MMA with two_ctas attribute = 2 CTA mode
// - Expected: Error on inconsistency
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// -----

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Category 3: 1 CTA MMA vs. 2 CTA MMA (NOT TESTABLE)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Reason: 2 CTA MMA operations require TMEM layouts with
//         CTASplitM/N = [2,1] parameters, which is beyond
//         the scope of simple LIT tests. These scenarios
//         are validated through E2E Python tests.
//
// Would test:
// - First MMA without two_ctas = 1 CTA mode
// - Second MMA with two_ctas attribute = 2 CTA mode
// - Expected: Error on inconsistency
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SUMMARY: 3 Test Categories, 2 Testable Cases
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Category 1: 2 CTA Copy vs. 1 CTA MMA
//   - Test 1a: MMA first, Copy second  ✅ TESTABLE
//   - Test 1b: Copy first, MMA second  ✅ TESTABLE
//
// Category 2: 1 CTA Copy vs. 2 CTA MMA
//   - Not testable (requires complex MMA layout)
//
// Category 3: 1 CTA MMA vs. 2 CTA MMA
//   - Not testable (requires complex MMA layout)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
