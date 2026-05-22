// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 \
// RUN:   | FileCheck %s

// Tests for the dataflow uniformity analysis driving AMD soffset split.
//
// Each buffer_load lowers to 4 llvm.mul ops (byte-offset per lane).
// When the splitter fires, it adds one more mul per lane for soffset:
//   SPLIT    => 8 llvm.mul
//   NO-SPLIT => 4 llvm.mul


// (1) Uniform loop IV: phi from uniform init + uniform update.
//     SPLIT (8 muls).
#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: llvm.func @load_uniform_loop_iv_split
    tt.func @load_uniform_loop_iv_split(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                        %base_offset : i32, %N : i32) {
        %c1 = arith.constant 1 : i32
        %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        cf.br ^body(%base_offset : i32)
    ^body(%iv: i32):
        %nxt = arith.addi %iv, %c1 : i32
        %done = arith.cmpi sge, %iv, %N : i32
        cf.cond_br %done, ^exit(%iv : i32), ^body(%nxt : i32)
    ^exit(%final: i32):
        %base = tt.splat %final : i32 -> tensor<128xi32, #blocked0>
        %offset = arith.addi %base, %range : tensor<128xi32, #blocked0>
        // CHECK-COUNT-8: llvm.mul
        // CHECK-NOT: llvm.mul
        %ret = amdg.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
    }
}

// -----

// (2) Phi with one uniform and one divergent incoming. NO-SPLIT (4 muls).
#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: llvm.func @load_phi_with_divergent_incoming_no_split
    tt.func @load_phi_with_divergent_incoming_no_split(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                       %base_offset : i32, %N : i32) {
        %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %lane = rocdl.workitem.id.x : i32
        %c0 = arith.constant 0 : i32
        %cond = arith.cmpi sge, %N, %c0 : i32
        cf.cond_br %cond, ^join(%base_offset : i32), ^join(%lane : i32)
    ^join(%mixed_scalar: i32):
        %mixed = tt.splat %mixed_scalar : i32 -> tensor<128xi32, #blocked0>
        %offset = arith.addi %mixed, %range : tensor<128xi32, #blocked0>
        // CHECK-COUNT-4: llvm.mul
        // CHECK-NOT: llvm.mul
        %ret = amdg.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
    }
}

// -----

// (3) Select with uniform condition and uniform arms. SPLIT (8 muls).
#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: llvm.func @load_select_uniform_cond_uniform_arms_split
    tt.func @load_select_uniform_cond_uniform_arms_split(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                         %u0 : i32, %u1 : i32, %N : i32) {
        %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %c0 = arith.constant 0 : i32
        %cond = arith.cmpi sge, %N, %c0 : i32
        %chosen = arith.select %cond, %u0, %u1 : i32
        %base = tt.splat %chosen : i32 -> tensor<128xi32, #blocked0>
        %offset = arith.addi %base, %range : tensor<128xi32, #blocked0>
        // CHECK-COUNT-8: llvm.mul
        // CHECK-NOT: llvm.mul
        %ret = amdg.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
    }
}

// -----

// (4) Select with divergent condition (threadIdx). NO-SPLIT (4 muls).
#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: llvm.func @load_select_divergent_cond_no_split
    tt.func @load_select_divergent_cond_no_split(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                 %u0 : i32, %u1 : i32) {
        %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %lane = rocdl.workitem.id.x : i32
        %c4 = arith.constant 4 : i32
        %cond = arith.cmpi slt, %lane, %c4 : i32
        %chosen = arith.select %cond, %u0, %u1 : i32
        %base = tt.splat %chosen : i32 -> tensor<128xi32, #blocked0>
        %offset = arith.addi %base, %range : tensor<128xi32, #blocked0>
        // CHECK-COUNT-4: llvm.mul
        // CHECK-NOT: llvm.mul
        %ret = amdg.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
    }
}

// -----

// (5) ExtractValue from insertvalue chain (splat+make_range lowering).
//     Transfer function chases the chain. SPLIT (8 muls).
#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: llvm.func @load_extractvalue_of_insertvalue_chain
    tt.func @load_extractvalue_of_insertvalue_chain(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                    %base_offset : i32, %N : i32) {
        %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %base = tt.splat %base_offset : i32 -> tensor<128xi32, #blocked0>
        %offset = arith.addi %base, %range : tensor<128xi32, #blocked0>
        %n = tt.splat %N: i32 -> tensor<128xi32, #blocked0>
        %mask = arith.cmpi slt, %range, %n: tensor<128xi32, #blocked0>
        // CHECK-COUNT-8: llvm.mul
        // CHECK-NOT: llvm.mul
        %ret = amdg.buffer_load %arg0[%offset], %mask : tensor<128xf32, #blocked0>
        tt.return
    }
}

// -----

// (6) SOUNDNESS: phi at join with two uniform-but-different incomings
//     under a divergent cond_br. Phi is per-lane divergent.
//     NO-SPLIT (4 muls).
#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: llvm.func @load_cond_br_divergent_condition_no_split
    tt.func @load_cond_br_divergent_condition_no_split(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                       %u0 : i32, %u1 : i32) {
        %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %lane = rocdl.workitem.id.x : i32
        %c4 = arith.constant 4 : i32
        %cond = arith.cmpi slt, %lane, %c4 : i32
        cf.cond_br %cond, ^join(%u0 : i32), ^join(%u1 : i32)
    ^join(%mixed_scalar: i32):
        %mixed = tt.splat %mixed_scalar : i32 -> tensor<128xi32, #blocked0>
        %offset = arith.addi %mixed, %range : tensor<128xi32, #blocked0>
        // CHECK-COUNT-4: llvm.mul
        // CHECK-NOT: llvm.mul
        %ret = amdg.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
    }
}

// -----

// (7) arith.ori -> llvm.or WITHOUT the disjoint flag, so the splitter
//     can't treat it as additive. NO-SPLIT (4 muls).
//     See test (11) for the same pattern at the LLVM level.
#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: llvm.func @load_or_disjoint_split
    tt.func @load_or_disjoint_split(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                    %base_offset : i32) {
        %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %c8 = arith.constant 8 : i32
        %shifted = arith.shli %base_offset, %c8 : i32
        %base = tt.splat %shifted : i32 -> tensor<128xi32, #blocked0>
        %offset = arith.ori %base, %range : tensor<128xi32, #blocked0>
        // CHECK-COUNT-4: llvm.mul
        // CHECK-NOT: llvm.mul
        %ret = amdg.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
    }
}

// -----

// (8) Subtraction not decomposed. NO-SPLIT (4 muls).
#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: llvm.func @load_sub_uniform_minus_perlane_no_split
    tt.func @load_sub_uniform_minus_perlane_no_split(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                     %base_offset : i32) {
        %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %base = tt.splat %base_offset : i32 -> tensor<128xi32, #blocked0>
        %offset = arith.subi %base, %range : tensor<128xi32, #blocked0>
        // CHECK-COUNT-4: llvm.mul
        // CHECK-NOT: llvm.mul
        %ret = amdg.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
    }
}

// -----

// (9) Kernel arg splatted to offset. SPLIT (8 muls).
#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: llvm.func @load_kernel_args_uniform
    tt.func @load_kernel_args_uniform(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                      %base_offset : i32) {
        %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %base = tt.splat %base_offset : i32 -> tensor<128xi32, #blocked0>
        %offset = arith.addi %base, %range : tensor<128xi32, #blocked0>
        // CHECK-COUNT-8: llvm.mul
        // CHECK-NOT: llvm.mul
        %ret = amdg.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
    }
}

// -----

// (10) Private (non-kernel) function args not seeded uniform.
//      NO-SPLIT (4 muls).
#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: @load_non_kernel_func_no_split
    tt.func private @load_non_kernel_func_no_split(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                   %base_offset : i32) attributes {noinline} {
        %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %base = tt.splat %base_offset : i32 -> tensor<128xi32, #blocked0>
        %offset = arith.addi %base, %range : tensor<128xi32, #blocked0>
        // CHECK-COUNT-4: llvm.mul
        // CHECK-NOT: llvm.mul
        %ret = amdg.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
    }
    // Public caller keeps the private callee alive through lowering.
    tt.func public @_drive_non_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                      %base_offset : i32) {
        tt.call @load_non_kernel_func_no_split(%arg0, %base_offset) : (!tt.ptr<f32>, i32) -> ()
        tt.return
    }
}

// -----

// (11) Same as (7) but documents that arith.ori -> llvm.or still lacks
//      the disjoint flag. NO-SPLIT (4 muls).
#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: llvm.func @load_or_disjoint_llvm_split
    tt.func @load_or_disjoint_llvm_split(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                         %base_offset : i32) {
        %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        // Shift left 8 bits so base doesn't overlap [0,128).
        %c8 = arith.constant 8 : i32
        %shifted = arith.shli %base_offset, %c8 : i32
        %base = tt.splat %shifted : i32 -> tensor<128xi32, #blocked0>
        // arith.ori -> llvm.or without disjoint, so no split.
        %offset = arith.ori %base, %range : tensor<128xi32, #blocked0>
        // CHECK-COUNT-4: llvm.mul
        // CHECK-NOT: llvm.mul
        %ret = amdg.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
    }
}

// -----

// (12) Three uniform kernel args summed + per-lane range.
//      All three go into soffset. SPLIT (8 muls).
#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: llvm.func @load_multi_uniform_leaf_split
    tt.func @load_multi_uniform_leaf_split(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                           %u0 : i32, %u1 : i32, %u2 : i32) {
        %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        // Build uniform sum: u0 + u1 + u2
        %sum01 = arith.addi %u0, %u1 : i32
        %sum012 = arith.addi %sum01, %u2 : i32
        %base = tt.splat %sum012 : i32 -> tensor<128xi32, #blocked0>
        %offset = arith.addi %base, %range : tensor<128xi32, #blocked0>
        // CHECK-COUNT-8: llvm.mul
        // CHECK-NOT: llvm.mul
        %ret = amdg.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
    }
}
