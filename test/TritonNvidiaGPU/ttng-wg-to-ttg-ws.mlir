// RUN: triton-opt %s -split-input-file --triton-nvidia-ttng-wg-to-ttg-ws | FileCheck %s

module attributes {nvws.mma = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 8 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4: i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @ttg_wg_2warps(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    ttng.warp_group start_warp(4) num_warps(8) reg_count(160) :  {{
      %1 = arith.constant 1
      ttng.warp_group_return
    } {barId = 1 : i32, groups = [@nvws.tma_load]}}
    ttng.warp_group start_warp(0) num_warps(4) reg_count(32) : {{
      %2 = arith.constant 2
      ttng.warp_group_return
    } {barId = 2 : i32, groups = [@nvws.mma]}}

    // CHECK: ttg.warp_specialize() attributes {barIds = dense<[2, 1]> : tensor<2xi32>, requestedRegisters = array<i32: 160>, warpGroupStartIds = array<i32: 4>}
    // CHECK-NEXT: default {
    // CHECK-NEXT:  %[[c2:.*]] = arith.constant 2
    // CHECK: partition0() num_warps(8) {
    // CHECK-NEXT:  %[[c1:.*]] = arith.constant 1

    tt.return
  }
}

// -----

module attributes {nvws.epilogue = {num_warps = 4 : i32, start_warp = 8 : i32}, nvws.mma = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 4 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @ttg_wg_2warps(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    ttng.warp_group start_warp(4) num_warps(8) reg_count(0) :  {{
      %1 = arith.constant 3
      ttng.warp_group_return
    } {barId = 1 : i32, groups = [@nvws.tma_load]}}
    ttng.warp_group start_warp(0) num_warps(4) reg_count(0) : {{
      %2 = arith.constant 4
      ttng.warp_group_return
    } {barId = 2 : i32, groups = [@nvws.mma]}}
    ttng.warp_group start_warp(12) num_warps(4) reg_count(0) : {{
      %3 = arith.constant 5
      ttng.warp_group_return
    } {barId = 3 : i32, groups = [@nvws.mma]}}

    // CHECK: ttg.warp_specialize() attributes {barIds = dense<[2, 1, 3]> : tensor<3xi32>, warpGroupStartIds = array<i32: 4, 12>}
    // CHECK-NEXT: default {
    // CHECK-NEXT:  %[[c2:.*]] = arith.constant 4
    // CHECK: partition0() num_warps(8) {
    // CHECK-NEXT:  %[[c1:.*]] = arith.constant 3
    // CHECK: partition1() num_warps(4) {
    // CHECK-NEXT:  %[[c1:.*]] = arith.constant 5

    tt.return
  }
}