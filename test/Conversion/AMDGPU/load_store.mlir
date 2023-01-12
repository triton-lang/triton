// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm | FileCheck --check-prefixes=CHECK,GCN %s

// Check load instructin doesn't generate incorrect bitcast.
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: @test_float16_bitcast
  func public @test_float16_bitcast(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %true = arith.constant true
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = tt.addptr %arg0, %0 : !tt.ptr<f16>, i32
    %2 = tt.load %1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f16
    // GCN: llvm.bitcast {{.*}} : i16 to f16
    tt.store %1, %2 : f16
    // GCN: llvm.bitcast {{.*}} : i16 to f16
    return
  }
}
