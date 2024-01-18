// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=target=rocdl | FileCheck --check-prefixes=CHECK,GCN %s

// Check load instruction doesn't generate incorrect bitcast.
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: @test_float16_bitcast
  tt.func public @test_float16_bitcast(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %true = arith.constant true
    %0 = tt.get_program_id x : i32
    %1 = tt.addptr %arg0, %0 : !tt.ptr<f16>, i32
    %2 = tt.load %1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f16
    // GCN-NOT: llvm.bitcast {{.*}} : {{.*}}32{{.*}}16
    // GCN-NOT: llvm.bitcast {{.*}} : {{.*}}16{{.*}}32
    // GCN: llvm.addrspacecast {{.*}} : !llvm.ptr<f16, 1> to !llvm.ptr<vector<1xf16>
    // GCN: llvm.load {{.*}} : !llvm.ptr<vector<1xf16>
    // GCN: llvm.bitcast {{.*}} : f16 to f16
    // GCN: llvm.bitcast {{.*}} : vector<1xf16> to i16
    // GCN: llvm.store {{.*}} : !llvm.ptr<f16, 1>
    tt.store %1, %2 : f16
    tt.return
  }
}
