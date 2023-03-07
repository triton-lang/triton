// RUN: %PYTHON -m triton.tools.aot %s --num_warps=1 --target=amdgcn --gfx=gfx906 --triple=amdgcn-amd-amdhsa --features="+sramecc,-xnack" | FileCheck %s
// RUN: %PYTHON -m triton.tools.aot %s --num_warps=1 --target=amdgcn --gfx=gfx908 --triple=amdgcn-amd-amdhsa --features="+sramecc,-xnack" | FileCheck %s
// RUN: %PYTHON -m triton.tools.aot %s --num_warps=1 --target=amdgcn --gfx=gfx90a --triple=amdgcn-amd-amdhsa --features="+sramecc,-xnack" | FileCheck %s

// == LLVM IR check begin ==
// CHECK-LABEL: {{^}}test_float16_load:
// CHECK: global_load_dword
// CHECK: global_load_dword
// CHECK: global_store_dword

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  func.func public @test_float16_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %1 = tt.splat %arg1 : (!tt.ptr<f16>) -> tensor<128x!tt.ptr<f16>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    %3 = tt.load %2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128xf16, #blocked>
    %4 = tt.splat %arg2 : (!tt.ptr<f16>) -> tensor<128x!tt.ptr<f16>, #blocked>
    %5 = tt.addptr %4, %0 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    %6 = tt.load %5 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128xf16, #blocked>
    %7 = arith.addf %3, %6 : tensor<128xf16, #blocked>
    %8 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<128x!tt.ptr<f16>, #blocked>
    %9 = tt.addptr %8, %0 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    tt.store %9, %7 {cache = 1 : i32, evict = 1 : i32} : tensor<128xf16, #blocked>
    return
  }
}
