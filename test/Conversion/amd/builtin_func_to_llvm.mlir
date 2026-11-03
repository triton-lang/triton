// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942 ftz=true" --convert-builtin-func-to-llvm="ftz=true" | FileCheck %s --check-prefix=LLVM_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950 ftz=true" --convert-builtin-func-to-llvm="ftz=true" | FileCheck %s --check-prefix=LLVM_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942 ftz=false" --convert-builtin-func-to-llvm="ftz=false" | FileCheck %s --check-prefix=LLVM_NO_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1250 ftz=true" --convert-builtin-func-to-llvm="gfx-arch=gfx1250 ftz=true" | FileCheck %s --check-prefix=CDNA5
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1250 ftz=false" --convert-builtin-func-to-llvm="gfx-arch=gfx1250 ftz=false" | FileCheck %s --check-prefix=CDNA5
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950 ftz=false" --convert-builtin-func-to-llvm="ftz=false" | FileCheck %s --check-prefix=LLVM_NO_FTZ

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fast_expf(%arg0: tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked> {
    // CHECK-LABEL: test_fast_expf
    // LLVM_FTZ: rocdl.exp2
    // LLVM_NO_FTZ: llvm.intr.exp2
    %0 = tt.extern_elementwise %arg0 {libname = "libdevice", libpath = "", pure = true, symbol = "__triton_hip_fast_expf"} : (tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return %0 : tensor<64xf32, #blocked>
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fast_tanhf(%arg0: tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked> {
    // CHECK-LABEL: test_fast_tanhf
    // LLVM_FTZ: rocdl.exp2
    // LLVM_NO_FTZ: llvm.intr.exp2
    %0 = tt.extern_elementwise %arg0 {libname = "libdevice", libpath = "", pure = true, symbol = "__triton_hip_fast_tanhf"} : (tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return %0 : tensor<64xf32, #blocked>
  }
}

#blocked32 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_fast_tanhf_cdna5(%arg0: tensor<64xf32, #blocked32>) -> tensor<64xf32, #blocked32> {
    // CHECK-LABEL: test_fast_tanhf_cdna5
    // CDNA5: llvm.call_intrinsic "llvm.amdgcn.tanh.f32"
    // CDNA5-NOT: rocdl.exp2
    %0 = tt.extern_elementwise %arg0 {libname = "libdevice", libpath = "", pure = true, symbol = "__triton_hip_fast_tanhf"} : (tensor<64xf32, #blocked32>) -> tensor<64xf32, #blocked32>
    tt.return %0 : tensor<64xf32, #blocked32>
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_tanh_cdna5(%arg0: tensor<64xf32, #blocked32>) -> tensor<64xf32, #blocked32> {
    // CHECK-LABEL: test_tanh_cdna5
    // CDNA5: llvm.call @__ocml_tanh_f32
    // CDNA5-NOT: llvm.amdgcn.tanh
    %0 = tt.extern_elementwise %arg0 {libname = "libdevice", libpath = "", pure = true, symbol = "__ocml_tanh_f32"} : (tensor<64xf32, #blocked32>) -> tensor<64xf32, #blocked32>
    tt.return %0 : tensor<64xf32, #blocked32>
  }
}
