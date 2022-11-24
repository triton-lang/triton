// RUN: %PYTHON -m triton.tools.aot %s --target=amdgcn --gfx=906 | FileCheck %s

// == LLVM IR check begin ==
// CHECK-LABEL: {{^}}test_empty_kernel:
// CHECK-NEXT: s_endpgm

module attributes {"triton_gpu.num-warps" = 4 : i32} {

func @test_empty_kernel(%lb : index, %A : !tt.ptr<f16>) {

  return
}

}
