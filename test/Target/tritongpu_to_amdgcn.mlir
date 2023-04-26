// RUN: %PYTHON -m triton.tools.aot %s --target=amdgcn --gfx=gfx906 --triple=amdgcn-amd-amdhsa --features="+sramecc,-xnack" | FileCheck %s

// == LLVM IR check begin ==
// CHECK-LABEL: {{^}}test_empty_kernel:
// CHECK-NEXT: s_endpgm

module attributes {"triton_gpu.num-warps" = 4 : i32} {

tt.func @test_empty_kernel(%lb : index, %A : !tt.ptr<f16>) {

  tt.return
}

}
