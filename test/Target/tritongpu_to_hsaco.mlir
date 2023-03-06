// RUN: export ROCM_PATH=/opt/rocm
// RUN: HSACO_PATH=$(%PYTHON -m triton.tools.aot %s --target=amdgcn --gfx=gfx906 --triple=amdgcn-amd-amdhsa --features="+sramecc,-xnack" | head -n 1)
// RUN: llvm-readobj -a "${HSACO_PATH}" | FileCheck %s

// CHECK: Format: elf64-amdgpu
// TODO: Arch: unknown
// CHECK: AddressSize: 64bit
// CHECK: ElfHeader {
// CHECK-NEXT:   Ident {
// CHECK-NEXT:    Magic: (7F 45 4C 46)
// CHECK-NEXT:    Class: 64-bit (0x2)
// CHECK-NEXT:    DataEncoding: LittleEndian (0x1)
// CHECK-NEXT:    FileVersion: 1
// CHECK-NEXT:    OS/ABI: AMDGPU_HSA (0x40)
// CHECK-NEXT:    ABIVersion: 2
// CHECK-NEXT:    Unused: (00 00 00 00 00 00 00)
// CHECK-NEXT:  }
// CHECK-NEXT:  Type: SharedObject (0x3)
// CHECK-NEXT:  Machine: EM_AMDGPU (0xE0)
// CHECK-NEXT:  Version: 1

// CHECK: Name: test_empty_kernel
// CHECK: Size: 4
// CHECK: Binding: Global
// CHECK: Type: Function
// CHECK: Section: .text

// CHECK: Type: NT_AMDGPU_METADATA (AMDGPU Metadata)
// CHECK:     .group_segment_fixed_size: 0
// CHECK-NEXT:     .kernarg_segment_align: 8
// CHECK-NEXT:     .kernarg_segment_size: 16
// CHECK-NEXT:     .max_flat_workgroup_size: 1024
// CHECK-NEXT:     .name:           test_empty_kernel
// CHECK-NEXT:     .private_segment_fixed_size: 0

module attributes {"triton_gpu.num-warps" = 4 : i32} {

func.func @test_empty_kernel(%lb : index, %A : !tt.ptr<f16>) {

  return
}

}
