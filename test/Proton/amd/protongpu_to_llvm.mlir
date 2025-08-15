// RUN: triton-opt %s -split-input-file -convert-proton-amd-gpu-to-llvm="arch=gfx942" --verify-diagnostics | FileCheck %s --check-prefix=CHECK
// RUN: triton-opt %s -split-input-file -convert-proton-amd-gpu-to-llvm="arch=gfx942" --convert-builtin-func-to-llvm --verify-diagnostics | FileCheck -allow-unused-prefixes --check-prefix=CONVERT-BUILTIN %s

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, ttg.profile_scratch_memory_alignment = 128 : i32, ttg.profile_scratch_memory_size = 384 : i32} {
  // CHECK-LABEL: convert_smem_finalize
  // CHECK: llvm.inline_asm asm_dialect = att operand_attrs = [] "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 3)", "=s"  : () -> i32
  // CHECK: llvm.inline_asm asm_dialect = att operand_attrs = [] "s_getreg_b32 $0, hwreg(HW_REG_HW_ID, 8, 4)", "=s"  : () -> i32
  // CHECK: llvm.inline_asm asm_dialect = att operand_attrs = [] "s_getreg_b32 $0, hwreg(HW_REG_HW_ID, 13, 3)", "=s"  : () -> i32
  // CONVERT-BUILTIN: llvm.cond_br %{{.*}}, ^bb1, ^bb3
  // CONVERT-BUILTIN: ^bb1:
  // CONVERT-BUILTIN: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<1>
  // CONVERT-BUILTIN: llvm.br ^bb2(%{{.*}} : i32)
  // CONVERT-BUILTIN: ^bb2(%{{.*}}: i32):
  // CONVERT-BUILTIN: llvm.load %{{.*}} : !llvm.ptr<3> -> i32
  // CONVERT-BUILTIN: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<1>
  // CONVERT-BUILTIN: llvm.load %{{.*}} : !llvm.ptr<3> -> i32
  // CONVERT-BUILTIN: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<1>
  // CONVERT-BUILTIN: llvm.cond_br %{{.*}}, ^bb2(%{{.*}} : i32), ^bb3
  // CONVERT-BUILTIN: ^bb3:
  // CHECK: llvm.return
  llvm.func @convert_smem_finalize(%arg: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<512xi32, #shared, #smem, mutable>
    %1 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32, offset = 0 : i32} : !tt.ptr<i32>
    %2 = proton_gpu.segment_alloc %0 : !ttg.memdesc<512xi32, #shared, #smem, mutable> -> !proton_gpu.segment<2048, #smem, warp>
    proton_gpu.finalize %2, %1 : !proton_gpu.segment<2048, #smem, warp>, !tt.ptr<i32>
    llvm.return
  }
}

// -----
