import argparse
import sys

# matrix code version, mfma instruction name
configs = [(3, "mfma_f32_32x32x16_fp8_fp8"),
           (3, "mfma_f32_32x32x16_fp8_bf8"),
           (3, "mfma_f32_32x32x16_bf8_fp8"),
           (3, "mfma_f32_32x32x16_bf8_bf8"),
           (2, "mfma_f32_32x32x8f16"),
           (1, "mfma_f32_32x32x4bf16"),
           (2, "mfma_f32_32x32x8bf16_1k"),
           (2, "mfma_f32_32x32x2f32"),
           (2, "mfma_i32_32x32x8i8"),
           (3, "mfma_i32_32x32x16_i8"),
           (3, "mfma_f32_16x16x32_fp8_fp8"),
           (3, "mfma_f32_16x16x32_fp8_bf8"),
           (3, "mfma_f32_16x16x32_bf8_fp8"),
           (3, "mfma_f32_16x16x32_bf8_bf8"),
           (2, "mfma_f32_16x16x16f16"),
           (1, "mfma_f32_16x16x8bf16"),
           (2, "mfma_f32_16x16x16bf16_1k"),
           (2, "mfma_f32_16x16x4f32"),
           (2, "mfma_i32_16x16x16i8"),
           (3, "mfma_i32_16x16x32_i8"),
           (2, "mfma_f32_4x4x4f16"),
           (1, "mfma_f32_4x4x2bf16"),
           (2, "mfma_f32_4x4x4bf16_1k"),
           (2, "mfma_f32_4x4x1f32"),
           (2, "mfma_i32_4x4x4i8")]

def generate(output_file):
    print(f'// This file is generated: $ python3 {" ".join(sys.argv)}', file=output_file)
    print('// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm="target=rocdl" 2>/dev/null | FileCheck --check-prefixes=CHECK,GCN %s', file=output_file)
    
    for cfg_id in range(len(configs)):
        matrix_core_version = configs[cfg_id][0]
        cfg = configs[cfg_id][1]
        parts = cfg.split("_")
        if parts[-1] == "1k":
            parts = parts[:-1]
        shape = parts[2].split("x")
        for ty in ["bf8", "f8", "bf16", "f16", "f32", "i8"]:
            if ty in shape[-1]:
              shape[-1] = shape[-1][:-len(ty)]
              parts += [ty]
              break
        for i in range(len(shape)):
            shape[i] = int(shape[i])
# shape
        non_k_dim = shape[0]
        k_width = shape[2]
        if non_k_dim == 32:
            k_width //= 2
        if non_k_dim == 16:
            k_width //= 4
# types
        b_ty = parts[-1]
        if b_ty in ["fp8", "bf8"]:
            a_ty = parts[-2]
        else:
            a_ty = b_ty
        c_ty = parts[1]
    
        mlir_type_names = {
            "fp8": "f8E4M3FNUZ",
            "bf8": "f8E5M2FNUZ",
            "f16": "f16",
            "bf16": "bf16",
            "f32": "f32",
            "i8": "i8",
            "i32": "i32"}
        a_ty = mlir_type_names[a_ty]
        b_ty = mlir_type_names[b_ty]
        c_ty = mlir_type_names[c_ty]
    
# misc
        if "i" in c_ty:
            cst_val = "0"
        else:
            cst_val = "0.000000e+00"
    
# repeats
        if non_k_dim == 32:
            M = 128
            N = 32
            K = 256
        if non_k_dim == 16:
            M = 128
            N = 32
            K = 256
        if non_k_dim == 4:
            M = 128
            N = 32
            K = 256
    
        num_subgroups = 1
        if non_k_dim == 4:
            num_subgroups = 16
        num_reps = (M // non_k_dim) * (N // non_k_dim) * (K // (shape[2] * num_subgroups))
    
# mlir operation name
        cfg.split
        mlir_op_name = "rocdl." + cfg.replace("_", ".")
        case_text = f'''
!a_ty = {a_ty}
!b_ty = {b_ty}
!c_ty = {c_ty}
#k_width = {k_width}
#non_k_dim = {non_k_dim}
#mfmaVersion = {matrix_core_version}
#mfma = #triton_gpu.mfma<{{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}}>
#dot_operand_a = #triton_gpu.dot_op<{{opIdx=0, parent=#mfma, kWidth = #k_width}}>
#dot_operand_b = #triton_gpu.dot_op<{{opIdx=1, parent=#mfma, kWidth = #k_width}}>
module attributes {{"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32}} {{
  // CHECK-LABEL: convert_dot_{cfg}
  tt.func @convert_dot_{cfg}(%a: tensor<{M}x{K}x!a_ty, #dot_operand_a>, %b: tensor<{K}x{N}x!b_ty, #dot_operand_b>) {{
    %cst_c = arith.constant dense<{cst_val}> : tensor<{M}x{N}x!c_ty, #mfma>
    // GCN-COUNT-{num_reps}: {mlir_op_name}
    %D = tt.dot %a, %b, %cst_c {{allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false}} : tensor<{M}x{K}x!a_ty, #dot_operand_a> * tensor<{K}x{N}x!b_ty, #dot_operand_b> -> tensor<{M}x{N}x!c_ty, #mfma>
    tt.return
  }}
}}

'''
        if cfg_id == len(configs) - 1:
            print(case_text, end="", file=output_file)
        else:
            print(case_text, end="// -----\n", file=output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()
    with open(args.output_file, "w") as f:
        generate(output_file=f)
