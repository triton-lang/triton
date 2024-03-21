import argparse
import sys

# M N K a_ty b_ty c_ty
configs = [[32, 32, 32, "f16", "f16", "f32"],
           [32, 32, 32, "bf16", "bf16", "f32"],
           [32, 32, 32, "f32", "f32", "f32"],
           [32, 32, 32, "i8", "i8", "i32"],
           [32, 32, 32, "f8E4M3FNUZ", "f8E4M3FNUZ", "f32"],
           [32, 32, 32, "f8E4M3FNUZ", "f8E5M2FNUZ", "f32"],
           [32, 32, 32, "f8E5M2FNUZ", "f8E4M3FNUZ", "f32"],
           [32, 32, 32, "f8E5M2FNUZ", "f8E5M2FNUZ", "f32"],

           [16, 16, 32, "f16", "f16", "f32"],
           [16, 16, 32, "bf16", "bf16", "f32"],
           [16, 16, 32, "f32", "f32", "f32"],
           [16, 16, 32, "i8", "i8", "i32"],
           [16, 16, 32, "f8E4M3FNUZ", "f8E4M3FNUZ", "f32"],
           [16, 16, 32, "f8E4M3FNUZ", "f8E5M2FNUZ", "f32"],
           [16, 16, 32, "f8E5M2FNUZ", "f8E4M3FNUZ", "f32"],
           [16, 16, 32, "f8E5M2FNUZ", "f8E5M2FNUZ", "f32"],

           [4, 4, 64, "f16", "f16", "f32"],
           [4, 4, 64, "bf16", "bf16", "f32"],
           [4, 4, 64, "f32", "f32", "f32"],
           [4, 4, 64, "i8", "i8", "i32"],
           [4, 4, 64, "f8E4M3FNUZ", "f8E4M3FNUZ", "f32"],
           [4, 4, 64, "f8E4M3FNUZ", "f8E5M2FNUZ", "f32"],
           [4, 4, 64, "f8E5M2FNUZ", "f8E4M3FNUZ", "f32"],
           [4, 4, 64, "f8E5M2FNUZ", "f8E5M2FNUZ", "f32"],

           [64, 4, 4, "f16", "f16", "f32"],
           [64, 4, 4, "bf16", "bf16", "f32"],
           [64, 4, 4, "f32", "f32", "f32"],
           [64, 4, 4, "i8", "i8", "i32"],
           [64, 4, 4, "f8E4M3FNUZ", "f8E4M3FNUZ", "f32"],
           [64, 4, 4, "f8E4M3FNUZ", "f8E5M2FNUZ", "f32"],
           [64, 4, 4, "f8E5M2FNUZ", "f8E4M3FNUZ", "f32"],
           [64, 4, 4, "f8E5M2FNUZ", "f8E5M2FNUZ", "f32"],

           [4, 64, 4, "f16", "f16", "f32"],
           [4, 64, 4, "bf16", "bf16", "f32"],
           [4, 64, 4, "f32", "f32", "f32"],
           [4, 64, 4, "i8", "i8", "i32"],
           [4, 64, 4, "f8E4M3FNUZ", "f8E4M3FNUZ", "f32"],
           [4, 64, 4, "f8E4M3FNUZ", "f8E5M2FNUZ", "f32"],
           [4, 64, 4, "f8E5M2FNUZ", "f8E4M3FNUZ", "f32"],
           [4, 64, 4, "f8E5M2FNUZ", "f8E5M2FNUZ", "f32"]
          ]

def generate(cdna_version, output_file):
    arch_names = {0:"", 1: "gfx908", 2: "gfx90a", 3: "gfx940"}
    arch_name = arch_names[cdna_version]
    print(f"// This file is generated: $ python3 {' '.join(sys.argv)}", file=output_file)
    print(f"// RUN: (! triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul=arch-generation-name={arch_name} --mlir-pass-pipeline-crash-reproducer=%t 2>/dev/null) | FileCheck --check-prefixes=CHECK %s", file=output_file)

    for cfg_id in range(len(configs)):
        cfg = configs[cfg_id]

        cfg_name = "_".join([str(item) for item in cfg])

        M, N, K, a_ty, b_ty, c_ty = cfg
        if "i" in c_ty:
            cst_val = "0"
        else:
            cst_val = "0.000000e+00"
        
        supported = True
        if cdna_version < 3 and ("f8" in a_ty or "f8" in b_ty):
            supported = False

        if M >= 32 and N >= 32:
            m_dim = 32
            n_dim = 32
        elif M >= 16 and N >= 16:
            m_dim = 16
            n_dim = 16
        elif M >= 64 and N < 16:
            m_dim = 64
            n_dim = 4
        elif M < 16 and N >= 64:
            m_dim = 4
            n_dim = 64
        elif M < 16 and N < 16:
            m_dim = 4
            n_dim = 4
        if ("f8" in a_ty or "f8" in b_ty) and min(m_dim, n_dim) == 4:
            supported = False

        if cdna_version == 1:
            if a_ty == "f16":
                k_width = 4
            if a_ty == "bf16":
                k_width = 2
            if a_ty == "i8":
                k_width = 4
            if a_ty == "f32":
                k_width = 1
        if cdna_version == 2:
            if a_ty == "f16":
                k_width = 4
            if a_ty == "bf16":
                k_width = 4
            if a_ty == "i8":
                k_width = 4
            if a_ty == "f32":
                k_width = 1
        if cdna_version == 3:
            if "f8" in a_ty:
                k_width = 8
            if a_ty == "f16":
                k_width = 4
            if a_ty == "bf16":
                k_width = 4
            if a_ty == "i8":
                if min(m_dim, n_dim) == 4:
                    k_width = 4
                else:
                    k_width = 8
            if a_ty == "f32":
                k_width = 1

        if supported:
            mfma_check = f"// CHECK: #mfma = #triton_gpu.mfma<{{versionMajor = {cdna_version}, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [{m_dim}, {n_dim}], isTransposed = false}}>"
            label_check = f"// CHECK: convert_dot_{cfg_name}"
            checks =f"""// CHECK: triton_gpu.convert_layout {{{{.*}}}} : (tensor<{{{{.*}}}}, #blocked>) -> tensor<{{{{.*}}}}, #mfma>
// CHECK: triton_gpu.convert_layout {{{{.*}}}} : (tensor<{{{{.*}}}}, #triton_gpu.dot_op<{{opIdx = 0, parent = #blocked}}>>) -> tensor<{{{{.*}}}}, #triton_gpu.dot_op<{{opIdx = 0, parent = #mfma, kWidth = {k_width}}}>>
// CHECK: triton_gpu.convert_layout {{{{.*}}}} : (tensor<{{{{.*}}}}, #triton_gpu.dot_op<{{opIdx = 1, parent = #blocked}}>>) -> tensor<{{{{.*}}}}, #triton_gpu.dot_op<{{opIdx = 1, parent = #mfma, kWidth = {k_width}}}>>"""
        else:
            mfma_check = ""
            label_check = f"// CHECK-NOT: convert_dot_{cfg_name}"
            checks = ""

        case_text = f'''
!a_ty = {a_ty}
!b_ty = {b_ty}
!c_ty = {c_ty}
#blocked = #triton_gpu.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}}>
#dot_operand_a = #triton_gpu.dot_op<{{opIdx=0, parent=#blocked}}>
#dot_operand_b = #triton_gpu.dot_op<{{opIdx=1, parent=#blocked}}>
module attributes {{"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32}} {{
{mfma_check}
{label_check}
    tt.func @convert_dot_{cfg_name}(%a: tensor<{M}x{K}x!a_ty, #dot_operand_a>, %b: tensor<{K}x{N}x!b_ty, #dot_operand_b>) -> tensor<{M}x{N}x!c_ty, #blocked> {{
        %cst_c = arith.constant dense<{cst_val}> : tensor<{M}x{N}x!c_ty, #blocked>
{checks}
        %D = tt.dot %a, %b, %cst_c {{allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false}} : tensor<{M}x{K}x!a_ty, #dot_operand_a> * tensor<{K}x{N}x!b_ty, #dot_operand_b> -> tensor<{M}x{N}x!c_ty, #blocked>
        tt.return %D: tensor<{M}x{N}x!c_ty, #blocked>
    }}
}}

'''
        if cfg_id == len(configs) - 1:
            print(case_text, end="", file=output_file)
        else:
            print(case_text, end="// -----\n", file=output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cdna_version", type=int)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()
    with open(args.output_file, "w") as f:
        generate(cdna_version=args.cdna_version, output_file=f)
