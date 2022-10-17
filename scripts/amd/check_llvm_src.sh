shopt -s extglob
/opt/rocm/llvm/bin/llc -mcpu=gfx908 triton_rocm_kernels/*+([0-9]).ll
# /opt/rocm/llvm/bin/llc -mcpu=gfx908 triton_rocm_kernels/*_before_verify.ll