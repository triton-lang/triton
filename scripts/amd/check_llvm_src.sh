shopt -s extglob
${ROCM_PATH}/llvm/bin/llc -mcpu=gfx908 triton_rocm_kernels/*+([0-9]).ll
# ${ROCM_PATH}/llvm/bin/llc -mcpu=gfx908 triton_rocm_kernels/*_before_verify.ll
