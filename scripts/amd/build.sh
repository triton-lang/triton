set -x

cd python
pip uninstall -y triton

export MLIR_ENABLE_DUMP=1
export LLVM_IR_ENABLE_DUMP=1
export AMDGCN_ENABLE_DUMP=1

export TRITON_USE_ROCM=ON
# export MI_GPU_ARCH=gfx90a # not needed

pip install --verbose -e .
pip install -U matplotlib pandas filelock
