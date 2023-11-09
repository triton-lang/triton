cd /home/nhat/github/triton/python
export TRITON_SHARED_OPT_PATH="$(pwd)/build/$(ls $(pwd)/build | grep -i cmake)/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
export LLVM_BINARY_DIR="${HOME}/.triton/llvm/$(ls ${HOME}/.triton/llvm/ | grep -i llvm)/bin"
python3 ../third_party/triton_shared/python/examples/reduce.py
