clear
bash scripts/amd/clean.sh
bash scripts/amd/deps.sh
bash scripts/amd/build.sh
bash scripts/amd/test.sh
# bash scripts/amd/debug.sh
bash scripts/amd/collect_rocm_kernels.sh
bash scripts/amd/check_llvm_src.sh