set -e
cd python
pip uninstall -y triton
# export TRITON_USE_ROCM=ON

export TRITON_ROCM_DEBUG=ON
pip install --verbose -e .