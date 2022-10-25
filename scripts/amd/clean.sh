set -x
rm -rf core
rm -rf ptx.hip
rm -rf python/build/
rm -rf python/test/__pycache__/
rm -rf python/triton.egg-info/
rm -rf python/triton/_C/libtriton.so
rm -rf python/triton/__pycache__/
rm -rf python/triton/ops/__pycache__/
rm -rf python/triton/ops/blocksparse/__pycache__/
rm -rf *.isa
rm -rf *.gcn
rm -rf *.ptx
rm -rf *.ll
rm -rf *.s
rm -rf *.o
rm -rf *.hsaco
rm -rf *.ttir
sh scripts/amd/delete_hip_files.sh
rm -rf triton_rocm_kernels
rm -rf /tmp/*.ll
rm -rf /tmp/*.gcn
rm -rf /tmp/*.hsaco
rm -rf /tmp/*.o
rm -rf /tmp/*.ttir
rm -rf /tmp/*.s
rm -rf build
# rm -rf /root/.triton