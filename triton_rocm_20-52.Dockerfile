FROM rocm/pytorch:rocm5.2.3_ubuntu20.04_py3.7_pytorch_1.12.1

# build triton
RUN export TRITON_USE_ROCM=ON MI_GPU_ARCH=gfx90a

# Unit Tests 
# to run unit tests
# 1. build this Dockerfile
#    docker build --build-arg -f triton_rocm_20-52.Dockerfile -t triton_rocm52 .
# 2. run docker container
#    docker run -it --rm --network=host --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --name triton --ipc=host --shm-size 16G --device=/dev/kfd --device=/dev/dri triton_rocm52:latest
# 3. run core unit tests on a rocm machine
#    cd ~/triton/python
#    pytest --verbose test/unit/operators/test_matmul.py | tee test_core.log