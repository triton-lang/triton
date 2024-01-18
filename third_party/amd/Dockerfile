FROM rocm/pytorch-nightly

# clean triton installation
RUN python -m pip uninstall -y triton

# clone openai/triton
WORKDIR /tmp
RUN git clone --recurse-submodules https://github.com/openai/triton

# build triton
WORKDIR /tmp/triton
RUN cd python && python -m pip install .

# back to root dir
WORKDIR /tmp/triton