
## Uncomment to enable specific LLVM build
# FROM registry.access.redhat.com/ubi9/ubi:latest as build
# USER 0
# RUN dnf update -y
# RUN dnf -y install clang rpm-build git ninja-build cmake
# RUN git clone https://github.com/llvm/llvm-project
# WORKDIR /llvm-project
## UPDATE COMMIT SHA if required.
# RUN git checkout 49af6502
# RUN mkdir build
# WORKDIR /llvm-project/build
# RUN cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"
# RUN ninja

FROM registry.access.redhat.com/ubi9/python-312

ARG USERNAME=1001
ARG INSTALL_CUDNN=false
ARG USER_UID=1000
ARG USER_GID=$USER_UID

USER 0
COPY user.sh user.sh
# Create the user
RUN ./user.sh -u $USERNAME -g $USER_GID
# Set the user
USER $USERNAME

ENV PYTHON_VERSION=3.12 \
    PATH=$HOME/.local/bin/:$PATH \
    PYTHONUNBUFFERED=1

## Uncomment to enable specific LLVM build
# COPY --from=build /llvm-project/build /llvm-project/build
# ENV LLVM_BUILD_DIR=/llvm-project/build \
#     LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
#     LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
#     LLVM_SYSPATH=$LLVM_BUILD_DIR

# install dependencies
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install ninja cmake wheel pybind11;
RUN if [[ "$INSTALL_CUDNN" == "true" ]]; then \
	python3 -m pip install nvidia-cudnn-cu12; \
fi;
RUN pip install pre-commit
RUN echo "export MAX_JOBS=$(nproc --all)" >> "${HOME}"/.bashrc
