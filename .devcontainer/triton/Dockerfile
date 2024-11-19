ARG CUSTOM_LLVM=false

FROM registry.access.redhat.com/ubi9/ubi:latest AS llvm-build
ARG CUSTOM_LLVM
USER 0
# Conditionally execute the build based on CUSTOM_LLVM
RUN if [ "$CUSTOM_LLVM" = "true" ]; then \
        dnf update -y && \
        dnf -y install clang rpm-build git ninja-build cmake lld && \
        git clone https://github.com/llvm/llvm-project && \
        cd llvm-project && \
        COMMIT=$(curl -s https://raw.githubusercontent.com/triton-lang/triton/refs/heads/main/cmake/llvm-hash.txt) &&\
        git checkout $COMMIT && \
        mkdir build && \
        cd build && \
        cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" && \
        ninja; \
    else \
        echo "Skipping LLVM build because CUSTOM_LLVM is not true"; \
    fi

FROM registry.access.redhat.com/ubi9/python-312 AS base
ARG USERNAME=1001
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG CUSTOM_LLVM
ARG INSTALL_CUDNN

USER 0
COPY user.sh user.sh
# Create the user
RUN ./user.sh -u $USERNAME -g $USER_GID
RUN dnf update -y && \
    dnf -y install clang cmake lld ninja-build;
# Set the user
USER $USERNAME

# Stage for llvm-local-true
FROM base AS llvm-local-true
COPY --from=llvm-build /llvm-project/ /llvm-project/

# Stage for llvm-local-false
FROM base AS llvm-local-false
ENV TRITON_OFFLINE_BUILD=NO

# Use intermediate stage selection
FROM llvm-local-${CUSTOM_LLVM}

ENV PYTHON_VERSION=3.12 \
    PATH=$HOME/.local/bin/:$PATH \
    PYTHONUNBUFFERED=1

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install ninja cmake wheel pybind11
RUN if [[ "$INSTALL_CUDNN" == "true" ]]; then \
	python3 -m pip install nvidia-cudnn-cu12; \
fi
RUN pip install pre-commit
RUN echo "export MAX_JOBS=$(nproc --all)" >> "${HOME}"/.bashrc
# Conditionally set the env vars based on CUSTOM_LLVM
RUN if [ "$CUSTOM_LLVM" = "true" ]; then \
    echo "export LLVM_BUILD_DIR=/llvm-project/build " >> "${HOME}/.bashrc" && \
    echo "export LLVM_INCLUDE_DIRS=/llvm-project/build /include" >> "${HOME}/.bashrc" && \
    echo "export LLVM_LIBRARY_DIR=/llvm-project/build /lib" >> "${HOME}/.bashrc" && \
    echo "export LLVM_SYSPATH=/llvm-project/build " >> "${HOME}/.bashrc"; \
fi
