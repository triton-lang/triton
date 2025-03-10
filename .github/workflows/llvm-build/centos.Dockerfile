FROM centos:7
ARG llvm_dir=llvm-project
# Add the cache artifacts and the LLVM source tree to the container
ADD sccache /sccache
ADD "${llvm_dir}" /source/llvm-project
ENV SCCACHE_DIR="/sccache"
ENV SCCACHE_CACHE_SIZE="2G"

RUN echo -e "[llvmtoolset-build]\nname=LLVM Toolset 13.0 - Build\nbaseurl=https://buildlogs.centos.org/c7-llvm-toolset-13.0.x86_64/\ngpgcheck=0\nenabled=1" > /etc/yum.repos.d/llvmtoolset-build.repo

# Note: This is required patch since CentOS have reached EOL
# otherwise any yum install setp will fail
RUN sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
RUN sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
RUN sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo

# Install build dependencies
RUN yum install --assumeyes centos-release-scl

# The definition of insanity is doing the same thing and expecting a different result
RUN sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
RUN sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
RUN sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo

RUN yum install --assumeyes --nogpgcheck llvm-toolset-13.0
RUN yum install --assumeyes rh-python38-python-devel rh-python38-python-pip
SHELL [ "/usr/bin/scl", "enable", "llvm-toolset-13.0", "rh-python38" ]

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade cmake ninja sccache

# Install MLIR's Python Dependencies
RUN python3 -m pip install -r /source/llvm-project/mlir/python/requirements.txt

# Configure, Build, Test, and Install LLVM
RUN cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_ASM_COMPILER=clang \
  -DCMAKE_C_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CXX_FLAGS="-Wno-everything" \
  -DCMAKE_LINKER=lld \
  -DCMAKE_INSTALL_PREFIX="/install" \
  -DLLVM_BUILD_UTILS=ON \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;lld" \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  /source/llvm-project/llvm

RUN ninja -C build install
