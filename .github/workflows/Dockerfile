FROM centos:7
ARG llvm_dir=llvm-project

# Add the cache artifacts and the LLVM source tree to the container
ADD sccache /sccache
ADD "${llvm_dir}" /source/llvm-project
ENV SCCACHE_DIR="/sccache"
ENV SCCACHE_CACHE_SIZE="2G"

RUN echo -e "[llvmtoolset-build]\nname=LLVM Toolset 14.0 - Build\nbaseurl=https://buildlogs.centos.org/c7-llvm-toolset-14.0.x86_64/\ngpgcheck=0\nenabled=1" > /etc/yum.repos.d/llvmtoolset-build.repo
RUN echo -e "[llvmtoolset-build]\nname=Dev Toolset 12.0 - Build\nbaseurl=https://buildlogs.centos.org/c7-devtoolset-12.x86_64/devtoolset-12-gcc/\ngpgcheck=0\nenabled=1" > /etc/yum.repos.d/dev-toolset-12-build.repo
RUN yum install --nogpgcheck devtoolset-12-gcc-c++ llvm-toolset-14.0-clang-tools-extra llvm-toolset-14.0-clang
# Install build dependencies
RUN yum install --assumeyes centos-release-scl
RUN yum install --assumeyes rh-python38-python-devel rh-python38-python-pip
SHELL [ "/usr/bin/scl", "enable", "llvm-toolset-14.0", "rh-python38" ]

RUN rm /usr/bin/python
RUN ln -s /opt/rh/rh-python38/root/usr/bin/python3.8 /usr/bin/python3
RUN ln -s /opt/rh/rh-python38/root/usr/bin/pip3.8 /usr/bin/pip3
RUN ln -s /opt/rh/rh-python38/root/usr/bin/python3.8 /usr/bin/python
RUN ln -s /opt/rh/rh-python38/root/usr/bin/pip3.8 /usr/bin/pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade cmake ninja sccache

# Install MLIR's Python Dependencies
RUN python3 -m pip install -r /source/llvm-project/mlir/python/requirements.txt
RUN python3 -m pip install --upgrade numpy

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
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  /source/llvm-project/llvm

RUN ninja -C build check-mlir install
