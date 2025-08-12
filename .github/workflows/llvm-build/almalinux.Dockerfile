FROM almalinux:8
ARG llvm_dir=llvm-project
# Add the cache artifacts and the LLVM source tree to the container
ADD sccache /sccache
ADD "${llvm_dir}" /source/llvm-project
ENV SCCACHE_DIR="/sccache"
ENV SCCACHE_CACHE_SIZE="2G"

RUN dnf install --assumeyes llvm-toolset
RUN dnf install --assumeyes python38-pip python38-devel git

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade cmake ninja sccache lit

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
