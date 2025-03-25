[中文版](./README_cn.md)

## FlagTree

Flagtree is a multi-backend Triton compiler project dedicated to developing a diverse ecosystem of AI chip compilers and related tooling platforms, thereby fostering and strengthening the upstream and downstream Triton ecosystem. Currently in its initial phase, the project aims to maintain compatibility with existing adaptation solutions while unifying the codebase to rapidly implement single-version multi-backend support.

## Install from source
Installation dependencies (ensure you use the correct python3.x version):
```shell
apt install zlib1g zlib1g-dev libxml2 libxml2-dev  # ubuntu
cd python; python3 -m pip install -r requirements.txt
```

Compile and install. Currently supported backends (backendxxx) include iluvatar, xpu, mthreads, and cambricon (limited support):
```shell
cd python
export FLAGTREE_BACKEND=backendxxx
python3 -m pip install . --no-build-isolation -v
```

## Tips for building

Automatic dependency library downloads may be limited by network conditions. You can manually download to the cache directory ~/.flagtree (modifiable via the FLAGTREE_CACHE_DIR environment variable). No need to manually set LLVM environment variables such as LLVM_BUILD_DIR.
Complete build commands for each backend:
```shell
# iluvatar
# Recommended: Use Ubuntu 20.04
mkdir -p ~/.flagtree/iluvatar; cd ~/.flagtree/iluvatar
wget https://github.com/Galaxy1458/build_tools/releases/download/v1.0.0-build-deps/iluvatar-llvm18-x86_64.tar.gz
wget https://github.com/Galaxy1458/build_tools/releases/download/v1.0.0-build-deps/iluvatarTritonPlugin-cpython3.10-glibc2.30-glibcxx3.4.28-cxxabi1.3.12-ubuntu-x86_64.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=iluvatar
python3 -m pip install . --no-build-isolation -v
```
```shell
# kunlunxin（xpu）
# Recommended: Use the Docker image (22GB) https://su.bcebos.com/klx-sdk-release-public/xpytorch/docker/ubuntu2004_v030/ubuntu_2004_x86_64_v30.tar
mkdir -p ~/.flagtree/xpu; cd ~/.flagtree/xpu
wget https://github.com/Galaxy1458/build_tools/releases/download/v1.0.0-build-deps/XTDK-llvm18-ubuntu2004_x86_64.tar
wget https://github.com/Galaxy1458/build_tools/releases/download/v1.0.0-build-deps/xre-Linux-x86_64.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=xpu
python3 -m pip install . --no-build-isolation -v
```
```shell
# mthreads
# Recommended: Use the Dockerfile flagtree/dockerfiles/Dockerfile-ubuntu22.04-python3.10-mthreads
mkdir -p ~/.flagtree/mthreads; cd ~/.flagtree/mthreads
wget https://github.com/Galaxy1458/build_tools/releases/download/v1.0.0-build-deps/mthreads-llvm19-glibc2.34-glibcxx3.4.30-x64.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=mthreads
python3 -m pip install . --no-build-isolation -v
```

To build with default backends (nvidia, amd, triton_shared):
```shell
# manually download LLVM
cd ${YOUR_LLVM_DOWNLOAD_DIR}
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-10dc3a8e-ubuntu-x64.tar.gz
tar -zxvf llvm-10dc3a8e-ubuntu-x64.tar.gz
# build
cd ${YOUR_CODE_DIR}/flagtree/python
export LLVM_BUILD_DIR=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-10dc3a8e-ubuntu-x64
export LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib
export LLVM_SYSPATH=$LLVM_BUILD_DIR
unset FLAGTREE_BACKEND
python3 -m pip install . --no-build-isolation -v
# If you need to build other backends afterward, you should clear LLVM-related environment variables
unset LLVM_BUILD_DIR LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR LLVM_SYSPATH
```

## Running tests

After installation, you can run tests in the backend directory:
```shell
cd third_party/backendxxx/python/test
python3 -m pytest -s
```

## Contributing

Contributions to FlagTree development are welcome. Please refer to [CONTRIBUTING.md](/CONTRIBUTING_cn.md) for details.

## License

FlagTree is licensed under the [MIT license](/LICENSE).
