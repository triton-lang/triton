[English](./README.md)

## FlagTree

FlagTree 是面向多种 AI 芯片的开源、统一编译器。FlagTree 致力于打造多元 AI 芯片编译器及相关工具平台，发展和壮大 Triton 上下游生态。项目当前处于初期，目标是兼容现有适配方案，统一代码仓库，快速实现单版本多后端支持。

## 从源代码安装
安装依赖（注意使用正确的 python3.x 执行）：
```shell
apt install zlib1g zlib1g-dev libxml2 libxml2-dev  # ubuntu
cd python; python3 -m pip install -r requirements.txt
```

编译安装，目前支持的后端 backendxxx 包括 iluvatar、xpu、mthreads、cambricon（有限支持）：
```shell
cd python
export FLAGTREE_BACKEND=backendxxx
python3 -m pip install . --no-build-isolation -v
```

## 构建技巧

自动下载依赖库的速度可能受限于网络环境，编译前可自行下载至缓存目录 ~/.flagtree（可通过环境变量 FLAGTREE_CACHE_DIR 修改），无需自行设置 LLVM_BUILD_DIR 等环境变量。
各后端完整编译命令如下：
```shell
# iluvatar
# 推荐使用镜像 Ubuntu 20.04
mkdir -p ~/.flagtree/iluvatar; cd ~/.flagtree/iluvatar
wget https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/iluvatar-llvm18-x86_64.tar.gz
wget https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/iluvatarTritonPlugin-cpython3.10-glibc2.30-glibcxx3.4.28-cxxabi1.3.12-ubuntu-x86_64.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=iluvatar
python3 -m pip install . --no-build-isolation -v
```
```shell
# xpu (klx)
# 推荐使用镜像（22GB）https://su.bcebos.com/klx-sdk-release-public/xpytorch/docker/ubuntu2004_v030/ubuntu_2004_x86_64_v30.tar
mkdir -p ~/.flagtree/xpu; cd ~/.flagtree/xpu
wget https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/XTDK-llvm18-ubuntu2004_x86_64.tar
wget https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/xre-Linux-x86_64.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=xpu
python3 -m pip install . --no-build-isolation -v
```
```shell
# mthreads
# 推荐使用镜像 flagtree/dockerfiles/Dockerfile-ubuntu22.04-python3.10-mthreads
mkdir -p ~/.flagtree/mthreads; cd ~/.flagtree/mthreads
wget https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/mthreads-llvm19-glibc2.34-glibcxx3.4.30-x64.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=mthreads
python3 -m pip install . --no-build-isolation -v
```

使用默认的编译命令，可以编译安装 nvidia、amd、triton_shared 后端：
```shell
# 自行下载 llvm
cd ${YOUR_LLVM_DOWNLOAD_DIR}
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-10dc3a8e-ubuntu-x64.tar.gz
tar -zxvf llvm-10dc3a8e-ubuntu-x64.tar.gz
# 编译安装
cd ${YOUR_CODE_DIR}/flagtree/python
export LLVM_BUILD_DIR=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-10dc3a8e-ubuntu-x64
export LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib
export LLVM_SYSPATH=$LLVM_BUILD_DIR
unset FLAGTREE_BACKEND
python3 -m pip install . --no-build-isolation -v
# 如果接下来需要编译安装其他后端，应清空 LLVM 相关环境变量
unset LLVM_BUILD_DIR LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR LLVM_SYSPATH
```

## 运行测试

安装完成后可以在后端目录下运行测试：
```shell
cd third_party/backendxxx/python/test
python3 -m pytest -s
```

## 关于贡献

欢迎参与 FlagTree 的开发并贡献代码，详情请参考[CONTRIBUTING.md](/CONTRIBUTING_cn.md)。

## 许可证

FlagTree 使用 [MIT license](/LICENSE)。
