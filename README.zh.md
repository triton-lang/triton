| **`官方技术文档`** | **`每日构建 Wheels`** |
|-------------------- | -------------------- |
| [![Documentation](https://github.com/triton-lang/triton/actions/workflows/documentation.yml/badge.svg)](https://triton-lang.org/) | [![Wheels](https://github.com/triton-lang/triton/actions/workflows/wheels.yml/badge.svg)](https://github.com/triton-lang/triton/actions/workflows/wheels.yml) |

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

# Triton 开发者大会 2025 (Triton Conference 2025)

![Triton Banner](https://github.com/user-attachments/assets/b4b6972a-857c-417f-bf2c-f16f38a358c0)

第三届 Triton 开发者大会于 2025年10月21日 在加州山景城微软硅谷园区圆满举行。

### 大会资料

大会录播视频与演讲课件已全面上线：

- **大会视频回顾：** [YouTube 播放列表](https://www.youtube.com/playlist?list=PLc_vA1r0qoiQqCdWFDUDqI90oY5EjfGuO)
- **演讲幻灯片：** [Google Drive 资料夹](https://drive.google.com/drive/folders/1KB6tD3UM1J0_eUp-F-JSlGrargLBawIr)

历届大会资料归档：
- [2024 年开发者大会资料](docs/meetups/dev_conference_2024.md)
- [2023 年开发者大会资料](docs/meetups/dev-meetup-2023.md)

---

# Triton

这是 **Triton** 的官方开发代码仓库。Triton 是一套专为编写超高性能自定义深度学习算子（Primitives）而设计的编程语言与编译器。Triton 的核心愿景是提供一个开源开发环境，让开发者能够以显著高于传统 CUDA 的生产力编写极致性能的代码，同时具备比现有专用领域语言（DSL）更高的灵活性与可表达性。

本项目的核心理论基础详见 MAPL 2019 学术论文：[Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)。如果您在科研或工业生产中使用了 Triton，请引用该论文！

[官方技术文档](https://triton-lang.org) 提供了详尽的安装步骤与编程教程。此外，推荐尝试社区优秀的 [Triton Puzzles 谜题关卡](https://github.com/srush/Triton-Puzzles)——所有挑战均可直接在 Triton 解释器中运行，无需物理 GPU 即可上手练习！

---

# 🚀 快速安装 (Quick Installation)

您可以通过 pip 直接安装 Triton 最新的官方稳定版本：

```shell
pip install triton
```

官方预编译二进制 Wheel 包支持 CPython 3.10 至 3.14。

---

# 🛠️ 源码编译安装 (Install from source)

```shell
git clone https://github.com/triton-lang/triton.git
cd triton

pip install -r python/requirements.txt # 安装编译期依赖
pip install -e .
```

在 Python 虚拟环境中构建：

```shell
git clone https://github.com/triton-lang/triton.git
cd triton

python -m venv .venv --prompt triton
source .venv/bin/activate

pip install -r python/requirements.txt # 安装编译期依赖
pip install -e .
```

---

# ⚙️ 搭配自定义 LLVM 构建

Triton 使用 LLVM 为 GPU 和 CPU 生成底层机器代码。通常情况下，Triton 编译系统会自动下载官方预编译好的 LLVM 工具链；您也可以选择从源码自编译并集成 LLVM。

由于 LLVM 不具备长期稳定的 C++ API，Triton 只能兼容特定修订版本的 LLVM。

为方便开发者，您可直接使用以下单条命令自动编译匹配的 LLVM 并安装 Triton：

```shell
make dev-install-llvm
```

<details>
<summary>
或者，按照以下步骤手动自编译 LLVM：
</summary>

1. 查看当前 Triton 所依赖的精确 LLVM 版本。检查 `cmake/llvm-info.json` 中的 `llvm_hash` 字段。例如：
   `"llvm_hash": "49af6502c6dcb4a7f7520178bd14df396f78240c"`
   这表示当前 Triton 需基于 [LLVM](https://github.com/llvm/llvm-project) 的 `49af6502` 提交进行编译。

2. 检出对应版本的 LLVM 源码（可根据需要添加自研补丁）：
   ```shell
   cd $HOME/llvm-project
   git checkout 49af6502
   ```

3. 编译构建 LLVM：
   ```shell
   cd $HOME/llvm-project
   mkdir build && cd build
   cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld;clang" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"
   ninja
   ```

4. 耐心等待编译完成（视机器配置可能耗时较长）。

5. 配置环境变量并安装 Triton：
   ```shell
   export LLVM_BUILD_DIR=$HOME/llvm-project/build

   cd <triton-repo-dir>
   LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include    LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib    LLVM_SYSPATH=$LLVM_BUILD_DIR    pip install -e .
   ```

</details>

---

# 💡 编译加速与调试技巧 (Tips for building)

- 设置环境变量 `TRITON_BUILD_WITH_CLANG_LLD=true` 以使用 Clang 和 LLD。LLD 链接器能够极大缩短编译时间。
- 设置环境变量 `TRITON_BUILD_WITH_CCACHE=true` 启用 ccache 编译缓存。
- 设置 `TRITON_HOME=/some/path` 自定义 `.triton` 缓存与下载目录路径（默认为用户主目录）。
- 若编译期间内存耗尽（OOM），可在执行 `pip install -e .` 时指定环境变量 `MAX_JOBS=N` 限制并行编译线程数。
- 构建时向 `pip install` 传入 `--no-build-isolation` 可实现极快的增量空构建（nop build）。
- 编译系统会在仓库根目录下生成 `compile_commands.json` 文件，用于 VSCode IntelliSense 与 clangd 获得高品质 C++ 代码补全与符号跳转支持。

---

# 🧪 运行单元测试 (Running tests)

推荐测试执行流程：

```shell
# 一次性环境初始化配置（注意：这会重装本地 Triton，因为 torch 会覆盖公共版本）
make dev-install

# 运行全量测试（需物理 GPU 环境）
make test

# 在无 GPU 环境下运行通用测试
make test-nogpu
```

---

# 🔍 调试与深度开发指南 (Tips for hacking)

如需针对 Triton 前端（Frontend）进行断点调试，请参阅官方 [调试教程](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)。以下为 Triton 后端编译器开发的高级技巧。

### 核心调优与调试环境变量

完整配置清单请查阅 [`python/triton/knobs.py`](python/triton/knobs.py)。您可以在 Python 代码中动态设置，或直接通过环境变量传参：

- `MLIR_ENABLE_DUMP=1`：在每个 MLIR Pass 执行前将 IR 转储输出。可使用 `MLIR_ENABLE_DUMP=kernelName` 仅针对特定算子进行转储。
  - Triton 编译缓存可能会影响 Dump。如未生效，可清空缓存：`rm -r ~/.triton/cache/*`。
- `MLIR_DUMP_PATH`：指定 MLIR 转储的目标目录路径，未指定时默认输出至 stderr。
- `LLVM_IR_ENABLE_DUMP=1`：在每个 LLVM Pass 执行前转储 LLVM IR。
- `TRITON_REPRODUCER_PATH=<reproducer_path>`：在每个 MLIR 编译阶段前生成 Reproducer 复现文件。若任一 Pass 崩溃，将直接保留崩溃前的现场 IR。
- `TRITON_INTERPRET=1`：启用纯 Python 解释器模式执行 Triton 算子，可直接在算子代码中打 Python 断点调试，无需 GPU！
- `TRITON_ENABLE_LLVM_DEBUG=1`：向 LLVM 传递 `-debug` 参数，输出详细的底层编译日志。
- `TRITON_LLVM_DEBUG_ONLY=<comma-separated>`：等效于 LLVM 的 `-debug-only` 参数，仅输出指定 Pass（如 `"tritongpu-remove-layout-conversions,regalloc"`）的调试信息。
- `TRITON_ENABLE_ASAN=1`：启用 LLVM 地址检查器（AddressSanitizer）检测内存越界与泄漏（当前支持 AMD ROCm 后端）。
- `USE_IR_LOC={ttir,ttgir}`：重新解析 IR，使报错与位置追踪精确映射到特定扩展名 IR 文件的行号。
- `TRITON_PRINT_AUTOTUNING=1`：在自动调优（Autotuning）完成后，打印每个算子的最优配置及所耗时间。
- `DISABLE_LLVM_OPT`：禁用 LLVM 优化阶段（如 `DISABLE_LLVM_OPT="disable-lsr"` 可禁用循环强度削弱，部分高寄存器压力算子可提升 10% 性能）。
- `TRITON_ALWAYS_COMPILE=1`：无视缓存命中，强制每次全量重新编译算子。
- `MLIR_ENABLE_TIMING` 与 `LLVM_ENABLE_TIMING`：打印各个 MLIR / LLVM Pass 的执行耗时分析。
- `TRITON_KERNEL_DUMP` 与 `TRITON_DUMP_DIR`：将各编译阶段的中间 IR 及最终产物（PTX / AMDGCN）导出至指定目录。
- `TRITON_KERNEL_OVERRIDE` 与 `TRITON_OVERRIDE_DIR`：支持在编译阶段起始处以用户自定义的 IR/PTX 文件替换自动编译结果。
- `TRITON_F32_DEFAULT`：配置 32 位浮点 `tl.dot` 运算的默认输入精度（可选 `ieee`、`tf32` 或 `tf32x3`）。
- `TRITON_FRONT_END_DEBUGGING=1`：前端发生异常时不进行包装，直接打印完整原始调用栈。
- `PTXAS_OPTIONS`：向 NVIDIA PTX 汇编器 `ptxas` 传递额外的底层编译器标志。

> [!NOTE]
> 部分环境变量在 `knobs.py` 中没有对应的 Python 接口，这是因为它们专用于 C++ 编译器底层。

### 算子覆写（Kernel Override）实战流程

```bash
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=<dump_dir>
export TRITON_KERNEL_OVERRIDE=1
export TRITON_OVERRIDE_DIR=<override_dir>
# 第 1 步：运行一次算子，将中间 IR 与最终 ptx/amdgcn 导出至 $TRITON_DUMP_DIR
# 第 2 步：将 $TRITON_DUMP_DIR/<kernel_hash> 目录复制至 $TRITON_OVERRIDE_DIR
# 第 3 步：删除不需要修改的阶段文件，仅保留并编辑想要手工优化的阶段代码
# 第 4 步：再次运行算子，系统将自动加载经过人工调整后的覆写阶段结果
```

### 编译器流水线检查（Pipeline Inspection）

在算子运行前，可通过挂载 Hook 深入审查甚至动态干预流水线阶段：

```python
def inspect_stages(_self, stages, options, language, capability):
    # 在此处审查或动态修改 add_stages 流水线阶段
    pass

triton.knobs.runtime.add_stages_inspection_hook = inspect_stages
```

相关树外插件扩展范例请参阅 [lib/Plugins/README.md](lib/Plugins/README.md)。

---

# 📦 版本演进记录 (Changelog)

**Triton 2.0+ 重大更新特性：**
- 大量底层稳定性修复与编译期 Bug 清理
- 全面极致的性能提升
- 后端架构全面基于 **MLIR** 现代化框架重写
- 原生支持背靠背连续矩阵乘法算子（如 **FlashAttention** 机制）

---

# 🤝 参与贡献 (Contributing)

无论是修复 Bug、完善文档还是提交全新功能，我们都非常欢迎社区贡献！详情请查阅 [贡献者指南 (CONTRIBUTING.md)](CONTRIBUTING.md)。

---

# 💻 平台与硬件兼容性 (Compatibility)

**支持的操作系统：**
- Linux

**支持的硬件架构：**
- **NVIDIA GPU**（计算能力 Compute Capability 8.0+，即 Ampere、Ada Lovelace、Hopper、Blackwell 及以上）
- **AMD GPU**（ROCm 6.2+）
- *研发支持中：通用 CPU*

---

# 🐳 容器化开发镜像 (Dev Container)

Triton 官方开发容器镜像由 [triton-dev-containers 仓库](https://github.com/redhat-et/triton-dev-containers) 维护：

- **一致性**：确保所有开发者处于高度一致的环境，消除跨平台行为差异。
- **环境隔离**：容器与本机开发环境彻底隔离，避免环境依赖冲突。
- **快速上手**：新团队成员开箱即用，免去繁琐的底层环境配置。

使用指引详见 [Dev Container 使用指南](https://github.com/redhat-et/triton-dev-containers/blob/main/.devcontainer/devcontainer.md)。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年9月1日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
