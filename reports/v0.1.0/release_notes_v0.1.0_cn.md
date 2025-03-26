[English](./release_notes_v0.1.0.md)

## FlagTree 0.1.0 Release

### Highlights

FlagTree 首次发布，基于 Triton 3.1 版本接入多元 AI 芯片后端。项目当前处于初期，目标是兼容各芯片后端现有适配方案，统一代码仓库，打造代码共建平台，快速实现单版本多后端支持。

### New features

* 多后端支持
目前支持的后端包括 iluvatar、xpu (klx)、mthreads、cambricon。

* 两种编译路径支持
项目初期，对 TritonGPU dialect 或 Linalg dialect 两种编译路径作简单快速兼容。

* 高差异度模块插件化能力
支持芯片后端定制化的高差异度模块以插件形式提供，这些非通用模块的代码由对应的芯片提供商自行维护，并通过工程化手段与 FlagTree 主仓库可保持同构设计。

* 交叉编译与快速验证能力
为方便开发者简单快速验证，FlagTree 可以实现在任意硬件平台上编译及在 python3 中导入。如果编译环境和运行环境一致（一般指 cpython、glibc、glibcxx、cxxabi 版本对齐或兼容），可以实现交叉编译，即编译结果能够在实际搭载对应芯片的环
境中跨平台运行。

* CI/CD 能力
项目为 iluvatar、xpu、mthreads、nvidia 等后端搭建了 CI/CD，可以完整验证从编译到测试正确性的全流程。

* 质量管理能力
FlagTree 除了建设 CI/CD 覆盖多后端芯片外，还搭建了贡献者许可协议（CLA）签署、安全合规扫描等机制做质量与合规保障。

### Known issues

* triton-opt、proton 等工具目前不支持。

### Looking ahead

FlagTree 将持续投入发展 Triton 生态，包括跟进 Triton 版本更迭，接入 AI 芯片后端，提升编译效率，优化跨平台兼容性。同时，FlagTree 将对兼顾通用性和芯片极致优化需求进行探索，兼容式地在语言层提供硬件的存储层次、并行层次、加速单>元等关键特征的统一抽象表达和显示指定能力。
