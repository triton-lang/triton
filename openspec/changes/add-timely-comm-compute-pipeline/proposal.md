## Why

现有 Triton 前端要求用户直接表达 kernel 控制流，无法将调度参数作为独立常量，通过逻辑时间公式描述通信与计算的发射偏序。需要在 Triton 仓库内建立一个同时表达时间线、固有依赖和资源需求的最小 Timely DSL 与 TM Dialect 闭环，验证后端能够据此完成合法性检查、资源调度和同步插入。

## What Changes

- 在 Triton Python 前端中增加最小 Timely DSL，支持一维无序 shard domain、作为一等公民的编译期调度常量、逻辑时间表达式、异步 `allgather_shard` 和 Triton 计算任务。
- 在 Triton MLIR 中增加独立 TM Dialect，分别表示逻辑发射时间、通信完成事件、固有任务依赖和任务资源需求；计算区域直接承载并复用 TTIR。
- 将逻辑时间解释为发射偏序而非执行顺序，对绝对时间做保序稠密化，并保持同时间独立操作可并发发射。
- 通过 SSA 数据流、通信计划语义和前端数据依赖 annotation 构建依赖图；无法证明或与时间偏序冲突的固有依赖直接报错。
- 复用 Triton 的 memory-effect、alias、buffer-region 和 membar 分析能力，并在需要时扩展其接口以服务跨 task 依赖分析。
- 将时间偏序、固有依赖和目标资源约束转换为确定性发射/资源计划；后端分别生成同步原语并分配 SM、线程和共享内存等物理资源。
- 将计算任务 outline 为 Triton kernel，将通信任务降到轻量异步 runtime plan，通过 stream/event 生成可执行 overlap。
- 提供串行 reference、非法调度诊断和 overlap 检查，证明结果正确且通信与计算能够并发推进。
- `timely/` 原型保持不变，仅作为设计与测试参考；正式实现不依赖其库、二进制或构建系统。

## Capabilities

### New Capabilities

- `timely-comm-compute-pipeline`: 定义 Timely 计算-通信流水 DSL、发射时间、数据依赖 annotation、资源约束、合法性验证、资源/同步计划及最小可执行 overlap 行为。

### Modified Capabilities

- 无。

## Impact

变更位于 Triton 仓库，影响 Python DSL/AST lowering、MLIR dialect 注册、Triton 依赖分析、资源规划、同步插入、runtime launch plan 和编译器测试。首版增加一个 reference 异步 shard 通信后端；生产级 NCCL、多节点通信、自动性能搜索和普通循环回退机制不在本次范围内。
