## Purpose

使用户能够在 Triton 生态内用无序 shard 域、独立调度常量和逻辑时间公式表达异步通信与计算，并由编译器结合固有依赖和资源约束生成合法、可执行的重叠流水。

## ADDED Requirements

### Requirement: Minimal Timely pipeline DSL
系统 SHALL 提供最小 Timely Python DSL，使用户能够定义编译期大小的一维 shard domain、异步 `allgather_shard`、Triton 计算任务及输出存储，并将 `P`、`LAG`、`ROW/COL` 一类编译期调度参数作为时间表达式中的一等公民常量，而无需将其编码进专用串行、流水或 persistent 控制结构。

#### Scenario: Express an allgather and GEMM pipeline
- **WHEN** 用户为每个 shard 将通信映射到 `t(q)=q`，并将消费该 shard 的计算映射到 `t(q)+LAG`
- **THEN** 前端 SHALL 构造包含全部通信和计算任务及其逻辑时间的 TM IR

#### Scenario: Change scheduling without changing tasks
- **WHEN** 用户只修改编译期 `LAG` 的值
- **THEN** TM IR 中的任务和固有依赖 SHALL 保持不变，只有时间表达式及其生成的发射计划发生变化

### Requirement: Logical issue-time semantics
系统 SHALL 将逻辑时间解释为操作的请求发射偏序，而 MUST NOT 将其解释为物理周期、操作完成时间或实际执行顺序。系统 SHALL 只保留时间值之间的小于、等于和大于关系，并对绝对时间值做保序稠密化。

#### Scenario: Equivalent absolute gaps
- **WHEN** 一个程序仅有两个不同时间层，分别写为 `{t, t+1}` 和 `{t, t+100}`
- **THEN** 两个程序 SHALL 产生等价的规范化发射层和偏序

#### Scenario: Operations at the same time
- **WHEN** 多个操作具有相同的规范化逻辑时间
- **THEN** 它们 SHALL 属于同一发射层，且系统 MUST NOT 因稳定输出顺序而增加语义依赖

#### Scenario: Same-time issue with a blocked consumer
- **WHEN** `allgather(1)` 与 `gemm(0)` 具有相同逻辑时间，而 `gemm(0)` 依赖尚未完成的 `allgather(0)`
- **THEN** 两个操作 SHALL 均可在该发射层被提交，`allgather(1)` SHALL 可继续执行，而 `gemm(0)` SHALL 等待 `allgather(0)` 的完成依赖

### Requirement: Dependency identification and annotations
系统 SHALL 将时间线与固有依赖建模为独立信息。系统 SHALL 从 SSA 数据消费和已知异步操作契约自动识别依赖，并 SHALL 允许前端通过数据访问或显式依赖 annotation 表达无法从 SSA 直接识别的通信、读写和跨任务依赖。

#### Scenario: Infer communication-to-compute dependency
- **WHEN** Triton 计算任务消费 `allgather_shard` 返回的数据
- **THEN** 生成的任务图 SHALL 包含该通信 completion event 到计算任务的依赖，且用户无需显式声明 wait

#### Scenario: Preserve an annotated memory dependency
- **WHEN** 用户为非 SSA 可见的 buffer region 声明生产者写入和消费者读取 annotation
- **THEN** 前端 SHALL 将其保留为可验证的 IR 访问关系，并在任务图中建立对应数据依赖

#### Scenario: Reject an unresolved dependency
- **WHEN** 某个跨任务读写关系既不能由编译器证明，也没有足够的前端 annotation
- **THEN** 编译 SHALL 失败并指出无法证明的访问关系，而 MUST NOT 假定其独立

### Requirement: Logical schedule legality
系统 SHALL 在发射计划生成前构建并验证固有依赖图。对于每条生产者到消费者的数据依赖，生产者逻辑时间 MUST 不晚于消费者逻辑时间；系统 MUST 直接拒绝时间逆序、依赖环和缺失生产者，且 MUST NOT 自动修改用户时间公式。

#### Scenario: Reject an early consumer
- **WHEN** 某计算任务消费 shard `q` 的通信结果，但其逻辑时间早于该通信任务的逻辑时间
- **THEN** 编译 SHALL 失败，并诊断生产者、消费者和冲突时间表达式

#### Scenario: Reject a dependency cycle
- **WHEN** 固有依赖关系形成环
- **THEN** 编译 SHALL 失败并报告环中的任务

#### Scenario: Accept a completion wait at the same logical time
- **WHEN** 通信与其消费者具有相同逻辑时间且存在 completion event 依赖
- **THEN** 编译 SHALL 接受该调度，并在执行计划中保留运行期完成等待

#### Scenario: Accept a legal but suboptimal schedule
- **WHEN** 时间公式和依赖关系合法但产生额外串行化或较低并行度
- **THEN** 编译 SHALL 接受该程序，且 MUST NOT 以性能不优为由改变其语义

### Requirement: Resource-aware scheduling
系统 SHALL 为任务保留资源需求，并由目标后端根据可用 SM、线程、共享内存和通信资源生成物理资源计划。资源容量冲突 SHALL 可以延迟或串行化实际执行，但 MUST NOT 删除固有依赖或改变逻辑发射偏序。

#### Scenario: Serialize tasks that exceed capacity
- **WHEN** 同一发射层的可执行任务所需资源超过目标容量
- **THEN** 后端 SHALL 生成满足容量限制的资源顺序，并保持任务的时间与数据依赖语义

#### Scenario: Reject an infeasible resource request
- **WHEN** 单个任务的线程数或共享内存需求超过目标可表示上限
- **THEN** 编译 SHALL 失败并报告不可满足的资源需求

### Requirement: Synchronization materialization
系统 SHALL 根据固有依赖的生产者、消费者、地址空间和执行 scope 插入必要同步原语。系统 MUST NOT 仅因两个操作具有不同逻辑时间就假定数据已经可见或已经完成。

#### Scenario: Materialize a communication completion wait
- **WHEN** 计算任务依赖异步通信产生的数据
- **THEN** 后端 SHALL 在消费点生成与所选执行 scope 匹配的 completion signal/wait，并允许无依赖的通信继续推进

### Requirement: Deterministic partial-order plan
系统 SHALL 将合法 TM IR 转换为确定性的任务发射与资源计划。计划 MUST 区分逻辑时间边、固有依赖边和资源顺序边；同一发射层且无依赖、无资源冲突的任务 SHALL 保持可并发，而非被解释为额外的执行顺序。

#### Scenario: Build a pipelined plan
- **WHEN** `LAG` 允许后续 shard 通信与先前 shard 计算在偏序上并发
- **THEN** 执行计划 SHALL 将通信与计算放入独立异步执行序列，并仅在数据消费点建立必要等待

### Requirement: Executable correctness and overlap
系统 SHALL 能执行生成的 reference 计划，计算任务 MUST 通过标准 Triton 编译链生成，执行结果 MUST 与串行通信后计算 reference 一致。

#### Scenario: Observe overlap without violating correctness
- **WHEN** 使用可观测的异步 reference 通信后端运行至少两个 shard
- **THEN** 结果 SHALL 与串行 reference 一致，且执行轨迹 SHALL 显示后续 shard 通信在先前 shard 计算完成前已经发起

### Requirement: Prototype isolation
Triton 内的正式实现 MUST 不依赖仓库外 `timely` 原型的库、二进制或构建产物。

#### Scenario: Build without the prototype
- **WHEN** Triton 在不配置外部 `timely` 目录的环境中构建和测试
- **THEN** Timely DSL、TM Dialect 和最小流水测试 SHALL 正常构建并运行
