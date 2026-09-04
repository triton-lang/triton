## 1. TM Compiler Foundation

- [ ] 1.1 在 Triton 构建与 dialect registry 中接入独立 TM Dialect，并用 `triton-opt` parser/printer round-trip 测试验证该 dialect 可独立加载。
- [ ] 1.2 建立 Timely Python 入口及 `@tm.kernel`、`@tm.task`、`Const`、`Time` 的最小对象模型，并验证导入和装饰器元数据测试通过。

## 2. Minimal DSL And IR

- [ ] 2.1 定义逻辑发射时间、completion event、访问/依赖 annotation、资源需求、静态 domain、异步 shard 通信、定时 task 调用及 graph/plan 的最小 ops/types，并用合法与非法 IR verifier 测试覆盖其结构约束。
- [ ] 2.2 扩展/复用 Triton Python AST lowering，使示例中的 `tm.domain`、一等公民 `Const` 时间参数、`allgather_shard` 和定时 `@tm.task` 调用生成预期 TM IR，并用 golden IR 测试确认用户源码不含专用调度控制流或显式 wait/stream。
- [ ] 2.3 让 `@tm.task` 计算体接受普通 `triton.language` 运算并可 outline 为标准 Triton kernel，验证一个最小 `tl.dot` task 能通过既有 TTIR/TTGIR 编译链。
- [ ] 2.4 为 task/call-site 增加最小 `reads`、`writes`、`depends_on` annotation，并验证合法 annotation 进入 IR、缺失或矛盾 annotation 产生明确诊断。

## 3. Time And Dependence Analysis

- [ ] 3.1 实现静态 domain 和调度 `Const` 的 constexpr 特化及逻辑时间保序稠密化，并验证仅有两个时间层时 `{t,t+1}` 与 `{t,t+100}` 结果相同、运行期调度参数被拒绝。
- [ ] 3.2 定义跨 task 依赖摘要接口，复用 Triton `MemoryEffectOpInterface`、alias 和 buffer-region 分析收集 task 内访问，并验证可推导的 RAW/WAR/WAW 摘要正确。
- [ ] 3.3 合并 SSA/async completion、访问 annotation 和显式 `depends_on`，建立 `DataDep` 图；验证无法证明的别名关系被拒绝且通信到计算依赖无需源级 wait。
- [ ] 3.4 实现缺失生产者、依赖环及 `T(producer) > T(consumer)` 检查，并验证全部直接报错；同时验证同时间依赖和合法但次优的时间映射被接受。
- [ ] 3.5 生成可检查的 `TimeOrder` 与 `DataDep` 图，验证改变 `LAG` 只改变时间交错、同层稳定输出顺序不增加语义边。

## 4. Resource And Synchronization Planning

- [ ] 4.1 实现 task 资源需求摘要和目标容量查询，至少覆盖执行资源类别、线程/warp 数与共享内存，并验证超出单任务上限时产生诊断。
- [ ] 4.2 实现发射层与资源规划 pass，为容量冲突生成独立 `ResourceOrder`，并验证该 pass 不修改 `TimeOrder` 或删除 `DataDep`。
- [ ] 4.3 实现同步插入 pass，根据 `DataDep`、执行 scope 和 placement 物化 completion token/event/wait，复用 Triton Membar 处理 task 内共享内存 hazard，并验证不同时间但无依赖时不会虚构同步。

## 5. Executable Reference Plan

- [ ] 5.1 实现最小通信 runtime interface 和异步 shard-copy reference backend，验证 communication stream 能产生可供其他执行资源等待的 completion event。
- [ ] 5.2 实现 plan executor，按稠密发射层提交通信节点和 outline 后的 Triton kernel，并遵守 `DataDep` 与 `ResourceOrder`，验证 launch/event 序列与三类计划关系一致。
- [ ] 5.3 增加可控通信延迟与执行轨迹，验证 `AG1` 与 `GEMM0` 同层时二者均被提交、`AG1` 可推进而 `GEMM0` 等待 `AG0`。

## 6. End-To-End Acceptance

- [ ] 6.1 实现串行 `allgather -> GEMM` reference，并验证多个静态 shard 形状及 LAG 配置下 Timely 流水结果与 reference 一致。
- [ ] 6.2 增加时间间隔等价、同时间发射、依赖 annotation、时间逆序、依赖环、资源容量、同步插入、Triton kernel 编译和 overlap 的组合测试，并运行相关 lit/pytest/GPU 测试确认全部通过。
- [ ] 6.3 在不引用或构建外部 `timely/` 原型的环境中验证 Timely 能力，并记录调度参数独立性、三类计划关系、严格合法性规则、资源/同步边界及普通循环回退不属于首版。
