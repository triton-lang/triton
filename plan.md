# Triton Blackwell 2CTA/TMA/TMEM 深度研究与 Pi0.5 实施计划

日期：2026-07-22  
目标硬件：NVIDIA Jetson AGX Thor（SM110）  
Triton 开发仓库：`/workspace/triton`  
Pi0.5 仓库：`/workspace/realtime_vla/realtime-vla`

## 1. 执行摘要

本项目的目标不是继续在 Pi0.5 中手写一套越来越复杂的 Gluon GEMM，而是把已经验证有效的 Blackwell 2SM 机制逐步下沉到普通 Triton 编译链路，使 `tl.dot`、TMA descriptor、warp specialization 和 epilogue 能自动生成正确且可维护的 2CTA/TMEM 代码，最终形成可以提交给 Triton upstream 的通用实现。

当前最重要的结论如下。

1. 当前 Pi0.5 默认 Python 环境已经以 editable 方式直接绑定用户 fork 的本地工作树。Python 源码来自 `/workspace/triton/python/triton`，native compiler 来自 `/workspace/triton/python/triton/_C/libtriton.so`。因此 Python 层修改立即可见；C++/MLIR pass 修改必须重新编译 `libtriton.so` 后才会生效。
2. 当前 Triton 基线 `682f467a` 已经具备 SM100/SM110 的 TCGen05、TMEM、TMA、cluster barrier 等底层表示和 lowering，但普通 `tl.dot` 的自动 2CTA 选择仍在 `AccelerateMatmul.cpp` 中明确写死为 `false`。这正是第一阶段需要补齐的入口。
3. 现有 Pi0.5 Gluon winner 已经真实使用 `tcgen05.mma.cta_group::2`、TMEM accumulator、TMA load/store 和 warp specialization，但“2CTA MMA”不等于“TMA load multicast”。只有 PTX 出现 `cp.async.bulk.tensor...multicast::cluster`，才能证明发生了真正的 TMA load multicast。
4. 真正的 RHS TMA load multicast 更自然地对应 4CTA cluster：两个 2CTA MMA pair 复用同一份权重 tile。它不是第一阶段最小 2CTA pass 的组成部分，必须作为独立 A/B，避免把 cluster 扩大、尾 tile、并行度变化与 multicast 收益混在一起。
5. CUTLASS/CuTe 将 Blackwell 2SM 看作完整 schedule，而不是一个 MMA 开关：ClusterShape、2SM TCGen05、TMEM、TMA producer、producer/consumer barrier、warp roles、寄存器配置和 epilogue 必须一致。Triton 实现也必须遵守这个原则。
6. 官方 `keren/*` 开发分支比 2025 年的 WIP 分支完整得多，覆盖自动 two-CTA、barrier、TMA lowering、warp specialization、4CTA multicast 和 persistent 调优，应作为主要设计参考；但不能把开发分支整包合入后直接宣称收益，必须分阶段移植并逐项归因。

## 2. 仓库、身份与绑定状态

### 2.1 Triton Git

当前状态：

```text
path:     /workspace/triton
branch:   perf/thor-blackwell-2cta
HEAD:     682f467ab85e6e4e0c5050aa44ccf1a47a65f742
origin:   https://github.com/Tipsyscholar/triton.git
upstream: https://github.com/triton-lang/triton.git
identity: kao hsing po <tipsy.schloar@gmail.com>
```

`origin` 是用户 fork，用于未来个人分支和 PR；`upstream` 只用于跟踪官方更新。当前不 push。官方 `upstream/main` 比固定基线多 55 个提交，但在建立 compiler-only A/B 前不改变当前分支基线。

查看官方更新：

```bash
cd /workspace/triton
git fetch upstream
git log --oneline --decorate HEAD..upstream/main
git rev-list --left-right --count HEAD...upstream/main
```

### 2.2 Pi0.5 Git

```text
path:   /workspace/realtime_vla/realtime-vla
branch: triton380
HEAD:   b40e993af3f0176c2c6973e4d10d1eba7436a366
```

编译器改动只提交到 Triton 仓库；Pi0.5 kernel/集成改动只提交到 realtime-vla 仓库。不要在一个 Git 历史中混合两类改动。

### 2.3 默认 Python 的真实绑定

已在 Pi0.5 工作目录中核实：

```text
distribution:       triton 3.8.0+git682f467a
editable project:   /workspace/triton
direct_url.json:    file:///workspace/triton, editable=true
Python package:     /workspace/triton/python/triton/__init__.py
native libtriton:   /workspace/triton/python/triton/_C/libtriton.so
```

这意味着直接运行：

```bash
cd /workspace/realtime_vla/realtime-vla
python3 benchmark.py --num_views 3 --prompt_len 10 --chunk_size 50 --model_version pi05
```

会导入 `/workspace/triton` 中的 Python 包和 native compiler，而不是另一个 wheel。修改 Python 编译器代码会被新进程直接读取；修改 `lib/`、`include/` 或 `third_party/nvidia/` 下的 C++/MLIR pass 后，必须先重编 native library。

### 2.4 每次 pass 修改后的强制重编与核验

按照仓库 `AGENTS.md`，native/compiler 变更后从 Triton 根目录执行：

```bash
cd /workspace/triton
make
```

当前增量 build tree 为：

```text
/workspace/triton/build/cmake.linux-aarch64-cpython-3.12
```

它的 CMake 配置明确满足：

```text
CMAKE_HOME_DIRECTORY=/workspace/triton
CMAKE_LIBRARY_OUTPUT_DIRECTORY=/workspace/triton/python/triton/_C
TRITON_BUILD_PYTHON_MODULE=ON
CMAKE_BUILD_TYPE=TritonRelBuildWithAsserts
```

需要只重建 Python native module 时，可用：

```bash
ninja -C /workspace/triton/build/cmake.linux-aarch64-cpython-3.12 libtriton.so
```

但正式验证仍优先遵循仓库约定运行 `make`。重编前后都记录：

```bash
stat /workspace/triton/python/triton/_C/libtriton.so
sha256sum /workspace/triton/python/triton/_C/libtriton.so
```

然后从 Pi0.5 目录再次验证：

```bash
python3 - <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import triton
import triton._C.libtriton as libtriton

dist = metadata.distribution("triton")
print("version:", metadata.version("triton"))
print("package:", Path(triton.__file__).resolve())
print("libtriton:", Path(libtriton.__file__).resolve())
print("direct_url:", dist.read_text("direct_url.json"))
PY
```

验收条件：两个路径都必须位于 `/workspace/triton`；`direct_url.json` 必须是 `file:///workspace/triton` 且 `editable=true`；native 修改后 `.so` 的 mtime/hash 必须变化。

### 2.5 防止旧 cubin 掩盖新 pass

即使 `libtriton.so` 已更新，默认 Triton cache 仍可能命中旧 cubin。每个编译器版本和每个实验必须使用独立目录：

```bash
export TRITON_CACHE_DIR=/workspace/realtime_vla/profiling_reports/cache/<experiment-id>
mkdir -p "$TRITON_CACHE_DIR"
```

不要删除或复用共享的多 GB cache。一个 pass 被视为“进入 Pi0.5 端到端”的最低证据链是：

```text
editable package path
  -> rebuilt libtriton.so hash
  -> fresh TRITON_CACHE_DIR
  -> generated TTGIR contains two_ctas/TMEM layout
  -> PTX/SASS contains expected TCGen05/TMA/barrier instructions
  -> Pi0.5 CUDA Graph A/B changes consistently
```

## 3. 当前普通 Triton 的能力边界

当前源码已经能表示和 lower 多项 Blackwell 机制：

- `TensorMemoryEncodingAttr(twoCTAs=true)` 表示 2CTA TMEM layout。
- `ttng.tc_gen5_mma {two_ctas}` 可 lower 为 `tcgen05.mma.cta_group::2`。
- TMA load/store、cluster launch、proxy fence 和 mbarrier 已有底层支持。
- `CheckMatmulTwoCTAs.cpp` 要求同一 module 内所有 MMAv5 matmul 的 CTA mode 一致，并把结果写入 module attr。
- `MMAv5.cpp`、`TensorMemoryToLLVM.cpp` 和 `NVGPUToLLVMPass.cpp` 已支持 two-CTA MMA、TMEM copy、alloc/dealloc 的 PTX codegen。

但普通 `tl.dot` 的入口仍被关闭：

```cpp
// lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp
// change the line below to: bool useTwoCTAs = canUseTwoCTAs(dotOp);
bool useTwoCTAs = false;
```

不能只把这一行改成 `true`。自动选择还需要同步保证：

- shape、dtype、MMAv5 instruction shape 合法；
- cluster 至少包含相邻的 CTA pair；
- RHS 来源及 layout 能被正确拆分；
- module 内多个 dot 使用同一 CTA mode；
- TMEM accumulator、TMA barrier、completion count 和 proxy fence 一致；
- warp specialization 下 producer/consumer barrier 仍捕获所有跨 CTA 用户；
- epilogue 与 TMA store 不在 peer CTA 数据尚未完成时读取。

## 4. Triton 官方开发分支谱系

### 4.1 主要参考：2026 年 `keren/*`

| 提交 | 作用 | 本项目用途 |
|---|---|---|
| `798dfff0` | 自动启用 MMAv5 two-CTA matmul | Phase 1 的选择与 layout 主参考 |
| `28df5f4f` | 修复 two-CTA barrier handling | Phase 1 正确性必需 |
| `f8011b76` | 修复 completion barrier count | Phase 1/2 双 dot 与 pipeline 必需 |
| `b4e756db` | 修复 two-CTA TMA load barrier ordering | Phase 1/3 必需 |
| `05742b00` | 4CTA + two-CTA MMA 的 TMA multicast | Phase 4 主参考 |
| `9b79b240` | 建模 two-CTA pair 之间的 CTA split reuse | Phase 4 layout 主参考 |
| `cc631be5` | descriptor RHS dot 优先 two-CTA layout | Phase 1/4 自动 layout 参考 |
| `6383685a` | 修复 two-CTA MMAv5 用户的 TMA lowering | Phase 1/3 必需 |
| `db6464e5` | warp specialization 下 two-CTA TMA barrier | Phase 3 主参考 |
| `fb259269` | 修复 WS 的 register/TMEM layout | Phase 3 主参考 |
| `c4e4c880` | 优化 WS 2CTA TMA multicast matmul | Phase 3/4 主参考 |
| `31c23655` | persistent WS multi-CTA 配置调优 | Phase 5 参考 |

`keren/2cta-support` 是最小自动 two-CTA 起点；`keren/2cta-barriers` 处理同步；`keren/tma-multi-cast-cta-layouts` 扩展布局；`keren/ws-2cta-tma-multicast-opt` 则包含后续 WS/persistent 优化。实施时应按这个依赖顺序拆分，而不是直接合并最终分支。

### 4.2 次要参考与历史风险

- `pb/two-cta-matmul`：2025 年 WIP/HACK，只用于了解早期尝试，不作为移植底座。
- `disable-2cta`：记录自动 2CTA 曾因正确性/同步问题被关闭的历史，说明只打开 `useTwoCTAs` 不足以形成可靠实现。
- `two-cta-mbarrier`：用于核对 mbarrier 语义和测试。
- `gluon-clc-support`：只用于 Phase 6 CLC，不进入最小 2CTA 实现。

### 4.3 当前 upstream/main 的同步修复

当前基线比 `upstream/main` 落后 55 个提交，其中以下变更直接影响 multi-CTA 安全性：

- `ed9deeb5`：MultiCTA arrive/expect 增加 `from_ctas`。
- `d66def39`：改进 warp specialization 内 cluster barrier codegen。
- `63f3fd78`：多项 cross-CTA synchronization 修复。
- `56cac4e8`：barrier 添加 `.release/.acquire.cluster`。
- `96bc7e78`：回退“所有 mbarrier 都采用 cluster-scoped ordering”的过度修改。
- `18f97b6f`：修复共享 accumulator 的 back-to-back MMA pipeline。
- `3d6e0508`：修改 TMEM layout 前 materialize out-dimension names。

因此存在两条开发路线：

1. 在 `682f467a` 上最小移植，最适合与现有 Pi0.5/Gluon winner 做干净性能归因。
2. 在最新 `upstream/main` 上实现，最适合最终 PR，且天然包含后续同步修复。

建议先在当前分支完成 Phase 0 的 compiler-only A/B；随后从最新 upstream 新建 PR 分支重新落实现有补丁。不要把 rebase、pass 实现和 kernel 配置变化放在同一次性能对比中。

注：本仓库为 partial clone。本次已获取关键分支引用和提交元数据；部分对象在离线状态下无法按需展开。开始移植前需在网络可用时执行 `git fetch upstream` 补齐目标提交对象，并逐文件复核 diff。

## 5. CUTLASS/CuTe Blackwell 设计审查

本机 NVIDIA CUTLASS 参考仓库：

```text
/workspace/cutlass
HEAD 2802e228c23b8c09f946a3a46e56df35939d34e2
```

主要参考：

- `examples/70_blackwell_gemm/70_blackwell_fp16_gemm.cu`
- `examples/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu`
- `include/cutlass/gemm/collective/builders/sm100_umma_builder.inl`
- `include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp`
- `include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp`
- `include/cutlass/pipeline/sm100_pipeline.hpp`
- `include/cute/atom/copy_traits_sm90_tma.hpp`
- `media/docs/cpp/blackwell_functionality.md`
- `media/docs/cpp/blackwell_cluster_launch_control.md`

### 5.1 2SM 是 schedule，不是单指令开关

CUTLASS 的 2SM UMMA schedule 同时约束：

```text
ClusterShape
  -> CTA pair 的 M/N 映射
  -> tcgen05.mma.cta_group::2
  -> TMEM accumulator 分布
  -> TMA producer 和 multicast mask
  -> producer/consumer mbarrier
  -> load/MMA/epilogue warp roles
  -> register budget 与 pipeline stages
  -> epilogue store
```

典型 BF16/FP16 2SM tile 为 `256x128x64`，即 cluster M=256、每个 SM/CTA 负责 M=128。Cluster M 维必须包含成对 CTA。该形状与当前 Pi0.5 Gluon winner 的 `CLUSTER_M=256`、`BM_PER_CTA=128` 一致。

### 5.2 TMA multicast 的方向

对 `C[M,N] = A[M,K] @ B[K,N]`：

- A tile 通常沿 N 方向广播，因为多个 N CTA 复用同一 A。
- B tile 通常沿 M 方向广播，因为多个 M CTA 复用同一 B。
- multicast mask 必须由 cluster CTA layout 和 CTA coordinate 计算，不能由一个布尔开关替代。

2CTA MMA pair 本身沿 M 合作执行一个 2SM instruction，并不自动意味着 TMA B load 只发一次。要在两个 2CTA pair 间真正复用 B，通常需要 4CTA cluster，并在 pair 之间给 B layout 配置 broadcast basis。

### 5.3 CLC 的适用边界

Cluster Launch Control 主要服务 persistent、多 tile 和尾部负载均衡。它不会自动降低单 tile 的 L2 字节，也不应与首次启用 2CTA/TMA multicast 同时测试。当前 Pi0.5 的 Res CLC-only 负结果说明 CLC 不能默认打开；只有 Phase 5 persistent schedule 稳定后，Phase 6 才进行单变量 A/B。

### 5.4 SM100 与 Thor SM110

CUTLASS 示例主要以 SM100/B200 规则组织。Thor 是 SM110，不能仅因 PTX 能编译就认定调度同样最优。每个阶段必须在 Thor 实机核查：

- TTGIR layout 和 barrier 语义；
- PTX 的 `cta_group::2`、TMA multicast、cluster scope；
- SASS 的 UTCMMA/TMA 指令；
- NCU 的实际 L2 sectors、occupancy、stall 和 duration；
- CUDA Graph 内 34 次大 GEMM 的累计时间和节点间 gap。

## 6. 当前 Pi0.5 Gluon winner 的精确语义

文件：`/workspace/realtime_vla/realtime-vla/gluon_large_gemm.py`

实际大矩阵（3 views、prompt 10、chunk 50）：

```text
Gate encoder: [778, 2048] x 2 * [2048, 16384], 17 calls/inference
Res FFN-down: [778, 16384] * [16384, 2048] + residual, 17 calls/inference
```

当前生产配置：

| Kernel | Cluster tile | N tile | K tile | Stages | Warps | MMA regs | Group-M |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gate encoder | M256（每 CTA M128） | 256 | 64 | 4 | 8 | 32 | 4 |
| Res FFN-down | M256（每 CTA M128） | 256 | 128 | 3 | 8 | 48 | 4 |

共同机制：

- `num_ctas=2`；
- 显式 load/MMA/epilogue warp specialization；
- TMA descriptor load 与 TMA store；
- TMEM accumulator；
- `tcgen05.mma.cta_group::2`；
- Group-M/N-major 权重局部性；
- Gate 双 dot 后融合 GeGELU；Res 融合 residual add。

### 6.1 必须区分的三层语义

| 层次 | 证明指令/属性 | 当前 winner |
|---|---|---|
| 2CTA cooperative MMA | `tcgen05.mma.cta_group::2` | 已有 |
| 2CTA-group TMA transaction | `cp.async.bulk.tensor...cta_group::2` | 已有 |
| 真正 TMA load multicast | load 指令带 `.multicast::cluster` | 需要按最终生成 PTX 单独确认 |

当前已有 winner PTX 明确证明前两项。对真正 TMA load multicast，不能依据 Python 参数 `MULTICAST=True` 下结论；Gluon builder 会结合 `hasCGABroadcast(result_layout)` 决定是否发出 multicast。报告和 benchmark 以后统一以最终 PTX/SASS 与 NCU 计数为准。

## 7. 能力差距矩阵

| 能力 | 普通 Triton 当前基线 | Gluon winner | 目标普通 Triton |
|---|---|---|---|
| TCGen05/TMEM 1CTA | 支持 | 支持 | 保持 |
| 自动 2CTA MMA 选择 | 入口写死关闭 | 手工指定 | 自动、可验证、可回退 |
| module 内 CTA mode 一致 | 已有检查 pass | 手工统一 | 自动保证 |
| two-CTA TMEM layout | 底层支持 | 手工构造 | 自动推导 |
| TMA descriptor load/store | 支持 | 显式使用 | 自动 pipeline |
| WS two-CTA barrier | 基线不完整 | 手工 barrier | 正确自动生成 |
| Gate 双 dot 统一 2CTA | 未启用 | 手工实现 | 自动且共享 pipeline |
| 4CTA RHS multicast | 未作为生产路径 | 尚未证明收益 | 独立启发式和 A/B |
| Group-M locality | kernel 侧配置 | 固定为 4 | 保留/纳入调优 |
| Persistent multi-tile | 有基础设施 | 非当前默认 | 后续阶段 |
| CLC | 有开发分支 | 负 A/B | 最后单变量验证 |

## 8. 分阶段实施计划

每一阶段都必须产生独立 commit、独立 cache、独立报告和可一键回退开关。禁止一次同时改变 pass、tile、stages、warps、epilogue 和 graph 集成。

### Phase 0：冻结基线与建立编译器证据链

工作项：

1. 记录 Triton/realtime-vla/CUTLASS commit、GPU clocks、Python binding、`.so` hash。
2. 用当前 `682f467a` 重编一次无源码变化的 `libtriton.so`，验证 build tree 与 editable install 闭环。
3. 用独立 cache 重测两个 Gluon winner 和普通 Triton baseline。
4. 保存 TTIR/TTGIR/LLIR/PTX/cubin，以及 CUDA Graph、NSYS、NCU 报告。
5. 在不改 Pi0.5 kernel 的条件下，单独测试 `upstream/main` 编译器对同一 workload 的影响；结果不与 pass 收益合并。

验收：数值正确；默认 Python 路径和 `.so` hash 可追溯；相同源码/相同 cache 策略重复运行稳定；能从汇编区分 1CTA/2CTA/multicast。

### Phase 1：普通 Triton 最小自动 2CTA，先做 Res FFN-down

范围：只处理单 `tl.dot` 的 Res FFN-down；保持 tile、TMA descriptor、residual epilogue、Group-M 和 CUDA Graph 集成不变。

实现要点：

1. 从 `798dfff0` 恢复受约束的 `canUseTwoCTAs`，不得无条件打开。
2. module 级统一选择 CTA mode，避免同一 kernel 内 TCGen05/TMEM mode 冲突。
3. 要求 Blackwell target、合法 dtype/shape、RHS 来自可追踪 load/descriptor load。
4. 构造 RHS/accumulator CGA layout，并保持 `nPerCTA <= 256` 等硬件限制。
5. 移植 `28df5f4f`、`f8011b76`、`b4e756db`、`6383685a` 中与最小路径直接相关的 barrier/TMA 修复。
6. 增加 lit tests，覆盖合法 two-CTA、非法 shape fallback、mixed mode rejection 和 descriptor RHS。

明确不做：TMA multicast、warp 配置调优、persistent、CLC、Gate 双 dot。

验收：

- 正确性与 group-1 对齐；
- TTGIR 有 `two_ctas` 和 two-CTA TMEM encoding；
- PTX 有 `tcgen05.mma.cta_group::2`；
- 无意外 `.multicast::cluster`；
- Res 微基准、Graph 内累计时间和端到端均不回退超过 3%；
- 能通过环境变量或 compiler option 回退 1CTA。

### Phase 2：Gate encoder 双 dot

范围：把 Gate 和 Up 两个 dot 同时自动转成 2CTA，保留 GeGELU 与 TMA store，不改变数学边界。

风险：两个 accumulator 的 TMEM allocation、back-to-back MMA completion barrier、共享 X tile 生命周期和 epilogue handoff。

实现要点：

1. module 决策必须对两个 dot 给出相同结果；任一 dot 不合法时整体 fallback。
2. 验证两个 TMEM accumulator 的 layout/容量和 token 生命周期。
3. 核对 `18f97b6f` 所处理的共享 accumulator/back-to-back MMA pipeline 问题是否需要纳入基线。
4. 保持当前双 dot + GeGELU 输出的数值和 store 顺序。

验收：两个 dot 均生成 `cta_group::2`；无 barrier race；Gate isolated、Graph 累计与端到端三层结果一致；ConSan/相关 lit tests 通过。

### Phase 3：自动 pipeline 与 warp specialization

范围：在 Phase 1/2 正确后，复刻当前 Gluon 的 load/MMA/epilogue 角色分工。

参考：`db6464e5`、`fb259269`、`c4e4c880`。

调优维度按单变量批次推进：

```text
warps -> MMA registers -> stages -> Group-M
```

每一批固定其余变量。重点检查 TMA barrier capture、AREF、producer/consumer register budget、TMEM layout 和 epilogue shared memory。

验收：普通 Triton 在两个大 GEMM 上达到或超过 Gluon winner 的 CUDA Graph 累计时间；端到端 P50 不因节点 gap 或资源占用恶化；PTX/NCU 同时确认硬件路径。

### Phase 4：4CTA + 真正 RHS TMA load multicast

目标：两个 2CTA MMA pair 共享同一权重 tile，使 B/W 的 global-to-shared 请求在 cluster 内广播。

参考：`05742b00`、`9b79b240`、`cc631be5`、`c4e4c880`。

必须保持单变量：从 Phase 3 winner 出发，只把 cluster 从一个 pair 扩为两个 pair并改变 RHS CGA broadcast layout。tile K、stages、warps、epilogue保持不变。

关键验证：

- PTX load 出现 `.multicast::cluster`，不能只看 API 参数；
- multicast mask 与 CTA coordinate 正确；
- 两个 pair 的 RHS shared tile 生命周期和 barrier count 正确；
- NCU 看实际 L2 sectors/miss bytes，不能只看 delivered TMA bytes；
- 评估 M=778 的尾 cluster 浪费和 cluster 数量下降；
- Gate 的两份权重需要分别验证 multicast，Res 验证单份权重。

停止条件：若 L2 sectors 没有下降，说明只是改变 TMA 表达而未发生真实复用；若 kernel 加速但 Graph/端到端恶化，则记录为负结果，不进入生产默认。

### Phase 5：Persistent multi-tile 与 Group-M/N-major locality

只有 Phase 4 后仍受权重/L2 请求限制时进行。

目标：让 CTA cluster 连续处理多个逻辑 tile，摊薄 launch/schedule 开销，并保持权重 tile 的 L2/SMEM 局部性。先用静态 persistent schedule，不引入 CLC。

比较：

- M-major、N-major、Group-M；
- 每 cluster 处理 1/2/4 个逻辑 tile；
- Gate/Res 分开调优；
- CUDA Graph 内 34 次大 GEMM 的节点间 idle gap。

验收：不是只看 isolated kernel；NSYS 必须证明累计 kernel 时间或 gap 减少，端到端 P50/P95 同时改善。

### Phase 6：CLC-only A/B

从 Phase 5 最优 persistent 版本出发，只替换 tile 分配为 Cluster Launch Control，保持 2CTA、4CTA/multicast、tile、TMA、计算和 epilogue不变。

由于此前 Res CLC-only 没有收益，CLC 不设为默认；只有 Thor 上多次稳定降低 Graph P50 且无长尾回退才保留。

### Phase 7：上游 PR 化

1. 从最新 `upstream/main` 新建干净分支，重放通用补丁。
2. 去除所有 Pi0.5 shape、文件名和环境变量特判。
3. 提交最小且可审查的 commit 序列：选择/layout、barrier、WS、multicast、tests。
4. 增加 lit、Python correctness、GPU tests、ConSan；覆盖 fallback。
5. 在用户 fork 的独立分支准备 PR，但 push 只能在用户明确授权后执行。

## 9. 测试与性能验证合同

### 9.1 正确性顺序

每阶段严格按以下顺序：

```text
lit/compiler IR
  -> kernel numerical correctness
  -> isolated microbenchmark
  -> CUDA Graph A/B
  -> Pi0.5 endpoint P50/P95/mean
  -> NSYS/NCU 解释原因
```

对于模型输出，复用相同 image、prompt 和 diffusion noise，检查：

- action shape 正确；
- 无 NaN/Inf；
- cosine similarity 至少 0.999；
- max error 和每维 error；
- 2-view 与 3-view 分开报告。

### 9.2 当前 realtime-vla 标准命令

保持用户现有口径：

```bash
cd /workspace/realtime_vla/realtime-vla
python3 benchmark.py \
  --num_views 3 \
  --prompt_len 10 \
  --chunk_size 50 \
  --model_version pi05
```

当前 `benchmark.py` 为 3 次 warmup + 100 次 wall-clock iteration。不同 view、prompt、chunk、GPU 频率、cache、profiler状态或 timing boundary 的数字不得放在同一比较列。

### 9.3 CUDA Graph-aware A/B

Autotune 的 isolated timing 只能筛配置，最终 winner 必须在同一 CUDA Graph 中 A/B。需要报告：

- 17 次 Gate encoder 累计时间；
- 17 次 Res FFN-down 累计时间；
- 两类共 34 个节点之间的 idle gap；
- 完整 inference P50/P95/mean；
- fresh process 多次重复结果。

### 9.4 IR/PTX/SASS 检查

每个实验保存：

```text
*.ttir
*.ttgir
*.llir
*.ptx
*.cubin
nvdisasm output
```

关键匹配：

```text
two_ctas
tcgen05.mma.cta_group::2
tcgen05.cp/commit
cp.async.bulk.tensor
multicast::cluster       # 仅 Phase 4 必须出现
mbarrier
.release.cluster/.acquire.cluster
fence.proxy
```

### 9.5 NCU 指标

至少包含：

- kernel duration；
- L2 read/write sectors、hit rate、miss bytes；
- TMA load/store request/bytes；
- UTCMMA/TCGen05 与 TMEM 指令；
- long scoreboard、barrier、not selected 等 stall；
- registers/thread、shared memory/block；
- active CTAs、occupancy、waves/tail；
- shared bank conflicts/excessive wavefronts。

Profiler replay 是冷缓存环境，NCU duration 不直接替代 NSYS 稳态或 unprofiled P50。结论必须同时引用对应测量边界。

## 10. 产物目录规范

报告不放进 realtime-vla Git 仓库内部。统一放到外层：

```text
/workspace/realtime_vla/profiling_reports/<date>_<phase>_<experiment>/
```

每个目录至少包含：

```text
README_ZH.md
environment.txt
commands.sh
baseline.log
candidate.log
ir/
asm/
nsys/
ncu/
```

`environment.txt` 记录两个仓库 commit、branch、dirty status、Python/package/libtriton 路径、`.so` hash、GPU clocks、cache 路径和关键环境变量。

## 11. 决策规则

1. isolated kernel 更快不等于端到端更快；生产选择以 CUDA Graph 和端到端为准。
2. 任何超过 3% 的端到端回退都需要解释；超过 5% 视为回归，直到排除频率、功耗、其他进程、cache、autotune 和 graph差异。
3. 若 PTX 没有 `.multicast::cluster`，不得把收益归因于 TMA multicast。
4. 若 NCU 的 L2 sectors 不降，不得用 TMA delivered bytes 宣称减少了真实 DRAM/L2 流量。
5. 每阶段只改变一类机制；负结果保留报告和 commit，不反复无记录地改了又撤。
6. Triton pass 与 Pi0.5 kernel 改动分仓库提交，便于二分和未来 PR。
7. 未经用户明确授权，不 push 用户 fork，也不向 upstream 创建 PR。

## 12. 第一轮执行清单

- [x] 把 `origin` 绑定到 `Tipsyscholar/triton`，保留官方 `upstream`。
- [x] 创建本地分支 `perf/thor-blackwell-2cta`。
- [x] 设置 Git 身份为 `kao hsing po <tipsy.schloar@gmail.com>`。
- [x] 验证 Pi0.5 默认 `python3` editable 加载 `/workspace/triton`。
- [x] 验证 native compiler 路径为同一工作树的 `_C/libtriton.so`。
- [x] 在当前 build tree 执行 `make`，确认链接输出直接覆盖上述 `_C/libtriton.so`。
- [x] 从 Pi0.5 目录用 fresh cache 在 NVIDIA Thor（SM110）运行最小 Triton kernel，结果通过。
- [x] 审查 Triton 2CTA 开发分支与 CUTLASS/CuTe Blackwell schedule。
- [ ] 网络可用后补齐 partial-clone 对象，逐文件复核目标提交 diff。
- [ ] 执行 Phase 0 的重编、独立 cache 和 compiler-only 基线。
- [ ] 从 Res FFN-down 开始 Phase 1 最小自动 2CTA pass。
