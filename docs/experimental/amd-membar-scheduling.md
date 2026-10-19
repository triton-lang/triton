# AMD barrier scheduling after Membar

AMD Membar defers the local barrier following a shared-copy wait until a later
memory demand needs synchronization. If a global load is issued in between, a
use of its result may lower to a memory wait. Reaching the deferred barrier
only after that wait can delay rendezvous across warps.

The AMD Membar pass inserts barriers first, then calls `scheduleMembarBarriers`
on the resulting TTGIR. Both steps run before ConSan and LLVM lowering and
share one pass verification. Generic Membar analysis is unchanged.

The scheduler considers a wait when a direct global-read result is used before
the next local barrier. It also checks one incoming CFG edge for block arguments;
a branch transfer can become a register copy. It follows every forward path to
the first full local barrier, groups waits sharing destination barriers, and
checks backward that every incoming path reaches a wait in the group. It then
places barriers before the selected uses and removes the destination barriers.

Motion stops at shared or unknown effects, other barriers, region operations,
and waits or commits. TDM and non-TDM waits are both boundaries: the scheduler
currently does not split a barrier that may serve two different completions.
Loops or joins without complete path coverage are left unchanged. The check is
a latency heuristic, not a prediction of LLVM's final memory waits or a match
between async writes and wait groups.

The AMD tail-wait pipeliner change places synchronous LDS stores after the wait
when stores and loads occupy different stages. This allows the moved barrier
to precede those stores, avoiding an early LDS drain in FP8 GEMM.

On gfx950, blockscale GEMM (M=64, N=1280, K=8192) passed its numerical check
and measured 25.42 TFLOPS. FP8 GEMM at the same shape passed and measured a
median 43.90 TFLOPS. The selected AMD Membar and pipeline lit tests passed.
