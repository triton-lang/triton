"""Spill-aware cost model for choosing an AMDGPU codegen configuration.

Triton pins ``amdgpu-waves-per-eu`` to ``N,N`` so the LLVM AMDGPU backend targets
a single register budget. When the machine scheduler's register pressure exceeds
that budget the allocator spills to scratch memory, and scratch traffic landing
inside a hot loop can cost far more than the occupancy the budget was protecting.
Whether that happens depends on the LLVM version, the target, and the kernel, so
it cannot be predicted before instruction selection runs.

This module therefore scores *already generated* assembly. The backend compiles
the default configuration, and only if that result spills inside a loop does it
compile a few alternatives and keep whichever the model scores cheapest. Kernels
that do not spill never pay for the search and their code is unchanged.

The model is::

    cost = loop_issue_slots / occupancy**ALPHA + SPILL_COST * loop_spill_ops

Issue slots are hidden by having several waves resident, hence the sublinear
occupancy term. Spill traffic is not hidden the same way: every resident wave
adds its own scratch accesses to a shared memory pipe, so that term is charged
per wave.

Parameters were fitted on gfx950 against 14 measured variants of the aiter
``unified_attention`` kernel (Spearman 0.97, mean absolute error 13% on absolute
time). Only the ranking is relied upon, and the ranking is stable across a wide
range of both parameters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SPILL_COST = 30.0
ALPHA = 0.5

# Required improvement before an alternative displaces the default. The model
# ranks candidates well but its absolute error is around 15%, so a candidate has
# to win by more than that to be worth taking. Without this, kernels that spill
# only lightly get recompiled on predictions that measure as noise.
MIN_GAIN = 0.15

# Alternatives to the backend's default scheduler, tried in order when the
# default configuration spills inside a loop. "iterative-minreg" minimizes
# register pressure, which is what a kernel over its register budget needs.
#
# "max-ilp" and "iterative-ilp" were evaluated too. Neither won on any kernel
# that reached the search, and each costs a full extra codegen run, so they are
# left out; adding a strategy here is all it takes to reconsider.
CANDIDATE_SCHED_STRATEGIES = ("iterative-minreg", )

_LABEL_RE = re.compile(r"^(\.?[A-Za-z_][\w.$]*):\s*(;.*)?$")
_BRANCH_RE = re.compile(r"s_(?:cbranch\w*|branch)\s+(\.?[\w.$]+)")
_SPILL_RE = re.compile(r"\bscratch_(?:load|store)")
_OCCUPANCY_RE = re.compile(r"; Occupancy:\s*(\d+)")
_SPILL_COUNT_RE = re.compile(r"\.vgpr_spill_count:\s*(\d+)")


@dataclass(frozen=True)
class AsmCost:
    occupancy: int
    loop_spill_ops: int
    loop_issue_slots: int
    total_spills: int

    @property
    def issue_cost(self) -> float:
        return self.loop_issue_slots / max(self.occupancy, 1)**ALPHA

    @property
    def spill_cost(self) -> float:
        return SPILL_COST * self.loop_spill_ops

    @property
    def cost(self) -> float:
        return self.issue_cost + self.spill_cost

    @property
    def spilling_dominates(self) -> bool:
        """Whether spill traffic outweighs all other issue slots in the loop."""
        return self.spill_cost > self.issue_cost


def _loops(lines: list[str]) -> list[tuple[int, int]]:
    """Return (start, end) line ranges for backward branches, i.e. loop bodies."""
    labels: dict[str, int] = {}
    for i, line in enumerate(lines):
        m = _LABEL_RE.match(line)
        if m:
            labels[m.group(1)] = i

    found = []
    for i, line in enumerate(lines):
        m = _BRANCH_RE.search(line)
        if not m:
            continue
        start = labels.get(m.group(1))
        if start is not None and start < i:
            found.append((start, i))
    return found


def _innermost(loops: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep only loops that do not strictly contain another loop."""
    return [
        a for a in loops
        if not any(b is not a and a[0] <= b[0] and b[1] <= a[1] and (b[1] - b[0]) < (a[1] - a[0]) for b in loops)
    ]


def analyze(asm: str) -> AsmCost:
    """Extract the cost model's inputs from generated AMDGCN assembly.

    A kernel usually has several loops; only the dominant one matters. Trip
    counts are not known statically, so the loop with the highest modelled
    per-iteration cost is taken as representative. That keeps the choice stable
    when a candidate removes spills, which would otherwise make a small setup
    loop look like the hot one.
    """
    lines = asm.split("\n")
    occupancy = _OCCUPANCY_RE.search(asm)
    total_spills = _SPILL_COUNT_RE.search(asm)
    occ = int(occupancy.group(1)) if occupancy else 1

    hot_spills, hot_slots, hot_cost = 0, len(lines), -1.0
    for start, end in _innermost(_loops(lines)):
        slots = end - start
        spills = sum(1 for line in lines[start:end + 1] if _SPILL_RE.search(line))
        cost = slots / max(occ, 1)**ALPHA + SPILL_COST * spills
        if cost > hot_cost:
            hot_spills, hot_slots, hot_cost = spills, slots, cost

    return AsmCost(
        occupancy=occ,
        loop_spill_ops=hot_spills,
        loop_issue_slots=hot_slots,
        total_spills=int(total_spills.group(1)) if total_spills else 0,
    )


def set_sched_strategy(llvm_ir: str, kernel_name: str, strategy: str) -> str | None:
    """Return `llvm_ir` with the kernel's ``amdgpu-sched-strategy`` set.

    Rewrites only the attribute group referenced by the kernel definition, so
    other functions in the module are untouched. Returns None if the kernel's
    attribute group cannot be located.
    """
    define = re.search(
        r"^define\s+amdgpu_kernel\s+void\s+@" + re.escape(kernel_name) + r"\b.*?#(\d+).*$",
        llvm_ir,
        re.M,
    )
    if not define:
        return None
    group = define.group(1)

    attr_re = re.compile(r"^attributes #" + group + r" = \{(.*)\}$", re.M)
    attrs = attr_re.search(llvm_ir)
    if not attrs:
        return None

    body = attrs.group(1)
    body = re.sub(r'\s*"amdgpu-sched-strategy"="[^"]*"', "", body)
    body = f' "amdgpu-sched-strategy"="{strategy}"' + body
    return llvm_ir[:attrs.start()] + f"attributes #{group} = {{{body}}}" + llvm_ir[attrs.end():]


def select_best(llvm_ir: str, kernel_name: str, compile_asm, default_asm: str, log=None) -> str:
    """Pick the cheapest of the default assembly and the candidate strategies.

    `compile_asm` maps LLVM IR to AMDGCN. The default is returned unchanged
    unless spilling dominates its hot loop *and* some alternative beats it by
    more than MIN_GAIN, so a kernel is never recompiled differently on a
    marginal prediction.
    """
    default = analyze(default_asm)
    if log:
        log("default", default)

    if not default.spilling_dominates:
        return default_asm

    threshold = default.cost * (1.0 - MIN_GAIN)
    best_asm, best_cost = default_asm, threshold
    for strategy in CANDIDATE_SCHED_STRATEGIES:
        patched = set_sched_strategy(llvm_ir, kernel_name, strategy)
        if patched is None:
            break
        try:
            asm = compile_asm(patched)
        except Exception:  # noqa: BLE001 - a failed candidate must never break compilation
            continue
        candidate = analyze(asm)
        if log:
            log(strategy, candidate)
        if candidate.cost < best_cost:
            best_asm, best_cost = asm, candidate.cost

    return best_asm
