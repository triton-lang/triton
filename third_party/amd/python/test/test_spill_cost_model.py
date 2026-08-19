"""Unit tests for the AMD spill-aware codegen cost model.

These operate on assembly and LLVM IR text, so they need no GPU.
"""

from triton.backends.amd import spill_cost_model as scm


def _asm(occupancy, setup_loop, hot_loop, hot_spills, total_spills=0):
    """Build an AMDGCN-shaped listing with a small setup loop and a hot loop."""
    lines = [f"\t.vgpr_spill_count: {total_spills}"]
    lines.append(".Lsetup:")
    lines += ["\tv_add_u32 v0, v1, v2"] * setup_loop
    lines.append("\ts_cbranch_scc1 .Lsetup")
    lines.append(".Lhot:")
    lines += ["\tv_fma_f32 v3, v4, v5, v6"] * (hot_loop - hot_spills)
    lines += ["\tscratch_store_dword off, v7, s0"] * hot_spills
    lines.append("\ts_cbranch_scc1 .Lhot")
    lines.append(f"; Occupancy: {occupancy}")
    return "\n".join(lines)


def test_picks_hot_loop_not_setup_loop():
    stats = scm.analyze(_asm(occupancy=2, setup_loop=5, hot_loop=900, hot_spills=0))
    assert stats.loop_issue_slots > 800
    assert stats.loop_spill_ops == 0
    assert stats.occupancy == 2


def test_counts_spills_in_hot_loop():
    stats = scm.analyze(_asm(occupancy=2, setup_loop=5, hot_loop=900, hot_spills=40))
    assert stats.loop_spill_ops == 40


def test_spilling_is_more_expensive_than_extra_issue_slots():
    spilling = scm.analyze(_asm(occupancy=2, setup_loop=5, hot_loop=950, hot_spills=33))
    clean = scm.analyze(_asm(occupancy=2, setup_loop=5, hot_loop=1010, hot_spills=0))
    assert clean.cost < spilling.cost


def test_higher_occupancy_wins_when_neither_spills():
    low = scm.analyze(_asm(occupancy=1, setup_loop=5, hot_loop=1000, hot_spills=0))
    high = scm.analyze(_asm(occupancy=2, setup_loop=5, hot_loop=1000, hot_spills=0))
    assert high.cost < low.cost


IR = """
define amdgpu_kernel void @k(ptr addrspace(1) %0) #1 {
  ret void
}
attributes #0 = { nounwind }
attributes #1 = { nounwind "amdgpu-waves-per-eu"="2,2" }
"""


def test_set_sched_strategy_targets_only_the_kernel_attribute_group():
    out = scm.set_sched_strategy(IR, "k", "iterative-minreg")
    assert '"amdgpu-sched-strategy"="iterative-minreg"' in out
    assert out.count("amdgpu-sched-strategy") == 1
    assert 'attributes #0 = { nounwind }' in out
    assert '"amdgpu-waves-per-eu"="2,2"' in out


def test_set_sched_strategy_replaces_existing_value():
    once = scm.set_sched_strategy(IR, "k", "max-ilp")
    twice = scm.set_sched_strategy(once, "k", "iterative-minreg")
    assert twice.count("amdgpu-sched-strategy") == 1
    assert '"amdgpu-sched-strategy"="iterative-minreg"' in twice


def test_unknown_kernel_name_is_ignored():
    assert scm.set_sched_strategy(IR, "other", "max-ilp") is None


def test_spilling_dominates_only_at_high_spill_density():
    # Densities taken from measured aiter kernels: unified_attention spilled
    # heavily and benefited, while mha and extend_attention barely spilled and
    # measured as noise.
    assert scm.analyze(_asm(occupancy=2, setup_loop=5, hot_loop=951, hot_spills=33)).spilling_dominates
    assert not scm.analyze(_asm(occupancy=1, setup_loop=5, hot_loop=859, hot_spills=8)).spilling_dominates
    assert not scm.analyze(_asm(occupancy=2, setup_loop=5, hot_loop=726, hot_spills=1)).spilling_dominates


def test_select_best_leaves_non_spilling_kernels_untouched():
    default = _asm(occupancy=2, setup_loop=5, hot_loop=900, hot_spills=0)
    calls = []

    def compile_asm(ir):
        calls.append(ir)
        return _asm(occupancy=2, setup_loop=5, hot_loop=100, hot_spills=0)

    assert scm.select_best(IR, "k", compile_asm, default) is default
    assert calls == [], "no candidate should be compiled when the default does not spill"


def test_select_best_does_not_search_on_light_spilling():
    default = _asm(occupancy=1, setup_loop=5, hot_loop=859, hot_spills=8)
    calls = []

    def compile_asm(ir):
        calls.append(ir)
        return _asm(occupancy=1, setup_loop=5, hot_loop=859, hot_spills=0)

    assert scm.select_best(IR, "k", compile_asm, default) is default
    assert calls == [], "light spilling measures as noise; searching only costs compile time"


def test_search_compiles_once_per_candidate():
    # Every candidate costs a full codegen run, so keep that count visible: it
    # is what a kernel passing the gate pays in compile time.
    default = _asm(occupancy=2, setup_loop=5, hot_loop=950, hot_spills=33)
    calls = []

    def compile_asm(ir):
        calls.append(ir)
        return default

    scm.select_best(IR, "k", compile_asm, default)
    assert len(calls) == len(scm.CANDIDATE_SCHED_STRATEGIES)


def test_select_best_switches_when_a_candidate_is_much_cheaper():
    default = _asm(occupancy=2, setup_loop=5, hot_loop=950, hot_spills=33)
    better = _asm(occupancy=2, setup_loop=5, hot_loop=1010, hot_spills=0)

    assert scm.select_best(IR, "k", lambda ir: better, default) is better


def test_select_best_ignores_marginal_improvements():
    default = _asm(occupancy=2, setup_loop=5, hot_loop=950, hot_spills=33)
    default_cost = scm.analyze(default).cost
    # Cheaper, but by less than MIN_GAIN, so not worth changing codegen for.
    marginal_spills = int((default_cost * 0.95 - 950 / 2**scm.ALPHA) / scm.SPILL_COST)
    marginal = _asm(occupancy=2, setup_loop=5, hot_loop=950, hot_spills=marginal_spills)
    assert scm.analyze(marginal).cost < default_cost
    assert scm.analyze(marginal).cost > default_cost * (1 - scm.MIN_GAIN)

    assert scm.select_best(IR, "k", lambda ir: marginal, default) is default


def test_select_best_keeps_default_when_candidates_are_worse():
    default = _asm(occupancy=2, setup_loop=5, hot_loop=950, hot_spills=33)
    worse = _asm(occupancy=2, setup_loop=5, hot_loop=950, hot_spills=64)

    assert scm.select_best(IR, "k", lambda ir: worse, default) is default


def test_candidate_compile_failure_does_not_break_compilation():
    default = _asm(occupancy=2, setup_loop=5, hot_loop=950, hot_spills=33)

    def boom(ir):
        raise RuntimeError("backend crashed on this candidate")

    assert scm.select_best(IR, "k", boom, default) is default
