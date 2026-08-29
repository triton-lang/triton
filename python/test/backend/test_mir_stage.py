import triton
import triton.language as tl

import pytest
import torch


def is_hip():
    try:
        return triton.runtime.driver.active.get_current_target().backend == "hip"
    except Exception:
        return False


# This applies to ALL tests in this file
pytestmark = pytest.mark.skipif(not is_hip(), reason="MIR tests require AMD/HIP backend")


def verify_mir_content(mir_content, kernel_name):
    import re

    # Verify basic MIR format
    assert len(mir_content) > 0, f"MIR for {kernel_name} should not be empty"
    assert mir_content.strip().startswith("---"), f"MIR for {kernel_name} should start with YAML document marker"
    assert "name:" in mir_content, f"MIR for {kernel_name} should contain function names"
    assert "body:" in mir_content, f"MIR for {kernel_name} should contain machine basic blocks"

    # The scheduling DAG is emitted after this marker as a (bb, position)-keyed
    # edge list: `region <bb>` / `node <pos> <def-reg|-> <opcode>` /
    # `edge <src-pos> <dst-pos> <Type> <latency> [<reg>]`.
    assert "========== SCHEDULING DAG ==========" in mir_content, \
        f"MIR for {kernel_name} should contain the scheduling DAG section"
    dag_section = mir_content.split("========== SCHEDULING DAG ==========", 1)[1]

    assert re.search(r'^region \d+$', dag_section, re.MULTILINE), \
        f"Scheduling DAG for {kernel_name} should contain region headers"
    assert re.search(r'^node \d+ \S+ \S+', dag_section, re.MULTILINE), \
        f"Scheduling DAG for {kernel_name} should contain node entries"
    edges = re.findall(r'^edge \d+ \d+ (Data|Anti|Output|Memory|Barrier)\b', dag_section, re.MULTILINE)
    assert len(edges) > 0, \
        f"Scheduling DAG for {kernel_name} should contain typed dependency edges"

    # Artificial/Cluster edges are dropped (not real ordering constraints).
    assert "Artificial" not in dag_section and "Cluster" not in dag_section, \
        f"Scheduling DAG for {kernel_name} should not contain Artificial/Cluster edges"

    # The DAG is built from the pre-RA MIR, so no post-RA physical-reg renaming.
    assert "renamable" not in mir_content, \
        f"MIR for {kernel_name} should not contain entries from post-RA scheduler"


# --- DAG cross-check against LLVM's own machine-scheduler DAG -----------------
#
# Our emitted DAG and LLVM's `-misched-print-dags` output are both built by the
# same ScheduleDAGInstrs machinery, so for the same MIR their dependency edges
# must agree -- modulo edges we deliberately drop (Artificial/Cluster/Weak hint
# edges; memory edges removed by alias analysis; barrier edges removed by
# S_WAITCNT counter semantics).
#
# Two structural differences make a naive region/position comparison impossible,
# so we key instructions by (opcode, ordinal-within-function) instead -- a
# parse-invariant identity that survives both virtual-register renumbering and
# region re-partitioning:
#
#  1. Region decomposition differs. We emit one DAG per basic block; LLVM's
#     scheduler splits each block into sub-regions at scheduling boundaries
#     (calls, terminators, isSchedulingBoundary instrs). So SU indices and our
#     (bb, pos) indices do not correspond.
#  2. Boundary/terminator instructions. LLVM excludes them from its DAGs; our
#     per-block DAG includes them. Any edge touching such an instruction exists
#     only on our side and is excluded from the subset check.

def _find_llc():
    import os
    import shutil
    for cand in (
        os.path.join(os.environ.get("LLVM_SYSPATH", ""), "bin", "llc"),
        os.path.join(os.environ.get("LLVM_BUILD_DIR", ""), "bin", "llc"),
        shutil.which("llc") or "",
    ):
        if cand and os.path.isfile(cand):
            return cand
    return None


def _our_edges_by_opkey(dag_section):
    """Parse our DAG into a set of (src_key, dst_key, type) edges, where each key
    is (opcode, ordinal) assigned in whole-file node order -- matching how the
    LLVM side keys instructions, so the two are directly comparable."""
    import re
    from collections import Counter
    seen = Counter()
    key = {}          # (region, pos) -> (opcode, ordinal)
    pending_edges = []  # (region, src_pos, dst_pos, type)
    region_nodes = {}   # region -> {pos: opcode}
    cur = None
    for line in dag_section.split("\n"):
        m = re.match(r'^region (\d+)', line)
        if m:
            cur = int(m.group(1))
            region_nodes[cur] = {}
            continue
        m = re.match(r'^node (\d+) \S+ (\S+)', line)
        if m and cur is not None:
            pos, op = int(m.group(1)), m.group(2)
            region_nodes[cur][pos] = op
            key[(cur, pos)] = (op, seen[op])
            seen[op] += 1
            continue
        m = re.match(r'^edge (\d+) (\d+) (\S+)', line)
        if m and cur is not None:
            pending_edges.append((cur, int(m.group(1)), int(m.group(2)),
                                  m.group(3)))
    edges = set()
    for region, s, d, t in pending_edges:
        sk, dk = key.get((region, s)), key.get((region, d))
        if sk is None or dk is None:
            continue
        edges.add((sk, dk, t))
    return edges, region_nodes


def _llvm_edges_by_opkey(text):
    """Parse `-misched-print-dags` into a set of (src_key, dst_key, type) edges,
    keyed by (opcode, ordinal) in whole-output SU order. Types are mapped to our
    vocabulary; hint edges become 'FILTERED'."""
    import re
    from collections import Counter

    FLAGS = {"early-clobber", "dead", "undef", "renamable", "internal", "nuw",
             "nsw", "disjoint", "exact", "nneg", "nofpexcept", "reassoc",
             "nnan", "ninf", "nsz", "arcp", "contract", "afn", "killed"}

    def opcode(mir_text):
        t = mir_text.split("::", 1)[0]
        rhs = t.split("=", 1)[1].strip() if "=" in t else t.strip()
        for tok in rhs.split():
            if tok in FLAGS or tok.startswith(("%", "$")):
                continue
            return tok
        return "?"

    # First pass: assign (opcode, ordinal) to each SU, per region (SU(0) resets),
    # in whole-output order.
    seen = Counter()
    su_key = []      # list of dicts: region_index -> {su: key}
    cur = None
    pending = []     # (region_index, src_su, dst_su, type)
    ri = -1
    cur_su = None
    in_succ = False  # only edges under "Successors:" define src->dst
    for line in text.split("\n"):
        m = re.match(r'^SU\((\d+)\):\s*(.*)$', line)
        if m:
            su = int(m.group(1))
            op = opcode(m.group(2))
            if su == 0:
                ri += 1
                su_key.append({})
            cur_su = su
            in_succ = False
            su_key[ri][su] = (op, seen[op])
            seen[op] += 1
            continue
        # Track Predecessors:/Successors: sections; the same edge is listed under
        # both (mirrored), so only count it once, from the Successors side.
        if re.match(r'^\s+Predecessors:', line):
            in_succ = False
            continue
        if re.match(r'^\s+Successors:', line):
            in_succ = True
            continue
        m = re.match(r'^\s+SU\((\d+)\): (Data|Anti|Out|Ord)\b\s*(.*)$', line)
        if m and ri >= 0 and cur_su is not None and in_succ:
            dst, kind, rest = int(m.group(1)), m.group(2), m.group(3)
            if kind == "Data":
                t = "Data"
            elif kind == "Anti":
                t = "Anti"
            elif kind == "Out":
                t = "Output"
            elif "Barrier" in rest:
                t = "Barrier"
            elif "Memory" in rest:
                t = "Memory"
            else:
                t = "FILTERED"
            pending.append((ri, cur_su, dst, t))

    edges = set()
    for r, s, d, t in pending:
        sk, dk = su_key[r].get(s), su_key[r].get(d)
        if sk is None or dk is None:  # e.g. ExitSU
            continue
        edges.add((sk, dk, t))
    allkeys = {k for region in su_key for k in region.values()}
    return edges, allkeys


def _our_opkeys(dag_section):
    """The set of (opcode, ordinal) keys for all nodes in our DAG, assigned in
    whole-file node order (same scheme as _our_edges_by_opkey)."""
    import re
    from collections import Counter
    seen = Counter()
    keys = set()
    for line in dag_section.split("\n"):
        m = re.match(r'^node (\d+) \S+ (\S+)', line)
        if m:
            op = m.group(2)
            keys.add((op, seen[op]))
            seen[op] += 1
    return keys


def _cross_check_dag_against_llvm(mir_content, llc_path):
    import subprocess
    import tempfile
    import os

    mir_only, _, dag_section = mir_content.partition(
        "========== SCHEDULING DAG ==========")
    assert dag_section, "dump should contain a scheduling DAG section"

    mir_only = mir_only.rstrip()
    if mir_only.endswith("---"):
        mir_only = mir_only[:-3].rstrip()
    if mir_only.endswith("..."):
        mir_only = mir_only[:-3].rstrip()

    arch = triton.runtime.driver.active.get_current_target().arch
    with tempfile.NamedTemporaryFile("w", suffix=".mir", delete=False) as f:
        f.write(mir_only)
        mir_path = f.name
    try:
        proc = subprocess.run(
            [llc_path, "-mtriple=amdgcn-amd-amdhsa", f"-mcpu={arch}",
             "-start-before=machine-scheduler", "-stop-after=machine-scheduler",
             "-misched-print-dags", mir_path, "-o", os.devnull],
            capture_output=True, text=True, timeout=120)
    finally:
        os.unlink(mir_path)
    assert proc.returncode == 0, f"llc failed: {proc.stderr[:800]}"

    our_edges, our_nodes = _our_edges_by_opkey(dag_section)
    llvm_edges, llvm_keys = _llvm_edges_by_opkey(proc.stderr)
    assert our_edges, "our DAG produced no edges"
    assert llvm_edges, "llc produced no DAG edges"

    # LLVM's scheduling regions end at the first terminator / scheduling boundary,
    # so a suffix of each block's instructions never appears as an SU. Any opkey we
    # emitted that LLVM never emitted is such a boundary/terminator instruction;
    # edges touching one exist only on our side and are excluded from the check.
    # (This is self-calibrating -- no hardcoded opcode list needed.)
    boundary_keys = _our_opkeys(dag_section) - llvm_keys

    def touches_boundary(edge):
        sk, dk, _ = edge
        return sk in boundary_keys or dk in boundary_keys

    KEPT = {"Data", "Anti", "Output", "Memory", "Barrier"}
    llvm_kept = {e for e in llvm_edges if e[2] in KEPT}

    # 1. Every real (non-boundary) edge we emit must exist in LLVM's DAG.
    missing = {e for e in our_edges - llvm_kept if not touches_boundary(e)}
    assert not missing, \
        f"edges in our DAG but not LLVM's: {sorted(missing)[:10]}"

    # 2. Every LLVM Data/Anti/Output edge must be one we emit (those are never
    #    filtered). Missing Memory/Barrier edges are allowed (AA / waitcnt
    #    filtering).
    for e in llvm_kept - our_edges:
        _, _, t = e
        if t in ("Memory", "Barrier"):
            continue
        assert False, \
            f"LLVM has a {t} edge {e} we did not emit (regdeps must match)"


def test_dag_matches_llvm(tmp_path, monkeypatch):
    llc_path = _find_llc()
    if llc_path is None:
        pytest.skip("llc not found (set LLVM_SYSPATH or LLVM_BUILD_DIR); "
                    "DAG cross-check needs the LLVM tools")

    monkeypatch.setenv("TRITON_DUMP_MIR", str(tmp_path))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    @triton.jit
    def cc_kernel(a_ptr, b_ptr, c_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        c = tl.load(c_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, a * b + c, mask=mask)

    # Compile only (no kernel launch): the MIR + scheduling DAG are dumped at
    # compile time, so this cross-check does not depend on a working GPU runtime.
    from triton.compiler import ASTSource
    src = ASTSource(
        fn=cc_kernel,
        signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
                   "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 256})
    triton.compile(src)

    mir_files = list(tmp_path.glob("cc_kernel_*.txt"))
    assert len(mir_files) == 1, "exactly one MIR file should have been dumped"
    _cross_check_dag_against_llvm(mir_files[0].read_text(), llc_path)


def test_mir_dump_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_DUMP_MIR", str(tmp_path))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    @triton.jit
    def mul_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x * y
        tl.store(output_ptr + offsets, output, mask=mask)

    # Run kernel
    size = 128
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=128)

    # Verify kernel executed correctly
    expected = x + y
    torch.testing.assert_close(output, expected)

    # Run mul kernel
    output_mul = torch.empty_like(x)
    mul_kernel[grid](x, y, output_mul, size, BLOCK_SIZE=128)

    # Verify mul kernel executed correctly
    expected_mul = x * y
    torch.testing.assert_close(output_mul, expected_mul)

    # Check that both kernels generated separate MIR files
    add_mir_files = list(tmp_path.glob("add_kernel_*.txt"))
    mul_mir_files = list(tmp_path.glob("mul_kernel_*.txt"))

    assert len(add_mir_files) == 1, "Exactly one MIR file should exist for add_kernel"
    assert len(mul_mir_files) == 1, "Exactly one MIR file should exist for mul_kernel"

    add_mir_path = add_mir_files[0]
    mul_mir_path = mul_mir_files[0]

    # Verify add_kernel MIR content
    add_mir_content = add_mir_path.read_text()
    verify_mir_content(add_mir_content, "add_kernel")

    # Verify mul_kernel MIR content
    mul_mir_content = mul_mir_path.read_text()
    verify_mir_content(mul_mir_content, "mul_kernel")


def test_mir_swap_pipeline(tmp_path, monkeypatch):
    # First, dump a MIR file to use for swapping
    monkeypatch.setenv("TRITON_DUMP_MIR", str(tmp_path))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    @triton.jit
    def copy_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        # Simple copy operation
        tl.store(output_ptr + offsets, x, mask=mask)

    # Run kernel once to generate MIR file
    size = 128
    x = torch.randn(size, device='cuda')
    output1 = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )
    copy_kernel[grid](x, output1, size, BLOCK_SIZE=128)

    # Verify first execution
    torch.testing.assert_close(output1, x)

    # Find the generated MIR file
    mir_files = list(tmp_path.glob("copy_kernel_*.txt"))
    assert len(mir_files) == 1, "Exactly one MIR file should have been dumped"

    original_mir_path = mir_files[0]
    mir_content = original_mir_path.read_text()
    verify_mir_content(mir_content, "copy_kernel")

    # Now test MIR swapping
    monkeypatch.setenv("TRITON_SWAP_MIR", str(tmp_path))
    # Remove TRITON_DUMP_MIR to test pure swap functionality
    monkeypatch.delenv("TRITON_DUMP_MIR", raising=False)
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    # Run kernel with MIR swap
    output2 = torch.empty_like(x)
    copy_kernel[grid](x, output2, size, BLOCK_SIZE=128)

    torch.testing.assert_close(output2, x)


_SIMPLE_KERNEL_SCRIPT = '''
import triton
import triton.language as tl
import torch

@triton.jit
def simple_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)

size = 128
x = torch.randn(size, device='cuda')
output = torch.empty_like(x)
grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )
simple_kernel[grid](x, output, size, BLOCK_SIZE=128)
'''


def test_mir_swap_pipeline_passes(tmp_path):
    """Test that MIR swap pipeline starts before machine-scheduler and disables schedulers."""
    import re
    import os
    import subprocess

    script_file = tmp_path / "test_kernel.py"
    script_file.write_text(_SIMPLE_KERNEL_SCRIPT)

    # Phase 1: Dump MIR
    env = os.environ.copy()
    env["TRITON_DUMP_MIR"] = str(tmp_path)
    env["TRITON_ALWAYS_COMPILE"] = "1"

    result = subprocess.run(["python", str(script_file)], capture_output=True, text=True, env=env, timeout=120)

    assert result.returncode == 0, \
        f"Dump phase should succeed. stderr: {result.stderr[:1000]}"

    # Verify MIR file was created
    mir_files = list(tmp_path.glob("simple_kernel_*.txt"))
    assert len(mir_files) == 1, "Exactly one MIR file should have been dumped"

    # Strip scheduling DAG and trailing "..." from MIR file (they break YAML parsing)
    mir_file = mir_files[0]
    mir_content = mir_file.read_text()
    dag_marker = "\n---\n=========="
    if dag_marker in mir_content:
        mir_content = mir_content.split(dag_marker)[0]
    # Remove trailing "..." which LLVM MIR parser doesn't accept
    if mir_content.rstrip().endswith("..."):
        mir_content = mir_content.rstrip()[:-3]
    mir_file.write_text(mir_content)

    # Phase 2: Swap MIR with LLVM_IR_ENABLE_DUMP to capture pass sequence
    env = os.environ.copy()
    env["TRITON_SWAP_MIR"] = str(tmp_path)
    env["TRITON_ALWAYS_COMPILE"] = "1"
    env["LLVM_IR_ENABLE_DUMP"] = "1"

    result = subprocess.run(["python", str(script_file)], capture_output=True, text=True, env=env, timeout=120)

    assert result.returncode == 0, \
        f"Swap phase should succeed. stderr: {result.stderr[:1000]}"

    all_output = result.stderr

    # Find the first "# Machine code for function" line and check the preceding IR Dump
    lines = all_output.split('\n')
    machine_code_indices = [i for i, line in enumerate(lines) if "# Machine code for function" in line]
    assert len(machine_code_indices) > 0, \
        f"Should find '# Machine code for function' in output. Stderr length: {len(all_output)}"

    first_machine_code_idx = machine_code_indices[0]

    # Find the immediately preceding "IR Dump After" line
    ir_dump_pattern = r"# \*\*\* IR Dump After (.+) \*\*\*"
    preceding_ir_dump = None
    for i in range(first_machine_code_idx - 1, -1, -1):
        match = re.search(ir_dump_pattern, lines[i])
        if match:
            preceding_ir_dump = match.group(1).strip()
            break

    assert preceding_ir_dump is not None, \
        f"Should find 'IR Dump After' before first Machine code. Lines before: {lines[max(0, first_machine_code_idx-10):first_machine_code_idx]}"

    assert "slotindexes" in preceding_ir_dump.lower() or "slot index" in preceding_ir_dump.lower(), \
        f"First MIR pass should be slotindexes, got: '{preceding_ir_dump}'"

    # Verify machine-scheduler pass does NOT modify MIR (disabled via enable-misched=false).
    # The scheduler passes still appear in the pipeline output but return early without
    # making changes when enable-misched=false is set. This is the expected LLVM behavior -
    # we verify the MIR is unchanged rather than checking for pass absence.
    dumps = re.split(r'# \*\*\* IR Dump After ([^*]+) \*\*\*', all_output)

    machine_sched_idx = None
    for i, part in enumerate(dumps):
        if 'Machine Instruction Scheduler' in part and 'PostRA' not in part:
            machine_sched_idx = i
            break

    if machine_sched_idx and machine_sched_idx >= 1 and machine_sched_idx + 1 < len(dumps):
        before_content = dumps[machine_sched_idx - 1]
        after_content = dumps[machine_sched_idx + 1]

        # Extract machine code sections
        def extract_machine_code(text):
            match = re.search(r'# Machine code for function.*', text, re.DOTALL)
            return match.group(0).strip() if match else text.strip()

        before_mc = extract_machine_code(before_content)
        after_mc = extract_machine_code(after_content)

        assert before_mc == after_mc, \
            "machine-scheduler should not modify MIR when disabled, but MIR changed"

    # Verify post-RA machine scheduler does NOT modify MIR (disabled via enable-post-misched=false).
    # Same as above - the pass appears but returns early without changes.
    post_ra_idx = None
    for i, part in enumerate(dumps):
        if 'PostRA Machine Instruction Scheduler' in part:
            post_ra_idx = i
            break

    if post_ra_idx and post_ra_idx >= 1 and post_ra_idx + 1 < len(dumps):
        before_content = dumps[post_ra_idx - 1]
        after_content = dumps[post_ra_idx + 1]

        def extract_machine_code(text):
            match = re.search(r'# Machine code for function.*', text, re.DOTALL)
            return match.group(0).strip() if match else text.strip()

        before_mc = extract_machine_code(before_content)
        after_mc = extract_machine_code(after_content)

        assert before_mc == after_mc, \
            "post-RA scheduler should not modify MIR when disabled, but MIR changed"


def _dump_and_prepare_mir(tmp_path, script_file):
    """Dump MIR for a kernel script and strip it for swapping. Returns the cleaned MIR file path."""
    import os
    import subprocess

    env = os.environ.copy()
    env["TRITON_DUMP_MIR"] = str(tmp_path)
    env["TRITON_ALWAYS_COMPILE"] = "1"

    result = subprocess.run(["python", str(script_file)], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, \
        f"Dump phase should succeed. stderr: {result.stderr[:1000]}"

    mir_files = list(tmp_path.glob("complex_kernel_*.txt"))
    assert len(mir_files) == 1, "Exactly one MIR file should have been dumped"

    mir_file = mir_files[0]
    mir_content = mir_file.read_text()
    dag_marker = "\n---\n=========="
    if dag_marker in mir_content:
        mir_content = mir_content.split(dag_marker)[0]
    if mir_content.rstrip().endswith("..."):
        mir_content = mir_content.rstrip()[:-3]
    mir_file.write_text(mir_content)
    return mir_file


def _swap_mir_and_get_output(tmp_path, script_file, enable_misched):
    """Swap MIR with LLVM_IR_ENABLE_DUMP and return stderr output."""
    import os
    import subprocess

    env = os.environ.copy()
    env["TRITON_SWAP_MIR"] = str(tmp_path)
    env["TRITON_ALWAYS_COMPILE"] = "1"
    env["LLVM_IR_ENABLE_DUMP"] = "1"
    if enable_misched:
        env["TRITON_SWAP_MIR_ENABLE_MISCHED"] = "1"

    result = subprocess.run(["python", str(script_file)], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, \
        f"Swap phase (misched={'enabled' if enable_misched else 'disabled'}) should succeed. stderr: {result.stderr[:1000]}"
    return result.stderr


def _extract_mc_around_sched(output_text):
    """Extract machine code before and after the Machine Instruction Scheduler pass."""
    import re

    dumps = re.split(r'# \*\*\* IR Dump After ([^*]+) \*\*\*', output_text)

    machine_sched_idx = None
    for i, part in enumerate(dumps):
        if 'Machine Instruction Scheduler' in part and 'PostRA' not in part:
            machine_sched_idx = i
            break

    if machine_sched_idx is None or machine_sched_idx < 1 or machine_sched_idx + 1 >= len(dumps):
        return None, None

    def extract_machine_code(text):
        match = re.search(r'# Machine code for function.*', text, re.DOTALL)
        return match.group(0).strip() if match else text.strip()

    before_mc = extract_machine_code(dumps[machine_sched_idx - 1])
    after_mc = extract_machine_code(dumps[machine_sched_idx + 1])
    return before_mc, after_mc


# Kernel script with enough independent operations for the scheduler to reorder
_COMPLEX_KERNEL_SCRIPT = '''
import triton
import triton.language as tl
import torch

@triton.jit
def complex_kernel(a_ptr, b_ptr, c_ptr, d_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Multiple independent loads
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    # Independent arithmetic chains
    ab = a * b + c
    cd = c * d + a
    bd = b + d
    ac = a - c
    # Merge results
    result = ab * cd + bd * ac
    tl.store(output_ptr + offsets, result, mask=mask)

size = 1024
a = torch.randn(size, device='cuda')
b = torch.randn(size, device='cuda')
c = torch.randn(size, device='cuda')
d = torch.randn(size, device='cuda')
output = torch.empty_like(a)
grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )
complex_kernel[grid](a, b, c, d, output, size, BLOCK_SIZE=256)

expected = (a * b + c) * (c * d + a) + (b + d) * (a - c)
torch.testing.assert_close(output, expected)
'''


def test_mir_swap_enable_misched(tmp_path):
    """Test that TRITON_SWAP_MIR_ENABLE_MISCHED=1 causes the machine scheduler to actually modify MIR."""
    script_file = tmp_path / "test_kernel.py"
    script_file.write_text(_COMPLEX_KERNEL_SCRIPT)

    # Phase 1: Dump and prepare MIR
    _dump_and_prepare_mir(tmp_path, script_file)

    # Phase 2: Swap with misched DISABLED (default) — scheduler should be a no-op
    disabled_output = _swap_mir_and_get_output(tmp_path, script_file, enable_misched=False)
    before_disabled, after_disabled = _extract_mc_around_sched(disabled_output)

    assert before_disabled is not None and after_disabled is not None, \
        "Should find machine code around scheduler pass (disabled case)"
    assert before_disabled == after_disabled, \
        "Scheduler should NOT modify MIR when misched is disabled"

    # Phase 3: Swap with misched ENABLED — scheduler should actually reschedule
    enabled_output = _swap_mir_and_get_output(tmp_path, script_file, enable_misched=True)
    before_enabled, after_enabled = _extract_mc_around_sched(enabled_output)

    assert before_enabled is not None and after_enabled is not None, \
        "Should find machine code around scheduler pass (enabled case)"
    assert before_enabled != after_enabled, \
        "Scheduler SHOULD modify MIR when misched is enabled"


def test_mir_swap_enable_misched_requires_swap_mir(tmp_path):
    """Test that TRITON_SWAP_MIR_ENABLE_MISCHED raises an error without TRITON_SWAP_MIR."""
    import os
    import subprocess

    script_file = tmp_path / "test_kernel.py"
    script_file.write_text(_SIMPLE_KERNEL_SCRIPT)

    env = os.environ.copy()
    env["TRITON_SWAP_MIR_ENABLE_MISCHED"] = "1"
    env["TRITON_ALWAYS_COMPILE"] = "1"
    # TRITON_SWAP_MIR is NOT set

    result = subprocess.run(["python", str(script_file)], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode != 0
    assert "TRITON_SWAP_MIR_ENABLE_MISCHED requires TRITON_SWAP_MIR" in result.stderr
