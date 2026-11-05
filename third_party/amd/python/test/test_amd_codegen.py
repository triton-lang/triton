import pytest

from triton._C.libtriton import llvm
from triton.backends.amd import compiler


def test_fp_fusion():
    arch = "gfx1250"
    triple = compiler.amd.get_target_triple(arch)
    source = f'''
target triple = "{triple}"

define amdgpu_kernel void @fusion_kernel(ptr addrspace(1) %out, float %a, float %b, float %c) {{
  %product = fmul float %a, %b
  %sum = fadd float %product, %c
  store volatile float %sum, ptr addrspace(1) %out, align 4
  ret void
}}
'''
    options = dict(flags=[], disable_optimization=False, canonicalize_gep=False, disabled_passes="", dump_ir=False,
                   enable_timing=False)

    without_fusion = compiler.compile_amdgpu(source, triple, arch, "", enable_fp_fusion=False, **options)
    with_fusion = compiler.compile_amdgpu(source, triple, arch, "", enable_fp_fusion=True, **options)
    explicit_contract_source = source.replace("fmul float",
                                              "fmul contract float").replace("fadd float", "fadd contract float")
    with_explicit_contract = compiler.compile_amdgpu(explicit_contract_source, triple, arch, "", enable_fp_fusion=False,
                                                     **options)

    fused_opcodes = ("_fma_f32", "_fmac_f32", "_mad_f32")
    assert not any(opcode in without_fusion for opcode in fused_opcodes)
    assert any(opcode in with_fusion for opcode in fused_opcodes)
    assert any(opcode in with_explicit_contract for opcode in fused_opcodes)


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("description", [False, True])
@pytest.mark.parametrize("disjoint", [False, True])
def test_alias_scope_domains(named, description, disjoint):
    arch = "gfx950"
    triple = compiler.amd.get_target_triple(arch)
    identity = '!"domain"' if named else "!0"
    extra = ', !"description"' if description else ""
    distinct = "" if named else "distinct "
    source = f'''
target triple = "{triple}"

declare void @llvm.experimental.noalias.scope.decl(metadata)

define amdgpu_kernel void @alias_scope_kernel(ptr addrspace(1) %a, ptr addrspace(1) %b) {{
  call void @llvm.experimental.noalias.scope.decl(metadata !3)
  %value = load i32, ptr addrspace(1) %a, !alias.scope !3, !noalias !4
  store i32 %value, ptr addrspace(1) %b, !alias.scope !4, !noalias !3
  ret void
}}

!0 = {distinct}!{{{identity}, i1 {str(disjoint).lower()}{extra}}}
!1 = !{{!"a", !0}}
!2 = !{{!"b", !0}}
!3 = !{{!1}}
!4 = !{{!2}}
'''
    assembly = compiler.compile_amdgpu(source, triple, arch, "", flags=[], enable_fp_fusion=False,
                                       disable_optimization=False, canonicalize_gep=False, disabled_passes="",
                                       dump_ir=False, enable_timing=False)
    assert "alias_scope_kernel:" in assembly


@pytest.mark.parametrize("arch", ["gfx90a", "gfx942", "gfx950", "gfx1250", "gfx1250-strict"])
@pytest.mark.parametrize("enable_misched", [False, True])
def test_mir_roundtrip(arch, enable_misched, tmp_path):
    triple = compiler.amd.get_target_triple(arch)
    source = f'''
target triple = "{triple}"

declare void @llvm.amdgcn.s.barrier()

define amdgpu_kernel void @mir_kernel(ptr addrspace(1) %out, i32 %a, i32 %b) {{
  %sum = add i32 %a, %b
  call void @llvm.amdgcn.s.barrier()
  store volatile i32 %sum, ptr addrspace(1) %out
  ret void
}}
'''
    mir = compiler.translate_to_mir(source, triple, arch, "", [], False, "mir_kernel", str(tmp_path), "", False)
    assert mir.startswith("---")
    assert "S_BARRIER" in mir
    assert "S_WAIT" not in mir
    path = tmp_path / "mir_kernel.txt"
    assert compiler.MIR_DAG_MARKER in path.read_text()
    path.write_text(mir)
    assembly = compiler.translate_mir_to_asm(str(path), triple, arch, "", [], False, False, enable_misched, False)
    assert "mir_kernel:" in assembly
    assert "s_barrier" in assembly


def test_strict_target_triple():
    # has_architected_sgprs looks the triple up in core LLVM's target registry.
    llvm.init_targets()
    assert compiler.amd.get_target_triple("gfx1250-strict") == "amdgpu12.50s-amd-amdhsa"
    assert not compiler.amd.has_architected_sgprs("gfx942")
    assert compiler.amd.has_architected_sgprs("gfx1250-strict")


def test_named_barrier_address_space():
    arch = "gfx1250"
    triple = compiler.amd.get_target_triple(arch)
    source = f'''
target triple = "{triple}"

@bar = internal addrspace(15) global target("amdgcn.named.barrier", 0) poison

declare void @llvm.amdgcn.s.barrier.join(ptr addrspace(15))
declare void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(15), i32)

define amdgpu_kernel void @named_barrier_kernel() {{
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(15) @bar)
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(15) @bar, i32 4)
  ret void
}}
'''
    assembly = compiler.compile_amdgpu(source, triple, arch, "", flags=[], enable_fp_fusion=False,
                                       disable_optimization=False, canonicalize_gep=False, disabled_passes="",
                                       dump_ir=False, enable_timing=False)

    assert "s_barrier_join" in assembly
    assert "s_barrier_signal" in assembly
    assert ".amdhsa_named_barrier_count 1" in assembly


def _compile_named_barriers(body):
    arch = "gfx1250"
    triple = compiler.amd.get_target_triple(arch)
    source = f'''
target triple = "{triple}"

declare void @llvm.amdgcn.s.barrier.init(ptr addrspace(15), i32)
declare void @llvm.amdgcn.s.barrier.join(ptr addrspace(15))
declare void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(15), i32)
declare i32 @llvm.amdgcn.s.get.named.barrier.state(ptr addrspace(15))
declare void @llvm.amdgcn.s.wakeup.barrier(ptr addrspace(15))
{body}
!llvm.module.flags = !{{!0}}
!0 = !{{i32 2, !"Debug Info Version", i32 3}}
'''
    return compiler.compile_amdgpu(source, triple, arch, "", flags=[], enable_fp_fusion=False,
                                   disable_optimization=False, canonicalize_gep=False, disabled_passes="",
                                   dump_ir=False, enable_timing=False)


@pytest.mark.parametrize("forward", [False, True])
def test_named_barrier_function_argument(forward):
    forwarding = """
define internal void @forward(ptr addrspace(15) %bar, ptr addrspace(3) %shared) noinline {
  call void @join(ptr addrspace(15) %bar, ptr addrspace(3) %shared)
  ret void
}
""" if forward else ""
    callee = "forward" if forward else "join"
    assembly = _compile_named_barriers(f"""
@bar = internal addrspace(15) global target("amdgcn.named.barrier", 0) poison
@shared = internal addrspace(3) global i32 undef

define internal void @join(ptr addrspace(15) %bar, ptr addrspace(3) %shared) noinline {{
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(15) %bar)
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(15) %bar, i32 4)
  store volatile i32 7, ptr addrspace(3) %shared
  ret void
}}
{forwarding}
define amdgpu_kernel void @named_barrier_kernel() {{
  call void @{callee}(ptr addrspace(15) @bar, ptr addrspace(3) @shared)
  ret void
}}
""")

    assert "s_barrier_join" in assembly
    assert "s_barrier_signal" in assembly
    assert ".amdhsa_named_barrier_count 1" in assembly
    assert "ds_store_b32" in assembly


def test_named_barrier_intrinsics():
    assembly = _compile_named_barriers("""
@bar = internal addrspace(15) global target("amdgcn.named.barrier", 0) poison

define internal void @sync(ptr addrspace(15) %bar, ptr addrspace(1) %out) noinline {
  call void @llvm.amdgcn.s.barrier.init(ptr addrspace(15) %bar, i32 4)
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(15) %bar)
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(15) %bar, i32 4)
  call void @llvm.amdgcn.s.wakeup.barrier(ptr addrspace(15) %bar)
  %state = call i32 @llvm.amdgcn.s.get.named.barrier.state(ptr addrspace(15) %bar)
  store i32 %state, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @named_barrier_kernel(ptr addrspace(1) %out) {
  call void @sync(ptr addrspace(15) @bar, ptr addrspace(1) %out)
  ret void
}
""")

    for instruction in ("s_barrier_init", "s_barrier_join", "s_barrier_signal", "s_wakeup_barrier",
                        "s_get_barrier_state"):
        assert instruction in assembly


def test_named_barrier_ssa_forwarding():
    assembly = _compile_named_barriers("""
@bar = internal addrspace(15) global target("amdgcn.named.barrier", 0) poison
@other = internal addrspace(15) global target("amdgcn.named.barrier", 0) poison

define amdgpu_kernel void @named_barrier_kernel(i1 %condition, i32 %count) {
entry:
  %first = select i1 %condition, ptr addrspace(15) @bar, ptr addrspace(15) @other
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%i.next, %loop]
  %current = phi ptr addrspace(15) [%first, %entry], [%next, %loop]
  %frozen = freeze ptr addrspace(15) %current
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(15) %frozen)
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(15) %frozen, i32 4)
  %next = select i1 %condition, ptr addrspace(15) @other, ptr addrspace(15) %frozen
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %count
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
""")

    assert "s_barrier_join m0" in assembly
    assert "s_barrier_signal m0" in assembly
    assert ".set .Lnamed_barrier_kernel.num_named_barrier, 2" in assembly


def test_named_barrier_unused_declarations():
    assembly = _compile_named_barriers("""
define amdgpu_kernel void @named_barrier_kernel() {
  ret void
}
""")

    assert "named_barrier_kernel:" in assembly
