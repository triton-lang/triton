import pytest

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


def test_legacy_named_barrier_address_space():
    arch = "gfx1250"
    triple = compiler.amd.get_target_triple(arch)
    source = f'''
target triple = "{triple}"

@bar = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison

declare void @llvm.amdgcn.s.barrier.join(ptr addrspace(3))
declare void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3), i32)

define amdgpu_kernel void @named_barrier_kernel() {{
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar)
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar, i32 4)
  ret void
}}
'''
    assembly = compiler.compile_amdgpu(source, triple, arch, "", flags=[], enable_fp_fusion=False,
                                       disable_optimization=False, canonicalize_gep=False, disabled_passes="",
                                       dump_ir=False, enable_timing=False)

    assert "s_barrier_join" in assembly
    assert "s_barrier_signal" in assembly
    assert ".amdhsa_named_barrier_count 1" in assembly


def _compile_legacy_named_barriers(body):
    arch = "gfx1250"
    triple = compiler.amd.get_target_triple(arch)
    source = f'''
target triple = "{triple}"

declare void @llvm.amdgcn.s.barrier.init(ptr addrspace(3), i32)
declare void @llvm.amdgcn.s.barrier.join(ptr addrspace(3))
declare void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3), i32)
declare i32 @llvm.amdgcn.s.get.named.barrier.state(ptr addrspace(3))
declare void @llvm.amdgcn.s.wakeup.barrier(ptr addrspace(3))
{body}
!llvm.module.flags = !{{!0}}
!0 = !{{i32 2, !"Debug Info Version", i32 3}}
'''
    return compiler.compile_amdgpu(source, triple, arch, "", flags=[], enable_fp_fusion=False,
                                   disable_optimization=False, canonicalize_gep=False, disabled_passes="",
                                   dump_ir=False, enable_timing=False)


@pytest.mark.parametrize("forward", [False, True])
def test_legacy_named_barrier_function_argument(forward):
    forwarding = """
define internal void @forward(ptr addrspace(3) %bar, ptr addrspace(3) %shared) noinline {
  call void @join(ptr addrspace(3) %bar, ptr addrspace(3) %shared)
  ret void
}
""" if forward else ""
    callee = "forward" if forward else "join"
    assembly = _compile_legacy_named_barriers(f"""
@bar = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison
@shared = internal addrspace(3) global i32 undef

define internal void @join(ptr addrspace(3) %bar, ptr addrspace(3) %shared) noinline {{
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) %bar)
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) %bar, i32 4)
  store volatile i32 7, ptr addrspace(3) %shared
  ret void
}}
{forwarding}
define amdgpu_kernel void @named_barrier_kernel() {{
  call void @{callee}(ptr addrspace(3) @bar, ptr addrspace(3) @shared)
  ret void
}}
""")

    assert "s_barrier_join" in assembly
    assert "s_barrier_signal" in assembly
    assert ".amdhsa_named_barrier_count 1" in assembly
    assert "ds_store_b32" in assembly


def test_legacy_named_barrier_intrinsics():
    assembly = _compile_legacy_named_barriers("""
@bar = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison

define internal void @sync(ptr addrspace(3) %bar, ptr addrspace(1) %out) noinline {
  call void @llvm.amdgcn.s.barrier.init(ptr addrspace(3) %bar, i32 4)
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) %bar)
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) %bar, i32 4)
  call void @llvm.amdgcn.s.wakeup.barrier(ptr addrspace(3) %bar)
  %state = call i32 @llvm.amdgcn.s.get.named.barrier.state(ptr addrspace(3) %bar)
  store i32 %state, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @named_barrier_kernel(ptr addrspace(1) %out) {
  call void @sync(ptr addrspace(3) @bar, ptr addrspace(1) %out)
  ret void
}
""")

    for instruction in ("s_barrier_init", "s_barrier_join", "s_barrier_signal", "s_wakeup_barrier",
                        "s_get_barrier_state"):
        assert instruction in assembly


def test_legacy_named_barrier_ssa_forwarding():
    assembly = _compile_legacy_named_barriers("""
@bar = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison
@other = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison

define amdgpu_kernel void @named_barrier_kernel(i1 %condition, i32 %count) {
entry:
  %first = select i1 %condition, ptr addrspace(3) @bar, ptr addrspace(3) @other
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%i.next, %loop]
  %current = phi ptr addrspace(3) [%first, %entry], [%next, %loop]
  %frozen = freeze ptr addrspace(3) %current
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) %frozen)
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) %frozen, i32 4)
  %next = select i1 %condition, ptr addrspace(3) @other, ptr addrspace(3) %frozen
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


def test_legacy_named_barrier_unused_declarations():
    assembly = _compile_legacy_named_barriers("""
define amdgpu_kernel void @named_barrier_kernel() {
  ret void
}
""")

    assert "named_barrier_kernel:" in assembly


@pytest.mark.parametrize("body", [
    pytest.param(
        """
define internal ptr addrspace(3) @identity(ptr addrspace(3) %bar) noinline {
  ret ptr addrspace(3) %bar
}

define amdgpu_kernel void @named_barrier_kernel() {
  %bar = call ptr addrspace(3) @identity(ptr addrspace(3) @bar)
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) %bar)
  ret void
}
""", id="returned"),
    pytest.param(
        """
define internal void @join(ptr addrspace(3) %bar) noinline {
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) %bar)
  ret void
}

define amdgpu_kernel void @named_barrier_kernel(ptr addrspace(1) %out) {
  call void @join(ptr addrspace(3) @bar)
  store ptr @join, ptr addrspace(1) %out
  ret void
}
""", id="address_taken_helper"),
])
def test_legacy_named_barrier_unsupported_use(body):
    with pytest.raises(RuntimeError, match="unsupported use of a legacy named barrier"):
        _compile_legacy_named_barriers(
            '@bar = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison\n' + body)


def test_legacy_named_barrier_mixed_with_shared_memory():
    with pytest.raises(RuntimeError, match="invalid LLVM IR"):
        _compile_legacy_named_barriers("""
@bar = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison
@shared = internal addrspace(3) global i32 undef

define amdgpu_kernel void @named_barrier_kernel(i1 %condition) {
  %pointer = select i1 %condition, ptr addrspace(3) @bar, ptr addrspace(3) @shared
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) %pointer)
  ret void
}
""")
