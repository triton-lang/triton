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
