"""Unit tests for the LLVM IR to MSL translator.

These tests exercise the LLVMIRToMSLTranslator class directly without
requiring a full Triton build. They verify that the translator correctly
converts LLVM IR operations to valid Metal Shading Language constructs.
"""

import re
import struct
import sys
import os

import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason="Metal requires macOS")

# Load the translator class from compiler.py without importing triton
_compiler_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'compiler.py')
with open(_compiler_path) as _f:
    _content = _f.read()
_start = _content.index('class LLVMIRToMSLTranslator:')
_end = _content.index('\n@dataclass(frozen=True)')
_ns = {'re': re, 'struct': struct}
exec(_content[_start:_end], _ns)
LLVMIRToMSLTranslator = _ns['LLVMIRToMSLTranslator']


class _FakeOpt:
    num_warps = 4
    warp_size = 32


def _translate(llvm_ir, shared=0, num_warps=4):
    """Helper: translate LLVM IR to MSL."""
    class Opt:
        pass
    opt = Opt()
    opt.num_warps = num_warps
    opt.warp_size = 32
    metadata = {'name': 'test_kernel', 'shared': shared}
    t = LLVMIRToMSLTranslator(llvm_ir, metadata, opt, 8)
    return t.translate(), t


class TestKernelSignature:
    """Test kernel signature extraction and buffer binding generation."""

    def test_empty_kernel(self):
        ir = 'define void @empty_kernel() {\n  ret void\n}\n'
        msl, t = _translate(ir)
        assert '[[kernel' in msl
        assert 'void empty_kernel(' in msl
        assert t.kernel_name == 'empty_kernel'

    def test_pointer_args_get_buffer_bindings(self):
        ir = 'define void @k(ptr %a, ptr %b, ptr %c) {\n  ret void\n}\n'
        msl, _ = _translate(ir)
        assert '[[buffer(0)]]' in msl
        assert '[[buffer(1)]]' in msl
        assert '[[buffer(2)]]' in msl

    def test_pointer_becomes_device_ptr(self):
        ir = 'define void @k(ptr %input) {\n  ret void\n}\n'
        msl, _ = _translate(ir)
        assert 'device float*' in msl

    def test_scalar_becomes_constant_ref(self):
        ir = 'define void @k(i32 %n) {\n  ret void\n}\n'
        msl, _ = _translate(ir)
        assert 'constant int32_t&' in msl

    def test_mixed_args(self):
        ir = 'define void @k(ptr %buf, i32 %size, float %scale) {\n  ret void\n}\n'
        msl, _ = _translate(ir)
        assert 'device float*' in msl
        assert 'int32_t' in msl
        assert 'float' in msl
        assert msl.count('[[buffer(') == 3

    def test_max_threads_attribute(self):
        ir = 'define void @k() {\n  ret void\n}\n'
        msl, _ = _translate(ir, num_warps=8)
        assert 'max_total_threads_per_threadgroup(256)' in msl

    def test_thread_position_attributes(self):
        ir = 'define void @k() {\n  ret void\n}\n'
        msl, _ = _translate(ir)
        assert '[[thread_position_in_grid]]' in msl
        assert '[[thread_position_in_threadgroup]]' in msl
        assert '[[threadgroup_position_in_grid]]' in msl
        assert '[[thread_index_in_simdgroup]]' in msl
        assert '[[simdgroup_index_in_threadgroup]]' in msl


class TestArithmeticOps:
    """Test arithmetic operation translation."""

    def test_integer_add(self):
        ir = '''define void @k(i32 %a, i32 %b) {
  %0 = add i32 %a, %b
  ret void
}'''
        msl, _ = _translate(ir)
        assert '+' in msl

    def test_integer_mul(self):
        ir = '''define void @k(i32 %a, i32 %b) {
  %0 = mul i32 %a, %b
  ret void
}'''
        msl, _ = _translate(ir)
        assert '*' in msl

    def test_float_add(self):
        ir = '''define void @k(float %a, float %b) {
  %0 = fadd float %a, %b
  ret void
}'''
        msl, _ = _translate(ir)
        assert '+' in msl

    def test_float_mul(self):
        ir = '''define void @k(float %a, float %b) {
  %0 = fmul float %a, %b
  ret void
}'''
        msl, _ = _translate(ir)
        assert '*' in msl

    def test_shift_ops(self):
        ir = '''define void @k(i32 %a) {
  %0 = shl i32 %a, 2
  %1 = lshr i32 %a, 1
  ret void
}'''
        msl, _ = _translate(ir)
        assert '<<' in msl
        assert '>>' in msl

    def test_bitwise_ops(self):
        ir = '''define void @k(i32 %a, i32 %b) {
  %0 = and i32 %a, %b
  %1 = or i32 %a, %b
  %2 = xor i32 %a, %b
  ret void
}'''
        msl, _ = _translate(ir)
        assert '&' in msl
        assert '|' in msl
        assert '^' in msl


class TestMemoryOps:
    """Test memory operation translation."""

    def test_load(self):
        ir = '''define void @k(ptr %p) {
  %0 = load float, ptr %p, align 4
  ret void
}'''
        msl, _ = _translate(ir)
        assert '*' in msl

    def test_store(self):
        ir = '''define void @k(ptr %p, float %v) {
  store float %v, ptr %p, align 4
  ret void
}'''
        msl, _ = _translate(ir)
        assert '*' in msl
        assert '=' in msl

    def test_gep(self):
        ir = '''define void @k(ptr %p, i32 %idx) {
  %0 = getelementptr inbounds float, ptr %p, i32 %idx
  ret void
}'''
        msl, _ = _translate(ir)
        assert '+' in msl

    def test_threadgroup_memory(self):
        ir = 'define void @k() {\n  ret void\n}\n'
        msl, _ = _translate(ir, shared=2048)
        assert 'threadgroup float shared_mem[512]' in msl


class TestControlFlow:
    """Test control flow translation."""

    def test_conditional_branch(self):
        ir = '''define void @k(i1 %cond) {
  br i1 %cond, label %then, label %else
then:
  ret void
else:
  ret void
}'''
        msl, _ = _translate(ir)
        assert 'goto' in msl or 'if' in msl

    def test_unconditional_branch(self):
        ir = '''define void @k() {
  br label %end
end:
  ret void
}'''
        msl, _ = _translate(ir)
        assert 'goto' in msl

    def test_select(self):
        ir = '''define void @k(i1 %c, i32 %a, i32 %b) {
  %0 = select i1 %c, i32 %a, i32 %b
  ret void
}'''
        msl, _ = _translate(ir)
        assert '?' in msl


class TestComparisons:
    """Test comparison instruction translation."""

    def test_icmp_eq(self):
        ir = '''define void @k(i32 %a, i32 %b) {
  %0 = icmp eq i32 %a, %b
  ret void
}'''
        msl, _ = _translate(ir)
        assert '==' in msl

    def test_icmp_slt(self):
        ir = '''define void @k(i32 %a, i32 %b) {
  %0 = icmp slt i32 %a, %b
  ret void
}'''
        msl, _ = _translate(ir)
        assert '<' in msl

    def test_fcmp_ogt(self):
        ir = '''define void @k(float %a, float %b) {
  %0 = fcmp ogt float %a, %b
  ret void
}'''
        msl, _ = _translate(ir)
        assert '>' in msl


class TestCasts:
    """Test type cast translation."""

    def test_sext(self):
        ir = '''define void @k(i32 %a) {
  %0 = sext i32 %a to i64
  ret void
}'''
        msl, _ = _translate(ir)
        assert 'static_cast<int64_t>' in msl

    def test_trunc(self):
        ir = '''define void @k(i64 %a) {
  %0 = trunc i64 %a to i32
  ret void
}'''
        msl, _ = _translate(ir)
        assert 'static_cast<int32_t>' in msl

    def test_sitofp(self):
        ir = '''define void @k(i32 %a) {
  %0 = sitofp i32 %a to float
  ret void
}'''
        msl, _ = _translate(ir)
        assert 'static_cast<float>' in msl


class TestIntrinsics:
    """Test LLVM intrinsic to Metal stdlib mapping."""

    def test_sqrt_intrinsic(self):
        ir = '''define void @k(float %x) {
  %0 = call float @llvm.sqrt.f32(float %x)
  ret void
}'''
        msl, _ = _translate(ir)
        assert 'sqrt' in msl

    def test_fabs_intrinsic(self):
        ir = '''define void @k(float %x) {
  %0 = call float @llvm.fabs.f32(float %x)
  ret void
}'''
        msl, _ = _translate(ir)
        assert 'abs' in msl

    def test_fma_intrinsic(self):
        ir = '''define void @k(float %a, float %b, float %c) {
  %0 = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret void
}'''
        msl, _ = _translate(ir)
        assert 'fma' in msl


class TestBarriers:
    """Test barrier/fence translation."""

    def test_seq_cst_fence(self):
        ir = '''define void @k() {
  fence seq_cst
  ret void
}'''
        msl, _ = _translate(ir)
        assert 'threadgroup_barrier' in msl

    def test_acq_rel_fence(self):
        ir = '''define void @k() {
  fence acq_rel
  ret void
}'''
        msl, _ = _translate(ir)
        assert 'threadgroup_barrier' in msl

    def test_relaxed_fence(self):
        ir = '''define void @k() {
  fence monotonic
  ret void
}'''
        msl, _ = _translate(ir)
        assert 'simdgroup_barrier' in msl


class TestVectorAddKernel:
    """End-to-end test: vector addition kernel translation."""

    def test_vector_add(self):
        ir = '''
define dso_local void @triton_add(ptr %a, ptr %b, ptr %c, i32 %n) {
entry:
  %pid = call i32 @llvm.aarch64.metal.threadgroup.position.x()
  %tid = call i32 @llvm.aarch64.thread.id.in.simdgroup()
  %offset = mul i32 %pid, 128
  %idx = add i32 %offset, %tid
  %cmp = icmp slt i32 %idx, %n
  br i1 %cmp, label %body, label %end

body:
  %ext = sext i32 %idx to i64
  %pa = getelementptr inbounds float, ptr %a, i64 %ext
  %va = load float, ptr %pa, align 4
  %pb = getelementptr inbounds float, ptr %b, i64 %ext
  %vb = load float, ptr %pb, align 4
  %sum = fadd float %va, %vb
  %pc = getelementptr inbounds float, ptr %c, i64 %ext
  store float %sum, ptr %pc, align 4
  br label %end

end:
  ret void
}
'''
        msl, t = _translate(ir)
        assert t.kernel_name == 'triton_add'
        # Should have 4 buffer bindings (3 ptrs + 1 scalar)
        assert msl.count('[[buffer(') == 4
        # Should have arithmetic
        assert '+' in msl
        # Should have memory ops
        assert '*' in msl
        # Should have control flow
        assert 'goto' in msl or 'if' in msl
        # Should have return
        assert 'return' in msl
        # Should NOT have placeholder text
        assert 'placeholder' not in msl.lower()
        assert 'TODO' not in msl


class TestReductionKernel:
    """End-to-end test: reduction kernel with SIMD shuffles."""

    def test_reduction_with_shuffles(self):
        ir = '''
define void @reduce(ptr %in, ptr %out, i32 %n) {
entry:
  %pid = call i32 @llvm.aarch64.metal.threadgroup.position.x()
  %tid = call i32 @llvm.aarch64.thread.id.in.simdgroup()
  %idx = add i32 %pid, %tid
  %p = getelementptr float, ptr %in, i32 %idx
  %v = load float, ptr %p, align 4
  %s1 = call float @llvm.aarch64.simd.shuffle.xor(float %v, i32 16)
  %r1 = fadd float %v, %s1
  %s2 = call float @llvm.aarch64.simd.shuffle.xor(float %r1, i32 8)
  %r2 = fadd float %r1, %s2
  %is0 = icmp eq i32 %tid, 0
  br i1 %is0, label %store, label %done

store:
  %op = getelementptr float, ptr %out, i32 %pid
  store float %r2, ptr %op, align 4
  br label %done

done:
  ret void
}
'''
        msl, _ = _translate(ir, shared=1024)
        # Has SIMD shuffle calls
        assert 'shuffle' in msl
        # Has threadgroup memory
        assert 'threadgroup float shared_mem' in msl
        # Has reduction additions
        assert msl.count('+') >= 3
        # Has conditional
        assert '==' in msl
