# This is generated from Triton compiler.py
from triton._C.libtriton import ir, passes, llvm, nvidia
class GPUOverrideBackend:
    @staticmethod
    def make_ttir(mod, metadata, opt, capability):
        print('overrided ttir')
        pm = ir.pass_manager(mod.context)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        if capability // 10 < 9:
            passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        pm.run(mod, 'make_ttir')
        return mod
