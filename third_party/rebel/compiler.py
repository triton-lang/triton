from triton.backends.compiler import BaseBackend, GPUTarget, AttrsDescriptor, register_descriptor
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

@dataclass(frozen=True)
class RebelOptions:
    debug: bool = True
    backend_name: str = 'rebel'

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

class RebelBackend(BaseBackend):
    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "rebel"

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in RebelOptions.__dataclass_fields__.keys() if k in opts}
        return RebelOptions(**args)
    
    def add_stages(self, stages: dict, options: object) -> None:
        raise NotImplementedError
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)

    def load_dialects(self, context):
        """
        Load additional MLIR dialects into the provided `context`
        """
        raise NotImplementedError
    
    @staticmethod
    def make_ttir(mod, metadata, options):
        raise NotImplementedError
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        return mod

