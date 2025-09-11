import os
import inspect

from importlib.util import spec_from_file_location, module_from_spec
import sys

from triton import knobs
from triton.backends.compiler import Language

def dump_stages(self, stages, options, language, capability):
    source_code = "# This is generated from Triton compiler.py"
    source_code = source_code + '\n' + "from triton._C.libtriton import ir, passes, llvm, amd, nvidia"
    source_code = source_code + '\n' + "class GPUOverrideBackend:"
    source_code = source_code + '\n' + inspect.getsource(self.make_ttir)
    source_code = source_code + '\n' + inspect.getsource(self.make_ttgir)
    with open("compiler_override.py", "w") as file:
        file.write(source_code)

def override_stages(self, stages, options, language, capability):
    print('override')
    # Limit to TTIR and TTGIR for now
    if language != Language.TRITON: return
    full_name = "compiler_override.py"

    print(f"\nOverriding compile pass stages with file {full_name}")
    module_name = 'triton_override_compiler_stages'
    spec = spec_from_file_location(module_name, full_name) if os.path.isfile(full_name) else None
    if not spec: return

    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, 'GPUOverrideBackend'): return
    module = getattr(module, 'GPUOverrideBackend')

    has_func = lambda mod, name: hasattr(mod, name) and callable(getattr(mod, name))
    make_lambda = lambda f: lambda src, metadata: f(src, metadata, options, capability)
    if has_func(module, "make_ttir"): stages["ttir"] = make_lambda(module.make_ttir)
    if has_func(module, "make_ttgir"): stages["ttgir"] = make_lambda(module.make_ttgir)
    # make_llir is not static, it uses self.target.arch so we don't allow overriding it
    # for now


def inspect_stages(self, stages, options, language, capability):
    print('INSPECT STAGES')
    if os.getenv('TRITON_DUMP_PASS_STAGES', '0') != '0':
        dump_stages(self, stages, options, language, capability)
    if os.getenv('TRITON_OVERRIDE_PASS_STAGES', '0') != '0':
        override_stages(self, stages, options, language, capability)


def init():
        knobs.runtime.add_stages_inspection_hook = inspect_stages

init()
