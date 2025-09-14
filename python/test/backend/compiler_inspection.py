import os
import inspect

from importlib.util import spec_from_file_location, module_from_spec
import sys

from triton import knobs
from triton.backends.compiler import Language

def inspect_stages(self, stages, options, language, capability):
    if os.getenv('TRITON_MOCK_INSPECT_STAGES', '0') != '0':
        source_code = "# This is generated from Triton add_stages_inspection_hook"
        full_name = os.path.join(knobs.cache.dump_dir, "inspect_stages.py")
        with open(full_name, "w") as file:
            file.write(source_code)
    if os.getenv('TRITON_MOCK_OVERRIDE_STAGES', '0') != '0':
        make_lambda = lambda f: lambda src, metadata: f(src, metadata, options, capability)
        stages["ttir"] = make_lambda(self.make_ttir)
        stages["ttgir"] = make_lambda(self.make_ttgir)

def init():
    do_mock_inspect = os.getenv('TRITON_MOCK_INSPECT_STAGES', '0') != '0'
    do_mock_override = os.getenv('TRITON_MOCK_OVERRIDE_STAGES', '0') != '0'
    if do_mock_inspect or do_mock_override:
        knobs.runtime.add_stages_inspection_hook = inspect_stages

init()
