import os
from triton import knobs



def inspect_stages(self, stages, options, language, capability):
    def make_ttgir_wrapper(src, metadata, options, capability):
        print("Called make_ttgir_wrapper")
        return self.make_ttgir(src, metadata, options, capability)
    stages["ttgir"] = lambda src, metadata: make_ttgir_wrapper(src, metadata, options, capability)

def init():
    knobs.runtime.add_stages_inspection_hook = inspect_stages


init()
