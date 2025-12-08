from triton._C.libtriton import ir, passes
import hashlib
import pathlib


# These two methods must be implemented and returned by the plugin hook.
# any changes in this entire file and the the plugin pipeline
# will trigger a recompile since the hash will change. To be
# less conservative, we could use a hash of the inspect_stages_hook
# function but then changes outside of the function won't be considered
# potentially causing a stale kernel hash
def get_key():
    return pathlib.Path(__file__).read_text()


def get_hash():
    return hashlib.sha256(get_key().encode('utf-8')).hexdigest()

# Keep custom pipeline stages in a seperate file from kernels as any change to the file
# will trigger a recompile.


def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
    # If the hook is called with no arguments we assume were just after the key and hash and don't want to
    # actually execute the pipeline yet
    if all(arg is None for arg in (stages, options, language, capability)):
        return get_key(), get_hash()

    def make_ttir_wrapper(mod, metadata, opt, capability):
        mod = self.make_ttir(mod, metadata, opt, capability)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.plugin.add_plugin(pm)
        pm.run(mod, 'make_ttir_plugin')
        return mod

    stages["ttir"] = lambda src, metadata: make_ttir_wrapper(src, metadata, options, capability)

    return get_key(), get_hash()


def inspect_stages_hook_dialect(self=None, stages=None, options=None, language=None, capability=None):
    # If the hook is called with no arguments we assume were just after the key and hash and don't want to
    # actually execute the pipeline yet
    if all(arg is None for arg in (stages, options, language, capability)):
        return get_key(), get_hash()

    def make_ttgir_wrapper(mod, metadata, opt, capability):
        mod = self.make_ttgir(mod, metadata, opt, capability)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.plugin.plugingpu_conversion(pm)
        pm.run(mod, 'make_ttgir_plugin')
        return mod

    stages["ttgir"] = lambda src, metadata: make_ttgir_wrapper(src, metadata, options, capability)

    return get_key(), get_hash()
