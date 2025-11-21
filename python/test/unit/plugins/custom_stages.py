from triton._C.libtriton import ir, passes


# Keep custom pipeline stages in a seperate file from kernels are any change to the file
# will trigger a recompile.
def inspect_stages_hook(self, stages, options, language, capability):

    def make_ttir_wrapper(mod, metadata, opt, capability):
        mod = self.make_ttir(mod, metadata, opt, capability)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.plugin.add_plugin(pm)
        pm.run(mod, 'make_ttir_plugin')
        return mod

    stages["ttir"] = lambda src, metadata: make_ttir_wrapper(src, metadata, options, capability)
