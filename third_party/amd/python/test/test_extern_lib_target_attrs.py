import textwrap

import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


def _llvm_test_module(tmp_path, body):
    from triton._C.libtriton import ir, llvm

    path = tmp_path / "module.mlir"
    path.write_text("module {\n" + textwrap.dedent(body) + "\n}")
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    return llvm.to_module(module, llvm.context())


@pytest.mark.parametrize("settings, library_features, expected_cpu, expected_features", [
    pytest.param({}, "+packed-fp32-ops", "gfx942", "+packed-fp32-ops", id="unchanged"),
    pytest.param({"target_cpu": "gfx950", "target_features": "-packed-fp32-ops"}, "+packed-fp32-ops", "gfx950",
                 "-packed-fp32-ops,+packed-fp32-ops", id="override-library-last"),
    pytest.param({"target_features": "-packed-fp32-ops"}, None, "gfx942", "-packed-fp32-ops", id="no-library-features"),
])
def test_llvm_link_extern_libs_target_settings(tmp_path, settings, library_features, expected_cpu, expected_features):
    from triton._C.libtriton import llvm

    module = _llvm_test_module(
        tmp_path, """
        llvm.func @kernel(%x: i32) -> i32 {
          %result = llvm.call @helper(%x) : (i32) -> i32
          llvm.return %result : i32
        }
        llvm.func @helper(i32) -> i32
    """)
    path = tmp_path / "library.ll"
    feature_attr = f'"target-features"="{library_features}"' if library_features is not None else ''
    path.write_text(f'''
        define i32 @helper(i32 %x) #0 {{ ret i32 %x }}
        attributes #0 = {{ noinline "target-cpu"="gfx942" {feature_attr} }}
    ''')
    llvm.link_extern_libs(module, [str(path)], **settings)
    functions = {fn.name: fn for fn in module.get_functions()}
    assert functions["helper"].get_fn_attr("target-cpu") == expected_cpu
    assert functions["helper"].get_fn_attr("target-features") == expected_features
    assert functions["kernel"].get_fn_attr("target-cpu") is None
    assert functions["kernel"].get_fn_attr("target-features") is None


def _compile_hip_to_llir(source, target, option_values):
    from triton._C.libtriton import ir
    from triton.compiler.compiler import make_backend

    backend = make_backend(target)
    options = backend.parse_options(option_values)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)
    metadata = {"target": target, **vars(options)}
    stages = {}
    backend.add_stages(stages, options, source.language)
    # Stop before AMD assembly/object generation, without accessing a GPU driver.
    for kind, stage in stages.items():
        module = stage(module, metadata)
        if kind == "llir":
            return module
    pytest.fail("LLIR stage not found")


def test_compile_only_hip_builtin_libs_inherit_kernel_target_settings(monkeypatch):
    from pathlib import Path

    from triton import knobs
    from triton._C.libtriton import llvm
    from triton.language.extra import libdevice

    monkeypatch.setattr(knobs.compilation, "enable_asan", False)
    monkeypatch.setattr(knobs.amd, "bitcode_path", None)

    @triton.jit
    def kernel(out, x):
        tl.store(out, libdevice.exp(x) + tl.num_programs(0).to(tl.float32))

    original_link = llvm.link_extern_libs
    settings = {}

    def capture_link(module, paths, **kwargs):
        settings[Path(paths[0]).stem] = kwargs
        return original_link(module, paths, **kwargs)

    monkeypatch.setattr(llvm, "link_extern_libs", capture_link)
    source = ASTSource(kernel, signature={"out": "*fp32", "x": "fp32"})
    _compile_hip_to_llir(source, GPUTarget("hip", "gfx950", 64), {
        "llvm_fn_attrs": (("target-features", "-packed-fp32-ops"), ),
    })
    expected = {"target_cpu": "gfx950", "target_features": "-packed-fp32-ops"}
    assert settings == {"ocml": expected, "ockl": expected}


def test_compile_only_hip_replacement_libraries_preserve_target_settings(monkeypatch, tmp_path):
    from pathlib import Path

    from triton import knobs
    from triton._C.libtriton import llvm
    from triton.language.extra import libdevice

    monkeypatch.setattr(knobs.compilation, "enable_asan", False)
    monkeypatch.setattr(knobs.amd, "bitcode_path", str(tmp_path))
    # Inspect linked attributes without optimizing deliberately incompatible
    # replacement functions.
    monkeypatch.setenv("DISABLE_LLVM_OPT", "1")
    for name, definition in (
        ("ocml", "define float @__ocml_exp_f32(float %x) #0 { ret float %x }"),
        ("ockl", "define i64 @__ockl_get_num_groups(i32 %dim) #0 { ret i64 1 }"),
    ):
        (tmp_path / f"{name}.bc").write_text(definition + '''
            attributes #0 = { noinline "target-cpu"="gfx942" "target-features"="+packed-fp32-ops" }
        ''')

    @triton.jit
    def kernel(out, x):
        value = libdevice.exp(x) + tl.num_programs(0).to(tl.float32)
        tl.store(out, value)

    original_link = llvm.link_extern_libs
    original_optimize = llvm.optimize_module
    linked = []
    captured = []

    def capture_link(module, paths, **kwargs):
        assert kwargs == {}
        for path in paths:
            assert Path(path).parent == tmp_path
            linked.append(Path(path).stem)
        return original_link(module, paths, **kwargs)

    def capture_optimize(module, *args, **kwargs):
        functions = {fn.name: fn for fn in module.get_functions()}
        for name in ("__ockl_get_num_groups", "__ocml_exp_f32"):
            assert not functions[name].is_declaration()
            assert functions[name].get_fn_attr("target-cpu") == "gfx942"
            assert functions[name].get_fn_attr("target-features") == "+packed-fp32-ops"
        captured.append(True)
        return original_optimize(module, *args, **kwargs)

    monkeypatch.setattr(llvm, "link_extern_libs", capture_link)
    monkeypatch.setattr(llvm, "optimize_module", capture_optimize)
    source = ASTSource(kernel, signature={"out": "*fp32", "x": "fp32"})
    _compile_hip_to_llir(source, GPUTarget("hip", "gfx950", 64), {
        "llvm_fn_attrs": (("target-features", "-packed-fp32-ops"), ),
    })
    assert linked == ["ocml", "ockl"]
    assert captured == [True]
