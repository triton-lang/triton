from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union

from dataclasses import dataclass


def _exists(x):
    return x is not None


class LinkerError(Exception):
    pass


@dataclass
class KernelLinkerMeta:
    orig_kernel_name: str
    arg_names: Sequence[str]
    arg_ctypes: Sequence[str]
    sizes: Sequence[Union[int, None]]
    sig_hash: str
    triton_suffix: str
    suffix: str
    num_specs: int
    """ number of specialized arguments """


class HeaderParser:

    def __init__(self) -> None:
        import re

        # [kernel_name, c signature]
        self.linker_directives = re.compile("//[\\s]*tt-linker:[\\s]*([\\w]+):(.+):(.+)")
        # [name, hash, suffix]
        self.kernel_name = re.compile("^([\\w]+)_([\\w]+)_([\\w]+)$")
        # [(type, name)]
        self.c_sig = re.compile("[\\s]*(\\w+)\\s(\\w+)[,]?")
        # [d|c]
        self.arg_suffix = re.compile("[c,d]")

        self.kernels = defaultdict(list)

    def extract_linker_meta(self, header: str):
        for ln in header.splitlines():
            if ln.startswith("//"):
                m = self.linker_directives.match(ln)
                if _exists(m):
                    ker_name, c_sig, algo_info = m.group(1), m.group(2), m.group(3)
                    name, sig_hash, suffix = self._match_name(ker_name)
                    c_types, arg_names = self._match_c_sig(c_sig)
                    num_specs, sizes = self._match_suffix(suffix, c_sig)
                    self._add_kernel(
                        "_".join([name, algo_info]),
                        KernelLinkerMeta(
                            orig_kernel_name=name,
                            arg_names=arg_names,
                            arg_ctypes=c_types,
                            sizes=sizes,
                            sig_hash=sig_hash,
                            triton_suffix=suffix,
                            suffix=suffix,
                            num_specs=num_specs,
                        ),
                    )

    def _match_name(self, ker_name: str):
        m = self.kernel_name.match(ker_name)
        if _exists(m):
            name, sig_hash, suffix = m.group(1), m.group(2), m.group(3)
            return name, sig_hash, suffix
        raise LinkerError(f"{ker_name} is not a valid kernel name")

    def _match_c_sig(self, c_sig: str):
        m = self.c_sig.findall(c_sig)
        if len(m):
            tys, args = [], []
            for ty, arg_name in m:
                tys.append(ty)
                args.append(arg_name)
            return tys, args

        raise LinkerError(f"{c_sig} is not a valid argument signature")

    def _match_suffix(self, suffix: str, c_sig: str):
        args = c_sig.split(",")
        s2i = {"c": 1, "d": 16}
        num_specs = 0
        sizes = []
        # scan through suffix, first find the index,
        # then see if it is followed by d or c
        for i in range(len(args)):
            pos = suffix.find(str(i))
            if pos == -1:
                raise LinkerError(f"{suffix} is not a valid kernel suffix")
            pos += len(str(i))
            if self.arg_suffix.match(suffix, pos):
                num_specs += 1
                sizes.extend([None] * (i - len(sizes)))
                sizes.append(s2i[suffix[pos]])
                pos += 1
            if i < len(args) - 1:
                suffix = suffix[pos:]
            else:
                sizes.extend([None] * (len(args) - len(sizes)))
        return num_specs, sizes

    def _add_kernel(self, name: str, ker: KernelLinkerMeta):
        if name in self.kernels:
            last: KernelLinkerMeta = self.kernels[name][-1]

            for cur, new_ in zip(last.arg_ctypes, ker.arg_ctypes):
                if cur != new_:
                    raise LinkerError(
                        f"Mismatched signature for kernel {name}: \n\texisting sig is: {','.join(last.arg_ctypes)}\n\tcurrent is: {','.join(ker.arg_ctypes)}"
                    )

        self.kernels[name].append(ker)


def gen_signature_with_full_args(m):
    return ", ".join([f"{ty} {arg}" for ty, arg in zip(m.arg_ctypes, m.arg_names)])


def gen_signature(m):
    arg_types = [ty for ty, hint in zip(m.arg_ctypes, m.sizes) if hint != 1]
    arg_names = [arg for arg, hint in zip(m.arg_names, m.sizes) if hint != 1]
    sig = ", ".join([f"{ty} {arg}" for ty, arg in zip(arg_types, arg_names)])
    return sig


# generate declarations of kernels with meta-parameter and constant values
def make_algo_decls(name: str, metas: Sequence[KernelLinkerMeta]) -> str:
    return f"""
CUresult {name}(CUstream stream, {gen_signature_with_full_args(metas[-1])});
void load_{name}();
void unload_{name}();
    """


# generate declarations of kernels with meta-parameter and constant values
def make_global_decl(meta: KernelLinkerMeta) -> str:
    return f"""
CUresult {meta.orig_kernel_name}_default(CUstream stream, {gen_signature_with_full_args(meta)});
CUresult {meta.orig_kernel_name}(CUstream stream, {gen_signature_with_full_args(meta)}, int algo_id);
void load_{meta.orig_kernel_name}();
void unload_{meta.orig_kernel_name}();
    """


# generate dispatcher function for kernels with different meta-parameter and constant values
def make_default_algo_kernel(meta: KernelLinkerMeta) -> str:
    src = f"CUresult {meta.orig_kernel_name}_default(CUstream stream, {gen_signature_with_full_args(meta)}){{\n"
    src += (f"  return {meta.orig_kernel_name}(stream, {', '.join(meta.arg_names)}, 0);\n")
    src += "}\n"
    return src


# generate dispatcher function for kernels with different integer value hints
def make_kernel_hints_dispatcher(name: str, metas: Sequence[KernelLinkerMeta]) -> str:
    src = f"// launcher for: {name}\n"
    for meta in sorted(metas, key=lambda m: -m.num_specs):
        src += f"CUresult {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}(CUstream stream, {gen_signature(meta)});\n"
    src += "\n"

    src += (f"CUresult {name}(CUstream stream, {gen_signature_with_full_args(metas[-1])}){{")
    src += "\n"
    for meta in sorted(metas, key=lambda m: -m.num_specs):
        cond_fn = (  #
            lambda val, hint: f"({val} % {hint} == 0)"  #
            if hint == 16  #
            else f"({val} == {hint})"  #
            if hint == 1  #
            else None)
        conds = " && ".join([  #
            cond_fn(val, hint)  #
            for val, hint in zip(meta.arg_names, meta.sizes)  #
            if hint is not None
        ])
        src += (f"  if ({conds})\n" if any(meta.sizes) else "if (1)\n"
                )  # Edge case where no specializations hence no dispatching required
        arg_names = [arg for arg, hint in zip(meta.arg_names, meta.sizes) if hint != 1]
        src += f"    return {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}(stream, {', '.join(arg_names)});\n"
    src += "\n"
    src += "  return CUDA_ERROR_INVALID_VALUE;\n"
    src += "}\n"

    for mode in ["load", "unload"]:
        src += f"\n// {mode} for: {name}\n"
        for meta in sorted(metas, key=lambda m: -m.num_specs):
            src += f"void {mode}_{meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}();\n"
        src += f"void {mode}_{name}() {{"
        src += "\n"
        for meta in sorted(metas, key=lambda m: -m.num_specs):
            src += (f"  {mode}_{meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}();\n")
        src += "}\n"
    return src


# generate dispatcher function for kernels with different meta-parameter and constant values
def make_kernel_meta_const_dispatcher(meta: KernelLinkerMeta) -> str:
    src = f"CUresult {meta.orig_kernel_name}(CUstream stream, {gen_signature_with_full_args(meta)}, int algo_id){{\n"
    src += f"  assert (algo_id < (int)sizeof({meta.orig_kernel_name}_kernels));\n"
    src += f"  return {meta.orig_kernel_name}_kernels[algo_id](stream, {', '.join(meta.arg_names)});\n"
    src += "}\n"
    return src


# generate definition of function pointers of kernel dispatchers based on meta-parameter and constant values
def make_func_pointers(names: str, meta: KernelLinkerMeta) -> str:
    # the table of hint dispatchers
    src = f"typedef CUresult (*kernel_func_t)(CUstream stream, {gen_signature_with_full_args(meta)});\n"
    src += f"kernel_func_t {meta.orig_kernel_name}_kernels[] = {{\n"
    for name in names:
        src += f"  {name},\n"
    src += "};\n"
    return src


# generate definition for load/unload functions for kernels with different meta-parameter and constant values
def make_kernel_load_def(names: str, meta: KernelLinkerMeta) -> str:
    src = ""
    for mode in ["load", "unload"]:
        src += f"void {mode}_{meta.orig_kernel_name}(void){{\n"
        for name in names:
            src += f"  {mode}_{name}();\n"
        src += "}\n\n"
    return src


def make_get_num_algos_decl(meta: KernelLinkerMeta) -> str:
    src = f"int {meta.orig_kernel_name}_get_num_algos(void);"
    return src


def make_get_num_algos_def(meta: KernelLinkerMeta) -> str:
    src = f"int {meta.orig_kernel_name}_get_num_algos(void){{\n"
    src += f"  return (int)(sizeof({meta.orig_kernel_name}_kernels) / sizeof({meta.orig_kernel_name}_kernels[0]));\n"
    src += "}\n"
    return src


desc = """
Triton ahead-of-time linker:

This program takes in header files generated by compile.py, and generates a
single entry-point responsible for dispatching the user's input to the right
kernel given the specializations that were compiled.

Example usage:
python link.py /path/to/headers/*.h -o kernel_name
"""

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description=desc)
    parser.add_argument(
        "headers",
        nargs="+",
        help="Paths to header files to link. Must include linker directive annotations (autogenerated by ttc)",
    )
    parser.add_argument("--out", "-o", type=Path, help="Out filename")
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="String to prefix kernel dispatcher names",
    )
    args = parser.parse_args()

    # metadata
    parser = HeaderParser()
    includes = []
    for header in args.headers:
        h_path = Path(header)
        h_str = h_path.read_text()
        includes.append(h_path.name)
        parser.extract_linker_meta(h_str)

    # generate headers
    algo_decls = [make_algo_decls(name, meta) for name, meta in parser.kernels.items()]
    meta_lists = [meta for name, meta in parser.kernels.items()]
    meta = meta_lists[0][0]
    get_num_algos_decl = make_get_num_algos_decl(meta)
    global_decl = make_global_decl(meta)
    with args.out.with_suffix(".h").open("w") as fp:
        out = "#include <cuda.h>\n"
        out += "\n".join(algo_decls)
        out += "\n"
        out += get_num_algos_decl
        out += "\n"
        out += global_decl
        fp.write(out)

    # generate source
    defs = [make_kernel_hints_dispatcher(name, meta) for name, meta in parser.kernels.items()]
    names = [name for name in parser.kernels.keys()]
    func_pointers_def = make_func_pointers(names, meta)
    meta_const_def = make_kernel_meta_const_dispatcher(meta)
    load_unload_def = make_kernel_load_def(names, meta)
    get_num_algos_def = make_get_num_algos_def(meta)
    default_algo_kernel = make_default_algo_kernel(meta)
    with args.out.with_suffix(".c").open("w") as fp:
        out = ""
        out += "#include <cuda.h>\n"
        out += "#include <stdint.h>\n"
        out += "#include <assert.h>\n"
        out += "\n"
        out += "\n".join(defs)
        out += "\n"
        out += func_pointers_def
        out += "\n"
        out += get_num_algos_def
        out += "\n"
        out += meta_const_def
        out += "\n"
        out += load_unload_def
        out += "\n"
        out += default_algo_kernel
        fp.write(out)
