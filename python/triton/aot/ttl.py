from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union

from dataclasses import dataclass


def _exists(x):
    return x is not None


class LinkerError(Exception):
    pass


@dataclass
class KernelLinkerMeta:
    arg_names: Sequence[str]
    arg_ctypes: Sequence[str]
    sizes: Sequence[Union[int, None]]
    suffix: str
    num_specs: int
    """ number of specialized arguments """


class HeaderParser:
    def __init__(self) -> None:
        import regex as re

        # [kernel_name, c signature]
        self.linker_directives = re.compile("//[\s]*tt-linker:[\s]*([\w]+):(.+)")

        # [name, suffix]
        self.kernel_name = re.compile("([\w]+)_([\w]+)")
        # [(argnum, d|c)]
        self.kernel_suffix = re.compile("([0-9]+)([c,d])")

        # [(type, name)]
        self.c_sig = re.compile("[\s]*(\w+)\s(\w+)[,]?")

        self._kernels = defaultdict(list)

    def all_kernels(self) -> Iterable[Tuple[str, Sequence[KernelLinkerMeta]]]:
        for k, v in self._kernels.items():
            yield k, v

    def extract_linker_meta(self, header: str):

        for ln in header.splitlines():
            if ln.startswith("//"):
                m = self.linker_directives.match(ln)
                if _exists(m):
                    ker_name, c_sig = m.group(1), m.group(2)
                    name, suffix = self._match_name(ker_name)
                    c_types, arg_names = self._match_c_sig(c_sig)
                    num_specs, sizes = self._match_suffix(suffix)
                    self._add_kernel(
                        name,
                        KernelLinkerMeta(
                            arg_names=arg_names,
                            arg_ctypes=c_types,
                            sizes=sizes,
                            suffix=suffix,
                            num_specs=num_specs,
                        ),
                    )

    def _match_name(self, ker_name: str):
        m = self.kernel_name.match(ker_name)
        if _exists(m):
            name, suffix = m.group(1), m.group(2)
            return name, suffix
        raise LinkerError(f"{ker_name} is not a valid kernel name")

    def _match_c_sig(self, c_sig: str):
        m = self.c_sig.findall(c_sig)
        if len(m):
            tys, args = [], []
            for (ty, arg_name) in m:
                tys.append(ty)
                args.append(arg_name)
            return tys, args

        raise LinkerError(f"{c_sig} is not a valid argument signature")

    def _match_suffix(self, suffix: str):
        m = self.kernel_suffix.findall(suffix)
        if not len(m):
            raise LinkerError(f"{suffix} is not a valid kernel suffix")
        sizes = []
        num_specs = len(m)
        s2i = {"c": 1, "d": 16}
        for (argnum, arg_size_ann) in m:
            while len(sizes) < int(argnum):
                sizes.append(None)

            sizes.append(s2i[arg_size_ann])
        return num_specs, sizes

    def _add_kernel(self, name: str, ker: KernelLinkerMeta):
        if name not in self._kernels:
            pass
        else:
            last: KernelLinkerMeta = self._kernels[name][-1]

            for (cur, new_) in zip(last.arg_ctypes, ker.arg_ctypes):
                if cur != new_:
                    raise LinkerError(
                        f"Mismatched signature for kernek {name}: \n\texisting sig is: {','.join(last.arg_ctypes)}\n\tcurrenr is: {','.join(ker.arg_ctypes)}"
                    )

        self._kernels[name].append(ker)
        return


def _make_dispatch_condition_expr(attr_names, attr_sizes: Sequence[int]):
    conds = []
    for arg, size_ in zip(attr_names, attr_sizes):

        if size_ is None:
            # non-specialized arg, no condition
            continue

        if size_ == 1:
            conds.append(f"{arg} == 1")
        else:
            conds.append(f"{arg} % {size_} == 0")

    return " & ".join(conds)


def link_headers_to_dispatcher(headers: Sequence[str]):

    includes = []
    parser = HeaderParser()

    for header in headers:
        h_path = Path(header)
        h_str = h_path.read_text()
        includes.append(h_path.name)
        parser.extract_linker_meta(h_str)

    dispatchers = []
    for name, kers in parser.all_kernels():
        dispatchers.append(make_kernel_dispatcher(name=name, metas=kers))

    dispatchers = "\n\n".join(dispatchers)
    includes = "\n".join([f'#include "{inc_}"' for inc_ in includes])

    linked_csrc = """
{includes}

{dispatchers}
    
    """
    return linked_csrc.format(includes=includes, dispatchers=dispatchers)


def make_kernel_dispatcher(name: str, metas: Sequence[KernelLinkerMeta]) -> str:
    src = """
CUresult {kernel_name}(CUstream stream, unsigned int gX,unsigned int gY,unsigned int gZ,unsigned int numWarps, {signature}) {{
    {conds}
}} 
    """

    cond = """
    if ({cond}) {{
        return {call_name}(stream, gX, gY, gZ, numWarps, {arg_names});
        }}
    """
    conds = []
    for meta in sorted(metas, key=lambda m: -m.num_specs):
        conds.append(
            cond.format(
                cond=_make_dispatch_condition_expr(meta.arg_names, meta.sizes),
                call_name=f"{name}_{meta.suffix}",
                arg_names=", ".join(meta.arg_names),
            )
        )

    conds = "\n\n".join(conds)
    m = metas[-1]
    signature = ", ".join([f"{ty} {arg}" for ty, arg in zip(m.arg_ctypes, m.arg_names)])

    return src.format(kernel_name=name, signature=signature, conds=conds)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Triton AOT Linker")
    parser.add_argument(
        "headers",
        nargs="+",
        help="Paths to header files to link. Must include linker directive annotations (autogenerated by ttc)",
    )
    args = parser.parse_args()
    print(link_headers_to_dispatcher(args.headers))
