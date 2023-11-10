import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union


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
    num_specs: int  # number of specialized arguments


@dataclass
class HeaderParsingPatterns:
    """Header Parsing Regexes"""

    # [kernel_name, c signature]
    LINKER_DIRECTIVES = re.compile("//[\\s]*tt-linker:[\\s]*([\\w]+):(.+):(.+)")

    # [name, hash, suffix]
    KERNEL_NAME = re.compile("^([\\w]+)_([\\w]+)_([\\w]+)$")

    # [(type, name)]
    C_SIG = re.compile("[\\s]*(\\w+)\\s(\\w+)[,]?")

    # [d|c]
    ARG_SUFFIX = re.compile("[c,d]")


class HeaderParser(ABC):
    def __init__(self, patterns=HeaderParsingPatterns):
        self.patterns = patterns
        self.kernels = defaultdict(list)

    @abstractmethod
    def parse(self, headers: List[Path | str]) -> Dict[str, List[KernelLinkerMeta]]:
        ...


class C_CUDA_HeaderParser(HeaderParser):
    def parse(self, headers: List[Path | str]) -> Dict[str, List[KernelLinkerMeta]]:
        for header in headers:
            h_path = Path(header)
            h_str = h_path.read_text()
            self.extract_linker_meta(h_str)

        return self.kernels

    def extract_linker_meta(self, header: str):
        for ln in header.splitlines():
            if ln.startswith("//"):
                m = self.patterns.LINKER_DIRECTIVES.match(ln)
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
        m = self.patterns.KERNEL_NAME.match(ker_name)
        if _exists(m):
            name, sig_hash, suffix = m.group(1), m.group(2), m.group(3)
            return name, sig_hash, suffix
        raise LinkerError(f"{ker_name} is not a valid kernel name")

    def _match_c_sig(self, c_sig: str):
        m = self.patterns.C_SIG.findall(c_sig)
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
            if self.patterns.ARG_SUFFIX.match(suffix, pos):
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
