import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class Symbol:
    _name: str
    _op_name: str
    _ret_type: str
    _arg_names: List[str]
    _arg_types: List[str]

    def __init__(
        self,
        name: str,
        op_name: str,
        ret_type: str,
        arg_names: List[str],
        arg_types: List[str],
    ) -> None:
        '''
        A symbol is a function declaration.
        :param name: name of the symbol
        :param op_name: name of the operation
        :param ret_type: return type of the operation
        :param arg_names: names of the arguments
        :param arg_types: types of the arguments
        '''
        self._name = name
        self._op_name = op_name
        self._ret_type = ret_type
        self._arg_names = list(arg_names)
        self._arg_types = list(arg_types)

    @property
    def name(self) -> str:
        return self._name

    @property
    def op_name(self) -> str:
        return self._op_name

    @property
    def ret_type(self) -> str:
        return self._ret_type

    @property
    def arg_names(self) -> List[str]:
        return self._arg_names

    @property
    def arg_types(self) -> List[str]:
        return self._arg_types


def convert_type(type_str) -> Optional[str]:
    if type_str == "i32":
        return "int32"
    elif type_str == "u32":
        return "uint32"
    elif type_str == "i64":
        return "int64"
    elif type_str == "u64":
        return "uint64"
    elif type_str == "float":
        return "fp32"
    elif type_str == "double":
        return "fp64"
    else:
        # ignore other types, such as pointer types
        return None


def to_unsigned(type_str) -> str:
    if type_str == "int32":
        return "uint32"
    elif type_str == "int64":
        return "uint64"
    else:
        return type_str


class ExternLibrary(ABC):
    _name: str
    _path: str
    _symbols: Dict[str, Symbol]
    _format: bool
    _grouping: bool

    def __init__(
        self,
        name: str,
        path: str,
        format: bool = True,
        grouping: bool = True,
    ) -> None:
        '''
        Abstract class for extern library.
        :param name: name of the library
        :param path: path of the library
        :param format: whether to format the generated stub file
        '''
        self._name = name
        self._path = path
        self._symbols = {}
        self._format = format
        self._grouping = grouping

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    @property
    def symbols(self) -> Dict[str, Symbol]:
        return self._symbols

    @property
    def grouping(self) -> bool:
        return self._grouping

    @abstractmethod
    def parse_symbols(self, input_file) -> None:
        pass

    @abstractmethod
    def _output_stubs(self) -> str:
        pass

    def generate_stub_file(self, output_dir) -> None:
        file_str = self._output_stubs()
        if file_str is None or len(file_str) == 0:
            raise Exception("file_str is empty")

        output_file = f"{output_dir}/{self._name}.py"
        with open(output_file, "w") as f:
            f.write(file_str)
            f.close()
            if self._format:
                subprocess.Popen(["autopep8", "-a", "-r", "-i", output_file], stdout=subprocess.PIPE).communicate()
                subprocess.Popen(["isort", output_file], stdout=subprocess.PIPE).communicate()


class Libdevice(ExternLibrary):
    _symbol_groups: Dict[str, List[Symbol]]

    def __init__(self, path) -> None:
        '''
        Constructor for Libdevice.
        :param path: path of the libdevice library
        '''
        super().__init__("libdevice", path)
        self._symbol_groups = {}
        self.is_pure = True

    @staticmethod
    def _extract_symbol(line) -> Optional[Symbol]:
        # Extract symbols from line in the following format:
        # "define [internal] <ret_type> @<name>(<arg_types>,)"
        entries = line.split("@")
        ret_str = entries[0]
        func_str = entries[1]
        # Get ret_type, skip internal symbols
        ret_strs = ret_str.split()
        if ret_strs[1] == "internal":
            return None
        ret_type = convert_type(ret_strs[1])
        if ret_type is None:
            return None
        # Get function name
        func_strs = func_str.split("(")
        func_name = func_strs[0].replace("@", "")
        op_name = func_name.replace("__nv_", "")
        if 'ieee' in op_name:
            return None
        # Get arg_types
        arg_strs = func_strs[1].split(",")
        arg_types = []
        arg_names = []
        for i, arg_str in enumerate(arg_strs):
            arg_type = convert_type(arg_str.split()[0])
            if arg_type is None:
                return None
            arg_name = 'arg' + str(i)
            arg_types.append(arg_type)
            arg_names.append(arg_name)
        if op_name == "sad":
            # Special case for sad, where the last argument is an unsigned int
            arg_types[-1] = to_unsigned(arg_types[-1])
        elif op_name.startswith("u"):
            # LLVM does not differentiate between signed and unsigned integer type.
            # We have to convert the types to unsigned
            ret_type = to_unsigned(ret_type)
            for i, arg_type in enumerate(arg_types):
                arg_types[i] = to_unsigned(arg_type)
        return Symbol(func_name, op_name, ret_type, arg_names, arg_types)

    def _group_symbols(self) -> None:
        symbol_set = {}
        for symbol in self._symbols.values():
            op_name = symbol.op_name
            symbol_set[op_name] = symbol

        # Group functions together by renaming.
        renaming = {
            'llabs': 'abs', 'acosf': 'acos', 'acoshf': 'acosh', 'dadd_rd': 'add_rd', 'fadd_rd': 'add_rd', 'dadd_rn':
            'add_rn', 'fadd_rn': 'add_rn', 'dadd_ru': 'add_ru', 'fadd_ru': 'add_ru', 'dadd_rz': 'add_rz', 'fadd_rz':
            'add_rz', 'asinf': 'asin', 'asinhf': 'asinh', 'atanf': 'atan', 'atan2f': 'atan2', 'atanhf': 'atanh',
            'brevll': 'brev', 'cbrtf': 'cbrt', 'ceilf': 'ceil', 'clzll': 'clz', 'copysignf': 'copysign', 'cosf': 'cos',
            'coshf': 'cosh', 'cospif': 'cospi', 'cyl_bessel_i0f': 'cyl_bessel_i0', 'cyl_bessel_i1f': 'cyl_bessel_i1',
            'fdiv_rd': 'div_rd', 'ddiv_rd': 'div_rd', 'fdiv_rn': 'div_rn', 'ddiv_rn': 'div_rn', 'fdiv_ru': 'div_ru',
            'ddiv_ru': 'div_ru', 'fdiv_rz': 'div_rz', 'ddiv_rz': 'div_rz', 'erff': 'erf', 'erfcf': 'erfc', 'erfcinvf':
            'erfcinv', 'erfcxf': 'erfcx', 'erfinvf': 'erfinv', 'expf': 'exp', 'exp10f': 'exp10', 'exp2f': 'exp2',
            'expm1f': 'expm1', 'fabsf': 'abs', 'fabs': 'abs', 'fast_fdividef': 'fast_dividef', 'fdimf': 'fdim', 'ffsll':
            'ffs', 'floorf': 'floor', 'fmaf': 'fma', 'fmaf_rd': 'fma_rd', 'fmaf_rn': 'fma_rn', 'fmaf_ru': 'fma_ru',
            'fmaf_rz': 'fma_rz', 'fmodf': 'fmod', 'uhadd': 'hadd', 'hypotf': 'hypot', 'ilogbf': 'ilogb', 'isinff':
            'isinf', 'isinfd': 'isinf', 'isnanf': 'isnan', 'isnand': 'isnan', 'j0f': 'j0', 'j1f': 'j1', 'jnf': 'jn',
            'ldexpf': 'ldexp', 'lgammaf': 'lgamma', 'llrintf': 'llrint', 'llroundf': 'llround', 'logf': 'log', 'log10f':
            'log10', 'log1pf': 'log1p', 'log2f': 'log2', 'logbf': 'logb', 'umax': 'max', 'llmax': 'max', 'ullmax':
            'max', 'fmaxf': 'max', 'fmax': 'max', 'umin': 'min', 'llmin': 'min', 'ullmin': 'min', 'fminf': 'min',
            'fmin': 'min', 'dmul_rd': 'mul_rd', 'fmul_rd': 'mul_rd', 'dmul_rn': 'mul_rn', 'fmul_rn': 'mul_rn',
            'dmul_ru': 'mul_ru', 'fmul_ru': 'mul_ru', 'dmul_rz': 'mul_rz', 'fmul_rz': 'mul_rz', 'umul24': 'mul24',
            'umulhi': 'mulhi', 'mul64hi': 'mulhi', 'umul64hi': 'mulhi', 'nearbyintf': 'nearbyint', 'nextafterf':
            'nextafter', 'norm3df': 'norm3d', 'norm4df': 'norm4d', 'normcdff': 'normcdf', 'normcdfinvf': 'normcdfinv',
            'popcll': 'popc', 'powif': 'pow', 'powi': 'pow', 'powf': 'pow', 'rcbrtf': 'rcbrt', 'frcp_rd': 'rcp_rd',
            'drcp_rd': 'rcp_rd', 'frcp_rn': 'rcp_rn', 'drcp_rn': 'rcp_rn', 'frcp_ru': 'rcp_ru', 'drcp_ru': 'rcp_ru',
            'frcp_rz': 'rcp_rz', 'drcp_rz': 'rcp_rz', 'remainderf': 'remainder', 'urhadd': 'rhadd', 'rhypotf': 'rhypot',
            'rintf': 'rint', 'rnorm3df': 'rnorm3d', 'rnorm4df': 'rnorm4d', 'roundf': 'round', 'rsqrtf': 'rsqrt',
            'frsqrt_rn': 'rsqrt_rn', 'usad': 'sad', 'scalbnf': 'scalbn', 'signbitf': 'signbit', 'signbitd': 'signbit',
            'sinf': 'sin', 'sinhf': 'sinh', 'sinpif': 'sinpi', 'sqrtf': 'sqrt', 'fsqrt_rd': 'sqrt_rd', 'dsqrt_rd':
            'sqrt_rd', 'fsqrt_rn': 'sqrt_rn', 'dsqrt_rn': 'sqrt_rn', 'fsqrt_ru': 'sqrt_ru', 'dsqrt_ru': 'sqrt_ru',
            'fsqrt_rz': 'sqrt_rz', 'dsqrt_rz': 'sqrt_rz', 'fsub_rd': 'sub_rd', 'dsub_rd': 'sub_rd', 'fsub_rn': 'sub_rn',
            'dsub_rn': 'sub_rn', 'fsub_ru': 'sub_ru', 'dsub_ru': 'sub_ru', 'fsub_rz': 'sub_rz', 'dsub_rz': 'sub_rz',
            'tanf': 'tan', 'tanhf': 'tanh', 'tgammaf': 'tgamma', 'truncf': 'trunc', 'y0f': 'y0', 'y1f': 'y1', 'ynf':
            'yn'
        }

        for symbol in self._symbols.values():
            op_name = symbol.op_name
            if op_name in renaming:
                op_name = renaming[op_name]
                symbol._op_name = op_name
            if op_name in self._symbol_groups:
                self._symbol_groups[op_name].append(symbol)
            else:
                self._symbol_groups[op_name] = [symbol]

    def parse_symbols(self, input_file) -> None:
        if len(self.symbols) > 0:
            return
        output = subprocess.check_output(["grep", "define", input_file]).decode().splitlines()
        for line in output:
            symbol = self._extract_symbol(line)
            if symbol is None:
                continue
            self._symbols[symbol.name] = symbol

        self._group_symbols()

    def _output_stubs(self) -> str:
        # Generate python functions in the following format:
        # @extern.extern
        # def <op_name>(<args>, _builder=None):
        #   arg_type_symbol_dict = {[arg_type]: {(symbol, ret_type)}}
        #   return core.extern_elementwise("libdevice", <path>, <args>, <arg_type_symbol_dict>, _builder)
        import_str = "from . import core\n"
        import_str += "import os\n"
        import_str += "import functools\n"

        header_str = ""
        header_str += "@functools.lru_cache()\n"
        header_str += "def libdevice_path():\n"
        header_str += "    import torch\n"
        header_str += "    third_party_dir =  os.path.join(os.path.dirname(os.path.abspath(__file__)), \"..\", \"third_party\")\n"
        header_str += "    if torch.version.hip is None:\n"
        header_str += "        default = os.path.join(third_party_dir, \"cuda\", \"lib\", \"libdevice.10.bc\")\n"
        header_str += "    else:\n"
        header_str += "        default = ''\n"
        header_str += "    return os.getenv(\"TRITON_LIBDEVICE_PATH\", default)\n"
        func_str = ""
        for symbols in self._symbol_groups.values():
            func_str += "@core.extern\n"
            func_name_str = f"def {symbols[0].op_name}("
            for arg_name in symbols[0].arg_names:
                func_name_str += f"{arg_name}, "
            func_name_str += "_builder=None):\n"

            return_str = f"\treturn core.extern_elementwise(\"{self._name}\", libdevice_path(), ["
            for arg_name in symbols[0].arg_names:
                return_str += f"{arg_name}, "
            return_str += "], \n"

            arg_type_symbol_dict_str = "{"
            for symbol in symbols:
                arg_type_symbol_dict_str += "("
                for arg_type in symbol.arg_types:
                    arg_type_symbol_dict_str += f'core.dtype("{arg_type}"),'
                ret_type = f'core.dtype("{symbol.ret_type}")'
                arg_type_symbol_dict_str += "): (\"" + symbol.name + "\", " + ret_type + "),\n"
            arg_type_symbol_dict_str += "}"

            return_str += arg_type_symbol_dict_str
            return_str += f", is_pure={self.is_pure}"
            return_str += ", _builder=_builder)\n"

            func_str += func_name_str + return_str + "\n"
        file_str = import_str + header_str + func_str

        return file_str


class LLVMDisassembler:
    _path: str
    _ll_file: str

    def __init__(self, path) -> None:
        '''
        Invoke llvm-dis to disassemble the given file.
        :param path: path to llvm-dis
        '''
        self._path = path
        self._ll_file = "/tmp/extern_lib.ll"

    def disasm(self, lib_path: str) -> None:
        subprocess.Popen([self._path, lib_path, "-o", self.ll_file], stdout=subprocess.PIPE).communicate()

    @property
    def ll_file(self) -> str:
        return self._ll_file

    @property
    def path(self) -> str:
        return self._path


extern_libs = ["libdevice"]


def build(
    llvm_dis_path: str,
    lib_path: str,
    lib_name: str,
    output_dir: str,
) -> None:
    '''
      Interface function to build the library file.
      :param llvm_dis_path: path to the llvm-dis binary
      :param lib_path: path to the external library file
      :param lib_name: name of the library
      :param output_dir: path to the output directory
    '''
    if lib_name == "libdevice":
        extern_lib = Libdevice(lib_path)
    else:
        raise Exception(f"Unknown extern library: {lib_name}")

    llvm_disassembler = LLVMDisassembler(llvm_dis_path)
    llvm_disassembler.disasm(lib_path)

    extern_lib.parse_symbols(llvm_disassembler.ll_file)
    extern_lib.generate_stub_file(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llvm-dis", dest="llvm_dis_path", help="Path to llvm-dis", default="llvm-dis")
    parser.add_argument("--lib-path", dest="lib_path", help="Path to the extern library")
    parser.add_argument("--lib-name", dest="lib_name", help="Name of the extern library")
    parser.add_argument("--output", dest="output_dir", help="Output file path", default="/tmp/")
    args = parser.parse_args()

    build(args.llvm_dis_path, args.lib_path, args.lib_name, args.output_dir)
