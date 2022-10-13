import argparse
import subprocess
from abc import ABC, abstractmethod


class Symbol:
    def __init__(self, name: str, op_name: str, ret_type: str, arg_names: list, arg_types: list) -> None:
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
        self._arg_names = arg_names
        self._arg_types = arg_types

    @property
    def name(self):
        return self._name

    @property
    def op_name(self):
        return self._op_name

    @property
    def ret_type(self):
        return self._ret_type

    @property
    def arg_names(self):
        return self._arg_names

    @property
    def arg_types(self):
        return self._arg_types


def convert_type(type_str):
    if type_str == "i32":
        return "int32"
    elif type_str == "u32":
        return "uint32"
    elif type_str == "i64":
        return "int64"
    elif type_str == "u64":
        return "uint64"
    elif type_str == "float":
        return "float32"
    elif type_str == "double":
        return "float64"
    else:
        # ignore other types, such as pointer types
        return None


def to_unsigned(type_str):
    if type_str == "int32":
        return "uint32"
    elif type_str == "int64":
        return "uint64"
    else:
        return type_str


class ExternLibrary(ABC):
    def __init__(self, name: str, path: str, format: bool = True, grouping: bool = True) -> None:
        '''
        Abstract class for extern library.

        :param name: name of the library
        :param path: path of the library
        :param format: whether to format the generated stub file
        '''
        self._name = name
        self._path = path
        self._symbols = {}
        self._format = True
        self._grouping = grouping

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._path

    @property
    def symbols(self):
        return self._symbols

    @property
    def grouping(self):
        return self._grouping

    @abstractmethod
    def parse_symbols(self, input_file):
        pass

    @abstractmethod
    def _output_stubs(self) -> str:
        pass

    def generate_stub_file(self, output_dir):
        file_str = self._output_stubs()
        if file_str is None or len(file_str) == 0:
            raise Exception("file_str is empty")

        output_file = f"{output_dir}/{self._name}.py"
        with open(output_file, "w") as f:
            f.write(file_str)
            f.close()
            if self._format:
                subprocess.Popen(["autopep8", "-a", "-r", "-i", output_file],
                                 stdout=subprocess.PIPE).communicate()
                subprocess.Popen(["isort", output_file], stdout=subprocess.PIPE).communicate()


class Libdevice(ExternLibrary):
    def __init__(self, path) -> None:
        '''
        Constructor for Libdevice.

        :param path: path of the libdevice library
        '''
        super().__init__("libdevice", path)
        self._symbol_groups = {}

    def _extract_symbol(self, line):
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

    def _group_symbols(self):
        symbol_set = {}
        for symbol in self._symbols.values():
            op_name = symbol.op_name
            symbol_set[op_name] = symbol
        # The following cases are grouped together:
        # op_name, <u/ull/ll>op_name<ll/f/i>
        for symbol in self._symbols.values():
            op_name = symbol.op_name
            if "max" in op_name:
                op_name = "max"
            elif "min" in op_name:
                op_name = "min"
            elif "abs" in op_name:
                op_name = "abs"
            elif "pow" in op_name and "fast" in op_name:
                op_name = "pow"
            elif "round" in op_name:
                if "llround" in op_name:
                    op_name = "llround"
                else:
                    op_name = "round"
            elif "rint" in op_name:
                if "llrint" in op_name:
                    op_name = "llrint"
                else:
                    op_name = "rint"
            elif op_name.startswith("ull"):
                if "2" not in op_name:
                    # e.g., ullmax->max
                    op_name = op_name[3:]
                else:
                    # e.g., ull2double->ll2double
                    op_name = op_name[1:]
            elif op_name.startswith("u"):
                if "2" not in op_name:
                    # e.g., uhadd->hadd
                    op_name = op_name[1:]
                else:
                    # e.g., uint2double_rn->int2double_rn
                    op_name = op_name[1:]
            elif op_name.startswith("ll"):
                if "2" not in op_name:
                    # e.g., llmax->max
                    op_name = op_name[2:]
            elif op_name.endswith("ll"):
                op_name = op_name[:-2]
            elif op_name.endswith("f"):
                op_name = op_name[:-1]
            if op_name in symbol_set:
                # Update op_name only if there's an existing symbol
                symbol._op_name = op_name
            else:
                op_name = symbol._op_name
            if op_name in self._symbol_groups:
                self._symbol_groups[op_name].append(symbol)
            else:
                self._symbol_groups[op_name] = [symbol]

    def parse_symbols(self, input_file):
        if len(self.symbols) > 0:
            return
        output = subprocess.check_output(["grep", "define", input_file]).decode().splitlines()
        for line in output:
            symbol = self._extract_symbol(line)
            if symbol is None:
                continue
            self._symbols[symbol.name] = symbol

        self._group_symbols()

    def _output_stubs(self):
        # Generate python functions in the following format:
        # @extern.extern
        # def <op_name>(<args>, _builder=None):
        #   arg_type_symbol_dict = {[arg_type]: {(symbol, ret_type)}}
        #   return extern.dispatch("libdevice", <path>, <args>, <arg_type_symbol_dict>, _builder)
        import_str = "from . import core, extern\n"
        import_str += "import os\n"
        header_str = "LIBDEVICE_PATH = os.path.dirname(os.path.abspath(__file__)) + \"/libdevice.10.bc\"\n"
        func_str = ""
        for symbols in self._symbol_groups.values():
            func_str += "@extern.extern\n"
            func_name_str = f"def {symbols[0].op_name}("
            for arg_name in symbols[0].arg_names:
                func_name_str += f"{arg_name}, "
            func_name_str += "_builder=None):\n"

            return_str = f"\treturn extern.elementwise(\"{self._name}\", LIBDEVICE_PATH, ["
            for arg_name in symbols[0].arg_names:
                return_str += f"{arg_name}, "
            return_str += "], \n"

            arg_type_symbol_dict_str = "{"
            for symbol in symbols:
                arg_type_symbol_dict_str += "("
                for arg_type in symbol.arg_types:
                    arg_type_symbol_dict_str += f"core.{arg_type},"
                ret_type = f"core.{symbol.ret_type}"
                arg_type_symbol_dict_str += "): (\"" + symbol.name + "\", " + ret_type + "),\n"
            arg_type_symbol_dict_str += "}"

            return_str += arg_type_symbol_dict_str
            return_str += ", _builder)\n"

            func_str += func_name_str + return_str + "\n"
        file_str = import_str + header_str + func_str

        return file_str


class LLVMDisassembler:
    def __init__(self, path):
        '''
        Invoke llvm-dis to disassemble the given file.

        :param path: path to llvm-dis
        '''
        self._path = path
        self._ll_file = "/tmp/extern_lib.ll"

    def disasm(self, lib_path):
        subprocess.Popen([self._path, lib_path, "-o", self.ll_file],
                         stdout=subprocess.PIPE).communicate()

    @property
    def ll_file(self):
        return self._ll_file

    @property
    def path(self):
        return self._path


extern_libs = ["libdevice"]


def build(llvm_dis_path, lib_path, lib_name, output_dir):
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
    parser.add_argument("-llvm", dest="llvm_dis_path", help="path to llvm-dis", default="llvm-dis")
    parser.add_argument("--lib-path", dest="lib_path", help="path to the extern library")
    parser.add_argument("--lib-name", dest="lib_name", help="name of the extern library")
    parser.add_argument("-o", dest="output_dir", help="output file path", default="/tmp/")
    args = parser.parse_args()

    build(args.llvm_dis_path, args.lib_path, args.lib_name, args.output_dir)
