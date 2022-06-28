import argparse
import subprocess


class Symbol:
  def __init__(self, name, ret_type, arg_names: list, arg_types: list) -> None:
    self._name = name
    self._op_name = name.split("nv_")[1]
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
  if type_str == 'i32':
    return 'int32'
  elif type_str == 'u32':
    return 'uint32'
  elif type_str == 'i64':
    return 'int64'
  elif type_str == 'u64':
    return 'uint64'
  elif type_str == 'float':
    return 'fp32'
  elif type_str == 'double':
    return 'fp64'
  else:
    # ignore other types, such as pointer types
    return None


class Libdevice:
  def __init__(self, path) -> None:
    self._path = path
    self._symbols = {}
    self._symbol_groups = {}

  def _extract_symbol(self, line):
    # Extract symbols from line in the following format:
    # "define [internal] <ret_type> @<name>(<arg_types>,)"
    entries = line.split('@')
    ret_str = entries[0]
    func_str = entries[1]
    # Get ret_type, skip internal symbols
    ret_strs = ret_str.split()
    if ret_strs[1] == 'internal':
      return None
    ret_type = convert_type(ret_strs[1])
    if ret_type is None:
      return None
    # Get function name
    func_strs = func_str.split('(')
    func_name = func_strs[0].replace('@', '')
    # Get arg_types
    arg_strs = func_strs[1].split(',')
    arg_types = []
    arg_names = []
    for arg_str in arg_strs:
      arg_type = convert_type(arg_str.split()[0])
      if arg_type is None:
        return None
      arg_name = arg_str.split()[1].replace('%', '')
      arg_types.append(arg_type)
      arg_names.append(arg_name)
    return Symbol(func_name, ret_type, arg_names, arg_types)

  def _group_symbols(self):
    # The following cases are grouped together:
    # op_name, <u>op_name<ll/f>
    for symbol in self._symbols.values():
      op_name = symbol.op_name
      if op_name.startswith('u'):
        op_name = op_name[1:]
      elif op_name.endswith('ll'):
        op_name = op_name[:-2]
      elif op_name.endswith('f'):
        op_name = op_name[:-1]
      if op_name in self._symbol_groups:
        self._symbol_groups[op_name].append(symbol)
      else:
        self._symbol_groups[op_name] = [symbol]

  def parse_symbols(self):
    if len(self.symbols) > 0:
      return
    output = subprocess.check_output(['grep', 'define', 'tmp/libdevice.ll']).decode().splitlines()
    for line in output:
      symbol = self._extract_symbol(line)
      if symbol is None:
        continue
      self._symbols[symbol.name] = symbol

    self._group_symbols()

  def generate_stubs(self, output_dir):
    # Generate python functions in the following format:
    # @core.builtin
    # def <op_name>(<args>, _builder=None):
    #   arg_type_symbol_dict = {[arg_type]: {(symbol, ret_type)}}
    #   return extern.dispatch(<path>, <args>, <arg_type_dict>, _builder)
    import_str = "from . import core, extern\n"
    functions_str = "core.builtin\n"
    for symbols in self._symbol_groups.values():
      func_name_str = f"def {symbols[0].op_name}("
      for arg_name in symbols[0].arg_names:
        func_name_str += f"{arg_name}, "
      func_name_str += "_builder=None):\n"

      return_str = f"return extern.dispatch({self.path}, "
      for arg_name in symbol.arg_names:
        return_str += f"[{arg_name}, "
      return_str += "], "

      arg_type_symbol_dict_str = "{"
      for symbol in symbols:
        arg_type_symbol_dict_str += "["
        for arg_type in symbol.arg_types:
          arg_type_symbol_dict_str += arg_type + ","
        arg_type_symbol_dict_str += f"]: {(symbol.name, symbol.ret_type)},"
      arg_type_symbol_dict_str += "}"
      
      return_str += arg_type_symbol_dict_str 
      return_str += ", _builder)\n"

      functions_str += func_name_str + return_str + "\n"
    
    with open(f"{output_dir}/libdevice.py", "w") as f:
      f.write(functions_str)
      f.close()


class LLVMDisassembler:
  def __init__(self, path):
    self.path = path

  def disasm(self):
    subprocess.Popen([self.path, libdevice, '-o', '/tmp/libdevice.ll'],
                     stdout=subprocess.PIPE).communicate()


def extract_symbols(output):
  for line in output:
    print(line)


parser = argparse.ArgumentParser()
parser.add_argument('--llvm-dis-path', help='path to llvm-dis', default='llvm-dis')
parser.add_argument('--libdevice-path', help='path to libdevice.10.bc',
                    default="/usr/local/cuda/nvvm/libdevice/libdevice.10.bc")
parser.add_argument('--output-dir', help='output file path', default='/tmp/')
args = parser.parse_args()

llvm_disassembler = LLVMDisassembler(args.llvm_dis_path)
llvm_disassembler.disasm()

libdevice = Libdevice(args.libdevice_path)
libdevice.parse_symbols()
libdevice.generate_stubs(args.output_dir)
