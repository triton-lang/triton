# import for cache
import os
import tempfile
import shutil
import hashlib
import sysconfig
import sys
import weakref
import contextlib
import io
# import for just-in-time compilation
import distutils
import setuptools.command.build_ext
import setuptools
# triton
import triton.frameworks as fw
import triton.utils
import triton._C.libtriton as libtriton


@contextlib.contextmanager
def quiet():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

def _build(src, path, name):
  ccdir = os.path.join(libtriton.__file__, os.path.pardir)
  ccdir = os.path.realpath(ccdir)
  # include directories
  triton_include_dirs = [os.path.join(ccdir, 'include')]
  include_dirs = triton_include_dirs 
  # library directories
  triton_library_dirs = [ccdir]
  library_dirs = triton_library_dirs
  # libraries
  libraries = ['triton']
  # add framework
  extra_compile_args = []
  if fw.has_torch():
    prefix = os.path.dirname(fw.torch.__file__)
    library_dirs += [os.path.join(prefix, 'lib')]
    include_dirs += ['/usr/local/cuda/include/',
                     os.path.join(prefix, 'lib', 'include'),
                     os.path.join(prefix, 'lib', 'include', 'torch', 'csrc', 'api', 'include'),
                     os.path.join(prefix, 'include'),
                     os.path.join(prefix, 'include', 'torch', 'csrc', 'api', 'include')]
    libraries += ['torch']
    abi = fw.torch._C._GLIBCXX_USE_CXX11_ABI
    extra_compile_args += ['-D_GLIBCXX_USE_CXX11_ABI={abi}'.format(abi=abi)]
  else:
    assert False
  # extra arguments
  extra_link_args = []
  # dependences
  depends = [os.path.realpath(libtriton.__file__)]
  # create extension module
  ext = setuptools.Extension(
      name = name,
      language = 'c++',
      sources = [src],
      include_dirs = include_dirs,
      extra_compile_args = extra_compile_args + ['-g0'],
      extra_link_args = extra_link_args,
      library_dirs = library_dirs,
      libraries = libraries,
      depends = depends
  )
  # build extension module
  args = ['build_ext']
  tmp = tempfile.mkdtemp()
  args.append('--build-temp=' + tmp)
  args.append('--build-lib=' + path)
  args.append('-q')
  args = dict(
      name = name,
      ext_modules = [ext],
      script_args = args,
  ) 
  with quiet():
    setuptools.setup(**args)
  shutil.rmtree(tmp)

def _cvt_to_def_str(obj):
  # bool
  if isinstance(obj, bool):
    return str(int(obj))
  # torch type
  if fw.has_torch():
    if isinstance(obj, fw.torch.dtype):
      return {fw.torch.int8: 'char',
              fw.torch.int16: 'short',
              fw.torch.int32: 'int',
              fw.torch.int64: 'long',
              fw.torch.float16: 'half',
              fw.torch.float32: 'float',
              fw.torch.float64: 'double'}[obj]
  else:
    assert False
  # default
  return str(obj)


def _encode(arg_types):
  codes = {
    libtriton.arg_type.int1:   'i1',
    libtriton.arg_type.int8:   'i8',
    libtriton.arg_type.int32:  'i32',
    libtriton.arg_type.int64:  'i64',
    libtriton.arg_type.half:   'f16',
    libtriton.arg_type.float:  'f32',
    libtriton.arg_type.double: 'f64',
    libtriton.arg_type.buffer: 'buf'
  }
  ret = '_'.join(map(codes.get, arg_types))
  return ret

def _make_framework_op(arg_types):
  name = _encode(arg_types)
  # path of .cpp and .so file
  home = os.path.expanduser('~')
  root = os.path.join(home, '.triton', 'torch', name)
  if not os.path.exists(root):
    os.makedirs(root)
  suffix = sysconfig.get_config_var('EXT_SUFFIX')
  so = os.path.join(root, f'op{suffix}')
  cpp = os.path.join(root, f'op.cpp')
  # handle cached .so file
  if os.path.exists(so):
    tt_mtime = os.stat(os.path.realpath(libtriton.__file__)).st_mtime
    so_mtime = os.stat(so).st_mtime
    # can use cached if libtriton is older than the .so
    if tt_mtime < so_mtime:
      fw.torch.ops.load_library(so)
      return getattr(fw.torch.ops.triton, name)
  # create torch source code
  src, _ = libtriton.make_torch_src(name, arg_types)
  with open(cpp, 'w+') as handle:
      handle.writelines(src)
  # compile torch source code
  _build(cpp, root, 'op')
  fw.torch.ops.load_library(so)
  return getattr(fw.torch.ops.triton, name)
  

class kernel:

  def __init__(self, src, defines = dict(), num_warps = [2, 4, 8]):
    self.src = src
    # create constants
    self.cst = dict()
    # create triton op
    macros = []
    for k, v in defines.items():
      cvt = lambda x: _cvt_to_def_str(x)
      if(isinstance(v, list)):
        values = list(map(cvt, v))
      else:
        values = [cvt(v)]
      macros.append((k, values))
    opt = libtriton.options_space()
    opt.defines = macros
    opt.num_warps = num_warps
    self.op_id = libtriton.make_op_id()
    libtriton.register_fn(self.op_id, self.src, opt, os.path.realpath(libtriton.__file__))
    # create pytorch hook
    arg_types = libtriton.get_fn_signature(self.src, opt)
    self.fw_op = _make_framework_op(arg_types)

  def set_constant(self, name, value):
    libtriton.register_cst(self.op_id, name, value)

  def __call__(self, *args, **kwargs):
    # launch options
    bench = kwargs['bench']         if 'bench'     in kwargs else 0
    bench_id = libtriton.make_scalar_id() if bench > 0 else -1
    # launch grid
    if 'grid' not in kwargs:
      raise RuntimeError('Must provide grid for kernel launch')
    grid = kwargs['grid']
    libtriton.register_grid(self.op_id, grid)
    # launch
    self.fw_op(self.op_id, bench, bench_id, *args)
    if bench > 0:
      return libtriton.retrieve_scalar(bench_id)