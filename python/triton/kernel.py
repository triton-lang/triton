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

def _make_framework_src(src, out, tmp, grid):
  if fw.has_tensorflow():
    return libtriton.make_tensorflow_src(src, out, tmp, grid)
  elif fw.has_torch:
    return libtriton.make_torch_src(src, out, tmp, grid)
  else:
    assert False

def _make_cache_path(src):
  md5 = hashlib.sha1(src.encode())
  hexhash = md5.hexdigest()
  home = os.path.expanduser('~')
  cacheroot = os.path.join(home, '.triton', 'cache')
  cachepath = os.path.join(cacheroot, str(hexhash))
  if not os.path.exists(cachepath):
    os.makedirs(cachepath)
  return cachepath

def _write_bindings(src, root):
  if fw.has_tensorflow():
    name = 'tensorflow'
  elif fw.has_torch():
    name = 'torch'
  else:
    assert False
  cpp = os.path.join(root, '{name}.cpp'.format(name=name))
  suffix = sysconfig.get_config_var('EXT_SUFFIX')
  so = os.path.join(root, '{name}{suffix}'.format(name=name, suffix=suffix))
  recompile = False
  # recompile if .so does not exist
  if not os.path.exists(cpp) or not os.path.exists(so):
    recompile = True
  # recompile if cpp was modified after .so
  elif max(cpp, so, key=os.path.getctime) == cpp:
    recompile = True
  # write cpp file
  if recompile:
    with open(cpp, 'w+') as handle:
      handle.writelines(src)
  # return path of cpp file
  return (cpp, so)

@contextlib.contextmanager
def quiet():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

def _build(src, path):
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
  if fw.has_tensorflow():
    library_dirs += [fw.tensorflow.sysconfig.get_lib()]
    include_dirs += [fw.tensorflow.sysconfig.get_include()]
    include_dirs += ['/usr/local/cuda/include/']
    libraries += [fw.tensorflow.sysconfig.get_link_flags()[1].replace('-l', '')]
    abi = fw.tensorflow.__cxx11_abi_flag__ if "__cxx11_abi_flag__" in fw.tensorflow.__dict__ else 0
    extra_compile_args += ['-D_GLIBCXX_USE_CXX11_ABI={abi}'.format(abi=abi)]
    name = 'tensorflow'
  elif fw.has_torch():
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
    name = 'torch'
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
  # tensorflow type
  if fw.has_tensorflow():
    if isinstance(obj, fw.tensorflow.DType):
      return {fw.tensorflow.int8: 'char',
              fw.tensorflow.int16: 'short',
              fw.tensorflow.int32: 'int',
              fw.tensorflow.int64: 'long',
              fw.tensorflow.float16: 'half',
              fw.tensorflow.float32: 'float',
              fw.tensorflow.float64: 'double'}[obj]
  # torch type
  elif fw.has_torch():
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


def _make_framework_op(src, outputs, tmp, options):
  src, name = _make_framework_src(src, outputs, tmp, options)
  cache_path = _make_cache_path(src)
  cpp, so = _write_bindings(src, cache_path)
  _build(cpp, cache_path)
  if fw.has_tensorflow():
    return fw.tensorflow.load_op_library(so).__dict__[name]
  elif fw.has_torch():
    fw.torch.ops.load_library(so)
    return getattr(fw.torch.ops.triton, name)
  else:
    assert False

def _make_grid(args) :
  scalars = [x for x in args[:-1] if isinstance(x, triton.utils.scalar)]
  def grid(opt):
    for x in scalars:
      x.set_assume_initialized()
    result = args[-1](opt)
    for x in scalars:
      x.unset_assume_initialized()
    return result
  return grid


bench_registry = triton.utils.id_dict()

class kernel:

  def __init__(self, src, outputs, tmp=[]):
    self.fw_id = dict()
    self.fw_grids = dict()
    self.fw_op = None
    self.src = src
    self.outputs = outputs
    self.tmp = tmp
    self.cst = dict()

  def set_constant(self, name, value):
    self.cst[name] = value

  def __call__(self, *args, **kwargs):
    #########################
    # cache
    ########################
    # create a new framework op when defines are different
    key = '-'.join(['{key}-{val}'.format(key=key, val=val) for key, val in kwargs.items()])
    if key not in self.fw_id.keys():
      # code generation options
      defines = []
      for k, v in kwargs.items():
        cvt = lambda x: _cvt_to_def_str(x)
        if(isinstance(v, list)):
          values = list(map(cvt, v))
        else:
          values = [cvt(v)]
        defines.append((k, values))
      opt = libtriton.options_space()
      opt.defines = defines
      opt.num_warps = [2, 4]
      # create unique id for this op
      op_id = libtriton.make_op_id()
      self.fw_id[key] = op_id
      # register function
      libtriton.register_fn(op_id, self.src, opt)
      for name, value in self.cst.items():
        libtriton.register_cst(op_id, name, value)
      if self.fw_op is None:
        self.fw_op = _make_framework_op(self.src, self.outputs, self.tmp, opt)

    ########################
    # initialize
    ########################
    op_id = self.fw_id[key]
    libtriton.register_grid(op_id, args[-1])
    bench = kwargs['bench'] if 'bench' in kwargs else 0
    bench_id = libtriton.make_scalar_id() if bench > 0 else -1

    #########################
    # call framework function
    #########################
    if fw.has_tensorflow():
      empty = [x for x in args[:-1] if isinstance(x, triton.utils.tf_empty_proxy)]
      if len(empty) != len(self.outputs):
        raise ValueError('Number of empty arguments does not much number of outputs provided')
      # operands
      operands = [x.shape if isinstance(x, triton.utils.tf_empty_proxy) else x for x in args[:-1]]
      # output data types
      kwargs = {'id': op_id, 'bench': bench, 'bench_id': bench_id}
      for i, x in enumerate(args[:-1]):
        if isinstance(x, triton.utils.tf_empty_proxy):
          kwargs['T' + str(i)] = x.dtype
      # launch
      ret = self.fw_op(*operands, **kwargs)
      ret = [ret] if isinstance(ret, fw.tensorflow.Tensor) else ret
      op_def = ret[0].op.op_def
       # fill empty tensors with corresponding values
      for j, y in enumerate(op_def.output_arg):
        found = False
        for i, x in enumerate(op_def.input_arg):
          if y.name + '_shape' == x.name:
            args[i].tensor = ret[j]
            found = True
        assert found
      # store timing information
      if bench > 0:
        for y in ret: 
          bench_registry[y] = triton.utils.id_dict.lazy_entry(bench_id)
    
    ############################
    # call torch function
    ############################
    elif fw.has_torch():
      args = [x if isinstance(x, fw.torch.Tensor) else x for x in args[:-1]]
      ret = self.fw_op(op_id, bench, bench_id, *args)
      if bench > 0:
        bench_registry[ret] = libtriton.retrieve_scalar(bench_id)

    else:
      assert False