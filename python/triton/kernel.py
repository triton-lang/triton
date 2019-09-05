# import for cache
import os
import tempfile
import shutil
import hashlib
import sysconfig
import sys
# import for just-in-time compilation
import distutils
import setuptools.command.build_ext
import setuptools
# triton
import triton.frameworks as fw
import triton.utils
import libtriton

def _make_framework_src(src, out, grid):
  if fw.has_tensorflow():
    return libtriton.make_tensorflow_src(src, out, grid)
  elif fw.has_torch:
    return libtriton.make_torch_src(src, out, grid)
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
  
def _build(src, path):
  # include directories
  triton_include_dirs = ['/home/philippe/development/triton/include']
  include_dirs = triton_include_dirs 
  # library directories
  triton_library_dirs = [os.path.realpath(os.path.join(libtriton.__file__, os.path.pardir))]
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
      extra_compile_args = extra_compile_args,
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


def _make_framework_op(src, outputs, options):
  src, name = _make_framework_src(src, outputs, options)
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


class kernel:

  def __init__(self, src, outputs):
    self.fw_id = dict()
    self.fw_grids = dict()
    self.fw_op = None
    self.src = src
    self.outputs = outputs

  def _init_framework(self):
    if fw.has_tensorflow():
      fw._import_tensorflow()
      fw._import_tf_extra_ops()
    elif fw.has_torch():
      fw._import_torch()
    else:
      assert False

  def __call__(self, *args, **kwargs):
    self._init_framework()
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
      opt.num_warps = [4]
      # create unique id for this op
      op_id = libtriton.make_op_id()
      self.fw_id[key] = op_id
      # register function
      libtriton.register_fn(op_id, self.src, opt)
      if self.fw_op is None:
        self.fw_op = _make_framework_op(self.src, self.outputs, opt)

    # retrieve framework op
    op_id = self.fw_id[key]
    # register grid
    libtriton.register_grid(op_id, _make_grid(args))
    # create operands
    op_args = [x.handle if isinstance(x, triton.utils.scalar) else x for x in args[:-1]]
    # call framework function
    if fw.has_tensorflow():
      return self.fw_op(*op_args, id=op_id)
    elif fw.has_torch():
      return self.fw_op(op_id, *op_args)
    else:
      assert False