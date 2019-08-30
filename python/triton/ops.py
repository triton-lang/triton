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
import libtriton


torch_id = 'torch'
tensorflow_id = 'tensorflow'

torch = None
tensorflow = None
tf_extra_ops = None

def _import_torch():
  global torch
  if torch is None:
    import torch

def _import_tensorflow():
  global tensorflow
  if tensorflow is None:
    import tensorflow

def _import_tf_extra_ops():
  global tf_extra_ops
  if tf_extra_ops is None:
    path = os.path.dirname(libtriton.__file__)
    path = os.path.join(path, 'libextra_tf_ops.so')
    _import_tensorflow()
    tf_extra_ops = tensorflow.load_op_library(path)


def _find_framework(default = None):
    is_tf_imported = 'tensorflow' in sys.modules
    is_torch_imported = 'torch' in sys.modules
    if default:
      if default not in [tensorflow_id, torch_id]:
        raise ValueError('unsupported framework')
      else:
        return default
    elif is_tf_imported and not is_torch_imported:
      return tensorflow_id
    elif is_torch_imported and not is_tf_imported:
      return torch_id
    else:
      raise ValueError('cannot determine imported framework, '
                       'please provide framework argument')


def _make_framework_src(src, out, grid, framework):
  if framework == tensorflow_id:
    return libtriton.make_tensorflow_src(src, out, grid)
  elif framework == torch_id:
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

def _write_bindings(src, root, framework):
  cpp = os.path.join(root, '{framework}.cpp'.format(framework=framework))
  suffix = sysconfig.get_config_var('EXT_SUFFIX')
  so = os.path.join(root, '{framework}{suffix}'.format(framework=framework, suffix=suffix))
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
  
def _build(src, path, framework):
  # include directories
  triton_include_dirs = ['/home/philippe/development/triton/include']
  include_dirs = triton_include_dirs 
  # library directories
  triton_library_dirs = [os.path.realpath(os.path.join(libtriton.__file__, os.path.pardir))]
  library_dirs = triton_library_dirs
  # libraries
  libraries = ['triton']
  # add framework
  if framework == tensorflow_id:
    _import_tensorflow()
    library_dirs += [tensorflow.sysconfig.get_lib()]
    include_dirs += [tensorflow.sysconfig.get_lib()]
    libraries += ['tensorflow_framework']
  elif framework == torch_id:
    _import_torch()
    prefix = os.path.dirname(torch.__file__)
    library_dirs += [os.path.join(prefix, 'lib')]
    include_dirs += [os.path.join(prefix, 'lib', 'include'),
                     os.path.join(prefix, 'lib', 'include', 'torch', 'csrc', 'api', 'include'),
                     os.path.join(prefix, 'include'),
                     os.path.join(prefix, 'include', 'torch', 'csrc', 'api', 'include')]
    libraries += ['torch']
  else:
    assert False
  # extra arguments
  extra_compile_args = []
  extra_link_args = []
  # dependences
  depends = [os.path.realpath(libtriton.__file__)]
  # create extension module
  ext = setuptools.Extension(
      name = 'tensorflow',
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
      name = 'tensorflow',
      ext_modules = [ext],
      script_args = args,
  ) 
  setuptools.setup(**args)
  shutil.rmtree(tmp)

def _cvt_to_def_str(obj, framework):
  # bool
  if isinstance(obj, bool):
    return str(int(obj))
  # tensorflow type
  if framework == tensorflow_id:
    _import_tensorflow()
    if isinstance(obj, tensorflow.DType):
      return {tensorflow.int8: 'char',
              tensorflow.int16: 'short',
              tensorflow.int32: 'int',
              tensorflow.int64: 'long',
              tensorflow.float16: 'half',
              tensorflow.float32: 'float',
              tensorflow.float64: 'double'}[obj]
  # torch type
  elif framework == torch_id:
    _import_torch()
    if isinstance(obj, torch.dtype):
      return {torch.int8: 'char',
              torch.int16: 'short',
              torch.int32: 'int',
              torch.int64: 'long',
              torch.float16: 'half',
              torch.float32: 'float',
              torch.float64: 'double'}[obj]
  else:
    assert False
  # default
  return str(obj)


def _make_framework_op(src, outputs, options, framework):
  src, name = _make_framework_src(src, outputs, options, framework)
  cache_path = _make_cache_path(src)
  cpp, so = _write_bindings(src, cache_path, framework)
  _build(cpp, cache_path, framework)
  if framework == tensorflow_id:
    _import_tensorflow()
    return tensorflow.load_op_library(so).__dict__[name]
  elif framework == torch_id:
    _import_torch()
    torch.ops.load_library(so)
    return torch.ops.triton.__dict__[name]
  else:
    assert False

def _make_grid(args) :
  scalars = [x for x in args[:-1] if isinstance(x, scalar)]
  def grid(opt):
    for x in scalars:
      x.set_assume_initialized()
    result = args[-1](opt)
    for x in scalars:
      x.unset_assume_initialized()
    return result
  return grid

class op:

  def __init__(self, src, outputs, framework = None):
    self.fw_id = dict()
    self.fw_ops = dict()
    self.fw_grids = dict()
    self.src = src
    self.outputs = outputs
    self.framework = _find_framework(None)
      
  def __call__(self, *args, **kwargs):
    # create a new op when defines are different
    key = zip(kwargs.keys(), kwargs.values())
    if key not in self.fw_ops:
      # code generation options
      defines = []
      for k, v in kwargs.items():
        cvt = lambda x: _cvt_to_def_str(x, self.framework)
        try:
          values = list(map(cvt, v))
        except TypeError:
          values = [cvt(v)]
        defines.append((k, values))
      opt = libtriton.options_space()
      opt.defines = defines
      opt.num_warps = [1, 2, 4, 8]
      # create unique id for this op
      op_id = libtriton.make_op_id()
      self.fw_id[key] = op_id
      # register function
      libtriton.register_fn(op_id, self.src, opt)
      self.fw_ops[key] = _make_framework_op(self.src, self.outputs, opt, self.framework)

    # retrieve framework op
    op_id = self.fw_id[key]
    op = self.fw_ops[key]
    # register grid
    grid = _make_grid(args)
    self.fw_grids[key] = grid
    libtriton.register_grid(op_id, self.fw_grids[key])
    # create operands
    op_args = [x.handle if isinstance(x, scalar) else x for x in args[:-1]]
    # call framework op
    return op(*op_args, id=op_id)


# class register_gradient:

#   def __init__(self, op):
#     self.op = op

#   def __call__(self, f):
#     name = 'Dot'
#     ops.RegisterGradient(name)(f)


def empty(shapes, framework = None):
  framework = _find_framework(framework)
  if framework == tensorflow_id:
    _import_tensorflow()
    _import_tf_extra_ops
    args = [x.handle if isinstance(x, scalar) else x for x in shapes]
    args = tensorflow.stack(args)
    return tf_extra_ops.alloc_empty(args)
  elif framework == torch_id:
    _import_torch()
    return torch.empty(*shapes)

class scalar:
  
  def __init__(self, x):
    _import_tf_extra_ops()
    self.id = libtriton.make_scalar_id()
    self.handle = tf_extra_ops.register_scalar(x, id=self.id)
    self.assume_initialized = False
  
  def set_assume_initialized(self):
    self.assume_initialized = True
  
  def unset_assume_initialized(self):
    self.assume_initialized = False

  def get_value(self):
    if self.assume_initialized:
      return libtriton.retrieve_scalar(self.id)
    else:
      return self.handle

  def __add__(self, other):
    return self.get_value() + other

  def __radd__(self, other):
    return other + self.get_value()

  def __sub__(self, other):
    return self.get_value() - other
  
  def __rsub(self, other):
    return other - self.get_value()
  
  def __mul__(self, other):
    return self.get_value() * other
  
  def __rmul(self, other):
    return other * self.get_value()

  def __floordiv__(self, other):
    return self.get_value() // other
  
  def __rfloordiv__(self, other):
    return other // self.get_value()

  def __div__(self, other):
    return self.get_value() / other

  def __rdiv__(self, other):
    return other / self.get_value()

  def __truediv__(self, other):
    self.get_value().__truediv__(other)
  
  def __rtruediv__(self, other):
    other.__truediv__(self.get_value())
  
  def __neg__(self):
    return -self.get_value()

class lazy_shape:

  def __init__(self, shape):
    self.shape = shape
  
  def __getitem__(self, key):
    return scalar(self.shape[key])

def shape(A) :
  _import_tensorflow()
  return lazy_shape(tensorflow.shape(A))

