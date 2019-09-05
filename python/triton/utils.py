import triton.frameworks as fw
import libtriton

def cdiv(a, b):
    return -(-a // b)

def empty(shapes, dtype):
  if fw.has_tensorflow():
    args = [x.handle if isinstance(x, scalar) else x for x in shapes]
    args = fw.tensorflow.stack(args)
    return fw.tf_extra_ops.alloc_empty(args, T = dtype)
  elif fw.has_torch():
    return fw.torch.empty(*shapes).cuda()

class lazy_shape:

  def __init__(self, shape):
    self.shape = shape
  
  def __getitem__(self, key):
    return scalar(self.shape[key])

def shape(A) :
  if fw.has_tensorflow():
    return lazy_shape(fw.tensorflow.shape(A))
  elif fw.has_torch():
    return A.shape
  else:
    assert False


class scalar:
  
  def __init__(self, x):
    self.id = libtriton.make_scalar_id()
    self.handle = fw.tf_extra_ops.register_scalar(x, id=self.id)
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


