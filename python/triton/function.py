import triton.frameworks as fw
import triton.utils

class OpContext(object):

    def save_for_backward(self, *tensors):
      self.to_save = tensors
    
    @property
    def saved_tensors(self):
      return self.to_save

class function_meta(type):

    def __init__(cls, name, bases, attrs):
        cls.registered = False
        return super(function_meta, cls).__init__(name, bases, attrs)

ctx_registry = triton.utils.id_dict()

class function(metaclass = function_meta):
  
  @staticmethod
  def forward(ctx, *args, **kwargs):
    raise NotImplementedError

  @staticmethod
  def backward(ctx, grad_output):
    raise NotImplementedError

  @classmethod
  def apply_torch(cls, *args, **kwargs):
    class TorchFunction(fw.torch.autograd.Function):
      @staticmethod
      def forward(ctx, *targs, **tkwargs):
        y = cls.forward(ctx, *targs, **tkwargs)
        ctx_registry[y] = ctx
        return y
      @staticmethod
      def backward(ctx, grad_output):
        return cls.backward(ctx, grad_output)
    return TorchFunction.apply(*args, **kwargs)

  @classmethod
  def apply_tensorflow(cls, *args, **kwargs):
    ctx = OpContext()
    # Acquire a mutex here to ensure that calls to alloc_empty() 
    # are handled properly
    mutex = fw.gen_resource_variable_ops.mutex_v2()
    lock = fw.gen_resource_variable_ops.mutex_lock(mutex)
    with fw.tensorflow.python.ops.control_dependencies([lock]):
      result = cls.forward(ctx, *args, **kwargs)
    ctx_registry[result] = ctx
    # register backward
    name = result.op.op_def.name
    if not cls.registered:
      @fw.tensorflow.RegisterGradient(name)
      def gradient(op, dy):
        with fw.tensorflow.control_dependencies([op]):
          return cls.backward(ctx_registry[op.outputs[0]], dy)
      cls.registered = True
    # return result tensor
    return result

  @classmethod
  def apply(cls, *args, **kwargs):
    if fw.has_tensorflow():
        return cls.apply_tensorflow(*args, **kwargs)
    elif fw.has_torch():
        return cls.apply_torch(*args, **kwargs)
    else:
        assert False
