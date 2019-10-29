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
    with fw.tensorflow.control_dependencies([lock]):
      result = cls.forward(ctx, *args, **kwargs)
    # Find a mapping between ::forward arguments and tensorflow op arguments
    remap = dict()
    for i, ix in enumerate(result.op.inputs):
      for j, jx in enumerate(args):
        if ix is jx:
          remap[j] = i
    # register backward
    ctx_registry[result] = ctx
    name = result.op.op_def.name
    if not cls.registered:
      @fw.tensorflow.RegisterGradient(name)
      def gradient(op, dy):
        y = op.outputs[0]
        grad = cls.backward(ctx_registry[y], dy)
        # Remap gradient in the right order
        ret = [None] * len(op.inputs)
        for i in range(len(grad)):
          if i in remap:
            ret[remap[i]] = grad[i]
        # Return
        return ret
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
