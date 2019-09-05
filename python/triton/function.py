import triton.frameworks as fw

class OpContext(object):

    def save_for_backward(self, *tensors):
      self.to_save = tensors
    
    @property
    def saved_tensors(self):
      return self.to_save

class function_meta(type):

    def __init__(cls, name, bases, attrs):
        cls.contexts = dict()
        cls.registered = False
        return super(function_meta, cls).__init__(name, bases, attrs)

class function(metaclass = function_meta):
  
  @staticmethod
  def forward(ctx, *args, **kwargs):
    raise NotImplementedError

  @staticmethod
  def backward(ctx, grad_output):
    raise NotImplementedError

  @classmethod
  def apply_torch(cls, *args, **kwargs):
    fw._import_torch()
    class TorchFunction(fw.torch.autograd.Function):
      @staticmethod
      def forward(ctx, *targs, **tkwargs):
        return cls.forward(ctx, *targs, **tkwargs)
      @staticmethod
      def backward(ctx, grad_output):
        return cls.backward(ctx, grad_output)
    return TorchFunction.apply(*args, **kwargs)

  @classmethod
  def apply_tensorflow(cls, *args, **kwargs):
    fw._import_tensorflow()
    ctx = OpContext()
    result = cls.forward(ctx, *args, **kwargs)
    id = result.op.get_attr('id')
    cls.contexts[id] = ctx
    # register backward
    name = result.op.op_def.name
    if not cls.registered:
      @fw.tensorflow.RegisterGradient(name)
      def gradient(op, dy):
        id = op.get_attr('id')
        return cls.backward(cls.contexts[id], dy)
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
