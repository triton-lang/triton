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
  def extract_tf_tensors(cls, lst, err):
    for x in lst:
      if x and not isinstance(x, triton.utils.tf_empty_proxy):
        raise ValueError('Results of ' + err + ' must be created using triton.empty()')
      if x and x.tensor is None:
        raise ValueError('Empty tensor never filled during ' + err)
    return [x.tensor if x else None for x in lst]
  
  @classmethod
  def apply_tensorflow(cls, *args, **kwargs):
    ctx = OpContext()
    result = cls.forward(ctx, *args, **kwargs)

    # check that all the results stem from triton.empty
    # and get the corresponding TF tensors if possible
    result = result if isinstance(result, tuple) else (result, )
    result = function.extract_tf_tensors(result, 'forward')
    
    # Find a mapping between ::forward arguments and tensorflow op arguments
    op = result[0].op
    remap = dict()
    for i, ix in enumerate(op.inputs):
      for j, jx in enumerate(args):
        if ix is jx:
          remap[j] = i

    # Register backward pass
    ctx_registry[op] = ctx
    name = op.op_def.name
    if not cls.registered:
      @fw.tensorflow.RegisterGradient(name)
      def gradient(op, *dy):
        dy = dy if len(dy) > 1 else dy[0]
        grad = cls.backward(ctx_registry[op], dy)
        grad = function.extract_tf_tensors(grad, 'backward')

        # Remap gradient in the right order
        ret = [None] * len(op.inputs)
        for i in range(len(grad)):
          if i in remap:
            ret[remap[i]] = grad[i]
        # Return
        return ret
      cls.registered = True

    # Return tensor
    return result

  @classmethod
  def apply(cls, *args, **kwargs):
    if fw.has_tensorflow():
        return cls.apply_tensorflow(*args, **kwargs)
    elif fw.has_torch():
        return cls.apply_torch(*args, **kwargs)
    else:
        assert False
