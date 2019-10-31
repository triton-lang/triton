import triton.frameworks as fw
import triton.utils as utils

class OpContext(object):

    def __init__(self):
      self.to_save = []

    def save_for_backward(self, *tensors):
      self.to_save = [x.to_tensor() if isinstance(x, utils.tf_empty_proxy) else x 
                      for x in tensors]
    
    @property
    def saved_tensors(self):
      return self.to_save

class function_meta(type):

    def __init__(cls, name, bases, attrs):
        cls.registered = False
        return super(function_meta, cls).__init__(name, bases, attrs)

ctx_registry = utils.id_dict()

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
    ret = []
    for x in lst:
      if x is None:
        ret += [None]
      elif isinstance(x, fw.tensorflow.Tensor):
        ret += [x]
      elif isinstance(x, utils.tf_empty_proxy):
        if x.tensor is None:
          raise ValueError('Empty tensor never filled during ' + err)
        else:
          ret += [x.tensor]
      else:
        raise ValueError('Unsupported return type', type(x))
    return ret
  
  @classmethod
  def map_in_to_args(cls, op, args):
    ret = dict()
    for i, ix in enumerate(op.inputs):
      for j, jx in enumerate(args):
        if ix is jx:
          ret[j] = i
    return ret
  
  @classmethod
  def map_res_to_out(cls, op, result):
    ret = []
    for i, ix in enumerate(result):
      for j, jx in enumerate(op.outputs):
        if ix is jx:
          ret.append(j)
    return ret
    
  @classmethod
  def apply_tensorflow(cls, *args, **kwargs):
    ctx = OpContext()
    result = cls.forward(ctx, *args, **kwargs)

    # check that all the results stem from triton.empty
    # and get the corresponding TF tensors if possible
    result = result if isinstance(result, tuple) else (result, )
    result = function.extract_tf_tensors(result, 'forward')
    
    # Register backward pass
    key = result[0]
    op = result[0].op
    ctx_registry[key] = ctx
    remap_in = cls.map_in_to_args(op, args)
    remap_out = cls.map_res_to_out(op, result)
    name = op.op_def.name
    if not cls.registered:
      @fw.tensorflow.RegisterGradient(name)
      def gradient(op, *dy):
        # Remap gradient inputs in the right order
        dy = [dy[i] for i in remap_out]
        dy = dy if len(dy) > 1 else dy[0]
        # Execute gradient function
        grad = cls.backward(ctx_registry[key], dy)
        grad = function.extract_tf_tensors(grad, 'backward')
        # Remap gradient in the right order
        ret = [None] * len(op.inputs)
        for i in range(len(grad)):
          if i in remap_in:
            ret[remap_in[i]] = grad[i]
        # Return
        return ret
      cls.registered = True

    # Return tensor
    return result[0] if len(result)==1 else result

  @classmethod
  def apply(cls, *args, **kwargs):
    if fw.has_tensorflow():
        return cls.apply_tensorflow(*args, **kwargs)
    elif fw.has_torch():
        return cls.apply_torch(*args, **kwargs)
    else:
        assert False
