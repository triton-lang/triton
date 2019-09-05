import triton.frameworks as fw

class OpContext(object):

    def save_for_backward(self, *tensors):
      self.to_save = tensors

    def mark_dirty(self, *args):
      self.dirty_tensors = args
    
    @property
    def saved_tensors(self):
      return self.to_save


class function_meta(type):

    def __init__(cls, name, bases, attrs):
        cls.contexts = dict()
        cls.registered = False
        return super(function_meta, cls).__init__(name, bases, attrs)

class function(metaclass = function_meta):

  def __init__(self, framework = None):
    self.framework = _find_framework(framework)
    pass
  
  @staticmethod
  def forward(ctx, *args, **kwargs):
      raise NotImplementedError

  @staticmethod
  def backward(ctx, grad_output):
      raise NotImplementedError

  @classmethod
  def apply(cls, *args, **kwargs):
    # call forward
    ctx = OpContext()
    result = cls.forward(ctx, *args, **kwargs)
    id = result.op.get_attr('id')
    cls.contexts[id] = ctx
    # register backward
    fw._import_tensorflow()
    name = result.op.op_def.name
    if not cls.registered:
      @fw.tensorflow.RegisterGradient(name)
      def gradient(op, dy):
        id = op.get_attr('id')
        return cls.backward(cls.contexts[id], dy)
      cls.registered = True
    # return result tensor
    return result