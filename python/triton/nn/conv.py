import triton
import torch.nn as nn
import torch
import torch.nn.functional as F

class _conv2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, 
                stride, padding, dilation, groups,
                acc_bitmask):
      assert dilation == (1, 1)
      assert groups == 1
      assert bias == None
      pad_h, pad_w = padding
      stride_h, stride_w = stride
      n, c, h, w = x.size()
      k, c, r, s = weight.size()
      # allocate output
      p = (h + 2*padding[0] - r)//stride[0] + 1
      q = (w + 2*padding[1] - s)//stride[1] + 1
      output = torch.empty((n, k, p, q), dtype=x.dtype, device=x.device)
      # padding
      if pad_h or pad_w:
        x = triton.ops._einsum.pad(x, [pad_w, pad_w, pad_h, pad_h])
      # convolution
      triton.ops.einsum(f'nc(h*stride_h + r - pad_h)(w*stride_w + s - pad_w),kcrs->nkhw', 
                        x, weight, mask=acc_bitmask,
                        output=output,
                        values = {'pad_h': pad_h,
                                  'stride_h': stride_h,
                                  'pad_w': pad_w,
                                  'stride_w': stride_w})
      # prepare backprop
      ctx.save_for_backward(x, weight)
      ctx.stride = stride
      ctx.padding = padding
      ctx.acc_bitmask = acc_bitmask
      # return
      return output
    
    @staticmethod
    def backward(ctx, dy):
      # retrieve contextual information
      x, weight = ctx.saved_tensors
      stride = ctx.stride
      padding = ctx.padding
      acc_bitmask = ctx.acc_bitmask
      # gradient of the input
      dx = None
      if ctx.needs_input_grad[0]:
        # dy must be padded
        n, k, p, q = dy.size()
        n, c, h, w = x.size()
        k, c, r, s = weight.size()
        dypad = triton.ops._einsum.pad(dy, [4, 4, 4, 4])
        # have to be careful here
        # the gradient of strided conv is a conv over a sparse image
        # which can be decomposed as a set of smaller convs
        dx = torch.empty_like(x)
        for offh in range(stride[0]):
          for offw in range(stride[1]):
            poffh = (offh + padding[0]) % stride[0]
            poffw = (offw + padding[1]) % stride[1]
            pad_h = int((padding[0] + (stride[0] - 1)*offh) / stride[0])
            pad_w = int((padding[1] + (stride[1] - 1)*offw) / stride[1])
            if poffh >= r or poffw >= s:
              dx[:, :, offh::stride[0], offw::stride[1]] = 0
            else:
              triton.ops.einsum(f'nk(h - r + pad_h)(w - s + pad_w),kcrs->nchw', 
                                 dypad[:, :, :, :], 
                                 weight[:, :, poffh::stride[0], poffw::stride[1]],
                                 output = dx[:, :, offh::stride[0], offw::stride[1]],
                                 mask = acc_bitmask,
                                 values = {'pad_h': pad_h,
                                           'pad_w': pad_w})
        
      # gradient for the weight
      dw = None
      if ctx.needs_input_grad[1]:
        dw = torch.empty_like(weight)
        triton.ops.einsum(f'nc(p*{stride[0]}+r-{padding[0]})(q*{stride[1]}+s-{padding[1]}),nkpq->kcrs', 
                           x, dy, output = dw, mask = acc_bitmask)
        #print('dw: ', dw.view(-1)[0])
      return dx, dw, None, None, None, None, None, None
conv2d = _conv2d.apply

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 acc_bitmask = None):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.acc_bitmask = acc_bitmask

    def forward(self, input):
        #if self.kernel_size[0] == 3 and self.stride[0] != 1:
          #print(self.padding, self.stride, input.size(), self.weight.size())
        #  return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return conv2d(input, self.weight, self.bias, self.stride, 
                      self.padding, self.dilation, self.groups,
                      self.acc_bitmask)


def replace_conv2d(model, acc_bitmask = None):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            conv2d = Conv2d(child.in_channels, child.out_channels, child.kernel_size,
                            child.stride, child.padding, child.dilation, child.groups,
                            child.bias, child.padding_mode, 
                            acc_bitmask=acc_bitmask)
            for yparam, xparam in zip(conv2d.parameters(), child.parameters()):
                yparam.data.copy_(xparam.data)
            setattr(model, child_name, conv2d)
        else:
            replace_conv2d(child, acc_bitmask)

# initialize input
#N, C, H, W, K, RS = 16, 32, 24, 24, 64, 3
#torch.Size([128, 64, 30, 30]) torch.Size([128, 64, 3, 3])
#torch.Size([128, 128, 15, 15]) torch.Size([256, 128, 3, 3])
#torch.Size([128, 256, 8, 8]) torch.Size([512, 256, 3, 3])

if __name__ == '__main__':
  N, C, H, W, K, RS = 128, 64, 30, 30, 128, 1
  #N, C, H, W, K, RS = 128, 128, 15, 15, 256, 3
  #N, C, H, W, K, RS = 128, 256, 8, 8, 512, 3
  pad, stride = 0, 1
  torch.manual_seed(0)
  x = torch.randn((N, C, H, W)).cuda()
  x.requires_grad_(True)
  #x.data[:] = 1
  # initialize layers
  torch.manual_seed(0)
  rconv2d = nn.Conv2d(C, K, RS, stride, pad, bias=False).cuda()
  torch.manual_seed(0)
  tconv2d = Conv2d(C, K, RS, stride, pad, bias=False).cuda()
  #rconv2d.weight.data[:] = 1
  #tconv2d.weight.data[:] = 1
  ry = rconv2d(x)
  ty = tconv2d(x)
  # reference
  dy = torch.randn(ry.size()).cuda()
  #dy.data[:] = 1
  ry.backward(dy)
  rdx = x.grad.clone()
  rdw = rconv2d.weight.grad.clone()
  x.grad.zero_()
  # triton
  ty.backward(dy)
  tdx = x.grad.clone()
  tdw = tconv2d.weight.grad.clone()
  x.grad.zero_()
  # print error
  diff = lambda x, y: (x - y).abs().max()
  print(diff(ry, ty))
  print(diff(rdx, tdx))
  print(diff(rdw, tdw))
  #print((rdx - tdx).abs())

  #print((rdx[0,0,:,:] - tdx[0,0,:,:]))
  #print(rdx[0,0,:,:])
  #print(tdx[0,0,:,:])