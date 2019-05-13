import torch
torch.manual_seed(0)

class TritonConv(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = torch.ops.triton.conv_fprop(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.ops.triton.conv_bprop(grad_output.contiguous(), weight)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.ops.triton.conv_wgrad(input, grad_output.contiguous())
        return grad_input, grad_weight
        
        
torch.ops.load_library("/home/philippe/Development/triton/build/examples/python/pytorch/libtorch_triton.so")

x = torch.autograd.Variable(torch.randn(16, 64, 8, 8).cuda(), requires_grad=True)
w = torch.autograd.Variable(torch.randn(64, 3, 3, 64).cuda(), requires_grad=True)
cuw = torch.autograd.Variable(w.permute(3,0,1,2).cuda(), requires_grad=True)
y_target = torch.autograd.Variable(torch.randn(16, 64, 8, 8).cuda(), requires_grad=True)

def run(x, w, conv):
  y = conv(x, w)
  loss = (y - y_target).norm(2)
  loss.backward()
  return loss, y.clone(), x.grad.clone(), w.grad.clone()

ttyloss, tty, ttdx, ttdw = run(x, w, TritonConv.apply)
x.grad.zero_()
w.grad.zero_()
culoss, cuy, cudx, cudw = run(x, cuw, lambda x, w: torch.nn.functional.conv2d(x, w, padding=1))

print((tty - cuy).norm(2))
print((ttdx - cudx).norm(2))
print((ttdw.permute(3,0,1,2) - cudw).norm(2))
#print(ttdx)
#print(cudx)
#print(ttdw)
#print(cudw)
#print((ttdw.permute(3,0,1,2) - cudw).norm(2))
