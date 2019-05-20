import torch
import triton

x = torch.autograd.Variable(torch.randn(16, 64, 8, 8).cuda(), requires_grad=True)
bias = torch.autograd.Variable(torch.randn(64).cuda(), requires_grad=True)
w = torch.autograd.Variable(torch.randn(64, 3, 3, 64).cuda(), requires_grad=True)
cuw = torch.autograd.Variable(w.permute(3,0,1,2).cuda(), requires_grad=True)
y_target = torch.autograd.Variable(torch.randn(16, 64, 6, 6).cuda(), requires_grad=True)

def run(x, w, conv):
  y = conv(x, w)
  loss = (y - y_target).norm(2)
  loss.backward()
  return loss, y.clone(), x.grad.clone(), w.grad.clone(), bias.grad.clone()

ttyloss, tty, ttdx, ttdw, ttbias = run(x, w, lambda x, w: triton.ConvFunction.apply(x, w, bias, (1,1), (0,0)))
x.grad.zero_()
w.grad.zero_()
bias.grad.zero_()
culoss, cuy, cudx, cudw, cubias = run(x, cuw, lambda x, w: torch.nn.functional.conv2d(x, w, bias=bias, stride=1, padding=0))

print(ttdx[0,0,:,:], cudx[0,0,:,:])
print((tty - cuy).norm(2))
print((ttdx - cudx).norm(2))
print((ttdw.permute(3,0,1,2) - cudw).norm(2))
print((ttbias - cubias).norm(2))
