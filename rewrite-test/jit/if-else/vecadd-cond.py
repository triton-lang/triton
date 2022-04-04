import triton

@triton.jit
def if_else(lb, ub, value):
  if value > lb:
    a = 0.0
  else:
    a = 1.0
  c = a + a

@triton.jit
def only_if(lb, ub, value):
  a = -1.0
  if value > lb:
    a = 0.0
  c = a + a

@triton.jit
def only_if_invalid(lb, ub, value):
  if value > lb:
    a = 0.0
  c = a + a

@triton.jit
def nested_if(lb, ub, value):
  if value > lb:
    if value < ub:
      a = 2.0
    else:
      a = 1.0
  else:
    a = 0.0
  c = a + a


mod_if_else, ctx_if_else = if_else.compile_to_ttir(2, 4, 3, grid=(1,))
mod_if_else.dump()

mod_only_if, ctx_only_if = only_if.compile_to_ttir(2, 4, 3, grid=(1,))
mod_only_if.dump()

try:
  mod_only_if_invalid, ctx_only_if = only_if_invalid.compile_to_ttir(2, 4, 3, grid=(1,))
  mod_only_if_invalid.dump()
except:
  print('value error')

mod_nested_if, ctx_nested_if = nested_if.compile_to_ttir(2, 4, 3, grid=(1,))
mod_nested_if.dump()
