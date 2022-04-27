Traceback (most recent call last):
  File "matmul.py", line 1, in <module>
    import triton
  File "/home/da/miniconda3/envs/torch-src/lib/python3.7/site-packages/triton-2.0.0-py3.7-linux-x86_64.egg/triton/__init__.py", line 9, in <module>
    from .code_gen import cdiv, next_power_of_2, jit, autotune, heuristics, \
  File "/home/da/miniconda3/envs/torch-src/lib/python3.7/site-packages/triton-2.0.0-py3.7-linux-x86_64.egg/triton/code_gen.py", line 23, in <module>
    import triton._C.libtriton.triton as _triton
ImportError: /home/da/miniconda3/envs/torch-src/lib/python3.7/site-packages/triton-2.0.0-py3.7-linux-x86_64.egg/triton/_C/libtriton.so: undefined symbol: _ZN4mlir6triton5CatOp10getEffectsERN4llvm15SmallVectorImplINS_11SideEffects14EffectInstanceINS_13MemoryEffects6EffectEEEEE
