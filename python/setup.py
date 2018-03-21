import os, sys
from os.path import dirname
from distutils.core import setup, Extension
from glob import glob
from build import build_clib_subclass, build_ext_subclass


def recursive_glob(rootdir='.', suffix=''):
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]

def main():

    path = os.path.join(os.pardir, 'include')
    include = [path, os.path.join(path, 'isaac', 'external', 'CUDA')]
    src =  recursive_glob(os.path.join(os.pardir,'lib'), 'cpp')
    flags = ['-std=c++11', '-fPIC', '-D_GLIBCXX_USE_CXX11_ABI=0']
    core = ('core', {'sources': src, 'include_dirs': include, 'cflags': flags})

    # Extensions
    extensions = []

    # Isaac
    extensions += [Extension('_isaac',
                    sources=recursive_glob(os.path.join('src','bind'), 'cpp'),
                    libraries=[],
                    library_dirs=[],
                    extra_compile_args=flags,
                    extra_link_args=[],
                    include_dirs=include + [os.path.join('src', 'bind')])]

    # Tensorflow
    # try:
    #   import tensorflow as tf
    #   tf_include = tf.sysconfig.get_include()
    #   extensions += [Extension('_tensorflow',
    #                            sources=[os.path.join('src', 'extensions', 'tensorflow.cpp')],
    #                            libraries = ['tensorflow_framework'],
    #                            extra_compile_args= flags,
    #                            include_dirs = include + [tf_include, os.path.join(tf_include, 'external', 'nsync', 'public')],
    #                            library_dirs = [tf.sysconfig.get_lib()])]
    # except ImportError:
    #   pass

    # Pytorch
    try:
      import torch
      from torch.utils.ffi import create_extension
      ffi = torch.utils.ffi.create_extension('isaac.pytorch.c_lib',
                               language='c++',
                               sources=[os.path.join('src', 'extensions', 'pytorch.cpp')],
                               headers=[os.path.join('src', 'extensions', 'pytorch.h')],
                               include_dirs = include,
                               relative_to = __file__,
                               with_cuda=True,
                               extra_compile_args= flags)
      ffi = ffi.distutils_extension()
      try:
          ffi.include_dirs.remove('/usr/local/cuda/include')
      except ValueError:
          pass
      ffi.name = 'pytorch.c_lib._c_lib'
      extensions += [ffi]
    except ImportError:
      pass


    # Setup
    setup(
          name='isaac',
          version='1.0',
          description="ISAAC",
          author='Philippe Tillet',
          author_email='ptillet@g.harvard.edu',
          packages=['isaac', 'isaac.pytorch', 'isaac.pytorch.models', 'isaac.pytorch.c_lib'],
          libraries=[core],
          ext_package='isaac',
          ext_modules=extensions,
          cmdclass={'build_clib': build_clib_subclass, 'build_ext': build_ext_subclass},
          classifiers=['Environment :: Console',
                       'Development Status :: 4 - Beta',
                       'Intended Audience :: Developers',
                       'Intended Audience :: Other Audience',
                       'Intended Audience :: Science/Research',
                       'Natural Language :: English',
                       'Programming Language :: C++',
                       'Programming Language :: Python',
                       'Programming Language :: Python :: 3',
                       'Topic :: Scientific/Engineering',
                       'Topic :: Scientific/Engineering :: Mathematics',
                       'Topic :: Scientific/Engineering :: Physics',
                       'Topic :: Scientific/Engineering :: Machine Learning']
         )

if __name__ == "__main__":
    main()
