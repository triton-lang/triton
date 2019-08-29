import os
import re
import sys
import sysconfig
import platform
import subprocess
import distutils

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        self.debug = True
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # python directors
        python_include_dirs = distutils.sysconfig.get_python_inc()
        python_lib_dirs = distutils.sysconfig.get_config_var('LIBDIR')
        # tensorflow directories
        import tensorflow as tf
        tf_include_dirs = tf.sysconfig.get_include()
        tf_lib_dirs = tf.sysconfig.get_lib()
        tf_libs = 'tensorflow_framework'

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DBUILD_TESTS=OFF',
                      '-DBUILD_PYTHON_MODULE=ON',
                      '-DPYTHON_INCLUDE_DIRS=' + python_include_dirs,
                      '-DTF_INCLUDE_DIRS=' + tf_include_dirs,
                      '-DTF_LIB_DIRS=' + tf_lib_dirs,
                      '-DTF_LIBS=' + tf_libs]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        sourcedir =  os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        subprocess.check_call(['cmake', sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
       
setup(
    name='triton',
    version='0.1',
    author='Philippe Tillet',
    author_email='ptillet@g.harvard.edu',
    description='A language and compiler for custom Deep Learning operations',
    long_description='',
    packages=['triton'],
    ext_modules=[CMakeExtension('triton')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
