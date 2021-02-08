import os
import re
import sys
import sysconfig
import platform
import subprocess
import distutils
import glob
from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import include_paths, library_paths
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
import distutils.spawn
import torch

def find_llvm():
    versions = ['-10', '-10.0', '']
    supported = ['llvm-config{v}'.format(v=v) for v in versions]
    paths = [distutils.spawn.find_executable(cfg) for cfg in supported]
    paths = [p for p in paths if p is not None]
    if paths:
        return paths[0]
    config = distutils.spawn.find_executable('llvm-config')
    instructions = 'Please install llvm-10-dev'
    if config is None:
        raise RuntimeError('Could not find llvm-config. ' + instructions)
    version = os.popen('{config} --version'.format(config=config)).read()
    raise RuntimeError('Version {v} not supported. '.format(v=version) + instructions)

class CMakeExtension(Extension):
    def __init__(self, name, path, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.path = path

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
        #self.debug = True
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))
        # python directories
        python_include_dirs = distutils.sysconfig.get_python_inc()
        python_lib_dirs = distutils.sysconfig.get_config_var('LIBDIR')
        torch_include_dirs = include_paths(True)
        torch_library_dirs = library_paths(True)
        cxx11abi = str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DBUILD_TUTORIALS=OFF',
            '-DBUILD_PYTHON_MODULE=ON',
            #'-DPYTHON_EXECUTABLE=' + sys.executable,
            #'-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON,
            '-DPYTHON_INCLUDE_DIRS=' + ';'.join([python_include_dirs] + include_paths(True)),
            '-DPYTHON_LINK_DIRS=' + ';'.join(library_paths(True)),
            '-DTORCH_CXX11_ABI=' + cxx11abi,
            '-DTORCH_LIBRARIES=c10;c10_cuda;torch;torch_cuda;torch_cpu;torch_python;triton',
            '-DLLVM_CONFIG=' + find_llvm()
        ]
        # configuration
        cfg = 'Debug' if self.debug else 'Release'
        cfg = 'Release'
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
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        sourcedir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
        subprocess.check_call(['cmake', sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

class BenchCommand(distutils.cmd.Command):

    description = 'run benchmark suite'
    user_options = [
        ('result-dir=', None, 'path to output benchmark results'),
        ('with-plots', None, 'plot benchmark results'),
    ]

    def initialize_options(self):
        self.result_dir = 'results'
        self.with_plots = False

    def finalize_options(self):
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    def run(self):
        import sys
        import inspect
        import triton
        bench_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench')
        sys.path.append(bench_dir)
        for mod in os.listdir(bench_dir):
            if not mod.endswith('.py'):
                continue
            print(f'running {mod}...')
            mod = __import__(os.path.splitext(mod)[0])
            benchmarks = inspect.getmembers(mod, lambda x: isinstance(x, triton.testing.Mark))
            for _, bench in benchmarks:
                bench.run(self.result_dir, self.with_plots)

setup(
    name='triton',
    version='1.0.0',
    author='Philippe Tillet',
    author_email='phil@openai.com',
    description='A language and compiler for custom Deep Learning operations',
    long_description='',
    packages=['triton', 'triton/_C', 'triton/ops', 'triton/ops/blocksparse'],
    install_requires=['numpy', 'torch'],
    package_data={'triton/ops': ['*.c'], 'triton/ops/blocksparse': ['*.c']},
    include_package_data=True,
    ext_modules=[CMakeExtension('triton', 'triton/_C/')],
    cmdclass={'build_ext': CMakeBuild, 'bench': BenchCommand},
    zip_safe=False,
    # for PyPI
    keywords=['Compiler', 'Deep Learning'],
    url='https://github.com/ptillet/triton/',
    download_url='https://github.com/ptillet/triton/archive/v0.1.tar.gz',
    classifiers=[
        'Development Status :: 3 - Alpha',  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3.6',
    ],
)
