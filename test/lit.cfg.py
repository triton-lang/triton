# -*- Python -*-
# ruff: noqa: F821

import os
import sys

import lit.formats
import lit.util
from lit.llvm import llvm_config

if llvm_config is None:
    class _LitDummy:
        use_lit_shell = False
        def with_system_environment(self, *args, **kwargs):
            return None
        def with_environment(self, *args, **kwargs):
            return None
        def add_tool_substitutions(self, *args, **kwargs):
            return None
    llvm_config = _LitDummy()

# Safe default for config.triton_obj_root for venv-installed lit
config.triton_obj_root = getattr(config, 'triton_obj_root', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')))

# Fallbacks for local non-llvm lit runs (guarded defaults so venv-installed lit won't raise AttributeError)
config.llvm_shlib_dir = getattr(config, 'llvm_shlib_dir', '')
config.llvm_shlib_ext = getattr(config, 'llvm_shlib_ext', '')
config.llvm_tools_dir = getattr(config, 'llvm_tools_dir', os.path.join(config.triton_obj_root, 'bin'))
config.mlir_binary_dir = getattr(config, 'mlir_binary_dir', os.path.join(config.triton_obj_root, 'lib'))
config.python_executable = getattr(config, 'python_executable', sys.executable)
config.triton_tools_dir = getattr(config, 'triton_tools_dir', os.path.join(config.triton_obj_root, 'bin'))

from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner

# (config is an instance of TestingConfig created when discovering tests)
# name: The name of this test suite
config.name = 'TRITON'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir', '.ll']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.triton_obj_root, 'test')
config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(("%shlibdir", config.llvm_shlib_dir))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

# llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.triton_obj_root, 'test')
config.triton_tools_dir = os.path.join(config.triton_obj_root, 'bin')
config.filecheck_dir = os.path.join(config.triton_obj_root, 'bin', 'FileCheck')

# FileCheck -enable-var-scope is enabled by default in MLIR test
# This option avoids to accidentally reuse variable across -LABEL match,
# it can be explicitly opted-in by prefixing the variable name with $
config.environment["FILECHECK_OPTS"] = "--enable-var-scope"

tool_dirs = [config.triton_tools_dir, config.llvm_tools_dir, config.filecheck_dir]

# Tweak the PATH to include the tools dir.
for d in tool_dirs:
    llvm_config.with_environment('PATH', d, append_path=True)
tools = [
    'triton-opt',
    'triton-llvm-opt',
    'mlir-translate',
    ToolSubst('%PYTHON', config.python_executable, unresolved='ignore'),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# TODO: what's this?
llvm_config.with_environment('PYTHONPATH', [
    os.path.join(config.mlir_binary_dir, 'python_packages', 'triton'),
], append_path=True)

# Make repo third_party importable for lit Python snippets
# repo_root resolves to the repository root (one level above the test exec root)
repo_root = os.path.abspath(os.path.join(config.test_exec_root, '..'))
llvm_config.with_environment('PYTHONPATH', [
    repo_root,
    os.path.join(repo_root, 'third_party'),
], append_path=True)
