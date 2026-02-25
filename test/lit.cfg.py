# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import config
import lit.formats
import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import FindTool, ToolSubst

# Configuration file for the 'lit' test runner

# Name: The name of this test suite
config.name = 'TRITON'

# Test format configuration
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)  # getting an attribute error


# Suffixes: A list of file extensions to treat as test files
config.suffixes = ['.mlir', '.ll']

# Test source root: The root path where tests are located
config.test_source_root = os.path.dirname(__file__)

# Test execution root: The root path where tests should be run
config.test_exec_root = os.path.join(config.triton_obj_root, 'test')

# Substitutions for the environment
config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

# Configure environment variables
llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

# Exclusions: A list of directories to exclude from the testsuite
config.excludes = [
    'Inputs',
    'Examples',
    'CMakeLists.txt',
    'README.txt',
    'LICENSE.txt'
]

# Verify essential config variables are set
if config.triton_obj_root is None:
    raise ValueError("Error: 'triton_obj_root' is not set.")
if config.python_executable is None:
    raise ValueError("Error: 'python_executable' is not set.")
if config.mlir_binary_dir is None:
    raise ValueError("Error: 'mlir_binary_dir' is not set.")

# Tool directories setup
config.triton_tools_dir = os.path.join(config.triton_obj_root, 'bin')
config.filecheck_dir = os.path.join(config.triton_obj_root, 'bin', 'FileCheck')
tool_dirs = [
    config.triton_tools_dir,
    config.llvm_tools_dir,
    config.filecheck_dir
]

# Update the PATH to include the tools directories
for d in tool_dirs:
    llvm_config.with_environment('PATH', d, append_path=True)

# Tool substitutions
tools = [
    'triton-opt',
    'triton-llvm-opt',
    ToolSubst('%PYTHON', config.python_executable, unresolved='ignore'),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Configure PYTHONPATH for Triton
llvm_config.with_environment('PYTHONPATH', [
    os.path.join(config.mlir_binary_dir, 'python_packages', 'triton'),
], append_path=True)

for d in tool_dirs:
    llvm_config.tempfile_env('%PATH%', config.enviroment['PATH'])
