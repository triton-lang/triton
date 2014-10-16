import os
import sys
from setuptools import Extension, setup
from cx_Freeze import setup, Executable

def main():
    sys.path.append('/home/philippe/Development/ATIDLAS/build/python/pyatidlas/build/lib.linux-x86_64-2.7/')
    sys.path.append('/home/philippe/Development/pyviennacl-dev/build/lib.linux-x86_64-2.7/')
    sys.path.append(os.path.join('${CMAKE_CURRENT_SOURCE_DIR}','pysrc'))
    extdir = os.path.join('${CMAKE_CURRENT_SOURCE_DIR}','external')
    
    buildOptions = dict(packages = ['scipy.sparse.csgraph._validation',
                                    'scipy.special._ufuncs_cxx',
                                    'scipy.sparse.linalg.dsolve.umfpack',
                                    'scipy.integrate.vode',
                                    'scipy.integrate.lsoda',
                                    'sklearn.utils.sparsetools._graph_validation',
                                    'sklearn.utils.lgamma'],
                        excludes = ['matplotlib'],
                        bin_path_includes = ['/usr/lib/x86_64-linux-gnu/'],
                        include_files = [(os.path.abspath(os.path.join(extdir, x)), x) for x in os.listdir(extdir)])
    base = 'Console'
    executables = [
        Executable(os.path.join('${CMAKE_CURRENT_SOURCE_DIR}','pysrc','autotune.py'), base=base)
    ]
    setup(name='atidlas-tune',
          version = '1.0',
          description = 'Auto-tuner for ATIDLAS',
          options = dict(build_exe = buildOptions),
          executables = executables)

if __name__ == "__main__":
    main()
