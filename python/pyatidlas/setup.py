import os
from distutils.command.build_ext import build_ext
from setuptools import Extension, setup
from distutils.sysconfig import get_python_inc
from distutils import sysconfig

platform_cflags = {}
platform_ldflags = {}
platform_libs = {}
class build_ext_subclass(build_ext):
    """Shamelessly stolen from
    https://stackoverflow.com/questions/724664
    """
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c in platform_cflags.keys():
            for e in self.extensions:
                e.extra_compile_args = platform_cflags[c]
        if c in platform_ldflags.keys():
            for e in self.extensions:
                e.extra_link_args = platform_ldflags[c]
        if c in platform_libs.keys():
            for e in self.extensions:
                try:
                    e.libraries += platform_libs[c]
                except:
                    e.libraries = platform_libs[c]
        build_ext.build_extensions(self)

def main():

    def remove_prefixes(optlist, bad_prefixes):
        for bad_prefix in bad_prefixes:
            for i, flag in enumerate(optlist):
                if flag.startswith(bad_prefix):
                    optlist.pop(i)
                    break
        return optlist

    cvars = sysconfig.get_config_vars()
    cvars['OPT'] = "-DNDEBUG -O3 " + str.join(' ', remove_prefixes(cvars['OPT'].split(), ['-g', '-O', '-Wstrict-prototypes', '-DNDEBUG']))
    cvars["CFLAGS"] = cvars["BASECFLAGS"] + " " + cvars["OPT"]
    cvars["LDFLAGS"] = '-Wl,--no-as-needed ' + cvars["LDFLAGS"]
    
    DEFINES = [('VIENNACL_WITH_OPENCL',None), ('VIENNACL_WITH_OPENMP', None),
               ('boost','pyviennaclboost')]
    INCLUDE_DIRS = ['/home/philippe/Development/pyviennacl-dev/external/boost-python-ublas-subset/boost_subset/',
                    '${PROJECT_SOURCE_DIR}',
                    '/home/philippe/Development/pyviennacl-dev/external/viennacl-dev']
    LIBRARY_DIRS = ['/home/philippe/Development/pyviennacl-dev/build/lib.linux-x86_64-2.7/pyviennacl/']

    setup(
		name="pyatidlas",
		package_dir={ '': '${CMAKE_CURRENT_SOURCE_DIR}' },
		version=[],
		description="Auto-tuned input-dependent linear algebra subroutines",
		author='Philippe Tillet',
		author_email='ptillet@g.harvard.edu',
		classifiers=[
		    'Environment :: Console',
		    'Development Status :: 1 - Experimental',
		    'Intended Audience :: Developers',
		    'Intended Audience :: Other Audience',
		    'Intended Audience :: Science/Research',
		    'License :: OSI Approved :: MIT License',
		    'Natural Language :: English',
		    'Programming Language :: C++',
		    'Programming Language :: Python',
		    'Programming Language :: Python :: 2',
		    'Programming Language :: Python :: 2.6',
		    'Programming Language :: Python :: 2.7',
		    'Programming Language :: Python :: 3',
		    'Programming Language :: Python :: 3.2',
		    'Programming Language :: Python :: 3.3',
		    'Programming Language :: Python :: 3.4',
		    'Topic :: Scientific/Engineering',
		    'Topic :: Scientific/Engineering :: Mathematics',
		    'Topic :: Scientific/Engineering :: Physics',
		    'Topic :: Scientific/Engineering :: Machine Learning',
		],

		packages=["pyatidlas"],
		ext_package="pyatidlas",
		ext_modules=[Extension(
		    '_atidlas',[os.path.join('${CMAKE_CURRENT_SOURCE_DIR}', 'src', '_atidlas.cpp')],
		    extra_compile_args= [],
		    extra_link_args=[],
		    define_macros=DEFINES,
		    undef_macros=[],
		    include_dirs=INCLUDE_DIRS,
		    library_dirs=LIBRARY_DIRS,
		    libraries=['OpenCL',':_viennacl.so']
		)],
		cmdclass={'build_ext': build_ext_subclass}
    )

if __name__ == "__main__":
    main()
