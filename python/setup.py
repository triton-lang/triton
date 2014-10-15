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
    cvars['OPT'] = str.join(' ', remove_prefixes(cvars['OPT'].split(), ['-g', '-O', '-Wstrict-prototypes', '-DNDEBUG']))

    DEFINES = [('VIENNACL_WITH_OPENCL','1')]
    INCLUDE_DIRS = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]

    setup(
		name="atidlas",
		version=[],
		description="Auto-tuned input-dependent linear algebra subroutines",
		author='Philippe Tillet',
		author_email='ptillet@g.harvard.edu',
		classifiers=[
		    'Environment :: Console',
		    'Development Status :: 5 - Production/Stable',
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

		packages=["atidlas"],
		ext_package="atidlas",
		ext_modules=[Extension(
		    '_atidlas',[os.path.join("src", "_atidlas.cpp")],
		    extra_compile_args= [],
		    extra_link_args=[],
		    define_macros=DEFINES,
		    undef_macros=[],
		    include_dirs=INCLUDE_DIRS,
		    library_dirs=[],
		    libraries=['OpenCL']
		)],
		cmdclass={'build_ext': build_ext_subclass}
    )

if __name__ == "__main__":
    main()
