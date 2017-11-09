import os

from distutils import log
from distutils.errors import DistutilsSetupError

# Build CLIB -- Taken from setuptools
from distutils.command.build_clib import build_clib
from distutils.dep_util import newer_group

def newer_pairwise_group(sources_groups, targets):
    """Walk both arguments in parallel, testing if each source group is newer
    than its corresponding target. Returns a pair of lists (sources_groups,
    targets) where sources is newer than target, according to the semantics
    of 'newer_group()'.
    """
    if len(sources_groups) != len(targets):
        raise ValueError("'sources_group' and 'targets' must be the same length")

    # build a pair of lists (sources_groups, targets) where source is newer
    n_sources = []
    n_targets = []
    for i in range(len(sources_groups)):
        if newer_group(sources_groups[i], targets[i]):
            n_sources.append(sources_groups[i])
            n_targets.append(targets[i])

    return n_sources, n_targets

class build_clib_subclass(build_clib):
    """
    Override the default build_clib behaviour to do the following:

    1. Implement a rudimentary timestamp-based dependency system
       so 'compile()' doesn't run every time.
    2. Add more keys to the 'build_info' dictionary:
        * obj_deps - specify dependencies for each object compiled.
                     this should be a dictionary mapping a key
                     with the source filename to a list of
                     dependencies. Use an empty string for global
                     dependencies.
        * cflags   - specify a list of additional flags to pass to
                     the compiler.
    """
    
    def build_libraries(self, libraries):
        try:
            os.mkdir(self.build_temp)
        except OSError:
            pass
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        
        for (lib_name, build_info) in libraries:
            sources = build_info.get('sources')
            if sources is None or not isinstance(sources, (list, tuple)):
                raise DistutilsSetupError(
                       "in 'libraries' option (library '%s'), "
                       "'sources' must be present and must be "
                       "a list of source filenames" % lib_name)
            sources = list(sources)

            log.info("building '%s' library", lib_name)

            # Make sure everything is the correct type.
            # obj_deps should be a dictionary of keys as sources
            # and a list/tuple of files that are its dependencies.
            obj_deps = build_info.get('obj_deps', dict())
            if not isinstance(obj_deps, dict):
                raise DistutilsSetupError(
                       "in 'libraries' option (library '%s'), "
                       "'obj_deps' must be a dictionary of "
                       "type 'source: list'" % lib_name)
            dependencies = []

            # Get the global dependencies that are specified by the '' key.
            # These will go into every source's dependency list.
            global_deps = obj_deps.get('', list())
            if not isinstance(global_deps, (list, tuple)):
                raise DistutilsSetupError(
                       "in 'libraries' option (library '%s'), "
                       "'obj_deps' must be a dictionary of "
                       "type 'source: list'" % lib_name)

            # Build the list to be used by newer_pairwise_group
            # each source will be auto-added to its dependencies.
            for source in sources:
                src_deps = [source]
                src_deps.extend(global_deps)
                extra_deps = obj_deps.get(source, list())
                if not isinstance(extra_deps, (list, tuple)):
                    raise DistutilsSetupError(
                           "in 'libraries' option (library '%s'), "
                           "'obj_deps' must be a dictionary of "
                           "type 'source: list'" % lib_name)
                src_deps.extend(extra_deps)
                dependencies.append(src_deps)

            expected_objects = self.compiler.object_filenames(
                    sources,
                    output_dir=self.build_temp
                    )

            if newer_pairwise_group(dependencies, expected_objects) != ([], []):
                # First, compile the source code to object files in the library
                # directory.  (This should probably change to putting object
                # files in a temporary build directory.)
                macros = build_info.get('macros')
                include_dirs = build_info.get('include_dirs')
                cflags = build_info.get('cflags')
                objects = self.compiler.compile(
                        sources,
                        output_dir=self.build_temp,
                        macros=macros,
                        include_dirs=include_dirs,
                        extra_postargs=cflags,
                        debug=self.debug
                        )

            # Now "link" the object files together into a static library.
            # (On Unix at least, this isn't really linking -- it just
            # builds an archive.  Whatever.)
            self.compiler.create_static_lib(
                    expected_objects,
                    lib_name,
                    output_dir=self.build_clib,
                    debug=self.debug
                    )
            
# Build EXTENSION
from distutils.command.build_ext import build_ext
                    
class build_ext_subclass(build_ext):
    """
    Removes -Wstrict-prototypes from build_ext because compiling C++
    """
    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)
