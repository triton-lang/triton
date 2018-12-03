# - Find LLVM 
# This module can be used to find LLVM.
# It requires that the llvm-config executable be available on the system path.
# Once found, llvm-config is used for everything else.
#
# Typical usage could be:
#   find_package(LLVM QUIET REQUIRED COMPONENTS jit native interpreter)
#
# If the QUIET flag is not set, the specified components and LLVM version are
# outputted.
#
# If the COMPONENTS are not set, the default set of "all" is used.
#
# The following variables are set:
#
# LLVM_FOUND        - Set to YES if LLVM is found.
# LLVM_VERSION      - Set to the decimal version of the LLVM library.
# LLVM_C_FLAGS      - All flags that should be passed to a C compiler. 
# LLVM_CXX_FLAGS    - All flags that should be passed to a C++ compiler.
# LLVM_CPP_FLAGS    - All flags that should be passed to the C pre-processor.
# LLVM_LD_FLAGS     - Additional flags to pass to the linker.
# LLVM_LIBRARY_DIRS - A list of directories where the LLVM libraries are located.
# LLVM_INCLUDE_DIRS - A list of directories where the LLVM headers are located.
# LLVM_LIBRARIES    - A list of libraries which should be linked against.

# A macro to run llvm config
macro(_llvm_config _var_name)
	# Firstly, locate the LLVM config executable
	find_program(_llvm_config_exe 
		NAMES llvm-config
		PATHS /home/philippe/Development/llvm-tlvm/build/bin/
		DOC "llvm-config executable location"
	)

	# If no llvm-config executable was found, set the output variable to not
	# found.
	if(NOT _llvm_config_exe)
		set(${_var_name} "${_var_name}-NOTFOUND")
	else(NOT _llvm_config_exe)
		# Otherwise, run llvm-config
		execute_process(
			COMMAND ${_llvm_config_exe} ${ARGN}
			OUTPUT_VARIABLE ${_var_name}
			RESULT_VARIABLE _llvm_config_retval
			OUTPUT_STRIP_TRAILING_WHITESPACE
		)
		if(RESULT_VARIABLE)
			message(SEND_ERROR
				"Error running llvm-config with arguments: ${ARGN}")
		endif(RESULT_VARIABLE)
	endif(NOT _llvm_config_exe)
endmacro(_llvm_config)

# The default set of components
set(_llvm_components all)

# If components have been specified via find_package, use them
if(LLVM_FIND_COMPONENTS)
	set(_llvm_components ${LLVM_FIND_COMPONENTS})
endif(LLVM_FIND_COMPONENTS)

if(NOT LLVM_FIND_QUIETLY)
	message(STATUS "Looking for LLVM components: ${_llvm_components}")
endif(NOT LLVM_FIND_QUIETLY)

_llvm_config(LLVM_VERSION --version)
_llvm_config(LLVM_C_FLAGS --cflags)
_llvm_config(LLVM_CXX_FLAGS --cxxflags)
_llvm_config(LLVM_CPP_FLAGS --cppflags)
_llvm_config(LLVM_LD_FLAGS --ldflags)
_llvm_config(LLVM_LIBRARY_DIRS --libdir)
_llvm_config(LLVM_INCLUDE_DIRS --includedir)
_llvm_config(LLVM_LIBRARIES --libs)

if(NOT LLVM_FIND_QUIETLY)
	message(STATUS "Found LLVM version: ${LLVM_VERSION}")
endif(NOT LLVM_FIND_QUIETLY)

# handle the QUIETLY and REQUIRED arguments and set LLVM_FOUND to TRUE if 
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LLVM
	DEFAULT_MSG 
	LLVM_LIBRARIES
	LLVM_INCLUDE_DIRS 
	LLVM_LIBRARY_DIRS)

# vim:sw=4:ts=4:autoindent
