include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)

execute_process(COMMAND python -c "from os.path import dirname; import tensorflow as tf; print(dirname(dirname(tf.sysconfig.get_include())))"
                OUTPUT_VARIABLE TF_INC OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
                OUTPUT_VARIABLE TF_LIB OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND python -c "import tensorflow as tf; print(tf.__cxx11_abi_flag__ if \"__cxx11_abi_flag__\" in tf.__dict__ else 0)"
                OUTPUT_VARIABLE TF_ABI OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

find_package_handle_standard_args(TensorFlow DEFAULT_MSG TF_INC TF_LIB)

# set external variables for usage in CMakeLists.txt
if(TensorFlow_FOUND)
    set(TensorFlow_LIBRARIES ${TF_LIB})
    set(TensorFlow_INCLUDE_DIRS ${TF_INC})
    set(TensorFlow_ABI ${TF_ABI})
endif()

# hide locals from GUI
mark_as_advanced(TF_INC TF_LIB TF_ABI)
