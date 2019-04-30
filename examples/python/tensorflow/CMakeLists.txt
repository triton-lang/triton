execute_process(COMMAND python -c "from os.path import dirname; import tensorflow as tf; print(dirname(dirname(tf.sysconfig.get_include())))"
                OUTPUT_VARIABLE TF_INC OUTPUT_STRIP_TRAILING_WHITESPACE)
#execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
#                OUTPUT_VARIABLE TF_LIB)
#execute_process(COMMAND python -c "import tensorflow as tf; print(tf.__cxx11_abi_flag__ if \"__cxx11_abi_flag__\" in tf.__dict__ else 0)"
#                OUTPUT_VARIABLE TF_ABI)

set(CUDA_HOME "/usr/local/cuda")
include_directories("${TF_INC}/tensorflow/include")
include_directories("${CUDA_HOME}/include")
add_library(tf_blocksparse SHARED blocksparse.cpp)
#link_libraries(tf_blocksparse ${TF_LIB})
