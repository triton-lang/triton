execute_process(COMMAND python -c "from os.path import dirname; import tensorflow as tf; print(dirname(dirname(tf.sysconfig.get_include())))"
                OUTPUT_VARIABLE TF_INC OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
                OUTPUT_VARIABLE TF_LIB OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python -c "import tensorflow as tf; print(tf.__cxx11_abi_flag__ if \"__cxx11_abi_flag__\" in tf.__dict__ else 0)"
                OUTPUT_VARIABLE TF_ABI OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CUDA_HOME "/usr/local/cuda")
include_directories("${TF_INC}/tensorflow/include")
include_directories("${CUDA_HOME}/include")
link_directories(${TF_LIB})
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_library(tf_blocksparse SHARED blocksparse.cpp)
target_link_libraries(tf_blocksparse tensorflow_framework triton)
