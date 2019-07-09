find_package(TensorFlow)
if(${TensorFlow_FOUND})
  set(CUDA_HOME "/usr/local/cuda")
  include_directories("${TF_INC}/tensorflow/include")
  include_directories("${CUDA_HOME}/include")
  link_directories(${TF_LIB})
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${TF_ABI})
  add_library(tf_blocksparse SHARED dot.cpp conv.cpp shift.cpp batchnorm.cpp)
  target_link_libraries(tf_blocksparse tensorflow_framework triton)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run.py
                  ${CMAKE_CURRENT_BINARY_DIR}/run.py
                  COPYONLY)
endif()
