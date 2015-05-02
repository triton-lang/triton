set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

file( GLOB_RECURSE LIBISAAC_SRC *.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(ANDROID)
    add_library(isaac STATIC ${LIBISAAC_SRC})
else()
    add_library(isaac SHARED ${LIBISAAC_SRC})
endif()

if(OPENCL_FOUND)
    target_link_libraries(isaac ${OPENCL_LIBRARIES})
endif()
if(CUDA_FOUND)
    cuda_find_library_local_first(CUDA_NVRTC_LIBRARY nvrtc "\"nvrtc\" library")
    target_link_libraries(isaac ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIBRARY})
    #Cuda JIT headers to file
    file( GLOB_RECURSE CUDA_HELPERS_SRC driver/helpers/*.u)
    include("${CMAKE_MODULE_PATH}/CodeToH.cmake")
    CODE_TO_H(SOURCES ${CUDA_HELPERS_SRC} VARNAME kernel_files EXTENSION "h"
               OUTPUT_DIR driver/helpers TARGETS cl_kernel_targets EOF "0")
endif()



install(TARGETS isaac LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
set(INSTALL_INCLUDE_DIR /usr/local/include)
install(DIRECTORY isaac "${PROJECT_SOURCE_DIR}/include/isaac"
           DESTINATION "${INSTALL_INCLUDE_DIR}"  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
