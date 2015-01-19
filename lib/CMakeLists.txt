file( GLOB_RECURSE LIBATIDLAS_SRC *.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(atidlas SHARED ${LIBATIDLAS_SRC})
set_target_properties(atidlas PROPERTIES COMPILE_FLAGS "-Wno-sign-compare -D__CL_ENABLE_EXCEPTIONS -Wall -Wextra -pedantic")

#add_library(atidlas_static STATIC ${LIBATIDLAS_SRC})
#set_target_properties(atidlas_static PROPERTIES OUTPUT_NAME atidlas)
