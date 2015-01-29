set(CMAKE_BUILD_TYPE RELEASE)

file( GLOB_RECURSE LIBATIDLAS_SRC *.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(atidlas SHARED ${LIBATIDLAS_SRC})
set_target_properties(atidlas PROPERTIES
   COMPILE_FLAGS "-Wno-sign-compare -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -D__CL_ENABLE_EXCEPTIONS -Wall -Wextra -pedantic")


#install(TARGETS atidlas LIBRARY DESTINATION lib)
#set(INSTALL_INCLUDE_DIR /usr/local/include)
#install(DIRECTORY atidlas "${PROJECT_SOURCE_DIR}/include/atidlas"
#        DESTINATION "${INSTALL_INCLUDE_DIR}"  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

set(POSTINSTALL_PY "${CMAKE_CURRENT_BINARY_DIR}/postinstall.py")
configure_file("${PROJECT_SOURCE_DIR}/python/postinstall.py" ${POSTINSTALL_PY})
install(CODE "execute_process(COMMAND ${PYTHON} ${POSTINSTALL_PY})")
