include("${CMAKE_MODULE_PATH}/android/apk.cmake" REQUIRED)

# Sources
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories("${ANDROID_NDK}/sources/android/native_app_glue" "${ANDROID_SYSROOT}/usr/include")
set(SRC "test.cpp" "${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c")

add_library(cmakeandroidtest SHARED ${SRC})
target_link_libraries(cmakeandroidtest isaac log android ${OPENCL_LIBRARIES})
set_target_properties(cmakeandroidtest PROPERTIES COMPILE_DEFINITIONS "ANDROID")
android_create_apk(cmakeandroidtest "isaac.test" "${CMAKE_CURRENT_BINARY_DIR}/apk" "${CMAKE_CURRENT_SOURCE_DIR}/libs" "${CMAKE_CURRENT_SOURCE_DIR}/android" "${CMAKE_CURRENT_SOURCE_DIR}/assets")
