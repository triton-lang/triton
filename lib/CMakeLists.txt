#Database
set(DATABASE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/model/presets/")
file(GLOB_RECURSE JSON_FILES "${DATABASE_PATH}/json/*.json")
CODE_TO_H(SOURCES ${JSON_FILES} VARNAME json_files EXTENSION "hpp" OUTPUT_DIR "${DATABASE_PATH}"
        NAMESPACE "isaac presets" TARGET database EOF "1")

#Compilation
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/tools")
if(ANDROID)
    add_library(isaac STATIC ${LIBISAAC_SRC})
else()
    add_library(isaac SHARED ${LIBISAAC_SRC})
	if(MSVC)
		set_target_properties(isaac PROPERTIES COMPILE_FLAGS "-DISAAC_DLL")
		set_target_properties(isaac PROPERTIES LINK_FLAGS "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/wrap/clBLAS.def")
	endif()
endif()

#Linkage
if(OPENCL_FOUND)
    target_link_libraries(isaac ${OPENCL_LIBRARIES})
endif()
if(CUDA_FOUND)
    cuda_find_library_local_first(CUDA_NVRTC_LIBRARY nvrtc "\"nvrtc\" library")
    target_link_libraries(isaac ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIBRARY})
    #Cuda JIT headers to file
    set(CUDA_HELPERS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/driver/helpers/cuda/)
    file(GLOB_RECURSE CUDA_HELPERS_SRC ${CUDA_HELPERS_PATH}/*.cu)
    CODE_TO_H(SOURCES ${CUDA_HELPERS_SRC} VARNAME kernel_files EXTENSION "hpp"
              OUTPUT_DIR ${CUDA_HELPERS_PATH} NAMESPACE "isaac helpers cuda" TARGET cuda_headers EOF "0")
endif()

#Installation
install(TARGETS isaac LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
set(INSTALL_INCLUDE_DIR /usr/local/include)
install(DIRECTORY isaac "${PROJECT_SOURCE_DIR}/include/isaac"
           DESTINATION "${INSTALL_INCLUDE_DIR}"  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
