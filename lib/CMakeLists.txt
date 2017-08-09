#Compilation
if(ANDROID)
    add_library(isaac STATIC ${LIBISAAC_SRC})
else()
    add_library(isaac SHARED ${LIBISAAC_SRC})
    if(MSVC)
      set_target_properties(isaac PROPERTIES COMPILE_FLAGS "-DISAAC_DLL")
      set_target_properties(isaac PROPERTIES LINK_FLAGS "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/api/blas/clBLAS.def")
    endif()
endif()

#Database
if(NOT ANDROID)
    #Presets
    foreach(VENDOR unknown amd intel nvidia)
        set(DATABASE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/runtime/database/${VENDOR}/")
        file(GLOB_RECURSE JSON_FILES "${DATABASE_PATH}/json/*.json")
        CODE_TO_H(SOURCES ${JSON_FILES} VARNAME database EXTENSION "hpp" OUTPUT_DIR "${DATABASE_PATH}"
                NAMESPACE "isaac database ${VENDOR}" TARGET database_${VENDOR} EOF "1")
        add_dependencies(isaac database_${VENDOR})
    endforeach()
endif()

if(MSVC)
    target_link_libraries(isaac ${DLFCN_LIB_DIR})
else(MSVC)
    target_link_libraries(isaac "dl")
endif(MSVC)

#Cuda JIT headers to file
set(CUDA_HELPERS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/driver/helpers/cuda/)
file(GLOB_RECURSE CUDA_HELPERS_SRC ${CUDA_HELPERS_PATH}/*.cu)
CODE_TO_H(SOURCES ${CUDA_HELPERS_SRC} VARNAME kernel_files EXTENSION "hpp"
          OUTPUT_DIR ${CUDA_HELPERS_PATH} NAMESPACE "isaac helpers cuda" TARGET headers EOF "0")

#Installation
install(TARGETS isaac LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
set(INSTALL_INCLUDE_DIR include)
install(DIRECTORY isaac "${PROJECT_SOURCE_DIR}/include/isaac"
           DESTINATION "${INSTALL_INCLUDE_DIR}"  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/external/"
           DESTINATION "${INSTALL_INCLUDE_DIR}"
           FILES_MATCHING PATTERN "clBLAS*.h"
           PATTERN "cuda" EXCLUDE)
