SET(DLFCN_SEARCH_PATHS
    dlfcn-win32
    ../dlfcn-win32
    /
    )

FIND_PATH(DLFCN_INCLUDE_DIR NAMES dlfcn.h PATHS ${DLFCN_SEARCH_PATHS})
IF(WIN32)
    FIND_PATH(DLFCN_LIB_SEARCH_PATH NAMES build/Release/dl.lib PATHS ${DLFCN_SEARCH_PATHS})
ENDIF()

IF(NOT DLFCN_INCLUDE_DIR)
    MESSAGE(FATAL_ERROR "Could not find dlfcn-win32")
ELSE()
    MESSAGE(STATUS "Found dlfcn-win32 include: " ${DLFCN_INCLUDE_DIR})
    IF(WIN32)
        IF(NOT DLFCN_LIB_SEARCH_PATH)
            MESSAGE(FATAL_ERROR "Could not find dl.lib")
        ELSE()
            SET(DLFCN_LIB_DIR ${DLFCN_LIB_SEARCH_PATH}/build/Release/dl.lib)
            MESSAGE(STATUS "Found dl.lib: " ${DLFCN_LIB_DIR})
        ENDIF()
    ENDIF()
ENDIF()