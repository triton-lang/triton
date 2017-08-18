IF(WIN32)
    SET(DLFCN_SEARCH_PATHS
        dlfcn-win32
        ../dlfcn-win32
        ../../dlfcn-win32
	$ENV{DLFCN_PATH}
        /
        )
    FIND_PATH(DLFCN_INCLUDE_DIR NAMES dlfcn.h PATHS ${DLFCN_SEARCH_PATHS})
    FIND_LIBRARY(DLFCN_LIB_DIR
	         NAMES dl
		 PATHS ${DLFCN_SEARCH_PATHS}
		 PATH_SUFFIXES Release Debug
		 )
    IF(NOT DLFCN_INCLUDE_DIR)
        MESSAGE(FATAL_ERROR "Could not find dlfcn-win32")
    ELSE()
        MESSAGE(STATUS "Found dlfcn-win32 include: " ${DLFCN_INCLUDE_DIR})
	IF(NOT DLFCN_LIB_DIR)
            MESSAGE(FATAL_ERROR "Could not find dl.lib")
        ELSE()
            MESSAGE(STATUS "Found dl.lib: " ${DLFCN_LIB_DIR})
        ENDIF()
    ENDIF()
ENDIF()
