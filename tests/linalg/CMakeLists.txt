set(_CLAPACK_SRC slasr xerbla lsame)
set(CLAPACK_SRC "")
foreach(FNAME ${_CLAPACK_SRC})
    list(APPEND CLAPACK_SRC "${CMAKE_CURRENT_SOURCE_DIR}/external/${FNAME}.c")
endforeach()

add_library(lapack-subset SHARED ${CLAPACK_SRC})

foreach(PROG axpy dot ger gemv gemm gesvd)
     add_executable(${PROG} ${PROG}.cpp)
     add_test(${PROG}-test ${PROG})
     target_link_libraries(${PROG} isaac lapack-subset)
endforeach()
