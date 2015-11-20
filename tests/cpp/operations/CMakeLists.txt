foreach(NAME axpy dot ger gemv gemm)
    add_isaac_test(${NAME})
endforeach()
