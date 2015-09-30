foreach(PROG tie)
     add_executable(${PROG}-example ${PROG}.cpp)
     target_link_libraries(${PROG}-example isaac)
endforeach(PROG)
