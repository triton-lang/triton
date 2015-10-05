foreach(PROG simple)
     add_executable(example-${PROG} ${PROG}.cpp)
     target_link_libraries(example-${PROG} isaac)
endforeach(PROG)
