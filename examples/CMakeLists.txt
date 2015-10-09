foreach(PROG indexing)
     add_executable(example-${PROG} ${PROG}.cpp)
     target_link_libraries(example-${PROG} isaac)
endforeach(PROG)
