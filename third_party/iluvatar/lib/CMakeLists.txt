add_compile_options("-Wno-deprecated-declarations")
add_compile_options("-Wno-error=deprecated-declarations")

add_subdirectory(Analysis)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Target)
add_subdirectory(Tools)
