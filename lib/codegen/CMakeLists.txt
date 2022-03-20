file(GLOB_RECURSE CODEGEN_SRC *.cc)

add_library(TritonCodeGen
  ${CODEGEN_SRC}
)
