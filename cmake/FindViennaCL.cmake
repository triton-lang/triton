find_path(
    VIENNACL_INCLUDE_DIR
    NAMES viennacl/vector.hpp
)

set(VIENNACL_INCLUDE_DIRS ${VIENNACL_INCLUDE_DIR})
mark_as_advanced(VIENNACL_INCLUDE_DIRS)
