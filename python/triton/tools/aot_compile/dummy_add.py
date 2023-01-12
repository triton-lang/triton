from aot_compile.compile_metadata import jit, constexpr

@jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector
    y_ptr,  # *Pointer* to second input vector
    output_ptr,  # *Pointer* to output vector
    n_elements,  # Size of the vector
    BLOCK_SIZE: constexpr = 32  # Number of elements each program should process
    # NOTE: `constexpr` so it can be used as a shape value
):
    pass



