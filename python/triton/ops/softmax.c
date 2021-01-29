__global__ void forward(TYPE* X, TYPE* Y) {
    int pid = get_program_id(0);
    int off[BLOCK] = pid * BLOCK + 0 ... BLOCK;
    float x[BLOCK] = *(X + off);
    float shifted[BLOCK] = exp(x - x[max]);
    float sum = shifted[+];
    *(Y + off) = shifted / sum;
} 