__global__ void forward(TYPE *logit __aligned(16),
                        TYPE *modified_logit __aligned(16),
                        long *indices __readonly,
                        TYPE *result __aligned(16),
                        int n_cols __multipleof(N_COLS_MULT)) {
  int row = get_program_id(0);

  bool check[TILE] = ((0 ... TILE) < n_cols);
  int offset[TILE] = row * n_cols + 0 ... TILE;
  TYPE *px[TILE] = logit + offset;
  TYPE *pmodified[TILE] = modified_logit + offset;
  long local_ind = *(indices + row);

  TYPE F16[TILE] = check ? *px : -INFINITY;
  float shifted_logit[TILE] = F16 - F16[max];
  float neg_logprob[TILE] = log(exp(shifted_logit)[+]) - shifted_logit;
  *? (check)pmodified = neg_logprob;
  __debug_barrier();
  *(result + row) = *(modified_logit + (local_ind + n_cols * row));
}

__global__ void backward(TYPE *neg_logprobs __aligned(16),
                         long *indices __aligned(16),
                         TYPE *dneg_logprobs __aligned(16),
                         int n_cols __multipleof(N_COLS_MULT)) {

  int row = get_program_id(0);
  // pointer arithmetic
  bool check[TILE] = ((0 ... TILE) < n_cols);
  int offset[TILE] = row * n_cols + 0 ... TILE;
  TYPE *px[TILE] = neg_logprobs + offset;
  long local_ind = *(indices + row);
  TYPE local_dn = *(dneg_logprobs + row);
  // We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
  // and we have -log(p[k]) stored, so this is easy
  TYPE intermediate[TILE] = check ? exp(-(float[TILE]) * ? (check)px) : 0;
  // selected_logit_idx is selected logit index for our token
  bool find_one[TILE] = ((0 ... TILE) == local_ind);
  intermediate = intermediate - ((TYPE[TILE])find_one);
  // multiply by dneg_logprobs
  *? (check)px = intermediate * local_dn;
}