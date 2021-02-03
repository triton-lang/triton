__global__ void softmax_fwd(TYPE *logit,
                        TYPE *modified_logit,
                        long *indices __readonly,
                        TYPE *result,
                        int n_vocab) {

            int token_idx = get_program_id(0);

            bool check[TILE] = ((0 ... TILE) < n_vocab);
            int offset[TILE] = token_idx * n_vocab + 0 ... TILE;
            TYPE* px[TILE]  = logit + offset;
            TYPE* pmodified[TILE] = modified_logit + offset;
            long local_ind = *(indices + token_idx);

            TYPE F16[TILE] = check ? *px : -INFINITY;
            float shifted_logit[TILE] = F16 - F16[max];
            float neg_logprob[TILE] = log(exp(shifted_logit)[+]) - shifted_logit;
            *?(check)pmodified = neg_logprob;
            __debug_barrier();
            *(result + token_idx) = *(modified_logit + (local_ind + n_vocab * token_idx));
        }


__global__ void softmax_bwd(TYPE *neg_logprobs,
                    long *indices,
                    TYPE *dneg_logprobs) {

        int token_idx = get_program_id(0);

        int offset[TILE] = token_idx * TILE + 0 ... TILE;
        TYPE* px[TILE]  = neg_logprobs + offset;
        long local_ind = *(indices + token_idx);
        TYPE local_dn = *(dneg_logprobs + token_idx);

        // We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
        // and we have -log(p[k]) stored, so this is easy

        *px = exp(-(float[TILE]) *px);

        // Do one hot

        // selected_logit_idx is selected logit index for our token
        bool find_one[TILE] = ((0 ... TILE) == local_ind);
        *px = *px - (TYPE[TILE]) find_one;

        // multiply by dneg_logprobs
        *px = (*px) * local_dn;
    }