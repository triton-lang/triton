__global__ void softmax_fwd(TYPE *logit,
                        TYPE *modified_logit,
                        long *indices __readonly,
                        TYPE *result) {

            int pid = get_program_id(0);

            int offset[TILE] = pid * TILE + 0 ... TILE;
            TYPE* px[TILE]  = logit + offset;
            TYPE* pmodified[TILE] = logit + offset;
            long local_ind = *(indices + pid);

            TYPE F16[TILE] = *px;
            float shifted_logit[TILE] = F16 - F16[max];
            float neg_logprob[TILE] = log(exp(shifted_logit)[+]) - shifted_logit;
            *pmodified = neg_logprob;
            *(result + pid) = *(modified_logit + (local_ind + TILE * pid));
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

        // TYPE F16[TILE] = *px;
        // float probs[TILE] = exp(-F16);
        // TYPE result[TILE] = probs;
        *px = exp(-(float[TILE]) *px);  // result;

        // Do one hot

        // selected_logit_idx is selected logit index for our token
        bool find_one[TILE] = ((0 ... TILE) == local_ind);
        //__debug_barrier();
        //TYPE one_hot[TILE] = find_one;
        *px = *px - (TYPE[TILE]) find_one;

        // multiply by dneg_logprobs
        *px = (*px) * local_dn;
    }