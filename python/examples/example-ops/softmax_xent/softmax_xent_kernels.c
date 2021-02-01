__global__ void softmax_fwd(TYPE* x __aligned(16),
                    int* indices,
                    TYPE* result __aligned(16),
                    TYPE* result_neg_logprob __aligned(16)) {

            int pid = get_program_id(0);

            int local_offset[TILE] = 0 ... TILE;
            int offset[TILE] = pid * TILE + local_offset;
            TYPE* px[TILE]  = x + offset;
            TYPE* p_neg_logprob[TILE] = result_neg_logprob + offset;
            long local_ind = *(indices + pid);


            TYPE F16[TILE] = *px;
            float Fx[TILE] = F16;
            float max_x = Fx[max];
            float shifted_logit[TILE] = Fx - max_x;
            float exp_logit[TILE] = exp(shifted_logit);
            float sum = exp_logit[+];
            float tile_sum[TILE] = 0.0 * local_offset + sum;
            float neg_logprob[TILE] = log(tile_sum) - shifted_logit;
            TYPE final[TILE] = neg_logprob;
            *p_neg_logprob = final;

            bool check[TILE] = (local_offset==0);
            *?(check)(result + pid) = *(result_neg_logprob + local_ind + TILE * pid);
        }


__global__ void softmax_bwd(TYPE* neg_logprobs __aligned(16),
                    long* indices,
                    TYPE* dneg_logprobs __aligned(16)) {

        int token_idx = get_program_id(0);

        int local_offset[TILE] = 0 ... TILE;
        int offset[TILE] = token_idx * TILE + local_offset;
        TYPE* px[TILE]  = neg_logprobs + offset;
        long local_ind = *(indices + token_idx);
        TYPE local_dn = *(dneg_logprobs + token_idx);

        // We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
        // and we have -log(p[k]) stored, so this is easy

        TYPE F16[TILE] = *px;
        float Fx[TILE] = F16;               // neg_logprob values inside tile
        float neg[TILE] = -Fx;              // probabilities for p[k]
        float probs[TILE] = exp(neg);
        TYPE result[TILE] = probs;
        *px = result;

        // Do one hot

        // selected_logit_idx is selected logit index for our token
        int selected_logit_idx = local_ind;
        bool find_one[TILE] = (local_offset == selected_logit_idx);
        TYPE one_hot[TILE] = find_one;
        *px = *px - one_hot;

        // multiply by dneg_logprobs
        *px = (*px) * local_dn;
    }