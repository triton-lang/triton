__global__ void softmax_fwd(TYPE* x __aligned(16),
                    long* indices,
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