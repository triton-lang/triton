__global__ void forward(TYPE *X __readonly __noalias,
                        float scale,
                        int *LUT __readonly __noalias,
                        TYPE *RPE __readonly __noalias,
                        TYPE *KP_M __readonly __noalias,
                        TYPE *ATTN_M __readonly __noalias,
                        int sizemax,
                        long stride_zx,
                        long stride_zrpe,
                        int stride_hrpe,
                        int stride_srpe,
                        int stride_zkpm,
                        int stride_zattnm) {
  int pidhm = get_program_id(0);
  int pidz = get_program_id(1);
  // create index ranges
  int rxm = pidhm % BLOCK;
  int rbm = pidhm / BLOCK;
  int rxn[TN] = (0 ... TN) % BLOCK;
  int rbn[TN] = (0 ... TN) / BLOCK;
  // extract information from look-up table
  int *header = LUT + rbm * 2;
  int size = *(header + 0);
  int offset = *(header + 1);
  bool check[TN] = rbn < size;
  int rbmn[TN] = check ? rbn : size - 1;
  // block id and column id
  long blockid[TN] = *(LUT + offset + rbmn * 4 + 0);
  long columnid[TN] = *(LUT + offset + rbmn * 4 + 1);
  long rowid[TN] = *(LUT + offset + rbmn * 4 + 2);
  long headid[TN] = *(LUT + offset + rbmn * 4 + 3);
  // pointers to X
  TYPE *px[TN] = X + pidz * stride_zx + blockid * BLOCK * BLOCK + rxm * BLOCK + rxn;
#ifdef APPLY_RPE
  // pointers to relative position embedding
  TYPE *prpe[TN] = RPE + pidz * stride_zrpe + headid * stride_hrpe + columnid * BLOCK + rowid * BLOCK * stride_srpe + rxm * stride_srpe + rxn;
#endif
#ifdef APPLY_KP_MASK
  // pointers to key padding mask
  TYPE *pkp_m[TN] = KP_M + pidz * stride_zkpm + columnid * BLOCK + rxn;
#endif
#ifdef APPLY_ATTN_MASK
  // pointers to attention mask
  TYPE *pattn_m[TN] = ATTN_M + columnid * BLOCK + rowid * BLOCK * stride_zattnm + rxm * stride_zattnm + rxn;
#endif

  // load  input
  TYPE x[TN] = check ? *px : -INFINITY;
#ifdef APPLY_RPE
  // load relative position embedding
  TYPE rpe[TN] = check ? *prpe : 0;
#endif
#ifdef APPLY_KP_MASK
  // load key-padding mask
  TYPE kp_m[TN] = check ? *pkp_m : -INFINITY;
#endif
#ifdef APPLY_ATTN_MASK
  // load attention mask
  TYPE attn_m[TN] = check ? *pattn_m : -INFINITY;
#endif
  // compute softmax in float
#ifdef APPLY_RPE
  float Frpe[TN] = rpe;
#endif
#ifdef APPLY_KP_MASK
  float Fkp_m[TN] = kp_m;
#endif
#ifdef APPLY_ATTN_MASK
  float Fattn_m[TN] = attn_m;
#endif
#ifdef KP_MASK_MUL
  Fkp_m = (Fkp_m == 0) ? (float[TN]) - INFINITY : 0;
#endif
#ifdef ATTN_MASK_MUL
  Fattn_m = (Fattn_m == 0) ? (float[TN]) - INFINITY : 0;
#endif
  float Fx[TN] = x;
#ifdef APPLY_SCALE
  Fx = Fx * scale; // apply scale
#endif
#ifdef APPLY_RPE
  Fx = Fx + Frpe; // apply relative position embedding
#endif
#ifdef APPLY_KP_MASK
  Fx = Fx + Fkp_m; // apply key padding mask
#endif
#ifdef APPLY_ATTN_MASK
  Fx = Fx + Fattn_m; // apply attention mask
#endif
  float Fxmax = Fx[max];
  float Fy[TN] = exp(Fx - Fxmax);
  float Fysum = (check ? Fy : 0)[+];
  // write-back in half/float
  TYPE y[TN] = Fy;
  TYPE ysum = Fysum;
  *? (check)px = y / ysum;
}

__global__ void backward(TYPE *X __readonly __noalias,
                         float scale,
                         TYPE *DX __readonly __noalias,
                         int *LUT,
                         int sizemax,
                         long stride_zx,
                         long stride_zdx) {
  int pidhm = get_program_id(0);
  int pidz = get_program_id(1);
  // create index ranges
  int rxm = pidhm % BLOCK;
  int rbm = pidhm / BLOCK;
  int rxn[TN] = (0 ... TN) % BLOCK;
  int rbn[TN] = (0 ... TN) / BLOCK;
  // extract information from look-up table
  int *header = LUT + rbm * 2;
  int size = *(header + 0);
  int offset = *(header + 1);
  // bounds checking on lut
  bool check[TN] = rbn < size;
  int rbmn[TN] = check ? rbn : size - 1;
  // initialize pointers to block-sparse input
  long blockid[TN] = *(LUT + offset + rbmn * 4);
  TYPE *px[TN] = X + pidz * stride_zx + blockid * BLOCK * BLOCK + rxm * BLOCK + rxn;
  TYPE *pdx[TN] = DX + pidz * stride_zdx + blockid * BLOCK * BLOCK + rxm * BLOCK + rxn;
  // compute fused softmax backward
  TYPE x[TN] = check ? *px : 0;
  TYPE dx[TN] = check ? *pdx : 0;
  float Fdx[TN] = dx;
  float Fx[TN] = x;
  float Fxdx[TN] = Fdx * Fx;
  float Fxdxsum = Fxdx[+];
  float Fy[TN] = Fx * (Fdx - Fxdxsum) * scale;
  TYPE y[TN] = Fy;
  // write-back
  *? (check)pdx = y;
}
