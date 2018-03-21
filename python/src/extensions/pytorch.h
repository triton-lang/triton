// Convolutions
int isaac_conv_nd_float_float(THCudaTensor *inputs, THCudaTensor *filters, THCudaTensor **outputs, int num_outputs,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  THCudaTensor *bias,
                  const char * activation,
                  float alpha,
                  float iscale, float fscale, float* oscale, float zscale,
                  const char * residual, THCudaTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1,
                  size_t optimization_level);

int isaac_conv_nd_int_float(THCudaIntTensor *inputs, THCudaIntTensor *filters, THCudaTensor **outputs, int num_outputs,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  THCudaTensor *bias,
                  const char * activation,
                  float alpha,
                  float iscale, float fscale, float* oscale, float zscale,
                  const char * residual, THCudaTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1,
                  size_t optimization_level);


int isaac_conv_nd_float_int(THCudaTensor *inputs, THCudaTensor *filters, THCudaIntTensor **outputs, int num_outputs,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  THCudaTensor *bias,
                  const char * activation,
                  float alpha,
                  float iscale, float fscale, float* oscale, float zscale,
                  const char * residual, THCudaIntTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1,
                  size_t optimization_level);

int isaac_conv_nd_int_int(THCudaIntTensor *inputs, THCudaIntTensor *filters, THCudaIntTensor **outputs, int num_outputs,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  THCudaTensor *bias,
                  const char * activation,
                  float alpha,
                  float iscale, float fscale, float* oscale, float zscale,
                  const char * residual, THCudaIntTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1,
                  size_t optimization_level);

// Pooling
int isaac_pool_nd_float_float(THCudaTensor *inputs, THCudaTensor *outputs,
                      const char * type,
                      size_t window_d, size_t window_h, size_t window_w,
                      size_t pad_d, size_t pad_h, size_t pad_w,
                      float i_scale, float o_scale,
                      size_t stride_d, size_t stride_h, size_t stride_w,
                      size_t optimization_level);

int isaac_pool_nd_int_int(THCudaIntTensor *inputs, THCudaIntTensor *outputs,
                      const char * type,
                      size_t window_d, size_t window_h, size_t window_w,
                      size_t pad_d, size_t pad_h, size_t pad_w,
                      float i_scale, float o_scale,
                      size_t stride_d, size_t stride_h, size_t stride_w,
                      size_t optimization_level);


int isaac_pool_nd_int_float(THCudaIntTensor *inputs, THCudaTensor *outputs,
                      const char * type,
                      size_t window_d, size_t window_h, size_t window_w,
                      size_t pad_d, size_t pad_h, size_t pad_w,
                      float i_scale, float o_scale,
                      size_t stride_d, size_t stride_h, size_t stride_w,
                      size_t optimization_level);


// Linear
int isaac_linear_int_float(THCudaIntTensor *inputs, THCudaIntTensor *weights, THCudaTensor *outputs,
                           THCudaTensor *bias,
                           float alpha, float beta,
                           float a_scale, float b_scale, float c_scale,
                           size_t optimization_level);

int isaac_linear_float_float(THCudaTensor *inputs, THCudaTensor *weights, THCudaTensor *outputs,
                             THCudaTensor *bias,
                             float alpha, float beta,
                             float a_scale, float b_scale, float c_scale,
                             size_t optimization_level);

// Packing
int isaac_pack_nd(THCudaTensor* inputs, THCudaIntTensor* outputs, float a, float b);
