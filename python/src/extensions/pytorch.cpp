#include <THC/THC.h>
#include "isaac/api.h"

extern THCState *state;

#define WRAP0(RET, NAME, TYPE0) RET NAME(THCState *state, TYPE0 *x){ return TYPE0 ## _ ## NAME(state,x);}
#define WRAP1(RET, NAME, TYPE0, TYPE1) RET NAME(THCState *state, TYPE0 *x, TYPE1 arg0){ return TYPE0 ## _ ## NAME(state, x, arg0);}

WRAP0(int, nDimension, THCudaTensor)
//WRAP0(int, nDimension, THCudaIntTensor)
WRAP1(int, size, THCudaTensor, int)
//WRAP1(int, size, THCudaIntTensor, int)
WRAP1(int, stride, THCudaTensor, int)
//WRAP1(int, stride, THCudaIntTensor, int)
WRAP0(THCudaStorage*, storage, THCudaTensor)
//WRAP0(float*, data, THCudaStorage)

//WRAP0(THCudaIntStorage*, storage, THCudaIntTensor)
void resizeNd(THCState *state, THCudaTensor *tensor, int nDimension, long *size, long *stride)
{ return THCudaTensor_resizeNd(state, tensor, nDimension, size, stride);}
//void resizeNd(THCState *state, THCudaIntTensor *tensor, int nDimension, long *size, long *stride)
//{ return THCudaIntTensor_resizeNd(state, tensor, nDimension, size, stride);}

CUdeviceptr data(isaac::DType dtype, THCState *state, THCudaTensor *x){
  if(dtype == isaac::FLOAT_TYPE)
      return (CUdeviceptr)THCudaTensor_data(state, x);
  return (CUdeviceptr)THCudaIntTensor_data(state, x);
}

inline isaac::PoolType get_sc_pool(const std::string & activation){
    if(activation == "avg") return isaac::AvgPool;
    if(activation == "max") return isaac::MaxPool;
    throw std::runtime_error("Unknown pooling function");
}

inline isaac::ActivationType get_sc_activation(const std::string & activation){
    if(activation == "relu") return isaac::ReLU;
    if(activation == "linear") return isaac::Linear;
    if(activation == "sigmoid") return isaac::Sigmoid;
    if(activation == "elu") return isaac::ELU;
    throw std::runtime_error("Unknown activation function");
}

inline isaac::ResidualType get_sc_residual(const std::string & residual){
    if(residual == "") return isaac::NoResidual;
    if(residual == "cat") return isaac::CatResidual;
    if(residual == "add") return isaac::AddResidual;
    throw std::runtime_error("Unknown residual function");
}

/* Convolution */
template<typename IN_TYPE, typename OUT_TYPE>
int isaac_conv_nd_impl(isaac::DType in_dtype, isaac::DType out_dtype,  IN_TYPE *inputs, IN_TYPE *filters, OUT_TYPE **outputs, int num_outputs,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  THCudaTensor * bias,
                  const char * activation, float alpha,
                  float i_scale, float f_scale, float * o_scale, float z_scale,
                  const char * residual, OUT_TYPE *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1,
                  size_t optimization_level)
{
  int DIM = nDimension(state, inputs) - 2;
  isaac::ActivationType sc_activation = get_sc_activation(activation);
  isaac::ResidualType sc_residual = get_sc_residual(residual);

  // Datatype
  size_t vect_c = (in_dtype==isaac::INT8X4_TYPE)?4:1;
  size_t vect_k = (out_dtype==isaac::INT8X4_TYPE)?4:1;


  // Inputs
  size_t N = size(state, inputs, 0);
  size_t Ci = size(state, inputs, 1);
  size_t D = 1, H = 1, W = 1;
  if(DIM > 2) D = size(state, inputs, 2);
  if(DIM > 1) H = size(state, inputs, 2 + (DIM > 2));
  if(DIM > 0) W = size(state, inputs, 2 + (DIM > 2) + (DIM > 1));

  // Filter
  size_t T = 1, R = 1, S = 1;
  size_t Cf = size(state, filters, 0);
  if(DIM > 2) T = size(state, filters, 1);
  if(DIM > 1) R = size(state, filters, 1 + (DIM > 2));
  if(DIM > 0) S = size(state, filters, 1 + (DIM > 2) + (DIM > 1));
  long K = size(state, filters, 1 + DIM);

  if(Ci != Cf)
    return 0;
  size_t C = Ci;

  // Output shapes
  isaac::param_t M, P, Q;
  isaac::templates::Conv::output_shapes(D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, M, P, Q);

  // Create output
  size_t Zk = z?size(state, z, 1):0;

  long output_sizes[5];
  output_sizes[0] = N;
  output_sizes[1] = K/vect_k;
  if(sc_residual == isaac::CatResidual)
    output_sizes[1] += Zk;
  if(DIM > 2) output_sizes[2] = M;
  if(DIM > 1) output_sizes[2 + (DIM > 2)] = P;
  if(DIM > 0) output_sizes[2 + (DIM > 2) + (DIM > 1)] = Q;
  for(int i = 0; i < num_outputs; i++)
    resizeNd(state, outputs[i], 2 + DIM, output_sizes, NULL);

  // Wrap handles
  isaac::driver::Stream stream(THCState_getCurrentStream(state), false);
  isaac::driver::Buffer I(stream.context(), data(in_dtype, state, inputs), false);
  isaac::driver::Buffer F(stream.context(), data(in_dtype, state, filters), false);
  std::vector<isaac::driver::Buffer> O;
  for(int i = 0; i < num_outputs; i++)
    O.push_back(isaac::driver::Buffer(stream.context(), data(out_dtype, state, outputs[i]), false));
  std::unique_ptr<isaac::driver::Buffer> Z;
  if(z)
    Z.reset(new isaac::driver::Buffer(stream.context(), data(out_dtype, state, z), false));
  std::unique_ptr<isaac::driver::Buffer> Bias;
  if(bias)
    Bias.reset(new isaac::driver::Buffer(stream.context(), data(isaac::FLOAT_TYPE, state, bias), false));

  // Execute
  isaac::CONV(stream.context().device(), stream, in_dtype, out_dtype, N, K, M, P, Q, C*vect_c, T, R, S, D, H, W,
              pad_d, pad_h, pad_w,
              stride_d, stride_h, stride_w,
              upsample_d, upsample_h, upsample_w,
              I, F, O.data(), num_outputs,
              Bias.get(),
              sc_activation, alpha,
              i_scale, f_scale, std::vector<float>(o_scale, o_scale + num_outputs), z_scale,
              sc_residual, Zk*vect_k, crop_z_d0, crop_z_d1, crop_z_h0, crop_z_h1, crop_z_w0, crop_z_w1, Z.get(),
              NULL, optimization_level);

  return 1;
}

/* Pooling */
template<typename IN_TYPE, typename OUT_TYPE>
int isaac_pool_nd_impl(isaac::DType in_dtype, isaac::DType out_dtype, IN_TYPE *inputs, OUT_TYPE *outputs,
                      const char * type,
                      size_t window_d, size_t window_h, size_t window_w,
                      size_t pad_d, size_t pad_h, size_t pad_w,
                      float i_scale, float o_scale,
                      size_t stride_d, size_t stride_h, size_t stride_w,
                      size_t optimization_level){
  int DIM = nDimension(state, inputs) - 2;

  // Datatype
  size_t vect_c = (in_dtype==isaac::INT8X4_TYPE)?4:1;
  size_t vect_k = (out_dtype==isaac::INT8X4_TYPE)?4:1;

  // Inputs
  size_t N = size(state, inputs, 0);
  size_t C = size(state, inputs, 1);
  size_t D = 1, H = 1, W = 1;
  if(DIM > 2) D = size(state, inputs, 2);
  if(DIM > 1) H = size(state, inputs, 2 + (DIM > 2));
  if(DIM > 0) W = size(state, inputs, 2 + (DIM > 2) + (DIM > 1));
  size_t T = window_d, R = window_h, S = window_w;

  // Output shapes
  isaac::param_t M, P, Q;
  isaac::templates::Conv::output_shapes(D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, 1, 1, 1, M, P, Q);

  // Create output
  long output_sizes[5];
  output_sizes[0] = N;
  output_sizes[1] = C * vect_c / vect_k;
  if(DIM > 2) output_sizes[2] = M;
  if(DIM > 1) output_sizes[2 + (DIM > 2)] = P;
  if(DIM > 0) output_sizes[2 + (DIM > 2) + (DIM > 1)] = Q;
  resizeNd(state, outputs, 2 + DIM, output_sizes, NULL);

  // Wrap handles
  isaac::driver::Stream stream(THCState_getCurrentStream(state), false);
  isaac::driver::Buffer I(stream.context(), data(in_dtype, state, inputs), false);
  isaac::driver::Buffer O(stream.context(), data(out_dtype, state, outputs), false);

  // Execute
  isaac::POOL(stream.context().device(), stream,
              in_dtype, out_dtype,
              get_sc_pool(type),
              C*vect_c, M, P, Q, N, T, R, S, D, H, W,
              pad_d, pad_h, pad_w,
              stride_d, stride_h, stride_w,
              I, O,
              i_scale, o_scale,
              NULL, optimization_level);

  return 1;
}

/* Linear */
template<typename IN_TYPE, typename OUT_TYPE>
int isaac_linear_impl(isaac::DType in_dtype, isaac::DType out_dtype,
                      IN_TYPE *inputs, IN_TYPE *weights, OUT_TYPE *outputs,
                      THCudaTensor *bias,
                      float a, float b,
                      float input_scale, float weight_scale, float output_scale,
                      size_t optimization_level)
{
  size_t vect_in = (in_dtype==isaac::INT8X4_TYPE)?4:1;
//  size_t vect_k = (out_dtype==isaac::INT8X4_TYPE)?4:1;


  // Inputs
  long M = size(state, inputs, 0);
  long Ki = size(state, inputs, 1);

  // Filter
  long Kf = size(state, weights, 0);
  long N = size(state, weights, 1);

  // Check shapes
  if(Ki != Kf)
    return 0;
  long K = Ki;

  // Create output
  long output_sizes[2] = {M, N};
  resizeNd(state, outputs, 2, output_sizes, NULL);

  // Get strides
  size_t lda = stride(state, weights, 0);
  size_t ldb = stride(state, inputs, 0);
  size_t ldc = stride(state, outputs, 0);

  // Wrap handles
  isaac::driver::Stream stream(THCState_getCurrentStream(state), false);
  isaac::driver::Buffer A(stream.context(), data(in_dtype, state, weights), false);
  isaac::driver::Buffer B(stream.context(), data(in_dtype, state, inputs), false);
  isaac::driver::Buffer C(stream.context(), data(out_dtype, state, outputs), false);
  std::unique_ptr<isaac::driver::Buffer> Bias;
  if(bias)
    Bias.reset(new isaac::driver::Buffer(stream.context(), data(isaac::FLOAT_TYPE, state, bias), false));
  isaac::scalar alpha(a, isaac::FLOAT_TYPE);
  isaac::scalar beta(b, isaac::FLOAT_TYPE);

  isaac::GEMM(stream.context().device(), stream, in_dtype, out_dtype, isaac::ISAAC_OP_N, isaac::ISAAC_OP_N, N, M, K*vect_in, 0, lda, 0, ldb, 0, ldc, alpha, A, B, beta, C, weight_scale, input_scale, output_scale, Bias.get(), NULL, optimization_level);
  return 1;
}


extern "C"
{


int isaac_conv_nd_float_float(THCudaTensor *inputs, THCudaTensor *filters, THCudaTensor **outputs, int num_outputs,
                size_t upsample_d, size_t upsample_h, size_t upsample_w,
                size_t pad_d, size_t pad_h, size_t pad_w,
                size_t stride_d, size_t stride_h, size_t stride_w,
                THCudaTensor *bias,
                const char * activation,
                float alpha,
                float iscale, float fscale, float* oscale, float zscale,
                const char * residual, THCudaTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1,
                size_t optimization_level)
{
  return isaac_conv_nd_impl(isaac::FLOAT_TYPE, isaac::FLOAT_TYPE, inputs, filters, outputs, num_outputs, upsample_d, upsample_h, upsample_w, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, bias, activation, alpha,
                            iscale, fscale, oscale, zscale, residual, z, crop_z_d0, crop_z_d1, crop_z_h0, crop_z_h1, crop_z_w0, crop_z_w1, optimization_level);
}

int isaac_conv_nd_int_float(THCudaIntTensor *inputs, THCudaIntTensor *filters, THCudaTensor **outputs, int num_outputs,
                size_t upsample_d, size_t upsample_h, size_t upsample_w,
                size_t pad_d, size_t pad_h, size_t pad_w,
                size_t stride_d, size_t stride_h, size_t stride_w,
                THCudaTensor *bias,
                const char * activation,
                float alpha,
                float iscale, float fscale, float* oscale, float zscale,
                const char * residual, THCudaTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1,
                size_t optimization_level)
{
  return isaac_conv_nd_impl(isaac::INT8X4_TYPE, isaac::FLOAT_TYPE, inputs, filters, outputs, num_outputs, upsample_d, upsample_h, upsample_w, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, bias, activation, alpha,
                            iscale, fscale, oscale, zscale, residual, z, crop_z_d0, crop_z_d1, crop_z_h0, crop_z_h1, crop_z_w0, crop_z_w1, optimization_level);
}

int isaac_conv_nd_float_int(THCudaTensor *inputs, THCudaTensor *filters, THCudaIntTensor **outputs, int num_outputs,
                size_t upsample_d, size_t upsample_h, size_t upsample_w,
                size_t pad_d, size_t pad_h, size_t pad_w,
                size_t stride_d, size_t stride_h, size_t stride_w,
                THCudaTensor *bias,
                const char * activation,
                float alpha,
                float iscale, float fscale, float* oscale, float zscale,
                const char * residual, THCudaIntTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1,
                size_t optimization_level)
{
  return isaac_conv_nd_impl(isaac::FLOAT_TYPE, isaac::INT8X4_TYPE, inputs, filters, outputs, num_outputs, upsample_d, upsample_h, upsample_w, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, bias, activation, alpha,
                            iscale, fscale, oscale, zscale, residual, z, crop_z_d0, crop_z_d1, crop_z_h0, crop_z_h1, crop_z_w0, crop_z_w1, optimization_level);
}

int isaac_conv_nd_int_int(THCudaIntTensor *inputs, THCudaIntTensor *filters, THCudaIntTensor **outputs, int num_outputs,
                size_t upsample_d, size_t upsample_h, size_t upsample_w,
                size_t pad_d, size_t pad_h, size_t pad_w,
                size_t stride_d, size_t stride_h, size_t stride_w,
                THCudaTensor *bias,
                const char * activation,
                float alpha,
                float iscale, float fscale, float* oscale, float zscale,
                const char * residual, THCudaIntTensor *z, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1,
                size_t optimization_level)
{
  return isaac_conv_nd_impl(isaac::INT8X4_TYPE, isaac::INT8X4_TYPE, inputs, filters, outputs, num_outputs, upsample_d, upsample_h, upsample_w, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, bias, activation, alpha,
                            iscale, fscale, oscale, zscale, residual, z, crop_z_d0, crop_z_d1, crop_z_h0, crop_z_h1, crop_z_w0, crop_z_w1, optimization_level);
}

/* Pooling */
int isaac_pool_nd_float_float(THCudaTensor *inputs, THCudaTensor *outputs,
                      const char * type,
                      size_t window_d, size_t window_h, size_t window_w,
                      size_t pad_d, size_t pad_h, size_t pad_w,
                      float i_scale, float o_scale,
                      size_t stride_d, size_t stride_h, size_t stride_w,
                      size_t optimization_level){
  return isaac_pool_nd_impl(isaac::FLOAT_TYPE, isaac::FLOAT_TYPE, inputs, outputs, type, window_d, window_h, window_w, pad_d, pad_h, pad_w, i_scale, o_scale, stride_d, stride_h, stride_w, optimization_level);
}


int isaac_pool_nd_int_int(THCudaIntTensor *inputs, THCudaIntTensor *outputs,
                      const char * type,
                      size_t window_d, size_t window_h, size_t window_w,
                      size_t pad_d, size_t pad_h, size_t pad_w,
                      float i_scale, float o_scale,
                      size_t stride_d, size_t stride_h, size_t stride_w,
                      size_t optimization_level){
  return isaac_pool_nd_impl(isaac::INT8X4_TYPE, isaac::INT8X4_TYPE, inputs, outputs, type, window_d, window_h, window_w, pad_d, pad_h, pad_w, i_scale, o_scale, stride_d, stride_h, stride_w, optimization_level);
}


int isaac_pool_nd_int_float(THCudaIntTensor *inputs, THCudaTensor *outputs,
                      const char * type,
                      size_t window_d, size_t window_h, size_t window_w,
                      size_t pad_d, size_t pad_h, size_t pad_w,
                      float i_scale, float o_scale,
                      size_t stride_d, size_t stride_h, size_t stride_w,
                      size_t optimization_level){
  return isaac_pool_nd_impl(isaac::INT8X4_TYPE, isaac::FLOAT_TYPE, inputs, outputs, type, window_d, window_h, window_w, pad_d, pad_h, pad_w, i_scale, o_scale, stride_d, stride_h, stride_w, optimization_level);
}


/* Linear */
int isaac_linear_int_float(THCudaIntTensor *inputs, THCudaIntTensor *weights, THCudaTensor *outputs, THCudaTensor *bias,
                           float alpha, float beta,
                           float a_scale, float b_scale, float c_scale,
                           size_t optimization_level){
  return isaac_linear_impl(isaac::INT8X4_TYPE, isaac::FLOAT_TYPE, inputs, weights, outputs, bias, alpha, beta, a_scale, b_scale, c_scale, optimization_level);
}

int isaac_linear_float_float(THCudaTensor *inputs, THCudaTensor *weights, THCudaTensor *outputs, THCudaTensor *bias,
                            float alpha, float beta,
                             float a_scale, float b_scale, float c_scale,
                             size_t optimization_level){
  return isaac_linear_impl(isaac::FLOAT_TYPE, isaac::FLOAT_TYPE, inputs, weights, outputs, bias, alpha, beta, a_scale, b_scale, c_scale, optimization_level);
}


/* Transform */
int isaac_pack_nd(THCudaTensor* inputs, THCudaIntTensor *outputs, float a, float b){
  size_t DIM = THCudaTensor_nDimension(state, inputs);
  std::vector<long> sizes(DIM);
  for(size_t i = 0; i < DIM; i++)
    sizes[i] = THCudaTensor_size(state, inputs, i);

  // Allocate output
  if(sizes[1] % 4 != 0)
    return 0;
  sizes[1] /= 4;
  THCudaIntTensor_resizeNd(state, outputs, DIM, sizes.data(), NULL);

  // Wrap handles
  isaac::driver::Stream stream(THCState_getCurrentStream(state), false);
  isaac::driver::Buffer I(stream.context(), (CUdeviceptr)THCudaTensor_data(state, inputs), false);
  isaac::driver::Buffer O(stream.context(), (CUdeviceptr)THCudaIntTensor_data(state, outputs), false);
  isaac::scalar alpha(a, isaac::FLOAT_TYPE);
  isaac::scalar beta(b, isaac::FLOAT_TYPE);

  // Execute
  long D = (DIM > 4)?sizes[2]:1;
  long H = (DIM > 3)?sizes[2 + (DIM>4)]:1;
  long W = (DIM > 2)?sizes[2 + (DIM>4) + (DIM>3)]:1;
  isaac::driver::cudnnTransformTensor(stream, isaac::FLOAT_TYPE, isaac::INT8X4_TYPE, CUDNN_TENSOR_NCHW, CUDNN_TENSOR_NCHW_VECT_C,
                                      sizes[0], sizes[1]*4, D, H, W, alpha, I, beta, O);

  return 1;
}

}
