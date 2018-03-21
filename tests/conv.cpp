#include <sstream>
#include <chrono>
#include <exception>
#include <iomanip>
#include <string>
#include <iostream>
#include <iterator>
#include <cassert>
#include <cmath>
#include <cfenv>
#include <iterator>

#include "isaac/driver/backend.h"
#include "isaac/driver/error.h"
#include "isaac/driver/module.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/stream.h"
#include "isaac/driver/buffer.h"

#include "isaac/templates/conv.h"
#include "isaac/templates/error.hpp"

#include "isaac/driver/cublas.h"

#include "isaac/tools/collections.hpp"
#include "isaac/api.h"

#include "test_utils.hpp"

namespace sc = isaac;
namespace drv = isaac::driver;

template<class DTYPE>
inline void upsample(std::vector<DTYPE> const & in, std::vector<DTYPE>& out,
                     size_t N, size_t C, size_t D, size_t H, size_t W, size_t upsample_d, size_t upsample_h, size_t upsample_w){
    for(size_t n = 0; n < N; ++n)
    for(size_t c = 0; c < C ; ++c)
    for(size_t d = 0; d < D; ++d)
    for(size_t h = 0; h < H; ++h)
    for(size_t w = 0; w < W; ++w){
    for(size_t ud = 0; ud < upsample_d; ud++)
    for(size_t uh = 0; uh < upsample_h; uh++)
    for(size_t uw = 0; uw < upsample_w; uw++)
      out[idx(n, c, d*upsample_d + ud, h*upsample_h + uh, w*upsample_w + uw, N, C, D*upsample_d, H*upsample_h, W*upsample_w)] = in[idx(n, c, d, h, w, N, C, D, H, W)];
    }
}



template<class IN_DTYPE, class OUT_DTYPE>
void cpp_conv_nchw(int32_t C, int32_t N, int32_t K,
              int32_t D, int32_t H, int32_t W,
              int32_t T, int32_t R, int32_t S,
              int32_t pad_d, int32_t pad_h, int32_t pad_w,
              int32_t stride_d, int32_t stride_h, int32_t stride_w,
              int32_t M, int32_t P, int32_t Q,
              std::vector<std::vector<OUT_DTYPE>>& O, IN_DTYPE* I, IN_DTYPE* F,
              float* bias,
              float iscale, float fscale, std::vector<float> const & oscale,
              OUT_DTYPE* Z, float zscale, int32_t Zk, int32_t crop_z_m0, int32_t crop_z_m1, int32_t crop_z_p0,int32_t crop_z_p1, int32_t crop_z_q0, int32_t crop_z_q1,
              sc::ResidualType residual_type)
{
  size_t num_outputs = O.size();
  static const int PACK_IN = pack_increment<IN_DTYPE>::VALUE;
  static const int PACK_OUT = pack_increment<OUT_DTYPE>::VALUE;
  if(C % PACK_IN != 0) throw std::runtime_error("Number of input channels must be a multiple of 4");
  if(K % PACK_OUT != 0) throw std::runtime_error("Number of output channels must be a multiple of 4");
  if(Zk % PACK_OUT != 0) throw std::runtime_error("Number of residual channels must be a multiple of 4");
  C /= PACK_IN;
  K /= PACK_OUT;
  Zk /= PACK_OUT;
  int32_t Kout = (residual_type == sc::CatResidual)? K + Zk : K;
  IN_DTYPE accs[PACK_OUT];
  float tmp[PACK_OUT];
  float tmp_z[PACK_OUT];
  int32_t Zm = M + crop_z_m0 + crop_z_m1;
  int32_t Zp = P + crop_z_p0 + crop_z_p1;
  int32_t Zq = Q + crop_z_q0 + crop_z_q1;
  for(size_t o = 0; o < num_outputs; o++)
  for(int32_t m = 0 ; m < M; ++m)
  for(int32_t p = 0 ; p < P; ++p)
  for(int32_t q = 0; q < Q; ++q)
  for(int32_t n = 0; n < N; ++n)
  for(int32_t k = 0; k < Kout ; ++k)
  {
    for(int32_t i = 0; i < PACK_OUT; ++i)
      accs[i] = 0;
    int32_t mm = m*stride_d - pad_d;
    int32_t pp = p*stride_h - pad_h;
    int32_t qq = q*stride_w - pad_w;
    for(int32_t kk = 0; kk < PACK_OUT; ++kk)
    for(int32_t c = 0; c < C; ++c)
    for(int32_t t = 0; t < T; ++t)
    for(int32_t r = 0; r < R; ++r)
    for(int32_t s = 0; s < S; ++s){
      int32_t d = mm + t;
      int32_t h = pp + r;
      int32_t w = qq + s;
      bool in_bounds = (d >= 0 && h >= 0 && w >= 0 && d < D && h < H && w < W);
      IN_DTYPE i = in_bounds?I[idx(n, c, d, h, w, N, C, D, H, W)]:0;
      IN_DTYPE f = F[idx(c, t, r, s, k*PACK_OUT + kk, C, T, R, S, K*PACK_OUT)];
      accs[kk] = dot(i, f, accs[kk]);
    }
    for(int32_t kk = 0; kk < PACK_OUT; ++kk){
      tmp[kk] = accs[kk];
      tmp[kk] /= (iscale*fscale);
      tmp[kk] += bias[k*PACK_OUT + kk];
    }
    OUT_DTYPE result = pack<OUT_DTYPE>(tmp, oscale[o]);;
    if(residual_type == sc::NoResidual)
      O[o][idx(n, k, m, p, q, N, K, M, P, Q)] = result;
    else{
      size_t zm = m + crop_z_m0;
      size_t zp = p + crop_z_p0;
      size_t zq = q + crop_z_q0;
      size_t idx_z = idx(n, k - K, zm, zp, zq, N, Zk, Zm, Zp, Zq);
      if(residual_type == sc::CatResidual){
        OUT_DTYPE result = pack<OUT_DTYPE>(tmp, oscale[o]);;
        OUT_DTYPE residual = pack<OUT_DTYPE>(unpack(tmp, Z[idx_z], zscale), oscale[o]);
        O[o][idx(n, k, m, p, q, N, Kout, M, P, Q)] = (k < K)?result:residual;
      }
      if(residual_type == sc::AddResidual){
        size_t idx_z = idx(n, k, zm, zp, zq, N, Zk, Zm, Zp, Zq);
        unpack(tmp_z, Z[idx_z], zscale);
        for(int32_t i = 0; i < PACK_OUT; ++i)
          tmp[i] += tmp_z[i];
        O[o][idx(n, k, m, p, q, N, K, M, P, Q)] = pack<OUT_DTYPE>(tmp, oscale[o]);
      }
    }

  }
}

template<class T>
bool abs_cmp(T a, T b)
{ return std::abs(a) < std::abs(b);}

template<class IN_DTYPE, class OUT_DTYPE>
void do_test_impl(sc::driver::Context const & ctx, size_t N, size_t K, size_t D, size_t H, size_t W, size_t C, size_t T, size_t R, size_t S,
                  size_t pad_d, size_t pad_h, size_t pad_w,
                  size_t stride_d, size_t stride_h, size_t stride_w,
                  size_t upsample_d, size_t upsample_h, size_t upsample_w,
                  bool has_bias, size_t num_outputs,
                  sc::ResidualType residual, size_t Zk, size_t crop_z_m0, size_t crop_z_m1, size_t crop_z_p0, size_t crop_z_p1, size_t crop_z_q0, size_t crop_z_q1)
{
  srand(0);
  sc::DType in_dtype = sc::to_DType<IN_DTYPE>::value;
  sc::DType out_dtype = sc::to_DType<OUT_DTYPE>::value;

  size_t in_dtsize = sc::size_of(in_dtype);
  size_t out_dtsize = sc::size_of(out_dtype);

  sc::ActivationType activation = sc::Linear;
  drv::Stream stream(ctx);

  // Shapes
  sc::param_t M, P, Q;
  sc::templates::Conv::output_shapes(D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, M, P, Q);
  sc::param_t Dup = D*upsample_d, Hup = H*upsample_h, Wup = W*upsample_w;
  sc::param_t Zm = M + crop_z_m0 + crop_z_m1;
  sc::param_t Zp = P + crop_z_p0 + crop_z_p1;
  sc::param_t Zq = Q + crop_z_q0 + crop_z_q1;
  sc::param_t Kout = (residual == sc::CatResidual)? K + Zk : K;

  // CPU buffers
  size_t PACK_IN = pack_increment<IN_DTYPE>::VALUE;
  size_t PACK_OUT = pack_increment<OUT_DTYPE>::VALUE;
  std::vector<IN_DTYPE> image_c(N*C*H*W*D/PACK_IN);
  std::vector<IN_DTYPE> upsampled_c(N*C*Hup*Wup*Dup/PACK_IN);
  std::vector<IN_DTYPE> filters_c(K*C*R*S*T/PACK_IN);
  std::vector<float> bias_c(K);
  std::vector<OUT_DTYPE> z_c(N*Zk*Zm*Zp*Zq/PACK_OUT);
  std::vector<std::vector<OUT_DTYPE>> ground_truth_c(num_outputs, std::vector<OUT_DTYPE>(N*Kout*M*P*Q/PACK_OUT));
  std::vector<std::vector<OUT_DTYPE>> output_isaac_c(ground_truth_c);

  // Initialize
  srand(0);
  for(size_t i = 0; i < z_c.size(); ++i)
    z_c[i] = (out_dtype==sc::INT8X4_TYPE)?rand():50*(float)rand()/RAND_MAX;
  for(size_t i = 0; i < image_c.size(); ++i)
    image_c[i] = (in_dtype==sc::INT8X4_TYPE)?rand():(float)rand()/RAND_MAX - 0.1;
  for(size_t i = 0; i < filters_c.size(); ++i)
    filters_c[i] = (in_dtype==sc::INT8X4_TYPE)?rand():(float)rand()/RAND_MAX - 0.1;
  for(size_t i = 0; i < bias_c.size(); ++i)
    bias_c[i] = has_bias?(float)rand()/RAND_MAX:0;

  // Scales
  float i_scale = (in_dtype==sc::INT8X4_TYPE)?((float)127 / *std::max_element(image_c.begin(), image_c.end(), abs_cmp<IN_DTYPE>)):1;
  float f_scale = (in_dtype==sc::INT8X4_TYPE)?((float)127 / *std::max_element(filters_c.begin(), filters_c.end(), abs_cmp<IN_DTYPE>)):1;
  float z_scale = (Zk > 0)?((float)127 / *std::max_element(z_c.begin(), z_c.end(), abs_cmp<IN_DTYPE>)):1;
  std::vector<float> o_scale;
  for(size_t n = 0; n < num_outputs; n++)
      o_scale.push_back((out_dtype==sc::INT8X4_TYPE)?(float)127/(170 + 100*n):1);

  // Ground truth
  upsample(image_c, upsampled_c, N, C/PACK_IN, D, H, W, upsample_d, upsample_h, upsample_w);
  cpp_conv_nchw(C, N, K, Dup, Hup, Wup, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, M, P, Q,
                ground_truth_c, upsampled_c.data(), filters_c.data(), bias_c.data(), i_scale, f_scale, o_scale,
                z_c.data(), z_scale, Zk, crop_z_m0, crop_z_m1, crop_z_p0, crop_z_p1, crop_z_q0, crop_z_q1,
                residual);

  // Isaac
  drv::Buffer image(ctx, image_c.size()*in_dtsize);
  drv::Buffer filters(ctx, filters_c.size()*in_dtsize);
  drv::Buffer z(ctx, std::max<int>(1, z_c.size()*out_dtsize));
  drv::Buffer bias(ctx, std::max<int>(1, bias_c.size()*out_dtsize));
  std::vector<drv::Buffer> output;
  for(size_t n = 0; n < num_outputs; n++)
      output.push_back(drv::Buffer(ctx, ground_truth_c[n].size()*out_dtsize));

  drv::Buffer* pz = Zk>0?&z:NULL;
  drv::Buffer* pbias = has_bias?&bias:NULL;
  stream.write(image, false, 0, image_c);
  stream.write(filters, false, 0, filters_c);
  stream.write(z, false, 0, z_c);
  stream.write(bias, false, 0, bias_c);
  sc::CONV(ctx.device(), stream, in_dtype, out_dtype, N, K, M, P, Q, C, T, R, S, D, H, W,
           pad_d, pad_h, pad_w,
           stride_d, stride_h, stride_w,
           upsample_d, upsample_h, upsample_w,
           image, filters, output.data(), num_outputs,
           pbias,
           activation, 0,
           i_scale, f_scale, o_scale, z_scale,
           residual, Zk, crop_z_m0, crop_z_m1, crop_z_p0, crop_z_p1, crop_z_q0, crop_z_q1, pz);

  // Check correctness
  for(size_t n = 0; n < num_outputs; n++){
      stream.read(output[n], true, 0, output_isaac_c[n]);
      if(!is_correct(output_isaac_c[n], ground_truth_c[n], 1e-3))
        exit(EXIT_FAILURE);
  }


  std::vector<int> rv = {1, 2, 4};
  std::vector<int> rl = {1, 8};
  std::vector<int> rs = {1, 4};
  std::vector<int> rgrid = {1, 8};
  std::vector<int> r1 = {1};
  for(auto x: sc::cpp::cartesian({rv, rl, rl, rs, rs, rl, r1, rgrid, rgrid})){
    isaac::templates::Conv conv(in_dtype, out_dtype, C, D, H, W, N, K, M, P, Q, T, R, S,
                                pad_d, pad_h, pad_w,
                                stride_d, stride_h, stride_w,
                                upsample_d, upsample_h, upsample_w,
                                activation, num_outputs,
                                residual, Zk, crop_z_m0, crop_z_m1, crop_z_p0, crop_z_p1, crop_z_q0, crop_z_q1,
                                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]);
    //Compile
    std::string src;
    try{
      src = conv.dump(ctx.device(), "fprop");
    }catch(isaac::templates::invalid_parameters){
      continue;
    }
    //Compile
    drv::Module program(ctx, src);
    drv::Kernel kernel(program, "fprop");
    //Launch
    try{
      conv.enqueue(kernel, stream, image, filters, output.data(), pbias, 0, i_scale, f_scale, o_scale, z_scale, pz);
    }catch(isaac::driver::exception::cuda::launch_out_of_resources){
      continue;
    }
    //Test
    for(size_t n = 0; n < num_outputs; n++){
        stream.read(output[n], true, 0, output_isaac_c[n]);
        if(!is_correct(output_isaac_c[n], ground_truth_c[n], 1e-3))
          exit(EXIT_FAILURE);
    }
  }
}

template<class IN_DTYPE, class OUT_DTYPE>
int do_test(sc::driver::Context const & ctx, std::string const & prefix, size_t N, size_t K, size_t D, size_t H, size_t W, size_t C, size_t T, size_t R, size_t S,
            size_t pad_d, size_t pad_h, size_t pad_w,
            size_t stride_d, size_t stride_h, size_t stride_w,
            size_t upsample_d, size_t upsample_h, size_t upsample_w,
            bool has_bias, size_t num_outputs,
            sc::ResidualType residual, size_t Zk, size_t crop_z_d0, size_t crop_z_d1, size_t crop_z_h0, size_t crop_z_h1, size_t crop_z_w0, size_t crop_z_w1)
{
  auto params = {N, K, D, H, W, C, T, R, S};
  std::cout << "(";
  std::copy(params.begin(), params.end(), std::ostream_iterator<size_t>(std::cout, ", "));
  std::cout << "\b\b) [" << prefix << "]" << std::endl;
  do_test_impl<IN_DTYPE, OUT_DTYPE>(ctx, N, K, D, H, W, C, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, has_bias, num_outputs, residual, Zk, crop_z_d0, crop_z_d1, crop_z_h0, crop_z_h1, crop_z_w0, crop_z_w1);
  return EXIT_SUCCESS;
}

int main(){
  auto ctx = drv::backend::contexts::get_default();
  std::cout << "===============" << std::endl;
  std::cout << "CONV:" << std::endl;
  std::cout << "===============" << std::endl;
  std::cout << "---------------" << std::endl;
  do_test<float, float>(ctx, "core", 5, 13, 19, 11, 15, 17, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, false, 1, sc::NoResidual, 0, 0, 0, 0, 0, 0, 0);
  do_test<float, int>(ctx, "core + quantize", 5, 16, 19, 11, 15, 17, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, false, 1, sc::NoResidual, 0, 0, 0, 0, 0, 0, 0);
  do_test<float, int>(ctx, "core + dual-quantize", 5, 16, 19, 11, 15, 17, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, false, 2, sc::NoResidual, 0, 0, 0, 0, 0, 0, 0);
  do_test<int, int>(ctx, "int8x4", 5, 16, 19, 11, 15, 20, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, false, 1, sc::NoResidual, 0, 0, 0, 0, 0, 0, 0);
  do_test<int, float>(ctx, "int8x4 + dequantize", 5, 13, 19, 11, 15, 20, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, false, 1, sc::NoResidual, 0, 0, 0, 0, 0, 0, 0);
  do_test<float, float>(ctx, "upsample", 5, 13, 19, 11, 15, 17, 3, 3, 3, 0, 0, 0, 1, 1, 1, 3, 2, 4, false, 1, sc::NoResidual, 0, 0, 0, 0, 0, 0, 0);
  do_test<float, float>(ctx, "residual-cat", 5, 13, 19, 11, 15, 17, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, false, 1, sc::CatResidual, 77, 1, 3, 5, 4, 2, 6);
  do_test<int, int>(ctx, "int8x4 + residual-cat", 5, 16, 19, 11, 15, 20, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, false, 1, sc::CatResidual, 60, 1, 3, 5, 4, 2, 6);
  do_test<float, float>(ctx, "residual-add", 5, 13, 19, 11, 15, 17, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, false, 1, sc::AddResidual, 77, 1, 3, 5, 4, 2, 6);
  do_test<int, int>(ctx, "int8x4 + residual-add", 5, 16, 19, 11, 15, 20, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, false, 1, sc::AddResidual, 60, 1, 3, 5, 4, 2, 6);
  do_test<float, float>(ctx, "pad", 5, 13, 19, 11, 15, 17, 3, 3, 3, 5, 1, 2, 1, 1, 1, 1, 1, 1, false, 1, sc::NoResidual, 0, 0, 0, 0, 0, 0, 0);
  do_test<float, float>(ctx, "stride", 5, 13, 19, 11, 15, 17, 3, 3, 3, 0, 0, 0, 6, 3, 4, 1, 1, 1, false, 1, sc::NoResidual, 0, 0, 0, 0, 0, 0, 0);
  do_test<float, float>(ctx, "pad + stride + bias", 5, 13, 19, 11, 15, 17, 3, 3, 3, 5, 1, 2, 6, 3, 4, 1, 1, 1, true, 1, sc::NoResidual, 0, 0, 0, 0, 0, 0, 0);
  do_test<float, float>(ctx, "vectorized + bias", 5, 13, 36, 11, 15, 17, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, true, 1, sc::NoResidual, 0, 0, 0, 0, 0, 0, 0);
  do_test<float, float>(ctx, "pad + stride + residual-cat + bias", 5, 13, 19, 11, 15, 17, 3, 3, 3, 5, 1, 2, 6, 3, 4, 1, 1, 1, true, 1, sc::CatResidual, 77, 1, 3, 5, 4, 2, 6);
  do_test<float, float>(ctx, "upsample + residual-cat + bias", 5, 13, 19, 11, 15, 17, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, true, 1, sc::CatResidual, 77, 1, 3, 5, 4, 2, 6);
  do_test<float, float>(ctx, "pad + stride + residual-cat + bias", 5, 13, 19, 11, 15, 17, 1, 1, 1, 5, 1, 2, 6, 3, 4, 1, 1, 1, true, 1, sc::CatResidual, 77, 1, 3, 5, 4, 2, 6);
  do_test<float, float>(ctx, "upsample + residual-cat + bias", 5, 13, 19, 11, 15, 17, 1, 1, 1, 0, 0, 0, 1, 1, 1, 3, 2, 4, true, 1, sc::CatResidual, 77, 1, 3, 5, 4, 2, 6);
  std::cout << "---------------" << std::endl;
}
