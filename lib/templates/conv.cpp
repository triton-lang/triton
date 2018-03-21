/* Copyright 2015-2017 Philippe Tillet
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <sstream>
#include <exception>
#include <bitset>

#include "isaac/templates/common.hpp"
#include "isaac/templates/conv.h"
#define FMT_HEADER_ONLY
#include "../external/fmt/format.h"

#include "isaac/driver/backend.h"
#include "isaac/driver/module.h"
#include "isaac/driver/error.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/stream.h"
#include "isaac/driver/buffer.h"

#include "isaac/templates/error.hpp"

namespace drv = isaac::driver;
using fmt::format;

namespace isaac{
namespace templates{

const std::string Conv::id = "conv";
const size_t Conv::Nshapes = 5;
const size_t Conv::Ntune = 9;
const size_t Conv::Nparams = Nshapes + Ntune;

Conv::Conv(DType in_dtype, DType out_dtype, param_t C, param_t D, param_t H, param_t W, param_t N, param_t K, param_t M, param_t P, param_t Q, param_t T, param_t R, param_t S,
           param_t pad_d, param_t pad_h, param_t pad_w, param_t stride_d, param_t stride_h, param_t stride_w, param_t upsample_d, param_t upsample_h, param_t upsample_w,
           ActivationType activation, size_t num_outputs,
           ResidualType residual_type, param_t Zk, param_t z_crop_m0, param_t z_crop_m1, param_t z_crop_p0, param_t z_crop_p1, param_t z_crop_q0, param_t z_crop_q1,
           param_t vec, param_t bc0, param_t bc1, param_t cs0, param_t cs1, param_t u, param_t, param_t bz, param_t):
  in_dtype_(in_dtype), out_dtype_(out_dtype), activation_(activation), num_outputs_(num_outputs),
  residual_type_(residual_type), Zk_(Zk), z_crop_m0_(z_crop_m0), z_crop_m1_(z_crop_m1), z_crop_p0_(z_crop_p0), z_crop_p1_(z_crop_p1), z_crop_q0_(z_crop_q0), z_crop_q1_(z_crop_q1),
  C_(C), N_(N), K_(K), D_(D), H_(H), W_(W), M_(M), P_(P), Q_(Q), T_(T), R_(R), S_(S),
  pad_d_(pad_d), pad_h_(pad_h), pad_w_(pad_w),
  stride_d_(stride_d), stride_h_(stride_h), stride_w_(stride_w),
  upsample_d_(upsample_d), upsample_h_(upsample_h), upsample_w_(upsample_w),
  vec_(vec), bc0_(bc0), bc1_(bc1), cs0_(cs0), cs1_(cs1), u_(u), us_(u), zs_(1), bz_(bz), gridz_(1)
{
    // Handle packed layouts
    size_t vect_c = (in_dtype_==INT8X4_TYPE)?4:1;
    if(C_ % vect_c != 0)
      throw std::runtime_error("Number of input channels must be a multiple of VECT_C");
    C_ /= vect_c;

    size_t vect_k = (out_dtype_==INT8X4_TYPE)?4:1;
    if(K_ % vect_k != 0)
      throw std::runtime_error("Number of output channels must be a multiple of VECT_C");
    if(Zk_ % vect_k != 0)
      throw std::runtime_error("Number of concatenation channels must be a multiple of VECT_C");
    Kout_ = K_ / vect_k;
    Zk_ /= vect_k;

    // Cropping shapes
    Zm_ = M_ + z_crop_m0_ + z_crop_m1_;
    Zp_ = P_ + z_crop_p0_ + z_crop_p1_;
    Zq_ = Q_ + z_crop_q0_ + z_crop_q1_;

    // Resize LUT
    size_t block = u_*zs_*bz_;
    size_t Nfilt = T_*R_*S_;
    size_t nlut = (block + Nfilt - 1)/Nfilt * Nfilt;

    // I strides
    int32_t strideIw = size_of(in_dtype_);
    int32_t strideIh = W_*strideIw;
    int32_t strideId = H_*strideIh;
    int32_t strideIc = D_*strideId;

    // Init constant memory
    cLUT.resize(nlut + upsample_d_*upsample_h_*upsample_w_*nlut);
    masks_.resize(nlut + (2*pad_h+1)*(2*pad_w+1)*(2*pad_d+1)*nlut);
    init_constant_memory(cLUT, masks_, nlut, strideIc, strideIw, strideIh, strideId);

    // Sanity check
    bool has_pad = pad_d > 0 || pad_h > 0 || pad_w > 0;
    bool has_upsample = upsample_d > 1 || upsample_h > 1 || upsample_w > 1;
    if(has_pad && has_upsample)
      throw std::runtime_error("Using both padding and upsampling is not supported at the moment");

}

void Conv::init_constant_memory(std::vector<int32_t> &delta, std::vector<uint32_t> &masks, size_t nlut, int32_t strideIc, int32_t strideIw, int32_t strideIh, int32_t strideId){
   size_t zl = zs_*bz_;
   size_t block = u_*zl;
   size_t Nfilt = T_*R_*S_;

   auto unpack = [&](int32_t trs){
       int32_t tr = trs / S_;
       int32_t s = trs - tr*S_;
       int32_t t = tr / R_;
       int32_t r = tr - t*R_;
       return std::make_tuple(t, r, s);
   };

   /* Increments */
   for(size_t i = 0; i < nlut; ++i)
       delta[i] = 4*(((i + block) % nlut) - i);

   /* Deltas */
   size_t Ds0 = nlut;
   size_t Ds1 = upsample_w_;
   size_t Ds2 = upsample_h_;
   size_t Ds3 = upsample_d_;
   for(size_t pd = 0; pd < Ds3; ++pd)
   for(size_t ph = 0; ph < Ds2; ++ph)
   for(size_t pw = 0; pw < Ds1; ++pw){

       int32_t* deltas_ptr = &delta[nlut + pw*Ds0 + ph*Ds0*Ds1 + pd*Ds0*Ds1*Ds2];

       // Cumulative increments
       for(size_t i = 0; i < Ds0; ++i){
           int32_t ctrs = i;
           int32_t c = ctrs / Nfilt;
           int32_t t, r, s;
           std::tie(t, r, s) = unpack(ctrs % Nfilt);

           int32_t nextctrs = ctrs + block;
           int32_t nextc = nextctrs / Nfilt;
           int32_t nextt, nextr, nexts;
           std::tie(nextt, nextr, nexts) = unpack(nextctrs % Nfilt);

           int32_t cdiff = nextc - c;
           int32_t tdiff = (nextt + pd)/upsample_d_ - (t + pd)/upsample_d_;
           int32_t rdiff = (nextr + ph)/upsample_h_ - (r + ph)/upsample_h_;
           int32_t sdiff = (nexts + pw)/upsample_w_ - (s + pw)/upsample_w_;

           deltas_ptr[i] = cdiff*strideIc + sdiff*strideIw + rdiff*strideIh + tdiff*strideId;
       }
   }

   /* Masks */
   size_t Ms0 = nlut;
   size_t Ms1 = 2*pad_w_ + 1;
   size_t Ms2 = 2*pad_h_ + 1;
   size_t Ms3 = 2*pad_d_ + 1;

   for(size_t pd = 0; pd < Ms3; ++pd)
   for(size_t ph = 0; ph < Ms2; ++ph)
   for(size_t pw = 0; pw < Ms1; ++pw){
     uint32_t* masks_ptr = &masks[nlut + pw*Ms0 + ph*Ms0*Ms1 + pd*Ms0*Ms1*Ms2];
     for(size_t i = 0; i < Ms0; ++i){
        int32_t t, r, s;
        uint32_t mask = 0x0;
        for(size_t j = 0; j < block; ++j){
          std::tie(t, r, s) = unpack((i + j) % Nfilt);
          bool in_bounds_d = (t + pd) >= pad_d_ && (t + pd) < (T_ + pad_d_);
          bool in_bounds_h = (r + ph) >= pad_h_ && (r + ph) < (R_ + pad_h_);
          bool in_bounds_w = (s + pw) >= pad_w_ && (s + pw) < (S_ + pad_w_);
          mask |= (in_bounds_d && in_bounds_h && in_bounds_w) << j;
        }
        masks_ptr[i] = mask;
     }
   }
   for(size_t i = 0; i < nlut; ++i)
     masks[i] = 0x0;
}

void Conv::output_shapes(param_t D, param_t H, param_t W, param_t T, param_t R, param_t S,
                         param_t pad_d, param_t pad_h, param_t pad_w, param_t stride_d, param_t stride_h, param_t stride_w, param_t upsample_d, param_t upsample_h, param_t upsample_w,
                         param_t &M, param_t &P, param_t &Q)
{
    M = (D*upsample_d - T + 1 + 2*pad_d + stride_d - 1)/stride_d;
    P = (H*upsample_h - R + 1 + 2*pad_h + stride_h - 1)/stride_h;
    Q = (W*upsample_w - S + 1 + 2*pad_w + stride_w - 1)/stride_w;
}


std::vector<param_t> Conv::tuning_params() const
{ return {bc0_, bc1_, cs0_, cs1_, u_, us_, zs_, bz_, gridz_}; }


double Conv::tflops(param_t P, param_t Q, param_t M, param_t K, param_t N, param_t C, param_t R, param_t S, param_t T, double time)
{ return (double)2*M*P*Q*K*N*C*R*S*T/(time*1e3); }

void Conv::check_valid(driver::Device const & device, size_t M, param_t* params, uint8_t* valid){
  std::array<int, Nparams> x{0};
  for(size_t m = 0; m < M; ++ m){

    //Parameters
    for(size_t i = 0; i < x.size(); ++i)
      x[i] = params[m*x.size() + i];
    DType dtype = (DType)(x[0]);
    param_t vec = x[5], bc0 = x[6], bc1 = x[7], cs0 = x[8], cs1 = x[9], u = x[10], zs = x[11], bz = x[12], gridz = x[13];

    //Features
    param_t dtsize = size_of(dtype);

    param_t cl0 = bc0*cs0;
    param_t cl1 = cs1*bc1;
    param_t zl = zs*bz;

    param_t nthreads = bc0*bc1*bz;

    param_t cd_sharedi = dtsize*cl0;
    param_t cd_sharedf = dtsize*cl1;

    param_t block = u*zl;
    param_t size_sharedi = cd_sharedi*block;
    param_t size_sharedf = cd_sharedf*block;
    param_t param_tiles = next_pow2(2*(size_sharedi + size_sharedf));
    param_t size_redc = dtsize*cl0*cl1*(bz==1?0:bz);
    param_t size_pack = dtsize*bc0*cl1*(dtype==INT8X4_TYPE?1:0);
    param_t size_shmem = std::max(std::max(size_pack, size_redc), param_tiles);

    param_t bf_ctrs = block;
    param_t bf_k = nthreads/bf_ctrs;
    param_t bf_pqn = bf_k;

    auto n_instructions = [&]() { return cs0*cs1*u; };

    //Test
    bool is_valid = (bf_pqn*bf_ctrs == nthreads)
                  && (bf_k*bf_ctrs == nthreads)
                  && (block % bf_ctrs == 0)
                  && (cl0 % (bf_pqn*vec) == 0)
                  && (cl1 % (bf_k*vec) == 0)
                  && (nthreads % 32 == 0)
                  && (cs0 % (vec) == 0)
                  && (cs1 % (vec) == 0)
                  && (size_shmem <= device.max_shared_memory())
                  && (bc0 <= device.max_block_dim()[0])
                  && (bc1 <= device.max_block_dim()[1])
                  && (bz <= device.max_block_dim()[2])
                  && (nthreads <= device.max_threads_per_block())
                  && (n_instructions() <= 1024)
                  && (vec*dtsize <= 16)
                  && (block <= 32)
                  && (dtype!=INT8X4_TYPE || gridz==1)
                  && (dtype!=INT8X4_TYPE || cs1 % 4 == 0);
    valid[m] = is_valid;
  }
}


/* Code generation */
std::string Conv::dump(drv::Device const & device, std::string const & name){

  std::stringstream iss;
  size_t cl0 = bc0_*cs0_;
  size_t cl1 = cs1_*bc1_;
  size_t zl = zs_*bz_;

  // data types
  size_t in_dtsize = size_of(in_dtype_);
  std::string in_compute_type = arith_str(in_dtype_);
  std::string in_word_type = format("b{}", 8*in_dtsize);

  size_t out_dtsize = size_of(out_dtype_);
  std::string out_compute_type = arith_str(out_dtype_);
  std::string out_word_type = format("b{}", 8*out_dtsize);
  size_t vect_k = (out_dtype_==INT8X4_TYPE)?4:1;

  // helpers
  size_t nthreads = bc0_*bc1_*bz_;
  size_t Nfilt = R_*S_*T_;
  size_t cd_sharedi = in_dtsize*cl0;
  size_t cd_sharedf = in_dtsize*cl1;

  size_t block = u_*zl;
  size_t nlut = (block + Nfilt - 1)/Nfilt * Nfilt;
  size_t size_sharedi = cd_sharedi*block;
  size_t size_sharedf = cd_sharedf*block;
  size_t size_tiles = next_pow2(2*(size_sharedi + size_sharedf));
  size_t size_redc = in_dtsize*cl0*cl1*(bz_==1?0:bz_);
  size_t size_pack = in_dtsize*bc0_*cl1*(out_dtype_==INT8X4_TYPE?1:0);
  size_t size_shmem = std::max(std::max(size_pack, size_redc), size_tiles);
  size_t Bvec = vec_*in_dtsize;
  size_t addr_i = 0;
  size_t addr_f = size_sharedi;
  size_t double_buffer_off = size_tiles/2;
  // params
  size_t bf_ctrs = block;
  size_t bf_k = nthreads/bf_ctrs;
  size_t bf_pqn = bf_k;

  uint8_t is_valid;
  uint32_t params[] = {out_dtype_, N_*P_*Q_*M_, K_, C_, R_*S_*T_, vec_, bc0_, bc1_, cs0_, cs1_, u_, zs_, bz_, gridz_};
  check_valid(device, 1, params, &is_valid);
  if(!is_valid)
    throw invalid_parameters();

  std::string vv = vec_>1?format(".v{}", vec_):"";
  const char* vs[] = {".x", ".y", ".z", ".w"};
  if(vec_==1)
    vs[0] = "";

  auto declare_register_tile = [&](char x, size_t M, size_t N, size_t dtinc){
    for(size_t c = 0; c < zs_; c++){
      for(size_t m = 0 ; m < M ; m+=vec_*dtinc){
        iss << format("  .reg {}.{}", vv, in_word_type);
        for(size_t n = 0 ; n < N ; n++)
          iss << format("{} %r{}{}_{}_{}", n>0?",":"", x, c, m, n);
        iss << ";" << std::endl;
      }
    }
  };

  auto ptr_ldg_i = [&](){
      iss << format("  //  Helping quantities for axes offsets") << std::endl;
      iss << format("  mul.lo.s32 %mM, %M, -1;") << std::endl;
      iss << format("  mul.lo.s32 %mN, %N, -1;") << std::endl;
      iss << format("  mul.lo.s32 %mP, %P, -1;") << std::endl;
      iss << format("  mul.lo.s32 %mQ, %Q, -1;") << std::endl;
      iss << format("  sub .s32 %Dmpad, {}, %D;", T_) << std::endl;
      iss << format("  sub .s32 %Hmpad, {}, %H;", R_) << std::endl;
      iss << format("  sub .s32 %Wmpad, {}, %W;", S_) << std::endl;
      iss << format("  mul.lo.u32 %offIndhw, %bid0, {};", cl0) << std::endl;
      iss << format("  mad.lo.u32 %offIndhw, %afid0, {}, %offIndhw;", vec_) << std::endl;

      iss << std::endl;
      iss << format("  // Masks and delta pointers") << std::endl;
      iss << format("  mov.u32 %p_inc_mask, _LUT;") << std::endl;
      iss << "  mov.b32 %p_inc_delta, _LUT;" << std::endl;
      iss << format("  mad.lo.u32 %p_inc_delta, %idctrs, 4, %p_inc_delta;") << std::endl;

      iss << std::endl;
      iss << format("  // Unpack IDCTRS") << std::endl;
      iss << format("  div.s32 %c, %idctrs, {};", Nfilt) << std::endl;
      iss << format("  rem.s32 %trs, %idctrs, {};", Nfilt) << std::endl;
      iss << format("  div.s32 %tr, %trs, {};", S_) << std::endl;
      iss << format("  rem.s32 %s, %trs, {};", S_) << std::endl;
      iss << format("  div.s32 %t, %tr, {};", R_) << std::endl;
      iss << format("  rem.s32 %r, %tr, {};", R_) << std::endl;

      iss << std::endl;
      iss << format("  // Initial offset") << std::endl;
      iss << format("  mad.lo.s32 %offi, %c, %strideIc, 0;") << std::endl;
      if(gridz_ > 1)
        iss << format("  mad.lo.s32 %offi, %offc, %strideIc, %offi;") << std::endl;

      for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
      for(size_t s = 0; s < vec_; s++){
        iss << std::endl;
        iss << format("  // Unpack and compute offsets [{0}]", i + s) << std::endl;
        iss << format("  add.u32 %offIndhw{0}, %offIndhw, {0};", i + s) << std::endl;
        iss << format("  div.u32 %offIndh{0}, %offIndhw{0}, %Q;", i + s) << std::endl;
        iss << format("  mad.lo.s32 %offIw{0}, %mQ, %offIndh{0}, %offIndhw{0};", i + s) << std::endl;
        iss << format("  div.u32 %offInd{0}, %offIndh{0}, %P;", i + s, Q_) << std::endl;
        iss << format("  mad.lo.s32 %offIh{0}, %mP, %offInd{0}, %offIndh{0};", i + s) << std::endl;
        iss << format("  div.u32 %offIn{0}, %offInd{0}, %M;", i + s, P_) << std::endl;
        iss << format("  mad.lo.s32 %offId{0}, %mM, %offIn{0}, %offInd{0};", i + s) << std::endl;

        iss << std::endl;
        iss << format("  // Predicate [{0}]", i + s) << std::endl;
        iss << format("   setp.lt.u32 %predi{}, %offIndhw{}, %Npix;", i + s, i + s) << std::endl;

        iss << std::endl;
        iss << format("  // Padding and striding [{0}]", i + s) << std::endl;
        iss << format("  mul.lo.s32 %offId{0}, %offId{0}, %stride_d;", i + s) << std::endl;
        iss << format("  mul.lo.s32 %offIh{0}, %offIh{0}, %stride_h;", i + s) << std::endl;
        iss << format("  mul.lo.s32 %offIw{0}, %offIw{0}, %stride_w;", i + s) << std::endl;
        iss << format("  sub.s32 %offId{0}, %offId{0}, %pad_d;", i + s) << std::endl;
        iss << format("  sub.s32 %offIh{0}, %offIh{0}, %pad_h;", i + s) << std::endl;
        iss << format("  sub.s32 %offIw{0}, %offIw{0}, %pad_w;", i + s) << std::endl;

        iss << std::endl;
        iss << format("  // Masks [{0}]", i + s) << std::endl;
        iss << format("  min.s32 %dlo{0}, %offId{0}, 0;", i + s) << std::endl;
        iss << format("  min.s32 %hlo{0}, %offIh{0}, 0;", i + s) << std::endl;
        iss << format("  min.s32 %wlo{0}, %offIw{0}, 0;", i + s) << std::endl;
        iss << format("  add.s32 %dhi{0}, %offId{0}, %Dmpad;", i + s) << std::endl;
        iss << format("  add.s32 %hhi{0}, %offIh{0}, %Hmpad;", i + s) << std::endl;
        iss << format("  add.s32 %whi{0}, %offIw{0}, %Wmpad;", i + s) << std::endl;
        iss << format("  max.s32 %dhi{0}, %dhi{0}, 0;", i + s) << std::endl;
        iss << format("  max.s32 %hhi{0}, %hhi{0}, 0;", i + s) << std::endl;
        iss << format("  max.s32 %whi{0}, %whi{0}, 0;", i + s) << std::endl;
        iss << format("  add.s32 %maskd{0}, %pad_d, %dlo{0};", i + s) << std::endl;
        iss << format("  add.s32 %maskd{0}, %maskd{0}, %dhi{0};", i + s) << std::endl;
        iss << format("  add.s32 %maskh{0}, %pad_h, %hlo{0};", i + s) << std::endl;
        iss << format("  add.s32 %maskh{0}, %maskh{0}, %hhi{0};", i + s) << std::endl;
        iss << format("  add.s32 %maskw{0}, %pad_w, %wlo{0};", i + s) << std::endl;
        iss << format("  add.s32 %maskw{0}, %maskw{0}, %whi{0};", i + s) << std::endl;
        iss << "  mov.b32 %masks, _masks;" << std::endl;
        iss << format("  @!%predi{0} mov.s32 %p_mask{0}, %masks;", i + s) << std::endl;
        iss << format("  @%predi{0} add.s32 %p_mask{0}, {1}, %masks;", i + s, 4*nlut) << std::endl;
        iss << format("  @%predi{0} mad.lo.s32 %p_mask{0}, %maskd{0}, {1}, %p_mask{0};", i + s, 4*nlut*(2*pad_w_ + 1)*(2*pad_h_ + 1)) << std::endl;
        iss << format("  @%predi{0} mad.lo.s32 %p_mask{0}, %maskh{0}, {1}, %p_mask{0};", i + s, 4*nlut*(2*pad_w_ + 1)) << std::endl;
        iss << format("  @%predi{0} mad.lo.s32 %p_mask{0}, %maskw{0}, {1}, %p_mask{0};", i + s, 4*nlut) << std::endl;

        iss << std::endl;
        iss << format("  // Upsampling [{0}]", i + s) << std::endl;
        iss << format("  rem.s32 %offDeltad{0}, %offId{0}, {1};", i + s, upsample_d_) << std::endl;
        iss << format("  rem.s32 %offDeltah{0}, %offIh{0}, {1};", i + s, upsample_h_) << std::endl;
        iss << format("  rem.s32 %offDeltaw{0}, %offIw{0}, {1};", i + s, upsample_w_) << std::endl;

        iss << std::endl;
        iss << format("  // Deltas [{0}]", i + s) << std::endl;
        iss << format("  add.s32 %p_delta{0}, %p_inc_delta, {1};", i + s, 4*nlut) << std::endl;
        iss << format("  mad.lo.s32 %p_delta{0}, %offDeltad{0}, {1}, %p_delta{0};", i + s, 4*nlut*upsample_w_*upsample_h_) << std::endl;
        iss << format("  mad.lo.s32 %p_delta{0}, %offDeltah{0}, {1}, %p_delta{0};", i + s, 4*nlut*upsample_w_) << std::endl;
        iss << format("  mad.lo.s32 %p_delta{0}, %offDeltaw{0}, {1}, %p_delta{0};", i + s, 4*nlut) << std::endl;

        iss << std::endl;
        iss << format("  // Initial pointers [{0}]", i + s) << std::endl;
        iss << format("  add.s32 %offId{0}, %offId{0}, %t;", i + s) << std::endl;
        iss << format("  add.s32 %offIh{0}, %offIh{0}, %r;", i + s) << std::endl;
        iss << format("  add.s32 %offIw{0}, %offIw{0}, %s;", i + s) << std::endl;
        iss << format("  div.s32 %offId{0}, %offId{0}, {1};", i + s, upsample_d_) << std::endl;
        iss << format("  div.s32 %offIh{0}, %offIh{0}, {1};", i + s, upsample_h_) << std::endl;
        iss << format("  div.s32 %offIw{0}, %offIw{0}, {1};", i + s, upsample_w_) << std::endl;
        iss << format("  mad.lo.s32 %offi{0}, %offId{0}, %strideId, %offi;", i + s) << std::endl;
        iss << format("  mad.lo.s32 %offi{0}, %offIh{0}, %strideIh, %offi{0};", i + s) << std::endl;
        iss << format("  mad.lo.s32 %offi{0}, %offIw{0}, %strideIw, %offi{0};", i + s) << std::endl;
        iss << format("  mad.lo.s32 %offi{0}, %offIn{0}, %strideIn, %offi{0};", i + s) << std::endl;
        iss << format("  mad.wide.s32 %pi{0}, %offi{0}, 1, %pi;", i + s) << std::endl;
      }
  };

  auto ptr_ldg_f = [&](){
      iss << std::endl;
      iss << format("  // F offsets", cl1, vec_) << std::endl;
      iss << format("  mul.lo.u32 %offFk, %bid1, {};", cl1) << std::endl;
      iss << format("  mad.lo.u32 %offFk, %bfid1, {}, %offFk;", vec_) << std::endl;

      iss << std::endl;
      iss << format("  // F pointers") << std::endl;
      iss << format("  mad.lo.u32 %offF, %offFk, %strideFk, 0;") << std::endl;
      iss << format("  mad.lo.u32 %offF, %idctrs, %strideFs, %offF;") << std::endl;
      if(gridz_ > 1)
        iss << format("  mad.lo.u32 %offF, %offc, %strideFc, %offF;") << std::endl;
      iss << format("  mad.wide.u32 %pf, %offF, 1, %pf;") << std::endl;
      iss << format("  mul.lo.s32 %incf, {}, %strideFs;", block) << std::endl;
  };

  auto ldg_i = [&](){
    iss << std::endl;
    iss << format("  // Load I") << std::endl;
    for(size_t i = 0; i < cl0; i+=vec_*bf_pqn){
      for(size_t s = 0; s < vec_; ++s){
        iss << format("  mov.{1} %rri{2}{3}, 0x0;", i + s, in_word_type, i, vs[s]) << std::endl;
        iss << format("  @%predi{0} ld.global.nc.{1} %rri{2}{3}, [%pi{4}];", i + s, in_word_type, i, vs[s], i + s) << std::endl;
      }
      iss << format("  st.shared{0}.{1} [%writei + {2}], %rri{3};", vv, in_word_type, i*in_dtsize, i) << std::endl;
    }
  };

  auto ldg_f = [&](bool safe){
    std::vector<std::string> preds(vec_, "%predloop");
    if(safe) for(size_t s = 0; s < vec_ ; ++s)
      preds[s] = format("%pred{}", s);
    iss << std::endl;
    iss << format("  // Load F") << std::endl;
    for(size_t j = 0; j < cl1; j+=vec_*bf_k){
      for(size_t s = 0; s < (safe?vec_:0); ++s){
        iss << format("  setp.lt.and.s32 %pred{}, %offFk{}, %Km{}, %predcrs;", s, j, s) << std::endl;
        iss << format("  mov.{1} %rrf{2}{3}, 0;", preds[s], in_word_type, j, vs[s]) << std::endl;
      }
      for(size_t s = 0; s < vec_; ++s)
        iss << format("  @{0} ld.global.nc.{1} %rrf{2}{3}, [%pf + {4}];", preds[s], in_word_type, j, vs[s], (j+s)*in_dtsize) << std::endl;
      iss << format("  st.shared{1}.{2} [%writef + {4}], %rrf{3};", preds[0], vv, in_word_type, j, j*in_dtsize) << std::endl;
    }
  };

  auto ptr_sts = [&](char x, int cd_shared, int off, std::string const & fid0, std::string const & fid1){
    iss << format("  // write{0} = shared + {1} + fid{2}*{3} + fid{4}*{5}", x, off, fid0, Bvec, fid1, cd_shared) << std::endl;
    iss << format("  add.u32 %write{0}, %shared, {1};", x, off) << std::endl;
    iss << format("  mad.lo.u32  %write{0}, %{1}, {2}, %write{0};", x, fid0, Bvec) << std::endl;
    iss << format("  mad.lo.u32  %write{0}, %{1}, {2}, %write{0};", x, fid1, cd_shared) << std::endl;
  };

  auto ptr_lds = [&](char x, std::string const & id0, int off, int cd_shared){
    iss << format("  // read{0} = shared + {1} + {2}*{3}", x, off, id0, Bvec) << std::endl;
    iss << format("  add.u32 %read{0}, %shared, {1};", x, off) << std::endl;
    iss << format("  mad.lo.u32  %read{0}, %{1}, {2}, %read{0};", x, id0, Bvec) << std::endl;
    iss << format("  mad.lo.u32  %read{0}, %idz, {1}, %read{0};", x, cd_shared) << std::endl;
  };

  auto lds = [&](char x, size_t nx, size_t ctrs, size_t cd_shared, size_t bs){
    for(size_t c = 0; c < zs_; c++)
    for(size_t rx = 0; rx < nx; rx+=vec_)
      iss << format("  ld.shared{0}.{1} %r{2}{3}_{4}_{5}, [%read{2} + {6}];", vv, in_word_type, x, c, rx, ctrs%us_, rx*bs*in_dtsize + (c*bz_ + ctrs*zl)*cd_shared) << std::endl;
  };



  auto fma = [&](int cc){
    for(size_t c = 0; c < zs_; c++)
    for(size_t i = 0; i < cs0_; i+=vec_)
    for(size_t k = 0; k < cs1_; k+=vec_)
    for(size_t kk = 0 ; kk < vec_ ; ++kk)
    for(size_t ii = 0 ; ii < vec_ ; ++ii){
      std::string ro = format("%rc{}_{}_{}{}", c, i, k + kk, vs[ii]);
      std::string ri = format("%ri{}_{}_{}{}", c, i, cc, vs[ii]);
      std::string rf = format("%rf{}_{}_{}{}", c, k, cc, vs[kk]);
      if(in_dtype_==INT8X4_TYPE)
        iss << format("  dp4a.{0}.{0} {1}, {2}, {3}, {1};", in_compute_type, ro, ri, rf) << std::endl;
      else
        iss << format("  fma.rn.{0} {1}, {2}, {3}, {1};", in_compute_type, ro, ri, rf) << std::endl;

    }
  };

  bool c_aligned = (M_*P_*Q_) % vec_ == 0 && gridz_==1;
  size_t c_inc = c_aligned?vec_:1;

  bool z_aligned = false;
  size_t z_inc = z_aligned?vec_:1;

  iss << std::endl;
  iss << "/* Constant memory */" << std::endl;
  iss << ".const .b32 _masks[" << masks_.size() << "];" << std::endl;
  iss << ".const .b32 _LUT[" << cLUT.size() << "];" << std::endl;

  /* -------------------------------------------- */
  /*         GENERATE STORE_COL FUNCTION          */
  /* -------------------------------------------- */

  if(activation_ == Sigmoid || activation_ == ELU){
    iss << ".func (.reg .f32 %retval) exp(.param .f32 _x){" << std::endl;
    iss << "  .reg .f32 %f<15>;" << std::endl;
    iss << "  .reg .pred %p1, %p2;" << std::endl;
    iss << "  ld.param.f32 %f3, [_x];" << std::endl;
    iss << "  mul.f32 	%f4, %f3, 0f3FB8AA3B;" << std::endl;
    iss << "  cvt.rzi.f32.f32	%f5, %f4;" << std::endl;
    iss << "  mov.f32 	%f6, 0fBF317200;" << std::endl;
    iss << "  fma.rn.f32 	%f7, %f5, %f6, %f3;" << std::endl;
    iss << "  mov.f32 	%f8, 0fB5BFBE8E;" << std::endl;
    iss << "  fma.rn.f32 	%f9, %f5, %f8, %f7;" << std::endl;
    iss << "  mul.f32 	%f2, %f9, 0f3FB8AA3B;" << std::endl;
    iss << "  // inline asm" << std::endl;
    iss << "  ex2.approx.ftz.f32 %f1,%f2;" << std::endl;
    iss << "  // inline asm" << std::endl;
    iss << "  add.f32 	%f10, %f5, 0f00000000;" << std::endl;
    iss << "  ex2.approx.f32 	%f11, %f10;" << std::endl;
    iss << "  mul.f32 	%f12, %f1, %f11;" << std::endl;
    iss << "  setp.lt.f32	%p1, %f3, 0fC2D20000;" << std::endl;
    iss << "  selp.f32	%f13, 0f00000000, %f12, %p1;" << std::endl;
    iss << "  setp.gt.f32	%p2, %f3, 0f42D20000;" << std::endl;
    iss << "  selp.f32	%retval, 0f7F800000, %f13, %p2;" << std::endl;
    iss << "  ret;" << std::endl;
    iss << "}" << std::endl;
  }


  auto make_store_col = [&](std::string const & name, ResidualType residual_type){
      iss << ".func " << name << "(.reg .b64 %po, .reg .b32 %Cs0";
      for(size_t i = 0; i < cs0_; i++)
        iss << format(", .reg .b32 %offc{}", i);
      for(size_t i = 0; i < cs0_ - c_inc; i+=c_inc)
        iss << format(", .reg .b32 %diffc{}", i);
      if(residual_type != CatResidual){
        for(size_t j = 0; j < vect_k ; j++)
        for(size_t i = 0 ; i < cs0_ ; i+=vec_)
          iss << format(", .reg {}.{} %rc{}_{}", vv, in_word_type, i, j);
      }
      iss << ",  .reg .b32 %o_scale";
      if(residual_type != NoResidual){
        iss << ",  .reg .b64 %pz";
        iss << ",  .reg .b32 %z_scale";
        for(size_t i = 0; i < cs0_ - z_inc; i+=z_inc)
          iss << format(", .reg .b32 %diffz{}", i);
      }
      if(residual_type == AddResidual){
        iss << ",  .reg .b32 %alpha";
      }
      iss << "){" << std::endl;

      iss << format("  .reg .pred %predc<{}>;", cs0_) << std::endl;

      if(residual_type == CatResidual){
          for(size_t j = 0; j < vect_k ; j++)
          for(size_t i = 0 ; i < cs0_ ; i+=vec_)
            iss << format(".reg {}.{} %rc{}_{};", vv, in_word_type, i, j) << std::endl;
      }
      if(residual_type != NoResidual){
          for(size_t i = 0 ; i < cs0_ ; i+=vec_){
            iss << format("  .reg {}.{} %rz_{};", vv, in_word_type, i) << std::endl;
            iss << format("  .reg {0}.{1} %rz{2}_0, %rz{2}_1, %rz{2}_2, %rz{2}_3;", vv, in_word_type, i) << std::endl;
          }
      }

      iss << std::endl;
      iss << "  /* Predicates */" << std::endl;
      for(size_t i = 0 ; i < cs0_ ; i+=std::min(c_inc, z_inc))
        iss << format("  setp.lt.s32 %predc{0}, %offc{0}, %Cs0;", i) << std::endl;

      if(residual_type != NoResidual){
          iss << std::endl;
          iss << "  /* Handle residual */" << std::endl;
          if(residual_type == AddResidual){
            iss << "  .reg .u32 %idz, %bidz;" << std::endl;
            iss << "  .reg .pred %predgz;" << std::endl;
            iss << "  mov.u32 %idz, %tid.z;" << std::endl;
            iss << "  mov.u32 %bidz, %ctaid.z;" << std::endl;
            iss << format("  setp.eq.s32 %predgz, %idz, 0;") << std::endl;
            iss << format("  setp.eq.and.s32 %predgz, %bidz, 0, %predgz;") << std::endl;
            iss << format("  @!%predgz bra RESIDUAL_DONE;") << std::endl;
          }

          for(size_t i = 0 ; i < cs0_ ; i+=vec_)
          for(size_t s = 0; s < vec_; s+=z_inc){
            iss << format("  @%predc{} ld.global{}.{} %rz{}_0{}, [%pz];", i + s, z_aligned?vv:"", in_word_type, i, z_aligned?"":vs[s]) << std::endl;
            if(i + s < cs0_ - z_inc)
              iss << format("  mad.wide.s32 %pz, %diffz{0}, 1, %pz;", i + s) << std::endl;
          }

          if(out_dtype_==INT8X4_TYPE)
            for(size_t i = 0; i < cs0_; i+=vec_)
            for(size_t s = 0; s < vec_; ++s){
              iss << format("  mov.b32 %rz_{0}{1}, %rz{0}_0{1};", i, vs[s]) << std::endl;
              for(size_t jj = 0; jj < vect_k; ++jj){
                iss << format("  shr.b32 %rz{0}_{1}{2}, %rz_{0}{2}, {3};", i, jj, vs[s], 8*jj) << std::endl;
                iss << format("  and.b32 %rz{0}_{1}{2}, %rz{0}_{1}{2}, 0xff;", i, jj, vs[s]) << std::endl;
                iss << format("  cvt.rn.f32.s8 %rz{0}_{1}{2}, %rz{0}_{1}{2};", i, jj, vs[s]) << std::endl;
              }
            }

          for(size_t i = 0 ; i < cs0_ ; i+=vec_)
          for(size_t s = 0; s < vec_; s+=z_inc)
          for(size_t jj = 0; jj < vect_k; ++jj){
            iss << format("  div.approx.f32 %rz{0}_{1}{2}, %rz{0}_{1}{2}, %z_scale;", i, jj, vs[s]) << std::endl;
            if(residual_type_ == CatResidual)
              iss << format("  mov.f32 %rc{0}_{1}{2}, %rz{0}_{1}{2};", i, jj, vs[s]) << std::endl;
            if(residual_type_ == AddResidual)
              iss << format("  add.f32 %rc{0}_{1}{2}, %rz{0}_{1}{2}, %rc{0}_{1}{2};", i, jj, vs[s]) << std::endl;
          }

          if(residual_type_ == AddResidual){
            if(activation_ == ReLU || activation_ == ELU){
              iss << std::endl;
              iss << "  /* ---------------------------- */" << std::endl;
              iss << "  /* --------- Rectifier -------- */" << std::endl;
              iss << "  /* ---------------------------- */" << std::endl;
              iss << "  .reg .b32 %leakage, %arg;" << std::endl;
              for(size_t j = 0; j < vect_k ; j++)
              for(size_t i = 0 ; i < cs0_ ; i+=vec_)
              for(size_t s = 0; s < vec_; ++s){
                if(activation_ == ReLU)
                  iss << format("  mul.f32 %leakage, %rc{0}_{1}{2}, %alpha;", i, j, vs[s]) << std::endl;
                else{
                    iss << format("  mov.f32 %arg, %rc{0}_{1}{2};", i, j, vs[s]) << std::endl;
                    iss << format("  call (%leakage), exp, (%arg);") << std::endl;
                    iss << format("  sub.f32 %leakage, %leakage, 1.;") << std::endl;
                    iss << format("  mul.f32 %leakage, %leakage, %alpha;", i, j, vs[s]) << std::endl;
                }
                iss << format("  slct.f32.f32 %rc{0}_{1}{2}, %rc{0}_{1}{2}, %leakage, %rc{0}_{1}{2};", i, j, vs[s]) << std::endl;
              }
            }
          }
          iss << "RESIDUAL_DONE:" << std::endl;
      }

      iss << std::endl;
      iss << "  /* Rescaling */" << std::endl;
      for(size_t j = 0; j < vect_k ; j++)
      for(size_t i = 0 ; i < cs0_ ; i+=vec_)
      for(size_t s = 0; s < vec_; ++s)
        iss << format("  mul.f32 %rc{0}_{1}{2}, %rc{0}_{1}{2}, %o_scale;", i, j, vs[s]) << std::endl;

      if(out_dtype_==INT8X4_TYPE){
        iss << std::endl;
        iss << "  /* Handle INT8x4 */" << std::endl;
        for(size_t j = 0; j < vect_k ; j++)
        for(size_t i = 0 ; i < cs0_ ; i+=vec_)
        for(size_t s = 0; s < vec_; ++s){
          iss << format("  cvt.rni.sat.s8.f32 %rc{0}_{1}{2}, %rc{0}_{1}{2};", i, j, vs[s]) << std::endl;
          iss << format("  and.b32 %rc{0}_{1}{2}, %rc{0}_{1}{2}, 0xff;", i, j, vs[s]) << std::endl;
          iss << format("  shl.b32 %rc{0}_{1}{2}, %rc{0}_{1}{2}, {3};", i, j, vs[s], 8*j) << std::endl;
          iss << format("  or.b32 %rc{0}_0{2}, %rc{0}_0{2}, %rc{0}_{1}{2};", i, j, vs[s]) << std::endl;
        }
      }

      iss << std::endl;
      iss << "  /* Store */" << std::endl;
      for(size_t i = 0 ; i < cs0_ ; i+=vec_)
      for(size_t s = 0; s < vec_; s+=c_inc){
        iss << format("  @%predc{} {}{}.{} [%po], %rc{}_0{};", i + s, (gridz_>1 && residual_type!=CatResidual)?"red.add":"st.global", c_aligned?vv:"", out_compute_type, i, c_aligned?"":vs[s]) << std::endl;
        if(i + s < cs0_ - c_inc)
        iss << format("  mad.wide.s32 %po, %diffc{0}, 1, %po;", i + s) << std::endl;
      }
      iss << "}" << std::endl;
  };

  iss << std::endl;
  iss << "/* Store single column of the result tensor */" << std::endl;
  make_store_col("store_col", residual_type_!=CatResidual?residual_type_:NoResidual);

  if(residual_type_ == CatResidual){
    iss << std::endl;
    iss << "/* Concatenate residual column into result tensor */" << std::endl;
    make_store_col("cat_residual_col", residual_type_);
  }



  /* --------------------------------------------------- */



  iss << ".entry " << name << "(" << std::endl
      << "            .param .b64 _pi, .param .b64 _pf, ";
  for(size_t i = 0; i < num_outputs_; i++)
    iss << format(".param .b64 _po{},  ", i) << std::flush;
  iss << std::endl;
  iss << "            .param .b32 _Npix, .param .b32 _Nfilt, .param .b32 _K, .param .b32 _Kout, .param .b32 _C," << std::endl
      << "            .param .b32 _M, .param .b32 _P, .param .b32 _Q, .param .b32 _N," << std::endl
      << "            .param .b32 _D, .param .b32 _H, .param .b32 _W," << std::endl
      << "            .param .b32 _MPQ," << std::endl
      << "            .param .b32 _pad_d, .param .b32 _pad_h, .param .b32 _pad_w, " << std::endl
      << "            .param .b32 _stride_d, .param .b32 _stride_h, .param .b32 _stride_w," << std::endl
      << "            .param .b32 _upsample_d, .param .b32 _upsample_h, .param .b32 _upsample_w," << std::endl
      << "            .param .b32 _strideIc, .param .b32 _strideId, .param .b32 _strideIh, .param .b32 _strideIw, .param .b32 _strideIn, " << std::endl
      << "            .param .b32 _strideOk, .param .b32 _strideOm, .param .b32 _strideOp, .param .b32 _strideOq, .param .b32 _strideOn, " << std::endl
      << "            .param .b32 _strideFk, .param .b32 _strideFc, .param .b32 _strideFs, " << std::endl
      << "            .param .b64 _bias," << std::endl
      << "            .param .b32 _alpha," << std::endl
      << "            .param .b32 _Zk, .param .b32 _offZm, .param .b32 _offZp, .param .b32 _offZq, .param .b32 _strideZn, .param .b32 _strideZk, .param .b32 _strideZm, .param .b32 _strideZp, .param .b32 _strideZq, .param .b64 _pz," << std::endl
      << "            .param .b32 _bound," << std::endl
      << "            .param .b32 _if_scale,";
  for(size_t i = 0; i < num_outputs_; i++)
      iss << format("  .param .b32 _o_scale{},", i) << std::flush;
  iss << "            .param .b32 _z_scale" << std::endl;
  iss << ")" << std::endl;
  iss << "{" << std::endl;

  // Predicates
  iss << "  .reg.pred %in_bounds, %predcrs, %predloop, %predz, %predgz, %predlut;" << std::endl;
  iss << format("  .reg .pred %predk<{0}>, %predbias<{0}>;", cs1_) << std::endl;
  iss << format("  .reg .pred %pred<{}>;", vec_) << std::endl;
  for(size_t i = 0; i < cl0; i += vec_*bf_pqn)
  for(size_t s = 0; s < vec_; s++){
    iss << format("  .reg .pred %predi{};", i + s) << std::endl;
    iss << format("  .reg .b32 %mask{};", i + s) << std::endl;
  }
  iss << format("  .reg .b32 %bound;") << std::endl;
  // Grid IDs
  iss << "  .reg .b32 %bid0, %bid1, %bidz;" << std::endl;
  iss << "  .reg .b32 %id, %idslice, %idz, %id0, %id1;" << std::endl;
  iss << "  .reg .b32 %afid0, %bfid1, %idctrs;" << std::endl;
  // Split-K
  iss << "  .reg .b32 %div, %rem, %offc;" << std::endl;
  // Look-up table
  iss << "  .reg .b32 %trs, %t, %c, %tr, %r, %s;" << std::endl;
  iss << "  .reg .b32 %maskf, %masks, %inc_delta, %p_inc_delta, %inc_mask, %p_inc_mask;" << std::endl;
  for(size_t i = 0; i < cl0; i += vec_*bf_pqn)
  for(size_t s = 0; s < vec_; s++){
      iss << format("  .reg .b32 %maski{0}, %p_mask{0};", i + s) << std::endl;
      iss << format("  .reg .b32 %p_delta{0}, %inc_i{0};", i + s) << std::endl;
  }
  // Tensor shapes
  iss << "  .reg .b32 %pad_d, %pad_h, %pad_w, %stride_d, %stride_h, %stride_w, %upsample_d, %upsample_h, %upsample_w;" << std::endl;
  iss << "  .reg .b32 %Npix, %Nfilt, %K, %Kout, %C, %CTRS;" << std::endl;
  iss << "  .reg .b32 %D, %H, %W, %M, %P, %Q, %N, %MPQ, %Zk, %mN, %mP, %mM, %mQ, %mMPQ;" << std::endl;
  // Strides
  iss << format("  .reg .b32 %strideIc, %strideId, %strideIh, %strideIw, %strideIn;") << std::endl;
  iss << format("  .reg .b32 %strideFc, %strideFs, %strideFk;") << std::endl;
  iss << format("  .reg .b32 %strideOk, %strideOm, %strideOp, %strideOq, %strideOn;") << std::endl;
  iss << format("  .reg .b32 %strideZk, %strideZm, %strideZp, %strideZq, %strideZn;") << std::endl;
  // Pointers
  iss << format("  .reg .b64 %pi, %pf, %pz;") << std::endl;
  for(size_t i = 0; i < num_outputs_; i++)
    iss << format("  .reg .b64 %po_base{};", i) << std::endl;

  iss << format("  .reg .b32 %offi, %offF, %incf;") << std::endl;
  for(size_t i = 0; i < cl0; i += vec_*bf_pqn)
  for(size_t s = 0; s < vec_; s++)
     iss << format("  .reg .b32 %offi{};", i + s) << std::endl;
  for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
  for(size_t s = 0; s < vec_; s++)
      iss << format("  .reg .b64 %pi{};", i + s) << std::endl;
  for(size_t i = 0; i < cs0_; i++)
    iss << format("  .reg .b32 %offc_{0}, %offz_{0};", i) << std::endl;
  for(size_t i = 0; i < cs0_ - c_inc; i+=c_inc)
    iss << format("  .reg .b32 %diffc{0};", i) << std::endl;
  for(size_t i = 0; i < cs0_ - z_inc; i+=z_inc)
    iss << format("  .reg .b32 %diffz{0};", i) << std::endl;
  for(size_t j = 0; j < cs1_; j++)
    iss << format("  .reg .b64 %po{0}, %pz{0};", j) << std::endl;
  // Pointer offsets
  iss << format("  .reg .b32 %offIndhw;") << std::endl;
  iss << format("  .reg .b32 %Dmpad, %Hmpad, %Wmpad;") << std::endl;
  for(size_t i = 0; i < cl0; i+= bf_pqn*vec_)
  for(size_t s = 0; s < vec_; s++){
    iss << format("  .reg.u32 %offIndhw{};", i + s) << std::endl;
    iss << format("  .reg.u32 %offIndh{};", i + s) << std::endl;
    iss << format("  .reg.u32 %offInd{};", i + s) << std::endl;
    iss << format("  .reg.u32 %offId{};", i + s) << std::endl;
    iss << format("  .reg.u32 %offIh{};", i + s) << std::endl;
    iss << format("  .reg.u32 %offIw{};", i + s) << std::endl;
    iss << format("  .reg.u32 %offIn{};", i + s) << std::endl;
    iss << format("  .reg.u32 %offDeltad{};", i + s) << std::endl;
    iss << format("  .reg.u32 %offDeltah{};", i + s) << std::endl;
    iss << format("  .reg.u32 %offDeltaw{};", i + s) << std::endl;
    iss << format("  .reg.u32 %dlo{0}, %hlo{0}, %wlo{0};", i + s) << std::endl;
    iss << format("  .reg.u32 %dhi{0}, %hhi{0}, %whi{0};", i + s) << std::endl;
    iss << format("  .reg.u32 %maskd{0}, %maskh{0}, %maskw{0};", i + s) << std::endl;
  }
  iss << format("  .reg .b32 %offFk, %offFrs, %offFc;") << std::endl;
  for(size_t j = 0; j < cl1; j+=vec_*bf_k)
    iss << format("  .reg.u32 %offFk{};", j) << std::endl;
  iss << format("  .reg .b32 %offc1, %offc0;") << std::endl;
  iss << format("  .reg .b32 %offZm, %offZp, %offZq, %offZn;") << std::endl;
  for(size_t i = 0 ; i < cl0 ; i+= bc0_*vec_)
  for(size_t s = 0 ; s < vec_; s++){
    iss << format("  .reg.u32 %offc0_{0}, %offOn{0}, %offOmpq{0};", i + s) << std::endl;
    iss << format("  .reg.u32 %offZnmp{0}, %offZnm{0}, %offZn{0}, %offZm{0}, %offZp{0}, %offZq{0};", i + s) << std::endl;
  }
  iss << format("  .reg.b32 %offc1_<{}>;", cs1_) << std::endl;
  // Bounds checking
  iss << format("  .reg.s32 %Km<{0}>;", vec_) << std::endl;
  // LDG registers
  iss << format("  .reg .b32 %writei, %readi;") << std::endl;
  for(size_t pqn = 0; pqn < cl0; pqn+=vec_*bf_pqn)
    iss << format("  .reg {}.{} %rri{};", vv, in_word_type, pqn) << std::endl;
  iss << format("  .reg .b32 %writef, %readf;") << std::endl;
  for(size_t k = 0; k < cl1; k+=vec_*bf_k)
    iss << format("  .reg {}.{} %rrf{};", vv, in_word_type, k) << std::endl;
  // Tiles
  iss << "  // Convolution output tile" << std::endl;
  declare_register_tile('c', cs0_, cs1_, 1);
  iss << "  // Kernel output tile" << std::endl;
  declare_register_tile('o', cs0_, cs1_, 1);
  iss << "  // Image tile" << std::endl;
  declare_register_tile('i', cs0_, us_, 1);
  iss << "  // Filter tile" << std::endl;
  declare_register_tile('f', cs1_, us_, 1);
  // Bias
  iss << format("  .reg .b64 %bias, %pbias<{}>;", cs1_) << std::endl;
  iss << format("  .reg .{} %rbias<{}>;", in_word_type, cs1_) << std::endl;
  iss << format("  .reg .pred %has_bias;") << std::endl;
  // Quantization
  iss << "  .reg .b32 %readk, %writek, %rid_mn, %rid_k;" << std::endl;
  iss << "  .reg .pred %predc;" << std::endl;
  iss << "  .reg .b32 %scale, %if_scale;" << std::endl;
  for(size_t i = 0; i < num_outputs_; i++)
    iss << format("  .reg .b32 %o_scale{};", i) << std::endl;
  iss << "  .reg .b32 %z_scale;" << std::endl;

  iss << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  iss << "  /* -- Initialize Accumulator -- */" << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;

  for(size_t c = 0; c < zs_; c++)
  for(size_t i = 0 ; i < cs0_ ; i+=vec_)
  for(size_t k = 0; k < cs1_ ; ++k)
  for(size_t nn = 0; nn < vec_ ; ++nn)
      iss << format("  mov.{} %rc{}_{}_{}{}, 0x0;", in_word_type, c, i, k, vs[nn]) << std::endl;


  iss << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  iss << "  /* ------ Load Parameters ---- */" << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;

  iss << format("  ld.param.u64 %pz, [_pz];") << std::endl;
  for(size_t n = 0; n < num_outputs_; n++)
    iss << format("  ld.param.u64 %po_base{0}, [_po{0}];", n) << std::endl;
  iss << format("  ld.param.u64 %pi, [_pi];") << std::endl;
  iss << format("  ld.param.u64 %pf, [_pf];") << std::endl;
  iss << "  ld.param.s32 %Zk, [_Zk];" << std::endl;
  iss << "  ld.param.s32 %K, [_K];" << std::endl;
  iss << "  ld.param.s32 %Kout, [_Kout];" << std::endl;
  iss << "  ld.param.s32 %C, [_C];" << std::endl;
  iss << "  ld.param.s32 %D, [_D];" << std::endl;
  iss << "  ld.param.s32 %H, [_H];" << std::endl;
  iss << "  ld.param.s32 %W, [_W];" << std::endl;
  iss << "  ld.param.s32 %M, [_M];" << std::endl;
  iss << "  ld.param.s32 %P, [_P];" << std::endl;
  iss << "  ld.param.s32 %Q, [_Q];" << std::endl;
  iss << "  ld.param.s32 %N, [_N];" << std::endl;
  iss << "  ld.param.s32 %MPQ, [_MPQ];" << std::endl;

  iss << "  ld.param.s32 %Npix, [_Npix];" << std::endl;
  iss << "  ld.param.s32 %Nfilt, [_Nfilt];" << std::endl;
  iss << "  ld.param.s32 %pad_d, [_pad_d];" << std::endl;
  iss << "  ld.param.s32 %pad_h, [_pad_h];" << std::endl;
  iss << "  ld.param.s32 %pad_w, [_pad_w];" << std::endl;
  iss << "  ld.param.s32 %stride_d, [_stride_d];" << std::endl;
  iss << "  ld.param.s32 %stride_h, [_stride_h];" << std::endl;
  iss << "  ld.param.s32 %stride_w, [_stride_w];" << std::endl;
  iss << "  ld.param.s32 %upsample_d, [_upsample_d];" << std::endl;
  iss << "  ld.param.s32 %upsample_h, [_upsample_h];" << std::endl;
  iss << "  ld.param.s32 %upsample_w, [_upsample_w];" << std::endl;

  iss << "  ld.param.s32 %strideIc, [_strideIc];" << std::endl;
  iss << "  ld.param.s32 %strideId, [_strideId];" << std::endl;
  iss << "  ld.param.s32 %strideIh, [_strideIh];" << std::endl;
  iss << "  ld.param.s32 %strideIw, [_strideIw];" << std::endl;
  iss << "  ld.param.s32 %strideIn, [_strideIn];" << std::endl;

  iss << "  ld.param.s32 %strideFc, [_strideFc];" << std::endl;
  iss << "  ld.param.s32 %strideFk, [_strideFk];" << std::endl;
  iss << "  ld.param.s32 %strideFs, [_strideFs];" << std::endl;

  iss << "  ld.param.s32 %strideOk, [_strideOk];" << std::endl;
  iss << "  ld.param.s32 %strideOm, [_strideOm];" << std::endl;
  iss << "  ld.param.s32 %strideOp, [_strideOp];" << std::endl;
  iss << "  ld.param.s32 %strideOq, [_strideOq];" << std::endl;
  iss << "  ld.param.s32 %strideOn, [_strideOn];" << std::endl;

  iss << "  ld.param.s32 %bound, [_bound];" << std::endl;
  iss << "  ld.param.b32 %if_scale, [_if_scale];" << std::endl;
  for(size_t n = 0; n < num_outputs_; n++)
    iss << format("  ld.param.b32 %o_scale{0}, [_o_scale{0}];", n) << std::endl;
  iss << "  ld.param.b32 %z_scale, [_z_scale];" << std::endl;

  iss << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  iss << "  /* ------ Special Registers --- */" << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;

  iss << "  // Shared memory" << std::endl;
  iss << format("  .shared .align 16 .b8 _shared[{}];", size_shmem) << std::endl;
  iss << format("  .reg .b64 %shared64;") << std::endl;
  iss << format("  .reg .b32 %shared;") << std::endl;
  iss << format("  mov.u64 %shared64, _shared;") << std::endl;
  iss << format("  cvt.u32.u64 %shared, %shared64;") << std::endl;


  iss << std::endl;
  iss << "  // Thread ID" << std::endl;
  iss << "  mov.u32 %id0, %tid.x;" << std::endl;
  iss << "  mov.u32 %id1, %tid.y;" << std::endl;
  iss << "  mov.u32 %idz, %tid.z;" << std::endl;
  iss << format("  mad.lo.u32 %idslice, %id1, {}, %id0;", bc0_) << std::endl;
  iss << format("  mad.lo.u32 %id, %idslice, {}, %idz;", bz_) << std::endl;

  iss << std::endl;
  iss << "  // Block ID" << std::endl;
  iss << "  mov.u32 %bid0, %ctaid.x;" << std::endl;
  iss << "  mov.u32 %bid1, %ctaid.y;" << std::endl;
  iss << "  mov.u32 %bidz, %ctaid.z;" << std::endl;

  if(gridz_>1){
    iss << format("  // Split") << std::endl;
    iss << format("  div.u32 %div, %C, {};", gridz_) << std::endl;
    iss << format("  rem.u32 %rem, %C, {};", gridz_) << std::endl;
    iss << "  mov.s32 %C, %div;" << std::endl;
    iss << "  mul.lo.u32 %offc, %bidz, %div;" << std::endl;
    iss << "  setp.lt.u32 %pred0, %bidz, %rem;" << std::endl;
    iss << "  @%pred0 add.s32 %C, %C, 1;" << std::endl;
    iss << "  @%pred0 add.s32 %offc, %bidz, %offc;" << std::endl;
    iss << "  @!%pred0 add.s32 %offc, %rem, %offc;" << std::endl;
  }

  iss << std::endl;
  iss << format("  div.u32 %idctrs, %id, {};", bf_k) << std::endl;
  iss << format("  rem.u32 %bfid1, %id, {};", bf_k) << std::endl;
  iss << format("  mov.u32 %afid0, %bfid1;") << std::endl;
  iss << format("  shl.b32 %maskf, 0x1, %idctrs;") << std::endl;


  iss << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  iss << "  /* ------ STS Lanes ----------- */" << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  ptr_sts('i', cd_sharedi, addr_i, "afid0", "idctrs");
  ptr_sts('f', cd_sharedf, addr_f, "bfid1", "idctrs");

  iss << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  iss << "  /* ------ LDS Lanes ----------- */" << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  ptr_lds('i', "id0", addr_i, cd_sharedi);
  ptr_lds('f', "id1", addr_f, cd_sharedf);

  iss << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  iss << "  /* ------ LDG Lanes ----------- */" << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  ptr_ldg_i();
  ptr_ldg_f();


  iss << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  iss << "  /* ------ Accumulate ---------- */" << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  iss << "  // Bounds" << std::endl;
  iss << format("  mul.lo.s32 %CTRS, %C, %Nfilt;") << std::endl;

  iss << "  bar.sync 0;" << std::endl;

  iss << std::endl;
  iss << "  // Predicates initialization" << std::endl;
  iss << format("  setp.lt.s32 %predcrs, %idctrs, %CTRS;") << std::endl;
  iss << format("  setp.gt.s32 %predloop, %CTRS, %bound;") << std::endl;
  iss << format("  @!%predcrs mov.b32 %maskf, 0x0;") << std::endl;
  for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
  for(size_t s = 0; s < vec_; s++){
    iss << format("  // Pixel {0}", i + s) << std::endl;
    iss << format("  ld.const.b32 %maski{0}, [%p_mask{0}];", i + s) << std::endl;;
    iss << format("  and.b32 %mask{0}, %maskf, %maski{0};", i + s) << std::endl;
    iss << format("  setp.ne.b32 %predi{0}, %mask{0}, 0x0;", i + s) << std::endl;
  }

  iss << std::endl;
  iss << "  // First loads" << std::endl;
  ldg_i();
  iss << format("  @!%predloop bra LAST_ITER;") << std::endl;
  ldg_f(false);
  iss << std::endl;

  iss << " // Main loop" << std::endl;
  iss << "LOOP:" << std::endl;
  iss << "  bar.sync 0;" << std::endl;
  for(size_t c = 0; c < u_; c+=us_){
    //Load I to registers
    for(size_t cc = 0 ; cc < us_ ; ++cc)
      lds('i', cs0_, c + cc, cd_sharedi, bc0_);
    //Load F to registers
    for(size_t cc = 0 ; cc < us_ ; ++cc)
      lds('f', cs1_, c + cc, cd_sharedf, bc1_);
    //FFMA
    for(size_t cc = 0 ; cc < us_ ; ++cc)
      fma(cc);
  }

  iss << std::endl;
  iss << "  // Increment filter pointers" << std::endl;
  iss << format("  mad.wide.u32 %pf, %incf, {}, %pf;", 1) << std::endl;

  iss << std::endl;
  iss << "  // Increment image pointers" << std::endl;
  iss << format("  ld.const.b32 %inc_delta, [%p_inc_delta];") << std::endl;
  if(upsample_d_== 1 && upsample_h_ == 1 && upsample_w_ == 1){
    iss << format("  ld.const.b32 %inc_i0, [%p_inc_delta + {}];", 4*nlut) << std::endl;
    for(size_t pqn = 0; pqn < cl0; pqn += vec_*bf_pqn)
    for(size_t s = 0; s < vec_; s++)
        iss << format("  mad.wide.s32 %pi{0}, %inc_i0, {1}, %pi{0};", pqn + s, 1) << std::endl;
  }
  else{
    for(size_t pqn = 0; pqn < cl0; pqn += vec_*bf_pqn)
    for(size_t s = 0; s < vec_; s++){
      iss << format("  ld.const.b32 %inc_i{0}, [%p_delta{0}];", pqn + s) << std::endl;
      iss << format("  mad.wide.s32 %pi{0}, %inc_i{0}, {1}, %pi{0};", pqn + s, 1) << std::endl;
      iss << format("  add.s32 %p_delta{0}, %p_delta{0}, %inc_delta;", pqn + s) << std::endl;
    }
  }
  iss << format("  add.s32 %p_inc_delta, %p_inc_delta, %inc_delta;") << std::endl;

  // Swap buffers
  for(char x: std::vector<char>{'i', 'f'}){
    iss << format("  xor.b32 %write{0}, %write{0}, {1};", x, double_buffer_off) << std::endl;
    iss << format("  xor.b32 %read{0}, %read{0}, {1};", x, double_buffer_off) << std::endl;
  }

  // Compute predicates
  // Filters
  iss << format("  sub.s32 %CTRS, %CTRS, {};", block) << std::endl;
  iss << format("  setp.lt.s32 %predcrs, %idctrs, %CTRS;") << std::endl;
  iss << format("  setp.gt.s32 %predloop, %CTRS, %bound;") << std::endl;
  // Images
  if(pad_d_ == 0 && pad_h_ == 0 && pad_w_ == 0 && stride_d_ == 1 && stride_h_ == 1 && stride_w_ == 1){
      for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
      for(size_t s = 0; s < vec_; s++)
        iss << format("  and.pred %predi{}, %predi{}, %predcrs;", i + s, i + s) << std::endl;
  }
  else{
    iss << format("  ld.const.b32 %inc_mask, [%p_inc_mask];") << std::endl;
    iss << format("  @!%predcrs mov.b32 %maskf, 0x0;") << std::endl;
    for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
    for(size_t s = 0; s < vec_; s++){
        iss << format("  add.s32 %p_mask{0}, %p_mask{0}, %inc_mask;", i + s) << std::endl;
        iss << format("  ld.const.b32 %maski{0}, [%p_mask{0}];", i + s) << std::endl;
        iss << format("  and.b32 %mask{}, %maskf, %maski{};", i + s, i + s) << std::endl;
        iss << format("  setp.ne.b32 %predi{}, %mask{}, 0x0;", i + s, i + s, i + s) << std::endl;
    }
  }
  iss << format("  add.s32 %p_inc_mask, %p_inc_mask, %inc_mask;") << std::endl;

  ldg_i();
  ldg_f(false);
  iss << format("  @%predloop bra.uni LOOP;") << std::endl;
  iss << std::endl;
  iss << format("  setp.gt.s32 %predloop, %CTRS, 0;") << std::endl;
  iss << format("  @!%predloop bra.uni ENDLOOP;") << std::endl;

  iss << "LAST_ITER:" << std::endl;
  iss << format("  mul.lo.u32 %offFk, %bid1, {};", cl1) << std::endl;
  iss << format("  mad.lo.u32 %offFk, %bfid1, {}, %offFk;", vec_) << std::endl;
  for(size_t k = 0; k < cl1; k+=vec_*bf_k)
    iss << format("  add.u32 %offFk{0}, %offFk, {0};", k) << std::endl;
  for(size_t s = 0; s < vec_; ++s)
    iss << format("  sub.s32 %Km{0}, %K, {0};", s) << std::endl;
  ldg_f(true);
  iss << format("  setp.gt.s32 %predloop, %CTRS, 0;") << std::endl;
  iss << format("  @%predloop bra.uni LOOP;") << std::endl;
  iss << "ENDLOOP:" << std::endl;

  //Reduce in registers
  for(size_t c = 1; c < zs_; ++c)
  for(size_t j = 0; j < cs1_; j ++)
  for(size_t i = 0; i < cs0_; i += vec_)
  for(size_t s = 0; s < vec_; s++)
    iss << format("  add.{0} %rc0_{2}_{3}{4}, %rc{1}_{2}_{3}{4}, %rc0_{2}_{3}{4};", in_compute_type, c, i, j, vs[s]) << std::endl;

  //Reduce in shared memory
  if(bz_>1)
  {
    size_t bc = nthreads/bz_;
    for(size_t ij = 0; ij < cl0*cl1; ij += bc)
      iss << format("  .reg .{0} %rrk{1}_0, %rrk{1}_1;", in_word_type, ij) << std::endl;

    iss << format("  mad.lo.u32 %writek, %idz, {}, %shared;", cl0*cl1*in_dtsize) << std::endl;
    iss << format("  mad.lo.u32 %writek, %idslice, {}, %writek;", cs0_*cs1_*in_dtsize) << std::endl;

    iss << "  bar.sync 0;" << std::endl;
    for(size_t j = 0; j < cs1_; j ++)
    for(size_t i = 0; i < cs0_; i += vec_)
    for(size_t s = 0; s < vec_; s++){
      size_t mn = i + j*cs0_;
      iss << format("  st.shared.{} [%writek + {}], %rc0_{}_{}{};", in_word_type, (mn + s)*in_dtsize, i, j, vs[s]) << std::endl;
    }
    iss << "  bar.sync 0;" << std::endl;

    iss << std::endl;
    iss << format("  div.u32 %rid_mn, %id, {};", bz_) << std::endl;
    iss << format("  rem.u32 %rid_k, %id, {};", bz_) << std::endl;
    iss << format("  mad.lo.u32 %readk, %rid_k, {}, %shared;", cl0*cl1*in_dtsize) << std::endl;
    iss << format("  mad.lo.u32 %readk, %rid_mn, {}, %readk;", in_dtsize) << std::endl;
    for(size_t c = bz_/2; c > 0; c /=2){
      iss << format("  setp.lt.u32 %predc, %rid_k, {};", c) << std::endl;
      for(size_t ij = 0; ij < cl0*cl1; ij += bc){
        iss << format("  @%predc ld.shared.{} %rrk{}_0, [%readk + {}];", in_word_type, ij, (ij)*in_dtsize) << std::endl;
        iss << format("  @%predc ld.shared.{} %rrk{}_1, [%readk + {}];", in_word_type, ij, (ij + c*cl0*cl1)*in_dtsize) << std::endl;
        iss << format("  @%predc add.{0} %rrk{1}_0, %rrk{1}_0, %rrk{1}_1;", in_compute_type, ij) << std::endl;
        iss << format("  @%predc st.shared.{} [%readk + {}], %rrk{}_0;", in_word_type, ij*in_dtsize, ij) << std::endl;
      }
      iss << "  bar.sync 0;" << std::endl;
    }

    iss << format("  mad.lo.u32 %readk, %idslice, {}, %shared;", cs0_*cs1_*in_dtsize) << std::endl;
    for(size_t j = 0; j < cs1_; j ++)
    for(size_t i = 0; i < cs0_; i += vec_)
    for(size_t s = 0; s < vec_; s++){
      iss << format("  ld.shared.{} %rc0_{}_{}{}, [%readk + {}];", in_word_type, i, j, vs[s], ((i+s) + j*cs0_)*in_dtsize) << std::endl;
    }
  }

  if(in_dtype_ == INT8X4_TYPE){
    iss << std::endl;
    iss << "  /* ---------------------------- */" << std::endl;
    iss << "  /* ------ Convert to FP32 ----- */" << std::endl;
    iss << "  /* ---------------------------- */" << std::endl;
    for(size_t j = 0; j < cs1_ ; j++)
    for(size_t i = 0 ; i < cs0_ ; i+=vec_)
    for(size_t s = 0; s < vec_; ++s){
      iss << format("   cvt.rn.f32.s32 %rc0_{0}_{1}{2}, %rc0_{0}_{1}{2};", i, j, vs[s]) << std::endl;
      iss << format("  mul.f32 %rc0_{0}_{1}{2}, %if_scale, %rc0_{0}_{1}{2};", i, j, vs[s]) << std::endl;
    }
  }

  iss << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  iss << "  /* ------ Column Offsets ----- */" << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;

  iss << format("  mov.s32 %bid0, %ctaid.x;") << std::endl;
  iss << format("  mov.s32 %bid1, %ctaid.y;") << std::endl;
  iss << format("  mov.s32 %id0, %tid.x;") << std::endl;
  iss << format("  mov.s32 %id1, %tid.y;") << std::endl;

  iss << format("  mad.lo.s32 %offc1, %bid1, {}, 0;", cl1) << std::endl;
  iss << format("  mad.lo.s32  %offc1, %id1, {}, %offc1;", vec_) << std::endl;
  for(size_t j = 0; j < cs1_; j+= vec_)
  for(size_t s = 0; s < vec_; s++)
    iss << format("  add.u32 %offc1_{}, {}, %offc1;", j + s, j*bc1_ + s) << std::endl;

  iss << "  // Predicates" << std::endl;
  iss << format("  setp.eq.s32 %predz, %idz, 0;") << std::endl;
  iss << format("  setp.eq.and.s32 %predgz, %bidz, 0, %predz;") << std::endl;

  iss << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  iss << "  /* ------ Handle Bias -------- */" << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  iss << format("  ld.param.u64 %bias, [_bias];") << std::endl;
  iss << format("  setp.ne.b64 %has_bias, %bias, 0;") << std::endl;
  iss << format("  @!%has_bias bra.uni BIAS_DONE;") << std::endl;
  iss << "DO_BIAS:" << std::endl;
  for(size_t j = 0; j < cs1_ ; j++)
    iss << format("  mad.wide.u32 %pbias{0}, %offc1_{0}, {1}, %bias;", j, in_dtsize) << std::endl;
  for(size_t j = 0; j < cs1_ ; j++)
    iss << format("  setp.lt.and.s32 %predbias{0}, %offc1_{0}, %K, %predgz;", j) << std::endl;
  for(size_t j = 0; j < cs1_ ; j++){
    iss << format("  @%predbias{0} ld.global.{1} %rbias{0}, [%pbias{0}];", j, in_word_type) << std::endl;
    iss << format("  @!%predbias{0} mov.{1} %rbias{0}, 0;", j, in_word_type) << std::endl;
  }
  for(size_t j = 0; j < cs1_ ; j++)
  for(size_t i = 0 ; i < cs0_ ; i+=vec_)
  for(size_t s = 0; s < vec_; ++s)
    iss << format("  add.f32 %rc0_{0}_{1}{2}, %rc0_{0}_{1}{2}, %rbias{1};", i, j, vs[s]) << std::endl;
  iss << "BIAS_DONE:" << std::endl;


  iss << "  .reg .b32 %alpha;" << std::endl;
  iss << "  ld.param.b32 %alpha, [_alpha];" << std::endl;
  if(residual_type_ != AddResidual){
      if(activation_ == ReLU || activation_ == ELU){
        iss << std::endl;
        iss << "  /* ---------------------------- */" << std::endl;
        iss << "  /* --------- Rectifier -------- */" << std::endl;
        iss << "  /* ---------------------------- */" << std::endl;
        iss << "  .reg .b32 %leakage, %arg;" << std::endl;
        for(size_t j = 0; j < cs1_ ; j++)
        for(size_t i = 0 ; i < cs0_ ; i+=vec_)
        for(size_t s = 0; s < vec_; ++s){
          if(activation_ == ReLU)
            iss << format("  mul.f32 %leakage, %rc0_{0}_{1}{2}, %alpha;", i, j, vs[s]) << std::endl;
          else{
              iss << format("  mov.f32 %arg, %rc0_{0}_{1}{2};", i, j, vs[s]) << std::endl;
              iss << format("  call (%leakage), exp, (%arg);") << std::endl;
              iss << format("  sub.f32 %leakage, %leakage, 1.;") << std::endl;
              iss << format("  mul.f32 %leakage, %leakage, %alpha;", i, j, vs[s]) << std::endl;
          }
          iss << format("  slct.f32.f32 %rc0_{0}_{1}{2}, %rc0_{0}_{1}{2}, %leakage, %rc0_{0}_{1}{2};", i, j, vs[s]) << std::endl;
        }
      }
      if(activation_ == Sigmoid){
        iss << std::endl;
        iss << "  /* ---------------------------- */" << std::endl;
        iss << "  /* ------------ Sigmoid ------- */" << std::endl;
        iss << "  /* ---------------------------- */" << std::endl;
        iss << "  .reg .f32 %res, %arg;" << std::endl;
        for(size_t j = 0; j < cs1_ ; j++)
        for(size_t i = 0 ; i < cs0_ ; i+=vec_)
        for(size_t s = 0; s < vec_; ++s){
          iss << format("  mul.f32 %arg, %rc0_{0}_{1}{2}, -1.;", i, j, vs[s]) << std::endl;
          iss << format("  call (%res), exp, (%arg);", i, j, vs[s]) << std::endl;
          iss << format("  add.f32 %rc0_{0}_{1}{2}, 1., %res;", i, j, vs[s]) << std::endl;
          iss << format("  div.approx.f32 %rc0_{0}_{1}{2}, 1., %rc0_{0}_{1}{2};", i, j, vs[s]) << std::endl;
        }
      }
  }

  if(out_dtype_==INT8X4_TYPE){
    iss << std::endl;
    iss << "  /* ---------------------------- */" << std::endl;
    iss << "  /* ------------ Pack ---------- */" << std::endl;
    iss << "  /* ---------------------------- */" << std::endl;
    iss << format("  mad.lo.u32 %writek, %id0, {}, %shared;", out_dtsize) << std::endl;
    iss << format("  mad.lo.u32 %writek, %id1, {}, %writek;", bc0_*vec_*out_dtsize) << std::endl;
    iss << format("  mad.lo.u32 %readk, %id0, {}, %shared;", out_dtsize) << std::endl;
    iss << format("  mad.lo.u32 %readk, %id1, {}, %readk;", bc0_*cs1_*out_dtsize) << std::endl;

    for(size_t i = 0; i < cs0_; i += vec_)
    for(size_t ii = 0; ii < vec_; ii++){
      iss << format("  bar.sync 0;") << std::endl;
      for(size_t j = 0; j < cs1_; j += vec_)
      for(size_t jj = 0; jj < vec_; jj++)
        iss << format("  st.shared.{} [%writek + {}], %rc0_{}_{}{};", in_word_type, (j*bc1_ + jj)*bc0_*out_dtsize, i, j + jj, vs[ii]) << std::endl;
      iss << format("  bar.sync 0;") << std::endl;
      for(size_t j = 0; j < cs1_; j += vec_)
      for(size_t jj = 0; jj < vec_; jj++)
        iss << format("  ld.shared.{} %rc0_{}_{}{}, [%readk + {}];", in_word_type, i, j + jj, vs[ii], (j+ jj)*bc0_*out_dtsize) << std::endl;
    }

    iss << format("  mad.lo.s32 %offc1, %bid1, {}, 0;", cl1/vect_k) << std::endl;
    iss << format("  mad.lo.s32  %offc1, %id1, {}, %offc1;", cs1_/vect_k) << std::endl;
    for(size_t j = 0; j < cs1_/vect_k; j++)
      iss << format("  add.u32 %offc1_{0}, {0}, %offc1;", j) << std::endl;
  }


  iss << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  iss << "  /* ----- Row Offsets ---------- */" << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  iss << format("  mad.lo.s32 %offc0, %bid0, {}, 0;", cl0) << std::endl;
  iss << format("  mad.lo.s32  %offc0, %id0, {}, %offc0;", vec_) << std::endl;
  iss << format("  mul.lo.s32 %mMPQ, %MPQ, -1;") << std::endl;

  for(size_t i = 0; i < cl0; i+=vec_*bc0_)
  for(size_t s = 0; s < vec_; s++){
    iss << format("  add.s32 %offc0_{0}, %offc0, {0};", i) << std::endl;
    iss << format("  add.u32 %offc0_{0}, %offc0, {0};", i + s) << std::endl;
    iss << format("  div.u32 %offOn{0}, %offc0_{0}, %MPQ;", i + s) << std::endl;
    iss << format("  mad.lo.s32 %offOmpq{0}, %mMPQ, %offOn{0}, %offc0_{0};", i + s) << std::endl;
    iss << format("  mad.lo.s32 %offc_{0}, %offOn{1}, %strideOn, 0;", i/bc0_ + s, i + s) << std::endl;
    iss << format("  mad.lo.s32 %offc_{0}, %offOmpq{1}, %strideOq, %offc_{0};", i/bc0_ + s, i + s) << std::endl;
  }

  iss << "  // Pointer deltas along M, P, Q" << std::endl;
  for(size_t i = 0; i < cs0_ - c_inc; i+=c_inc)
    iss << format("  sub.s32 %diffc{0}, %offc_{1}, %offc_{0};", i, i + c_inc) << std::endl;

  if(Zk_ > 0){
      iss << std::endl;
      iss << "  // Residual" << std::endl;
      iss << "  ld.param.s32 %strideZn, [_strideZn];" << std::endl;
      iss << "  ld.param.s32 %strideZk, [_strideZk];" << std::endl;
      iss << "  ld.param.s32 %strideZm, [_strideZm];" << std::endl;
      iss << "  ld.param.s32 %strideZp, [_strideZp];" << std::endl;
      iss << "  ld.param.s32 %strideZq, [_strideZq];" << std::endl;
      iss << "  ld.param.s32 %offZm, [_offZm];" << std::endl;
      iss << "  ld.param.s32 %offZp, [_offZp];" << std::endl;
      iss << "  ld.param.s32 %offZq, [_offZq];" << std::endl;

      iss << format("  mul.lo.s32 %mM, %M, -1;") << std::endl;
      iss << format("  mul.lo.s32 %mP, %P, -1;") << std::endl;
      iss << format("  mul.lo.s32 %mQ, %Q, -1;") << std::endl;

      for(size_t i = 0; i < cl0; i+=vec_*bc0_)
      for(size_t s = 0; s < vec_; s++){
        iss << format("  // Unpack ID offset") << std::endl;
        iss << format("  div.u32 %offZnmp{0}, %offc0_{0}, %Q;", i + s) << std::endl;
        iss << format("  mad.lo.s32 %offZq{0}, %mQ, %offZnmp{0}, %offc0_{0};", i + s) << std::endl;
        iss << format("  div.u32 %offZnm{0}, %offZnmp{0}, %P;", i + s, Q_) << std::endl;
        iss << format("  mad.lo.s32 %offZp{0}, %mP, %offZnm{0}, %offZnmp{0};", i + s) << std::endl;
        iss << format("  div.u32 %offZn{0}, %offZnm{0}, %M;", i + s, P_) << std::endl;
        iss << format("  mad.lo.s32 %offZm{0}, %mM, %offZn{0}, %offZnm{0};", i + s) << std::endl;
        iss << format("  add.s32 %offZm{0}, %offZm{0}, %offZm;", i + s) << std::endl;
        iss << format("  add.s32 %offZp{0}, %offZp{0}, %offZp;", i + s) << std::endl;
        iss << format("  add.s32 %offZq{0}, %offZq{0}, %offZq;", i + s) << std::endl;

        iss << format("  // Compute pointer offsets") << std::endl;
        iss << format("  mad.lo.s32 %offz_{0}, %offZn{1}, %strideZn, 0;", i/bc0_ + s, i + s) << std::endl;
        iss << format("  mad.lo.s32 %offz_{0}, %offZm{1}, %strideZm, %offz_{0};", i/bc0_ + s, i + s) << std::endl;
        iss << format("  mad.lo.s32 %offz_{0}, %offZp{1}, %strideZp, %offz_{0};", i/bc0_ + s, i + s) << std::endl;
        iss << format("  mad.lo.s32 %offz_{0}, %offZq{1}, %strideZq, %offz_{0};", i/bc0_ + s, i + s) << std::endl;
      }

      iss << format("  // Compute pointer deltas") << std::endl;
      for(size_t i = 0; i < cs0_ - z_inc; i+=z_inc)
        iss << format("  sub.s32 %diffz{0}, %offz_{1}, %offz_{0};", i, i + z_inc) << std::endl;

      iss << format("  // Compute pointers") << std::endl;
      for(size_t j = 0; j < cs1_; j++){
        iss << format("  mad.wide.s32 %pz{0}, %offc1_{0}, %strideZk, %pz;", j) << std::endl;
        iss << format("  mad.wide.s32 %pz{0}, %offz_0, 1, %pz{0};", j) << std::endl;
      }
  }

  for(size_t n = 0; n < num_outputs_; n++){
    iss << std::endl;
    iss << "  /* ---------------------------- */" << std::endl;
    iss << "  /* ----- Store output " << n << " ------- */" << std::endl;
    iss << "  /* ---------------------------- */" << std::endl;

    iss << "  // Pointers " << std::endl;
    for(size_t j = 0; j < cs1_/vect_k; j++){
      iss << format("  mad.wide.s32 %po{0}, %offc1_{0}, %strideOk, %po_base{1};", j, n) << std::endl;
      iss << format("  mad.wide.s32 %po{0}, %offc_0, 1, %po{0};", j) << std::endl;
    }

    iss << std::endl;
    iss << "  // Write back" << std::endl;
    for(size_t j = 0; j < cs1_/vect_k; j++){
      iss << format("  setp.lt.and.s32 %predk{0}, %offc1_{0}, %Kout, %predz;", j) << std::endl;
      iss << format("  @%predk{0} call.uni store_col, (%po{0}, %Npix", j);
      for(size_t i = 0 ; i < cs0_ ; i+=vec_)
      for(size_t s = 0; s < vec_; s++)
        iss << format(", %offc0_{}", i*bc0_ + s);
      for(size_t i = 0; i < cs0_ - c_inc; i+=c_inc)
        iss << format(", %diffc{}", i);
      for(size_t jj = 0; jj < vect_k ; jj++)
      for(size_t i = 0 ; i < cs0_ ; i+=vec_)
        iss << format(", %rc0_{}_{}", i, j*vect_k + jj);
      iss << format(", %o_scale{}", n);
      if(residual_type_ == AddResidual){
        iss << format(", %pz{}, %z_scale", j);
        for(size_t i = 0; i < cs0_ - z_inc; i+=z_inc)
          iss << format(", %diffz{}", i);
        iss << format(", %alpha") << std::endl;
      }
      iss << ");" << std::endl;
    }


    if(residual_type_ == CatResidual && Zk_ > 0){
      iss << std::endl;
      iss << "  // Concatenate residual" << std::endl;
      iss << "  .reg .u32 %inc;" << std::endl;
      iss << "  mov.s32 %inc, %nctaid.y;" << std::endl;
      for(size_t j = 0; j < cs1_; j++)
        iss << format("  mad.wide.s32 %po{0}, %Kout, %strideOk, %po{0};", j) << std::endl;
      iss << std::endl;
      iss << "CROP_MERGE:" << std::endl;
      for(size_t j = 0; j < cs1_/vect_k; j++){
        iss << format("  setp.lt.and.s32 %predk{0}, %offc1_{0}, %Zk, %predz;", j) << std::endl;
        iss << format("  @%predk{0} call.uni cat_residual_col, (%po{0}, %Npix", j);
        for(size_t i = 0 ; i < cs0_ ; i+=vec_)
        for(size_t s = 0; s < vec_; s++)
            iss << format(", %offc0_{}", i*bc0_ + s);
        for(size_t i = 0; i < cs0_ - c_inc; i+=c_inc)
            iss << format(", %diffc{}", i);
        iss << format(", %o_scale{}", n);
        iss << format(", %pz{}, %z_scale", j);
        for(size_t i = 0; i < cs0_ - z_inc; i+=z_inc)
          iss << format(", %diffz{}", i);
        iss << ");" << std::endl;
        iss << format("  mad.wide.s32 %pz{0}, %inc, %strideZk, %pz{0};", j) << std::endl;
        iss << format("  mad.wide.s32 %po{0}, %inc, %strideOk, %po{0};", j) << std::endl;
      }
      iss << "  sub.s32 %Zk, %Zk, %inc;" << std::endl;
      iss << "  setp.gt.s32 %predloop, %Zk, 0;" << std::endl;
      iss << "@%predloop bra.uni CROP_MERGE;" << std::endl;
    }

  }
  iss << "}" << std::endl;

  return iss.str();
}

void Conv::enqueue(driver::Kernel& kernel, driver::Stream& stream,
                   driver::Buffer const & I, driver::Buffer const & F, driver::Buffer* O, // Conv
                   driver::Buffer const *bias, // Bias
                   float alpha, // Relu
                   float i_scale, float f_scale, std::vector<float> o_scale, float z_scale,// Quantization
                   driver::Buffer const *Z // Merge
                   )
{
  int32_t Ko = (residual_type_ == AddResidual)?Kout_:Kout_ + Zk_;
  // Data-type size
  // I strides
  int32_t strideIw = size_of(in_dtype_);
  int32_t strideIh = W_*strideIw;
  int32_t strideId = H_*strideIh;
  int32_t strideIc = D_*strideId;
  int32_t strideIn = C_*strideIc;

  // F strides
  int32_t strideFk = size_of(in_dtype_);
  int32_t strideFs = K_*strideFk;
  int32_t strideFr = S_*strideFs;
  int32_t strideFt = R_*strideFr;
  int32_t strideFc = T_*strideFt;

  // O strides
  int32_t strideOq = size_of(out_dtype_);
  int32_t strideOp = Q_*strideOq;
  int32_t strideOm = P_*strideOp;
  int32_t strideOk = M_*strideOm;
  int32_t strideOn = Ko*strideOk;

  // Z strides
  int32_t strideZq = size_of(out_dtype_);
  int32_t strideZp = Zq_*strideZq;
  int32_t strideZm = Zp_*strideZp;
  int32_t strideZk = Zm_*strideZm;
  int32_t strideZn = Zk_*strideZk;

  // Input information
  int32_t MPQ = M_*P_*Q_;
  int32_t Npix = M_*P_*Q_*N_;
  int32_t Nfilt = T_*R_*S_;

  // Grid dimensions
  int32_t cl0 = bc0_*cs0_;
  int32_t cl1 = bc1_*cs1_;
  int32_t zl = bz_*zs_;

  size_t grid0 = ceil(Npix, cl0);
  size_t grid1 = ceil(K_, cl1);

  // Leaky ReLU alpha
  scalar a(alpha, FLOAT_TYPE);

  // Last safe index
  int32_t depth = C_*Nfilt;
  // Last filter element
  int32_t lastj = (grid1 - 1)*cl1 + cl1 - 1;
  int32_t lastctrs = u_*zl - 1;
  // Last safe filter element
  int32_t last_safe_b = (depth*K_ - 1 - lastj)/K_ - lastctrs;
  int32_t bound = std::max<int32_t>(1, depth - last_safe_b);

  // Constant memory
  driver::Buffer LUT = kernel.module().symbol("_LUT");
  driver::Buffer masks = kernel.module().symbol("_masks");
  stream.write(LUT, false, 0, cLUT.size()*4, cLUT.data());
  stream.write(masks, false, 0, masks_.size()*4, masks_.data());

  // Enqueue
  size_t idx = 0;
  kernel.setArg(idx++, I);
  kernel.setArg(idx++, F);
  for(size_t i = 0; i < num_outputs_; i++)
    kernel.setArg(idx++, O[i]);
  kernel.setArg(idx++, Npix);
  kernel.setArg(idx++, Nfilt);
  kernel.setArg(idx++, K_);
  kernel.setArg(idx++, Kout_);

  kernel.setArg(idx++, C_);
  kernel.setArg(idx++, M_);
  kernel.setArg(idx++, P_);
  kernel.setArg(idx++, Q_);
  kernel.setArg(idx++, N_);
  kernel.setArg(idx++, D_*upsample_d_);
  kernel.setArg(idx++, H_*upsample_h_);
  kernel.setArg(idx++, W_*upsample_w_);
  kernel.setArg(idx++, MPQ);
  kernel.setArg(idx++, pad_d_);
  kernel.setArg(idx++, pad_h_);
  kernel.setArg(idx++, pad_w_);
  kernel.setArg(idx++, stride_d_);
  kernel.setArg(idx++, stride_h_);
  kernel.setArg(idx++, stride_w_);
  kernel.setArg(idx++, upsample_d_);
  kernel.setArg(idx++, upsample_h_);
  kernel.setArg(idx++, upsample_w_);
  kernel.setArg(idx++, strideIc);
  kernel.setArg(idx++, strideId);
  kernel.setArg(idx++, strideIh);
  kernel.setArg(idx++, strideIw);
  kernel.setArg(idx++, strideIn);
  // O strides
  kernel.setArg(idx++, strideOk);
  kernel.setArg(idx++, strideOm);
  kernel.setArg(idx++, strideOp);
  kernel.setArg(idx++, strideOq);
  kernel.setArg(idx++, strideOn);
  // F strides
  kernel.setArg(idx++, strideFk);
  kernel.setArg(idx++, strideFc);
  kernel.setArg(idx++, strideFs);
  // Bias
  kernel.setArg(idx++, bias?*bias:(uint64_t)0);
  // ReLU
  kernel.setArg(idx++, size_of(a.dtype()), a.data());
  // Crop-Merge
  kernel.setArg(idx++, Zk_);
  kernel.setArg(idx++, z_crop_m0_);
  kernel.setArg(idx++, z_crop_p0_);
  kernel.setArg(idx++, z_crop_q0_);
  kernel.setArg(idx++, strideZn);
  kernel.setArg(idx++, strideZk);
  kernel.setArg(idx++, strideZm);
  kernel.setArg(idx++, strideZp);
  kernel.setArg(idx++, strideZq);
  kernel.setArg(idx++, Z?*Z:(uint64_t)0);
  // Loop optimization
  kernel.setArg(idx++, bound);
  // Quantization
  kernel.setArg(idx++, (float)1/(i_scale*f_scale));
  for(size_t i = 0; i < num_outputs_; i++)
    kernel.setArg(idx++, o_scale[i]);
  kernel.setArg(idx++, z_scale);
  if(gridz_>1)
    for(size_t i = 0; i < num_outputs_; i++)
      O[i].set_zero(stream, N_*Ko*M_*P_*Q_*size_of(out_dtype_));
  stream.enqueue(kernel, {grid0, grid1, gridz_}, {bc0_, bc1_, bz_});
}

}
}
