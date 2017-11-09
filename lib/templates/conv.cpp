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

Conv::Conv(DType dtype, param_t C, param_t D, param_t H, param_t W, param_t N, param_t K, param_t M, param_t P, param_t Q, param_t T, param_t R, param_t S, param_t pad_d, param_t pad_h, param_t pad_w, param_t stride_d, param_t stride_h, param_t stride_w,
     param_t vec, param_t bc0, param_t bc1, param_t cs0, param_t cs1, param_t u, param_t, param_t bz, param_t gridz):
  dtype_(dtype), C_(C), N_(N), K_(K), D_(D), H_(H), W_(W), M_(M), P_(P), Q_(Q), T_(T), R_(R), S_(S),
  pad_d_(pad_d), pad_h_(pad_h), pad_w_(pad_w),
  stride_d_(stride_d), stride_h_(stride_h), stride_w_(stride_w),
  vec_(vec), bc0_(bc0), bc1_(bc1), cs0_(cs0), cs1_(cs1), u_(u), us_(u), zs_(1), bz_(bz), gridz_(gridz)
{
//    std::cout << vec << " " << bc0 << " " << bc1 << " " << cs0 << " " << cs1 << std::endl;
    // Resize LUT
    size_t block = u_*zs_*bz_;
    size_t Nfilt = T_*R_*S_;
    size_t nlut = (block + Nfilt - 1)/Nfilt * Nfilt;
    cLUT.resize(2*nlut);

    // Data-type size
    int32_t dtsize = size_of(dtype_);

    // I strides
    int32_t strideIw = dtsize;
    int32_t strideIh = W_*strideIw;
    int32_t strideId = H_*strideIh;
    int32_t strideIc = D_*strideId;
//    int32_t strideIn = C_*strideIc;

    // Init constant memory
    masks_.resize(nlut + (2*pad_h+1)*(2*pad_w+1)*(2*pad_d+1)*nlut);
    init_constant_memory(cLUT, masks_, strideIc, strideIw, strideIh, strideId);
}

void Conv::init_constant_memory(std::vector<int32_t> &delta, std::vector<uint32_t> &masks, int32_t strideIc, int32_t strideIw, int32_t strideIh, int32_t strideId){
   size_t nlut = delta.size()/2;
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

   // Delta Table
   for(size_t i = 0; i < nlut; ++i){
       int32_t ctrs = i;
       int32_t c = ctrs / Nfilt;
       int32_t t, r, s;
       std::tie(t, r, s) = unpack(ctrs % Nfilt);

       int32_t nextctrs = ctrs + block;
       int32_t nextc = nextctrs / Nfilt;
       int32_t nextt, nextr, nexts;
       std::tie(nextt, nextr, nexts) = unpack(nextctrs % Nfilt);

       int32_t cdiff = nextc - c;
       int32_t tdiff = nextt - t;
       int32_t rdiff = nextr - r;
       int32_t sdiff = nexts - s;

       delta[i] = cdiff*strideIc + sdiff*strideIw + rdiff*strideIh + tdiff*strideId;
       delta[nlut + i] = 4*((nextctrs % nlut) - ctrs);
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
        if(pd == 0 && ph == 0 && pw == 0){
//          std::cout << std::endl;
//          std::cout << std::bitset<16>(masks_ptr[i]) << " " << std::flush;
        }
     }
   }
//   std::cout << std::endl;
   for(size_t i = 0; i < nlut; ++i)
     masks[i] = 0x0;

}

void Conv::output_shapes(param_t D, param_t H, param_t W, param_t T, param_t R, param_t S,
                         param_t pad_d, param_t pad_h, param_t pad_w, param_t stride_d, param_t stride_h, param_t stride_w,
                         param_t &M, param_t &P, param_t &Q)
{
    M = (D - T + 1 + 2*pad_d + stride_d - 1)/stride_d;
    P = (H - R + 1 + 2*pad_h + stride_h - 1)/stride_h;
    Q = (W - S + 1 + 2*pad_w + stride_w - 1)/stride_w;
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
//    param_t  NPQ = x[1], K = x[2], C = x[3];
    param_t vec = x[5], bc0 = x[6], bc1 = x[7], cs0 = x[8], cs1 = x[9],
            u = x[10], zs = x[11], bz = x[12];

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
    param_t size_shmem = std::max(size_redc, param_tiles);

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
                  && (block <= 32);
    valid[m] = is_valid;
  }
}


/* Code generation */
std::string Conv::dump(drv::Device const & device, std::string const & name){

  std::stringstream iss;
  size_t dtsize = size_of(dtype_);
  std::string compute_dtype = arith_str(dtype_);
  std::string dtype = io_str(dtype_);
  size_t cl0 = bc0_*cs0_;
  size_t cl1 = cs1_*bc1_;
  size_t zl = zs_*bz_;

  // helpers
  size_t nthreads = bc0_*bc1_*bz_;
  size_t Nfilt = R_*S_*T_;
  size_t cd_sharedi = dtsize*cl0;
  size_t cd_sharedf = dtsize*cl1;

  size_t block = u_*zl;
  size_t nlut = (block + Nfilt - 1)/Nfilt * Nfilt;
  size_t size_sharedi = cd_sharedi*block;
  size_t size_sharedf = cd_sharedf*block;
  size_t size_tiles = next_pow2(2*(size_sharedi + size_sharedf));
  size_t size_redc = dtsize*cl0*cl1*(bz_==1?0:bz_);
  size_t size_shmem = std::max(size_redc, size_tiles);
  size_t Bvec = vec_*dtsize;
  size_t addr_i = 0;
  size_t addr_f = size_sharedi;
  size_t double_buffer_off = size_tiles/2;
  // params
  size_t bf_ctrs = block;
  size_t bf_k = nthreads/bf_ctrs;
  size_t bf_pqn = bf_k;

  uint8_t is_valid;
  uint32_t params[] = {(uint32_t)size_of(dtype_), N_*P_*Q_*M_, K_, C_, R_*S_*T_,
                     vec_, bc0_, bc1_, cs0_, cs1_, u_, zs_, bz_, gridz_};
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
        iss << format("  .reg {}.{}", vv, dtype);
        for(size_t n = 0 ; n < N ; n++)
          iss << format("{} %r{}{}_{}_{}", n>0?",":"", x, c, m, n);
        iss << ";" << std::endl;
      }
    }
  };

  auto ptr_ldg_i = [&](){
      iss << format("  // I offsets") << std::endl;
      iss << format("  mul.lo.s32 %mM, %M, -1;") << std::endl;
      iss << format("  mul.lo.s32 %mN, %N, -1;") << std::endl;
      iss << format("  mul.lo.s32 %mP, %P, -1;") << std::endl;
      iss << format("  mul.lo.s32 %mQ, %Q, -1;") << std::endl;

      iss << format("  mul.lo.u32 %offIndhw, %bid0, {};", cl0) << std::endl;
      iss << format("  mad.lo.u32 %offIndhw, %afid0, {}, %offIndhw;", vec_) << std::endl;
      for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
      for(size_t s = 0; s < vec_; s++){
        iss << format("  add.u32 %offIndhw{0}, %offIndhw, {0};", i + s) << std::endl;
        iss << format("  div.u32 %offIndh{0}, %offIndhw{0}, %Q;", i + s) << std::endl;
        iss << format("  mad.lo.s32 %offIw{0}, %mQ, %offIndh{0}, %offIndhw{0};", i + s) << std::endl;
        iss << format("  div.u32 %offInd{0}, %offIndh{0}, %P;", i + s, Q_) << std::endl;
        iss << format("  mad.lo.s32 %offIh{0}, %mP, %offInd{0}, %offIndh{0};", i + s) << std::endl;
        iss << format("  div.u32 %offIn{0}, %offInd{0}, %M;", i + s, P_) << std::endl;
        iss << format("  mad.lo.s32 %offId{0}, %mM, %offIn{0}, %offInd{0};", i + s) << std::endl;
      }

      iss << format("  sub .s32 %Dmpad, %M, %pad_d;") << std::endl;
      iss << format("  sub .s32 %Hmpad, %P, %pad_h;") << std::endl;
      iss << format("  sub .s32 %Wmpad, %Q, %pad_w;") << std::endl;
      iss << format("  sub .s32 %Dmpad, %Dmpad, 1;") << std::endl;
      iss << format("  sub .s32 %Hmpad, %Hmpad, 1;") << std::endl;
      iss << format("  sub .s32 %Wmpad, %Wmpad, 1;") << std::endl;

      for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
      for(size_t s = 0; s < vec_; s++)
        iss << format("   setp.lt.u32 %predi{}, %offIndhw{}, %Npix;", i + s, i + s) << std::endl;

      for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
      for(size_t s = 0; s < vec_; s++){
          iss << format("  sub.s32 %dlo{0}, %pad_d, %offId{0};", i + s) << std::endl;
          iss << format("  sub.s32 %hlo{0}, %pad_h, %offIh{0};", i + s) << std::endl;
          iss << format("  sub.s32 %wlo{0}, %pad_w, %offIw{0};", i + s) << std::endl;
          iss << format("  sub.s32 %dhi{0}, %offId{0}, %Dmpad;", i + s) << std::endl;
          iss << format("  sub.s32 %hhi{0}, %offIh{0}, %Hmpad;", i + s) << std::endl;
          iss << format("  sub.s32 %whi{0}, %offIw{0}, %Wmpad;", i + s) << std::endl;

          iss << format("  max.s32 %dlo{0}, %dlo{0}, 0;", i + s) << std::endl;
          iss << format("  max.s32 %hlo{0}, %hlo{0}, 0;", i + s) << std::endl;
          iss << format("  max.s32 %wlo{0}, %wlo{0}, 0;", i + s) << std::endl;

          iss << format("  max.s32 %dhi{0}, %dhi{0}, 0;", i + s) << std::endl;
          iss << format("  max.s32 %hhi{0}, %hhi{0}, 0;", i + s) << std::endl;
          iss << format("  max.s32 %whi{0}, %whi{0}, 0;", i + s) << std::endl;

          iss << format("  sub.s32 %maskd{0}, %pad_d, %dlo{0};", i + s) << std::endl;
          iss << format("  add.s32 %maskd{0}, %maskd{0}, %dhi{0};", i + s) << std::endl;

          iss << format("  sub.s32 %maskh{0}, %pad_h, %hlo{0};", i + s) << std::endl;
          iss << format("  add.s32 %maskh{0}, %maskh{0}, %hhi{0};", i + s) << std::endl;

          iss << format("  sub.s32 %maskw{0}, %pad_w, %wlo{0};", i + s) << std::endl;
          iss << format("  add.s32 %maskw{0}, %maskw{0}, %whi{0};", i + s) << std::endl;

          iss << format("  @!%predi{0} mov.s32 %pmask{0}, %masks;", i + s) << std::endl;
          iss << format("  @%predi{0} add.s32 %pmask{0}, {1}, %masks;", i + s, 4*nlut) << std::endl;
          iss << format("  @%predi{0} mad.lo.s32 %pmask{0}, %maskd{0}, {1}, %pmask{0};", i + s, 4*nlut*(2*pad_w_ + 1)*(2*pad_h_ + 1)) << std::endl;
          iss << format("  @%predi{0} mad.lo.s32 %pmask{0}, %maskh{0}, {1}, %pmask{0};", i + s, 4*nlut*(2*pad_w_ + 1)) << std::endl;
          iss << format("  @%predi{0} mad.lo.s32 %pmask{0}, %maskw{0}, {1}, %pmask{0};", i + s, 4*nlut) << std::endl;
      }

      for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
      for(size_t s = 0; s < vec_; s++){
          iss << format("  mul.lo.s32 %offId{0}, %offId{0}, %stride_d;", i + s) << std::endl;
          iss << format("  mul.lo.s32 %offIh{0}, %offIh{0}, %stride_h;", i + s) << std::endl;
          iss << format("  mul.lo.s32 %offIw{0}, %offIw{0}, %stride_w;", i + s) << std::endl;
          iss << format("  sub.s32 %offId{0}, %offId{0}, %pad_d;", i + s) << std::endl;
          iss << format("  sub.s32 %offIh{0}, %offIh{0}, %pad_h;", i + s) << std::endl;
          iss << format("  sub.s32 %offIw{0}, %offIw{0}, %pad_w;", i + s) << std::endl;
      }

      iss << format("  // I pointers") << std::endl;
      iss << format("  div.s32 %c, %idctrs, {};", Nfilt) << std::endl;
      iss << format("  rem.s32 %trs, %idctrs, {};", Nfilt) << std::endl;
      iss << format("  div.s32 %tr, %trs, {};", S_) << std::endl;
      iss << format("  rem.s32 %s, %trs, {};", S_) << std::endl;
      iss << format("  div.s32 %t, %tr, {};", R_) << std::endl;
      iss << format("  rem.s32 %r, %tr, {};", R_) << std::endl;
      iss << format("  mad.lo.s32 %offi, %t, %strideId, 0;") << std::endl;
      iss << format("  mad.lo.s32 %offi, %r, %strideIh, %offi;") << std::endl;
      iss << format("  mad.lo.s32 %offi, %s, %strideIw, %offi;") << std::endl;
      iss << format("  mad.lo.s32 %offi, %c, %strideIc, %offi;") << std::endl;
      if(gridz_ > 1)
        iss << format("  mad.lo.s32 %offi, %offc, %strideIc, %offi;", dtsize) << std::endl;
      for(size_t pqn = 0; pqn < cl0; pqn += vec_*bf_pqn)
      for(size_t s = 0; s < vec_; s++){
        iss << format("  mad.lo.s32 %offi{0}, %offId{0}, %strideId, %offi;", pqn + s) << std::endl;
        iss << format("  mad.lo.s32 %offi{0}, %offIh{0}, %strideIh, %offi{0};", pqn + s) << std::endl;
        iss << format("  mad.lo.s32 %offi{0}, %offIw{0}, %strideIw, %offi{0};", pqn + s) << std::endl;
        iss << format("  mad.lo.s32 %offi{0}, %offIn{0}, %strideIn, %offi{0};", pqn + s) << std::endl;
      }
      for(size_t pqn = 0; pqn < cl0; pqn += vec_*bf_pqn)
      for(size_t s = 0; s < vec_; s++)
        iss << format("  mad.wide.s32 %pi{0}, %offi{0}, 1, %pi;", pqn + s) << std::endl;
  };

  auto ptr_ldg_f = [&](){
      iss << format("  // F offsets", cl1, vec_) << std::endl;
      iss << format("  mul.lo.u32 %offFk, %bid1, {};", cl1) << std::endl;
      iss << format("  mad.lo.u32 %offFk, %bfid1, {}, %offFk;", vec_) << std::endl;

      iss << format("  // F pointers") << std::endl;
      iss << format("  mad.lo.u32 %offF, %offFk, %strideFk, 0;") << std::endl;
      iss << format("  mad.lo.u32 %offF, %idctrs, %strideFs, %offF;") << std::endl;
      if(gridz_ > 1)
        iss << format("  mad.lo.u32 %offF, %offc, %strideFc, %offF;", dtsize) << std::endl;
      iss << format("  mad.wide.u32 %pf, %offF, 1, %pf;") << std::endl;
      iss << format("  mul.lo.s32 %incf, {}, %strideFs;", block) << std::endl;
  };

  auto ldg_i = [&](){
    iss << format("  //Load I") << std::endl;
    for(size_t i = 0; i < cl0; i+=vec_*bf_pqn){
      for(size_t s = 0; s < vec_; ++s)
        iss << format("  mov.b{1} %rri{2}{3}, 0;", i + s, 8*dtsize, i, vs[s]) << std::endl;
      for(size_t s = 0; s < vec_; ++s)
        iss << format("  @%predi{0} ld.global.nc.{1} %rri{2}{3}, [%pi{4}];", i + s, dtype, i, vs[s], i + s, s*dtsize) << std::endl;
      iss << format("  st.shared{0}.{1} [%writei + {2}], %rri{3};", vv, dtype, i*dtsize, i) << std::endl;
    }
  };

  auto ldg_f = [&](bool safe){
    std::vector<std::string> preds(vec_, "%predloop");
    if(safe) for(size_t s = 0; s < vec_ ; ++s)
      preds[s] = format("%pred{}", s);
    iss << format("  //Load F") << std::endl;
    for(size_t j = 0; j < cl1; j+=vec_*bf_k){
      for(size_t s = 0; s < (safe?vec_:0); ++s)
        iss << format("  setp.lt.and.s32 %pred{}, %offFk{}, %Km{}, %predcrs;", s, j, s) << std::endl;
      for(size_t s = 0; s < (safe?vec_:0); ++s)
        iss << format("  mov.b{1} %rrf{2}{3}, 0;", preds[s], 8*dtsize, j, vs[s]) << std::endl;
      for(size_t s = 0; s < vec_; ++s)
        iss << format("  @{0} ld.global.nc.{1} %rrf{2}{3}, [%pf + {4}];", preds[s], dtype, j, vs[s], (j+s)*dtsize) << std::endl;
      iss << format("  st.shared{1}.{2} [%writef + {4}], %rrf{3};", preds[0], vv, dtype, j, j*dtsize) << std::endl;
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
      iss << format("  ld.shared{0}.{1} %r{2}{3}_{4}_{5}, [%read{2} + {6}];", vv, dtype, x, c, rx, ctrs%us_, rx*bs*dtsize + (c*bz_ + ctrs*zl)*cd_shared) << std::endl;
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
      iss << format("  fma.rn.{0} {1}, {2}, {3}, {1};", compute_dtype, ro, ri, rf) << std::endl;
    }
  };

  bool aligned = (M_*P_*Q_) % vec_ == 0 && gridz_==1;
  size_t inc = aligned?vec_:1;

  iss << ".const .b32 _masks[" << masks_.size() << "];" << std::endl;
  iss << ".const .b32 _LUT[" << cLUT.size() << "];" << std::endl;

  iss << std::endl;
  iss << ".func store_col(.reg .b64 %pc, .reg .b32 %Cs0";
  for(size_t i = 0; i < cs0_; i++) iss << format(", .reg .b32 %offc{}", i);
  for(size_t i = 0; i < cs0_ - inc; i+=inc) iss << format(", .reg .b32 %diffc{}", i);
  for(size_t i = 0 ; i < cs0_ ; i+=vec_) iss << format(", .reg {}.{} %rc{}", vv, dtype, i);
  iss << "){" << std::endl;
  iss << format("  .reg .pred %predc<{}>;", cs0_) << std::endl;

  iss << "// Predicates" << std::endl;
  for(size_t i = 0 ; i < cs0_ ; i+=inc)
    iss << format("  setp.lt.s32 %predc{0}, %offc{0}, %Cs0;", i) << std::endl;

  for(size_t i = 0 ; i < cs0_ ; i+=vec_)
  for(size_t s = 0; s < vec_; s+=inc){
    iss << format("  @%predc{} {}{}.{} [%pc], %rc{}{};", i + s, gridz_>1?"red.add":"st.global", aligned?vv:"", compute_dtype, i, aligned?"":vs[s]) << std::endl;
    if(i + s < cs0_ - inc)
    iss << format("  mad.wide.s32 %pc, %diffc{0}, 1, %pc;", i + s) << std::endl;
  }
  iss << "}" << std::endl;

  iss << ".entry " << name << "(.param ." << dtype << " _alpha, .param ." << dtype << " _beta," << std::endl
      << "            .param .b64 _pi, .param .b64 _pf, .param .b64 _po," << std::endl
      << "            .param .b32 _Npix, .param .b32 _Nfilt, .param .b32 _K, .param .b32 _C," << std::endl
      << "            .param .b32 _M, .param .b32 _P, .param .b32 _Q, .param .b32 _N," << std::endl
      << "            .param .b32 _D, .param .b32 _H, .param .b32 _W," << std::endl
      << "            .param .b32 _MPQ," << std::endl
      << "            .param .b32 _stride_d, .param .b32 _stride_h, .param .b32 _stride_w, .param .b32 _pad_d, .param .b32 _pad_h, .param .b32 _pad_w, " << std::endl
      << "            .param .b32 _strideIc, .param .b32 _strideId, .param .b32 _strideIh, .param .b32 _strideIw, .param .b32 _strideIn, " << std::endl
      << "            .param .b32 _strideOk, .param .b32 _strideOm, .param .b32 _strideOp, .param .b32 _strideOq, .param .b32 _strideOn, " << std::endl
      << "            .param .b32 _strideFk, .param .b32 _strideFc, .param .b32 _strideFs, " << std::endl
      << "            .param .b64 _bias," << std::endl
      << "            .param .b32 _bound)" << std::endl;
  iss << "{" << std::endl;
  // Predicates
  iss << "  .reg.pred %in_bounds, %predcrs, %predloop, %predz, %predlut;" << std::endl;
  iss << format("  .reg .pred %predk<{}>;", cs1_) << std::endl;
  iss << format("  .reg .pred %pred<{}>;", vec_) << std::endl;
  iss << format("  .reg .b32 %maskcrs;", vec_) << std::endl;
  for(size_t j = 0; j < cs1_ ; j++)
  for(size_t i = 0 ; i < cs0_ ; i++)
    iss << format("  .reg .pred %predc{}_{};", i, j) << std::endl;

  for(size_t i = 0; i < cl0; i += vec_*bf_pqn)
  for(size_t s = 0; s < vec_; s++){
    iss << format("  .reg .pred %predi{};", i + s) << std::endl;
    iss << format("  .reg .b32 %mask{};", i + s) << std::endl;
  }
  iss << format("  .reg .b32 %bound;") << std::endl;
  // Split-K
  iss << "  .reg .b32 %div, %rem, %offc;" << std::endl;
  // Special registers
  iss << "  .reg .b32 %bid0, %bid1, %bidz;" << std::endl;
  iss << "  .reg .b32 %id, %id1pqn, %idz, %id0, %id1;" << std::endl;
  iss << "  .reg .b32 %afid0, %bfid1, %idctrs;" << std::endl;
  // Look-up table
  iss << "  .reg .b32 %ctrs, %trs, %t, %c, %tr, %r, %s;" << std::endl;
  iss << "  .reg .b32 %nextctrs, %nexttrs, %nextt, %nextc, %nexttr, %nextr, %nexts;" << std::endl;
  iss << "  .reg .b32 %cdiff, %tdiff, %rdiff, %sdiff;" << std::endl;
  iss << "  .reg .b32 %maskf, %masks, %LUT, %writelut, %readlut, %inclut, %incmask, %pincmask;" << std::endl;
  for(size_t i = 0; i < cl0; i += vec_*bf_pqn)
  for(size_t s = 0; s < vec_; s++){
      iss << format("  .reg .b32 %maski{0}, %pmask{0};", i + s) << std::endl;
  }
  // Tensor shapes
  iss << "  .reg .b32 %pad_d, %pad_h, %pad_w, %stride_d, %stride_h, %stride_w;" << std::endl;
  iss << "  .reg .b32 %Npix, %K, %C, %Nfilt, %CTRS;" << std::endl;
  iss << "  .reg .b32 %D, %H, %W, %M, %P, %Q, %N, %MPQ, %mN, %mP, %mM, %mQ, %mMPQ;" << std::endl;
  // Scalars
  iss << format("  .reg .{} %alpha, %beta;", dtype) << std::endl;
  // Strides
  iss << format("  .reg .b32 %strideIc, %strideId, %strideIh, %strideIw, %strideIn;") << std::endl;
  iss << format("  .reg .b32 %strideFc, %strideFs, %strideFk;") << std::endl;
  iss << format("  .reg .b32 %strideOk, %strideOm, %strideOp, %strideOq, %strideOn;") << std::endl;
  // Pointers
  iss << format("  .reg .b64 %pi, %pf, %pc;") << std::endl;
  iss << format("  .reg .b32 %offi, %inci, %offF, %incf;") << std::endl;
  for(size_t i = 0; i < cl0; i += vec_*bf_pqn)
  for(size_t s = 0; s < vec_; s++)
     iss << format("  .reg .b32 %offi{};", i + s) << std::endl;
  for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
  for(size_t s = 0; s < vec_; s++)
      iss << format("  .reg .b64 %pi{};", i + s) << std::endl;
  for(size_t i = 0; i < cs0_; i++){
    iss << format("  .reg .b32 %offc_{};", i) << std::endl;
  }
  for(size_t i = 0; i < cs0_ - inc; i+=inc)
    iss << format("  .reg .b32 %diffc{};", i) << std::endl;

  for(size_t j = 0; j < cs1_; j++)
    iss << format("  .reg .b64 %pc{};", j) << std::endl;

  iss << "  .reg .b64 %bias;" << std::endl;
  for(size_t j = 0; j < cs1_; j++)
    iss << format("  .reg.b64 %pbias{};", j) << std::endl;
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
    iss << format("  .reg.u32 %dlo{0}, %hlo{0}, %wlo{0};", i + s) << std::endl;
    iss << format("  .reg.u32 %dhi{0}, %hhi{0}, %whi{0};", i + s) << std::endl;
    iss << format("  .reg.u32 %maskd{0}, %maskh{0}, %maskw{0};", i + s) << std::endl;
  }
  iss << format("  .reg .b32 %offFk, %offFrs, %offFc;") << std::endl;
  for(size_t j = 0; j < cl1; j+=vec_*bf_k)
    iss << format("  .reg.u32 %offFk{};", j) << std::endl;
  iss << format("  .reg .b32 %offc1, %offc0;") << std::endl;
  for(size_t i = 0 ; i < cl0 ; i+= bc0_*vec_){
  for(size_t s = 0 ; s < vec_; s++)
    iss << format("  .reg.u32 %offc0_{0}, %offOn{0}, %offOmpq{0};", i + s) << std::endl;
  }
  iss << format("  .reg.b32 %offc1_<{}>;", cs1_) << std::endl;
  // Bounds checking
  iss << format("  .reg.s32 %Npixm<{0}>, %Km<{0}>;", vec_) << std::endl;
  // LDG registers
  iss << format("  .reg .b32 %writei, %readi;") << std::endl;
  iss << format("  .reg .b32 %writef, %readf;") << std::endl;
  for(size_t pqn = 0; pqn < cl0; pqn+=vec_*bf_pqn)
    iss << format("  .reg {}.{} %rri{};", vv, dtype, pqn) << std::endl;
  for(size_t k = 0; k < cl1; k+=vec_*bf_k)
    iss << format("  .reg {}.{} %rrf{};", vv, dtype, k) << std::endl;
  // Tile registers
  iss << "  // For O tile" << std::endl;
  declare_register_tile('c', cs0_, cs1_, 1);
  iss << "  // For I tile" << std::endl;
  declare_register_tile('i', cs0_, us_, 1);
  iss << "  // For F tile" << std::endl;
  declare_register_tile('f', cs1_, us_, 1);
  // Bias
  iss << format("  .reg .{} %rbias<{}>;", dtype, cs1_) << std::endl;
  iss << format("  .reg .pred %has_bias;") << std::endl;

  iss << std::endl;
  iss << "  /* Initialize O */" << std::endl;
  for(size_t c = 0; c < zs_; c++)
  for(size_t i = 0 ; i < cs0_ ; i+=vec_)
  for(size_t k = 0; k < cs1_ ; ++k)
  for(size_t nn = 0; nn < vec_ ; ++nn)
      iss << format("  mov.{} %rc{}_{}_{}{}, 0x0;", dtype, c, i, k, vs[nn]) << std::endl;


  iss << "  // Load Param" << std::endl;
  iss << format("  ld.param.u64 %pc, [_po];") << std::endl;
  iss << format("  ld.param.u64 %pi, [_pi];") << std::endl;
  iss << format("  ld.param.u64 %pf, [_pf];") << std::endl;
  iss << "  ld.param.s32 %K, [_K];" << std::endl;
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

  iss << "  // Constant memory" << std::endl;
  iss << "  mov.b32 %LUT, _LUT;" << std::endl;
  iss << "  mov.b32 %masks, _masks;" << std::endl;

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
  iss << format("  mad.lo.u32 %id1pqn, %id1, {}, %id0;", bc0_) << std::endl;
  iss << format("  mad.lo.u32 %id, %id1pqn, {}, %idz;", bz_) << std::endl;

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
  iss << "  /* ---------------------------- */" << std::endl;
  iss << "  /* ------ Make lanes ------ */" << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  iss << std::endl;
  iss << format("  div.u32 %idctrs, %id, {};", bf_k) << std::endl;
  iss << format("  rem.u32 %bfid1, %id, {};", bf_k) << std::endl;
  iss << format("  mov.u32 %afid0, %bfid1;") << std::endl;
  iss << format("  shl.b32 %maskf, 0x1, %idctrs;") << std::endl;


  iss << std::endl;
  iss << format("  // STS Lanes") << std::endl;
  ptr_sts('i', cd_sharedi, addr_i, "afid0", "idctrs");
  ptr_sts('f', cd_sharedf, addr_f, "bfid1", "idctrs");

  iss << format("  // LDS lanes") << std::endl;
  ptr_lds('i', "id0", addr_i, cd_sharedi);
  ptr_lds('f', "id1", addr_f, cd_sharedf);

  iss << format("  // LDG lanes") << std::endl;
  ptr_ldg_i();
  ptr_ldg_f();

  iss << "  // Bounds" << std::endl;
  iss << format("  mul.lo.s32 %CTRS, %C, %Nfilt;") << std::endl;

  iss << format("  add.u32 %pincmask, %LUT, {};", 4*nlut) << std::endl;
  iss << format("  mad.lo.u32 %LUT, %idctrs, 4, %LUT;") << std::endl;

  iss << "bar.sync 0;" << std::endl;
  iss << std::endl;
  iss << " //First load" << std::endl;
  iss << format("  setp.lt.s32 %predcrs, %idctrs, %CTRS;") << std::endl;
  iss << format("  setp.gt.s32 %predloop, %CTRS, %bound;") << std::endl;
  iss << format("  @!%predcrs mov.b32 %maskf, 0x0;") << std::endl;
  for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
  for(size_t s = 0; s < vec_; s++)
    iss << format("  ld.const.b32 %maski{0}, [%pmask{0}];", i + s) << std::endl;;
  for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
  for(size_t s = 0; s < vec_ ; ++s)
    iss << format("  and.b32 %mask{}, %maskf, %maski{};", i + s, i + s) << std::endl;
  for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
  for(size_t s = 0; s < vec_ ; ++s)
    iss << format("  setp.ne.b32 %predi{}, %mask{}, 0x0;", i + s, i + s, i + s) << std::endl;
  ldg_i();
  iss << format("  @!%predloop bra LAST_ITER;") << std::endl;
  ldg_f(false);
  iss << std::endl;

  iss << " //Main loop" << std::endl;
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

  // Increment pointers
  iss << " // Increment filter pointers" << std::endl;
  iss << format("  mad.wide.u32 %pf, %incf, {}, %pf;", 1) << std::endl;
  iss << " // Increment image pointers" << std::endl;
  iss << format("  ld.const.b32 %inci, [%LUT];") << std::endl;
  iss << format("  ld.const.b32 %inclut, [%LUT + {}];", 4*nlut) << std::endl;
  iss << format("  add.s32 %LUT, %LUT, %inclut;") << std::endl;
  for(size_t pqn = 0; pqn < cl0; pqn += vec_*bf_pqn)
  for(size_t s = 0; s < vec_; s++)
    iss << format("  mad.wide.u32 %pi{}, %inci, {}, %pi{};", pqn + s, 1, pqn + s) << std::endl;

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
  iss << format("  ld.const.b32 %incmask, [%pincmask];") << std::endl;
  iss << format("  add.s32 %pincmask, %pincmask, %incmask;") << std::endl;
  for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
  for(size_t s = 0; s < vec_; s++)
      iss << format("  add.s32 %pmask{0}, %pmask{0}, %incmask;", i + s) << std::endl;
  for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
  for(size_t s = 0; s < vec_; s++)
      iss << format("  ld.const.b32 %maski{0}, [%pmask{0}];", i + s) << std::endl;
  iss << format("  @!%predcrs mov.b32 %maskf, 0x0;") << std::endl;
  for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
  for(size_t s = 0; s < vec_ ; ++s)
      iss << format("  and.b32 %mask{}, %maskf, %maski{};", i + s, i + s) << std::endl;
  for(size_t i = 0; i < cl0; i+=vec_*bf_pqn)
  for(size_t s = 0; s < vec_ ; ++s)
    iss << format("  setp.ne.b32 %predi{}, %mask{}, 0x0;", i + s, i + s, i + s) << std::endl;

  iss << " // Load" << std::endl;
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
  iss << format("  @%predloop bra.uni LOOP;") << std::endl;
  iss << "ENDLOOP:" << std::endl;

  //Reduce in registers
  for(size_t c = 1; c < zs_; ++c)
  for(size_t j = 0; j < cs1_; j ++)
  for(size_t i = 0; i < cs0_; i += vec_)
  for(size_t s = 0; s < vec_; s++)
    iss << format("  add.{0} %rc0_{2}_{3}{4}, %rc{1}_{2}_{3}{4}, %rc0_{2}_{3}{4};", compute_dtype, c, i, j, vs[s]) << std::endl;

  //Reduce in shared memory
  if(bz_>1)
  {
    size_t bc = nthreads/bz_;
    iss << ".reg .b32 %readk, %writek, %rid_mn, %rid_k;" << std::endl;
    iss << ".reg .pred %predc;" << std::endl;
    for(size_t ij = 0; ij < cl0*cl1; ij += bc)
      iss << format("  .reg .{0} %rrk{1}_0, %rrk{1}_1;", dtype, ij) << std::endl;

    iss << format("  mad.lo.u32 %writek, %idz, {}, %shared;", cl0*cl1*dtsize) << std::endl;
    iss << format("  mad.lo.u32 %writek, %id1pqn, {}, %writek;", cs0_*cs1_*dtsize) << std::endl;

    iss << "  bar.sync 0;" << std::endl;
    for(size_t j = 0; j < cs1_; j ++)
    for(size_t i = 0; i < cs0_; i += vec_)
    for(size_t s = 0; s < vec_; s++){
      size_t mn = i + j*cs0_;
      iss << format("  st.shared.{} [%writek + {}], %rc0_{}_{}{};", dtype, (mn + s)*dtsize, i, j, vs[s]) << std::endl;
    }
    iss << "  bar.sync 0;" << std::endl;

    iss << std::endl;
    iss << format("  div.u32 %rid_mn, %id, {};", bz_) << std::endl;
    iss << format("  rem.u32 %rid_k, %id, {};", bz_) << std::endl;
    iss << format("  mad.lo.u32 %readk, %rid_k, {}, %shared;", cl0*cl1*dtsize) << std::endl;
    iss << format("  mad.lo.u32 %readk, %rid_mn, {}, %readk;", dtsize) << std::endl;
    for(size_t c = bz_/2; c > 0; c /=2){
      iss << format("  setp.lt.u32 %predc, %rid_k, {};", c) << std::endl;
      for(size_t ij = 0; ij < cl0*cl1; ij += bc){
        iss << format("  @%predc ld.shared.{} %rrk{}_0, [%readk + {}];", dtype, ij, (ij)*dtsize) << std::endl;
        iss << format("  @%predc ld.shared.{} %rrk{}_1, [%readk + {}];", dtype, ij, (ij + c*cl0*cl1)*dtsize) << std::endl;
        iss << format("  @%predc add.{0} %rrk{1}_0, %rrk{1}_0, %rrk{1}_1;", compute_dtype, ij) << std::endl;
        iss << format("  @%predc st.shared.{} [%readk + {}], %rrk{}_0;", dtype, ij*dtsize, ij) << std::endl;
      }
      iss << "  bar.sync 0;" << std::endl;
    }

    iss << format("  mad.lo.u32 %readk, %id1pqn, {}, %shared;", cs0_*cs1_*dtsize) << std::endl;
    for(size_t j = 0; j < cs1_; j ++)
    for(size_t i = 0; i < cs0_; i += vec_)
    for(size_t s = 0; s < vec_; s++){
      iss << format("  ld.shared.{} %rc0_{}_{}{}, [%readk + {}];", dtype, i, j, vs[s], ((i+s) + j*cs0_)*dtsize) << std::endl;
    }
  }

  iss << std::endl;
  iss << "/* Scale */" << std::endl;
  iss << format("  ld.param.{} %alpha, [_alpha];", dtype) << std::endl;
  for(size_t j = 0; j < cs1_ ; j++)
  for(size_t i = 0 ; i < cs0_ ; i+=vec_)
  for(size_t s = 0; s < vec_; ++s)
    iss << format("  mul.{0} %rc0_{1}_{2}{3}, %rc0_{1}_{2}{3}, %alpha;", compute_dtype, i, j, vs[s]) << std::endl;


  iss << "/* Offsets */" << std::endl;
  iss << format("  mov.s32 %bid0, %ctaid.x;") << std::endl;
  iss << format("  mov.s32 %bid1, %ctaid.y;") << std::endl;
  iss << format("  mov.s32 %id0, %tid.x;") << std::endl;
  iss << format("  mov.s32 %id1, %tid.y;") << std::endl;

  iss << format("  mad.lo.s32 %offc1, %bid1, {}, 0;", cl1) << std::endl;
  iss << format("  mad.lo.s32  %offc1, %id1, {}, %offc1;", vec_) << std::endl;
  for(size_t j = 0; j < cs1_; j+= vec_)
  for(size_t s = 0; s < vec_; s++)
    iss << format("  add.u32 %offc1_{}, {}, %offc1;", j + s, j*bc1_ + s) << std::endl;

  iss << std::endl;
  iss << format("  mad.lo.s32 %offc0, %bid0, {}, 0;", cl0) << std::endl;
  iss << format("  mad.lo.s32  %offc0, %id0, {}, %offc0;", vec_) << std::endl;
  for(size_t i = 0 ; i < cl0 ; i+= bc0_*vec_)
    iss << format("  add.s32 %offc0_{0}, %offc0, {0};", i) << std::endl;


  iss << format("  mul.lo.s32 %mMPQ, %MPQ, -1;") << std::endl;
  for(size_t i = 0; i < cl0; i+=vec_*bc0_)
  for(size_t s = 0; s < vec_; s++){
    iss << format("  add.u32 %offc0_{0}, %offc0, {0};", i + s) << std::endl;
    iss << format("  div.u32 %offOn{0}, %offc0_{0}, %MPQ;", i + s) << std::endl;
    iss << format("  mad.lo.s32 %offOmpq{0}, %mMPQ, %offOn{0}, %offc0_{0};", i + s) << std::endl;
  }
  for(size_t i = 0; i < cs0_; i+=vec_)
  for(size_t s = 0; s < vec_; s++){
    iss << format("  mad.lo.s32 %offc_{0}, %offOn{1}, %strideOn, 0;", i + s, i*bc0_ + s) << std::endl;
    iss << format("  mad.lo.s32 %offc_{0}, %offOmpq{1}, %strideOq, %offc_{0};", i + s, i*bc0_ + s) << std::endl;
  }

  for(size_t j = 0; j < cs1_; j++){
    iss << format("  mad.wide.s32 %pc{0}, %offc1_{0}, %strideOk, %pc;", j) << std::endl;
    iss << format("  mad.wide.s32 %pc{0}, %offc_0, 1, %pc{0};", j) << std::endl;
  }

  iss << "// Pointer deltas" << std::endl;
  for(size_t i = 0; i < cs0_ - inc; i+=inc)
    iss << format("  sub.s32 %diffc{0}, %offc_{1}, %offc_{0};", i, i + inc) << std::endl;

  iss << "  // Predicates" << std::endl;
  iss << format("  setp.eq.s32 %predz, %idz, 0;") << std::endl;
  for(size_t j = 0; j < cs1_; j++)
    iss << format("  setp.lt.and.s32 %predk{0}, %offc1_{0}, %K, %predz;", j) << std::endl;

  iss << "  // Write back" << std::endl;
  for(size_t j = 0; j < cs1_; j++){
    iss << format("  @%predk{0} call.uni store_col, (%pc{0}, %Npix", j);
    for(size_t i = 0 ; i < cs0_ ; i+=vec_) for(size_t s = 0; s < vec_; s++) iss << format(", %offc0_{}", i*bc0_ + s);
    for(size_t i = 0; i < cs0_ - inc; i+=inc) iss << format(", %diffc{}", i);
    for(size_t i = 0 ; i < cs0_ ; i+=vec_) iss << format(", %rc0_{}_{}", i, j);
    iss << ");" << std::endl;
  }


  iss << std::endl;
  iss << "}" << std::endl;

  return iss.str();
}

void Conv::enqueue(driver::Kernel& kernel, driver::Stream& stream, scalar const & alpha, driver::Buffer const & I, driver::Buffer const & F, scalar const & beta, driver::Buffer& O){
  // Data-type size
  int32_t dtsize = size_of(dtype_);

  // I strides
  int32_t strideIw = dtsize;
  int32_t strideIh = W_*strideIw;
  int32_t strideId = H_*strideIh;
  int32_t strideIc = D_*strideId;
  int32_t strideIn = C_*strideIc;

  // F strides
  int32_t strideFk = dtsize;
  int32_t strideFs = K_*strideFk;
  int32_t strideFr = S_*strideFs;
  int32_t strideFt = R_*strideFr;
  int32_t strideFc = T_*strideFt;
  // O strides
  int32_t strideOq = dtsize;
  int32_t strideOp = Q_*strideOq;
  int32_t strideOm = P_*strideOp;
  int32_t strideOk = M_*strideOm;
  int32_t strideOn = K_*strideOk;

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
  DType alpha_dtype = (dtype_==DOUBLE_TYPE)?DOUBLE_TYPE:FLOAT_TYPE;
  kernel.setArg(0, size_of(alpha_dtype), alpha.data());
  kernel.setArg(1, size_of(alpha_dtype), beta.data());
  kernel.setArg(2, I);
  kernel.setArg(3, F);
  kernel.setArg(4, O);
  kernel.setArg(5, Npix);
  kernel.setArg(6, Nfilt);
  kernel.setArg(7, K_);
  kernel.setArg(8, C_);
  kernel.setArg(9, M_);
  kernel.setArg(10, P_);
  kernel.setArg(11, Q_);
  kernel.setArg(12, N_);
  kernel.setArg(13, D_);
  kernel.setArg(14, H_);
  kernel.setArg(15, W_);
  kernel.setArg(16, MPQ);

  kernel.setArg(17, stride_d_);
  kernel.setArg(18, stride_h_);
  kernel.setArg(19, stride_w_);
  kernel.setArg(20, pad_d_);
  kernel.setArg(21, pad_h_);
  kernel.setArg(22, pad_w_);
  kernel.setArg(23, strideIc);
  kernel.setArg(24, strideId);
  kernel.setArg(25, strideIh);
  kernel.setArg(26, strideIw);
  kernel.setArg(27, strideIn);
  // O strides
  kernel.setArg(28, strideOk);
  kernel.setArg(29, strideOm);
  kernel.setArg(30, strideOp);
  kernel.setArg(31, strideOq);
  kernel.setArg(32, strideOn);
  // F strides
  kernel.setArg(33, strideFk);
  kernel.setArg(34, strideFc);
  kernel.setArg(35, strideFs);
  // Bia
  kernel.setArg(36, (uint64_t)0);
  // Misc.
  kernel.setArg(37, bound);
  if(gridz_>1)
    O.set_zero(stream, N_*K_*M_*P_*Q_*dtsize);
  stream.enqueue(kernel, {grid0, grid1, gridz_}, {bc0_, bc1_, bz_});
}

}
}
