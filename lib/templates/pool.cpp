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

#include "isaac/templates/common.hpp"
#include "isaac/templates/pool.h"
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

const std::string Pool::id = "pool";
const size_t Pool::Nshapes = 3;
const size_t Pool::Ntune = 4;
const size_t Pool::Nparams = Nshapes + Ntune;

Pool::Pool(DType in_dtype, DType out_dtype, PoolType pool_type,
           param_t C, param_t D, param_t H, param_t W, param_t N, param_t M, param_t P, param_t Q, param_t T, param_t R, param_t S,
           param_t pad_d, param_t pad_h, param_t pad_w,
           param_t stride_d, param_t stride_h, param_t stride_w,
           param_t vec, param_t bc0, param_t cs0, param_t):
    in_dtype_(in_dtype), out_dtype_(out_dtype), pool_type_(pool_type),
    D_(D), H_(H), W_(W), N_(N), M_(M), P_(P), Q_(Q), T_(T), R_(R), S_(S),
    pad_d_(pad_d), pad_h_(pad_h), pad_w_(pad_w),
    stride_d_(stride_d), stride_h_(stride_h), stride_w_(stride_w),
    vec_(vec), bc0_(bc0), cs0_(cs0), u_(1)
{
  // Packed inputs
  size_t vect_c_in = (in_dtype_==INT8X4_TYPE)?4:1;
  if(C % vect_c_in != 0)
    throw std::runtime_error("Number of input channels must be a multiple of VECT_C");
  Cin_ =  C/vect_c_in;

  // Packed outputs
  size_t vect_c_out = (out_dtype_==INT8X4_TYPE)?4:1;
  if(C % vect_c_out != 0)
    throw std::runtime_error("Number of input channels must be a multiple of VECT_C");
  Cout_ =  C/vect_c_out;

  size_t Nfilt = T_*R_*S_;
  size_t nlut = Nfilt;

  // Data-type size
  int32_t dtsize = size_of(in_dtype);

  // I strides
  int32_t strideIw = dtsize;
  int32_t strideIh = W_*strideIw;
  int32_t strideId = H_*strideIh;
  int32_t strideIc = D_*strideId;

  // Init constant memory
  cLUT.resize(nlut);
  masks_.resize(nlut + (2*pad_h+1)*(2*pad_w+1)*(2*pad_d+1)*nlut);
  init_constant_memory(cLUT, masks_, nlut, strideIc, strideIw, strideIh, strideId);

}


// Constant memory
void Pool::init_constant_memory(std::vector<int32_t> &delta, std::vector<uint32_t> &masks, size_t nlut, int32_t, int32_t strideIw, int32_t strideIh, int32_t strideId){
   size_t block = u_;
   size_t Nfilt = T_*R_*S_;

   auto unpack = [&](int32_t trs){
       int32_t tr = trs / S_;
       int32_t s = trs - tr*S_;
       int32_t t = tr / R_;
       int32_t r = tr - t*R_;
       return std::make_tuple(t, r, s);
   };

   /* Deltas */
   for(size_t i = 0; i < nlut; ++i){
       int32_t t, r, s, nextt, nextr, nexts;
       std::tie(t, r, s) = unpack(i);
       std::tie(nextt, nextr, nexts) = unpack(i + block);
       int32_t tdiff = nextt - t;
       int32_t rdiff = nextr - r;
       int32_t sdiff = nexts - s;
       delta[i] = sdiff*strideIw + rdiff*strideIh + tdiff*strideId;
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

// Validity
void Pool::check_valid(driver::Device const & device, size_t M, param_t* params, uint8_t* valid){
  std::array<int, Nparams> x{0};
  for(size_t m = 0; m < M; ++ m){
    //Parameters
    for(size_t i = 0; i < x.size(); ++i)
      x[i] = params[m*x.size() + i];
    DType dtype = (DType)(x[0]);
    param_t vec = x[3], bc0 = x[4], cs0 = x[5];
    //Test
    bool is_valid =  (bc0 <= device.max_block_dim()[0])
                  && (bc0 <= device.max_threads_per_block())
                  && (vec*size_of(dtype) <= 16)
                  && (cs0 % vec == 0);
    valid[m] = is_valid;
  }
}


// Execution
std::string Pool::dump(driver::Device const &, std::string const & name){
    std::stringstream iss;
    size_t cl0 = cs0_*bc0_;
    size_t Nfilt = T_*R_*S_;
    size_t nlut = Nfilt;
    size_t in_dtsize = size_of(in_dtype_);
    std::string in_word_type = format("b{}", 8*in_dtsize);

    size_t vect_c_in = (in_dtype_==INT8X4_TYPE)?4:1;
    size_t vect_c_out = (out_dtype_==INT8X4_TYPE)?4:1;

    std::string vv = vec_>1?format(".v{}", vec_):"";
    const char* vs[] = {".x", ".y", ".z", ".w"};
    if(vec_==1)
      vs[0] = "";

    std::string neutral_element;
    if(pool_type_ == MaxPool)
        neutral_element = (in_dtype_==INT8X4_TYPE)?"0x80000000":"0xff800000";
    if(pool_type_ == AvgPool)
        neutral_element = "0x0";

    size_t vect_in = (in_dtype_==INT8X4_TYPE)?4:1;

    auto ptr_ldg_i = [&](){
        iss << format("  // I offsets") << std::endl;
        iss << format("  mul.lo.s32 %mM, %M, -1;") << std::endl;
        iss << format("  mul.lo.s32 %mC, %C, -1;") << std::endl;
        iss << format("  mul.lo.s32 %mP, %P, -1;") << std::endl;
        iss << format("  mul.lo.s32 %mQ, %Q, -1;") << std::endl;

        iss << format("  mul.lo.u32 %offIncdhw, %bid0, {};", cl0) << std::endl;
        iss << format("  mad.lo.u32 %offIncdhw, %id0, {}, %offIncdhw;", vec_) << std::endl;
        for(size_t i = 0; i < cl0; i+=vec_*bc0_)
        for(size_t s = 0; s < vec_; s++){
          iss << format("  add.u32 %offIncdhw{0}, %offIncdhw, {0};", i + s) << std::endl;
          iss << format("  div.u32 %offIncdh{0}, %offIncdhw{0}, %Q;", i + s) << std::endl;
          iss << format("  mad.lo.s32 %offIw{0}, %mQ, %offIncdh{0}, %offIncdhw{0};", i + s) << std::endl;
          iss << format("  div.u32 %offIncd{0}, %offIncdh{0}, %P;", i + s) << std::endl;
          iss << format("  mad.lo.s32 %offIh{0}, %mP, %offIncd{0}, %offIncdh{0};", i + s) << std::endl;
          iss << format("  div.u32 %offInc{0}, %offIncd{0}, %M;", i + s) << std::endl;
          iss << format("  mad.lo.s32 %offId{0}, %mM, %offInc{0}, %offIncd{0};", i + s) << std::endl;
          iss << format("  div.u32 %offIn{0}, %offInc{0}, %C;", i + s) << std::endl;
          iss << format("  mad.lo.s32 %offIc{0}, %mC, %offIn{0}, %offInc{0};", i + s) << std::endl;
        }

        for(size_t i = 0; i < cl0; i+=vec_*bc0_)
        for(size_t s = 0; s < vec_; s++)
          iss << format("   setp.lt.u32 %predi{0}, %offIncdhw{0}, %Npix;", i + s) << std::endl;

        for(size_t i = 0; i < cl0; i+=vec_*bc0_)
        for(size_t s = 0; s < vec_; s++){
            iss << format("  mul.lo.s32 %offId{0}, %offId{0}, %stride_d;", i + s) << std::endl;
            iss << format("  mul.lo.s32 %offIh{0}, %offIh{0}, %stride_h;", i + s) << std::endl;
            iss << format("  mul.lo.s32 %offIw{0}, %offIw{0}, %stride_w;", i + s) << std::endl;
            iss << format("  sub.s32 %offId{0}, %offId{0}, %pad_d;", i + s) << std::endl;
            iss << format("  sub.s32 %offIh{0}, %offIh{0}, %pad_h;", i + s) << std::endl;
            iss << format("  sub.s32 %offIw{0}, %offIw{0}, %pad_w;", i + s) << std::endl;
        }

        iss << format("  sub .s32 %Dmpad, {}, %D;", T_) << std::endl;
        iss << format("  sub .s32 %Hmpad, {}, %H;", R_) << std::endl;
        iss << format("  sub .s32 %Wmpad, {}, %W;", S_) << std::endl;
        for(size_t i = 0; i < cl0; i+=vec_*bc0_)
        for(size_t s = 0; s < vec_; s++){
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
        }


        iss << format("  // I deltas pointers") << std::endl;
        iss << "  mov.b32 %p_delta, _LUT;" << std::endl;

        iss << format("  // I pointers") << std::endl;
        for(size_t pqn = 0; pqn < cl0; pqn += vec_*bc0_)
        for(size_t s = 0; s < vec_; s++){
          iss << format("  mad.lo.s32 %offi{0}, %offId{0}, %strideId, 0;", pqn + s) << std::endl;
          iss << format("  mad.lo.s32 %offi{0}, %offIh{0}, %strideIh, %offi{0};", pqn + s) << std::endl;
          iss << format("  mad.lo.s32 %offi{0}, %offIw{0}, %strideIw, %offi{0};", pqn + s) << std::endl;
          iss << format("  mad.lo.s32 %offi{0}, %offIc{0}, %strideIc, %offi{0};", pqn + s) << std::endl;
          iss << format("  mad.lo.s32 %offi{0}, %offIn{0}, %strideIn, %offi{0};", pqn + s) << std::endl;
          iss << format("  mad.wide.s32 %pi{0}, %offi{0}, 1, %pi;", pqn + s) << std::endl;
        }
    };


    iss << ".const .b32 _masks[" << masks_.size() << "];" << std::endl;
    iss << ".const .b32 _LUT[" << cLUT.size() << "];" << std::endl;

    iss << ".entry " << name << "(" << std::endl
        << "            .param .b64 _pi, .param .b64 _pc," << std::endl
        << "            .param .b32 _Npix, .param .b32 _Nfilt," << std::endl
        << "            .param .b32 _M, .param .b32 _P, .param .b32 _Q, .param .b32 _K," << std::endl
        << "            .param .b32 _D, .param .b32 _H, .param .b32 _W, .param .b32 _C," << std::endl
        << "            .param .b32 _stride_d, .param .b32 _stride_h, .param .b32 _stride_w, .param .b32 _pad_d, .param .b32 _pad_h, .param .b32 _pad_w, " << std::endl
        << "            .param .b32 _strideIc, .param .b32 _strideId, .param .b32 _strideIh, .param .b32 _strideIw, .param .b32 _strideIn, " << std::endl
        << "            .param .b32 _strideOk, .param .b32 _strideOm, .param .b32 _strideOp, .param .b32 _strideOq, .param .b32 _strideOn, " << std::endl
        << "            .param .b32 _i_scale, .param .b32 _o_scale)" << std::endl;
    iss << "{" << std::endl;
    iss << "  .reg .pred %in_bounds, %predloop;" << std::endl;
    iss << "  .reg .b32 %id, %id0, %bid0;" << std::endl;
    iss << "  .reg .b32 %trs, %tr, %t, %r, %s, %nexttrs, %nexttr, %nextt, %nextr, %nexts, %tdiff, %rdiff, %sdiff;" << std::endl;
    iss << "  .reg .b32 %Npix, %fNfilt, %Nfilt, %Dmpad, %Hmpad, %Wmpad, %D, %H, %W, %C, %M, %P, %Q, %K, %mM, %mP, %mQ, %mC, %mK;" << std::endl;
    iss << "  .reg .b32 %pad_d, %pad_h, %pad_w;" << std::endl;
    iss << "  .reg .b32 %stride_d, %stride_h, %stride_w;" << std::endl;
    iss << "  .reg .b32 %strideIc, %strideId, %strideIh, %strideIw, %strideIn;" << std::endl;
    iss << "  .reg .b32 %strideOk, %strideOm, %strideOp, %strideOq, %strideOn;" << std::endl;
    iss << "  .reg .b32 %inci;" << std::endl;
    iss << "  .reg .b64 %pi, %pc;" << std::endl;
    iss << "  .reg .b32 %acc, %icvt<4>, %ccvt<4>;" << std::endl;
    iss << "  .reg .b8 %b8_icvt<4>, %b8_ccvt<4>;" << std::endl;
    for(size_t i = 0; i < cl0; i+=vec_*bc0_)
    for(size_t s = 0; s < vec_; s++)
      iss << format("  .reg .b64 %pi{0};", i + s) << std::endl;
    iss << "  .reg .b32 %offIncdhw;" << std::endl;
    for(size_t i = 0; i < cl0; i+=vec_*bc0_)
    for(size_t s = 0; s < vec_; s++){
      iss << format("  .reg .b32 %offIncdhw{0}, %offIncdh{0}, %offIncd{0}, %offInc{0}, %offIw{0}, %offIcd{0}, %offIh{0}, %offIc{0}, %offId{0}, %offIn{0}, %dlo{0}, %dhi{0}, %hlo{0}, %hhi{0}, %wlo{0}, %whi{0}, %maskd{0}, %maskh{0}, %maskw{0};", i + s) << std::endl;
    }
    iss << "  .reg .b32 %TRS;" << std::endl;
    for(size_t j = 0; j < vect_in; j++)
    for(size_t i = 0; i < cl0; i+=vec_*bc0_)
      iss << format("  .reg {0}.b32 %rc{1}_{2};", vv, i, j) << std::endl;

    for(size_t i = 0; i < cl0; i+=vec_*bc0_)
      iss << format("  .reg {0}.b32 %rri{1};", vv, i) << std::endl;

    for(size_t j = 0; j < vect_c_in / vect_c_out; j++)
    for(size_t s = 0; s < vec_; s++)
    for(size_t i = 0; i < cl0; i+=vec_*bc0_)
      iss << format("  .reg .b64 %pc{0}_{1};", i + s, j) << std::endl;

    iss << "  .reg .b32 %offc0;" << std::endl;
    for(size_t s = 0; s < vec_; s++)
    for(size_t i = 0; i < cl0; i+=vec_*bc0_)
        iss << format("  .reg .b32 %offc0_{0};", i + s) << std::endl;

    for(size_t i = 0; i < cl0; i += vec_*bc0_)
    for(size_t s = 0; s < vec_; s++){
        iss << format("  .reg .pred %predi{0}, %pred{0};", i + s) << std::endl;
        iss << format("  .reg .b32 %mask{};", i + s) << std::endl;
    }
    iss << "  .reg .b32 %offi, %maskf, %masks, %inc_i, %p_delta, %writelut, %readlut, %inc_delta, %p_inc_delta, %inc_mask, %p_inc_mask;" << std::endl;
    for(size_t i = 0; i < cl0; i += vec_*bc0_)
    for(size_t s = 0; s < vec_; s++)
        iss << format("  .reg .b32 %offi{0}, %p_delta{0}, %inc_i{0}, %maski{0}, %p_mask{0};", i + s) << std::endl;
    iss << "  .reg .b32 %i_scale, %o_scale;" << std::endl;

    iss << "  // Initialize C" << std::endl;
    for(size_t j = 0; j < vect_in; j++)
    for(size_t i = 0; i < cl0; i += vec_*bc0_)
    for(size_t s = 0; s < vec_; s++)
      iss << format("  mov.b32 %rc{0}_{1}{2}, {3};", i, j, vs[s], neutral_element) << std::endl;

    iss << std::endl;
    iss << "  ld.param.u64 %pc, [_pc];" << std::endl;
    iss << "  ld.param.u64 %pi, [_pi];" << std::endl;
    iss << "  ld.param.s32 %Npix, [_Npix];" << std::endl;
    iss << "  ld.param.s32 %Nfilt, [_Nfilt];" << std::endl;
    iss << std::endl;
    iss << "  // Output shapes" << std::endl;
    iss << "  ld.param.s32 %M, [_M];" << std::endl;
    iss << "  ld.param.s32 %P, [_P];" << std::endl;
    iss << "  ld.param.s32 %Q, [_Q];" << std::endl;
    iss << "  ld.param.s32 %K, [_K];" << std::endl;

    iss << "  // Input" << std::endl;
    iss << "  ld.param.s32 %D, [_D];" << std::endl;
    iss << "  ld.param.s32 %H, [_H];" << std::endl;
    iss << "  ld.param.s32 %W, [_W];" << std::endl;
    iss << "  ld.param.s32 %C, [_C];" << std::endl;

    iss << std::endl;
    iss << "  // Padding/Striding" << std::endl;
    iss << "  ld.param.s32 %pad_d, [_pad_d];" << std::endl;
    iss << "  ld.param.s32 %pad_h, [_pad_h];" << std::endl;
    iss << "  ld.param.s32 %pad_w, [_pad_w];" << std::endl;
    iss << "  ld.param.s32 %stride_d, [_stride_d];" << std::endl;
    iss << "  ld.param.s32 %stride_h, [_stride_h];" << std::endl;
    iss << "  ld.param.s32 %stride_w, [_stride_w];" << std::endl;
    iss << std::endl;
    iss << "  // Input strides" << std::endl;
    iss << "  ld.param.s32 %strideIc, [_strideIc];" << std::endl;
    iss << "  ld.param.s32 %strideId, [_strideId];" << std::endl;
    iss << "  ld.param.s32 %strideIh, [_strideIh];" << std::endl;
    iss << "  ld.param.s32 %strideIw, [_strideIw];" << std::endl;
    iss << "  ld.param.s32 %strideIn, [_strideIn];" << std::endl;
    iss << std::endl;
    iss << "  // Output strides" << std::endl;
    iss << "  ld.param.s32 %strideOk, [_strideOk];" << std::endl;
    iss << "  ld.param.s32 %strideOm, [_strideOm];" << std::endl;
    iss << "  ld.param.s32 %strideOp, [_strideOp];" << std::endl;
    iss << "  ld.param.s32 %strideOq, [_strideOq];" << std::endl;
    iss << "  ld.param.s32 %strideOn, [_strideOn];" << std::endl;
    iss << std::endl;
    iss << "  // Scales" << std::endl;
    iss << "  ld.param.f32 %i_scale, [_i_scale];" << std::endl;
    iss << "  ld.param.f32 %o_scale, [_o_scale];" << std::endl;

    iss << std::endl;
    iss << "  // Special registers" << std::endl;
    iss << "  mov.u32 %id0, %tid.x;" << std::endl;
    iss << "  mov.u32 %bid0, %ctaid.x;" << std::endl;
    iss << "  mov.u32 %id, %id0;" << std::endl;

    iss << std::endl;
    iss << "  /* LDG Lanes */" << std::endl;
    ptr_ldg_i();

    iss << std::endl;
    iss << "  /* ---------------------------- */" << std::endl;
    iss << "  /* ------ Accumulate ---------- */" << std::endl;
    iss << "  /* ---------------------------- */" << std::endl;
    iss << "  mov.u32 %TRS, %Nfilt;" << std::endl;
    iss << "  setp.gt.u32 %predloop, %TRS, 0;" << std::endl;
    iss << "  mov.b32 %maskf, 0x1;" << std::endl;

    iss << std::endl;
    iss << "  // Main loop" << std::endl;
    iss << "  @!%predloop bra ENDLOOP;" << std::endl;
    iss << "LOOP:" << std::endl;
    iss << std::endl;
    iss << "  // Compute predicates" << std::endl;
    if(pad_d_ == 0 && pad_h_ == 0 && pad_w_ == 0 && stride_d_ == 1 && stride_h_ == 1 && stride_w_ == 1){
        for(size_t i = 0; i < cl0; i+=vec_*bc0_)
        for(size_t s = 0; s < vec_; s++)
          iss << format("  and.pred %predi{}, %predi{}, %predloop;", i + s, i + s) << std::endl;
    }
    else{
      iss << format("  @!%predloop mov.b32 %maskf, 0x0;") << std::endl;
      for(size_t i = 0; i < cl0; i+=vec_*bc0_)
      for(size_t s = 0; s < vec_; s++){
          iss << format("  ld.const.b32 %maski{0}, [%p_mask{0}];", i + s) << std::endl;
          iss << format("  and.b32 %mask{}, %maskf, %maski{};", i + s, i + s) << std::endl;
          iss << format("  setp.ne.b32 %predi{}, %mask{}, 0x0;", i + s, i + s, i + s) << std::endl;
          iss << format("  add.s32 %p_mask{0}, %p_mask{0}, 4;", i + s) << std::endl;
      }
    }

    iss << std::endl;
    iss << "  // Load" << std::endl;
    for(size_t i = 0; i < cl0; i += vec_*bc0_)
    for(size_t s = 0; s < vec_; s++){
      iss << format("  @!%predi{0} mov.{1} %rri{2}{3}, 0x0;", i + s, in_word_type, i, vs[s])  << std::endl;
      iss << format("  @%predi{0} ld.global.cg.{1} %rri{2}{3}, [%pi{0}];", i + s, in_word_type, i, vs[s])  << std::endl;
    }

    iss << std::endl;
    iss << "  // Pool" << std::endl;
    for(size_t j = 0; j < vect_in; j++)
    for(size_t i = 0; i < cl0; i += vec_*bc0_)
    for(size_t s = 0; s < vec_; s++){
      if(in_dtype_ == FLOAT_TYPE){
        if(pool_type_ == MaxPool)
            iss << format("  max.f32 %rc{0}_{1}{2}, %rri{0}{2}, %rc{0}_{1}{2};", i, j, vs[s]) << std::endl;
        if(pool_type_ == AvgPool)
            iss << format("  add.f32 %rc{0}_{1}{2}, %rri{0}{2}, %rc{0}_{1}{2};", i, j, vs[s]) << std::endl;
      }
      if(in_dtype_ == INT8X4_TYPE){
        iss << format("  and.b32 %icvt{0}, %rri{1}{2}, {3};", j, i, vs[s], (0xFF) << 8*j) << std::endl;
        iss << format("  shr.b32 %icvt{0}, %icvt{0}, {1};", j, 8*j) << std::endl;
        iss << format("  cvt.s32.s8 %icvt{0}, %icvt{0};", j) << std::endl;
        if(pool_type_ == MaxPool)
          iss << format("  max.s32 %rc{0}_{1}{2}, %rc{0}_{1}{2}, %icvt{1};", i, j, vs[s]) << std::endl;
        if(pool_type_ == AvgPool)
          iss << format("  add.s32 %rc{0}_{1}{2}, %rc{0}_{1}{2}, %icvt{1};", i, j, vs[s]) << std::endl;
      }
    }


    iss << std::endl;
    iss << "  // Increment image pointers " << std::endl;
    iss << format("  ld.const.b32 %inc_i, [%p_delta];") << std::endl;
    iss << format("  add.s32 %p_delta, %p_delta, 4;") << std::endl;
    for(size_t pqn = 0; pqn < cl0; pqn += vec_*bc0_)
    for(size_t s = 0; s < vec_; s++)
      iss << format("  mad.wide.s32 %pi{0}, %inc_i, 1, %pi{0};", pqn + s) << std::endl;

    iss << "  // Loop back" << std::endl;
    iss << format("  sub.s32 %TRS, %TRS, 1;") << std::endl;
    iss << "  setp.gt.u32 %predloop, %TRS, 0;" << std::endl;
    iss << "  @%predloop bra.uni LOOP;" << std::endl;

    iss << std::endl;
    iss << "ENDLOOP:" << std::endl;


    if(in_dtype_ == INT8X4_TYPE){
      for(size_t j = 0; j < vect_in; j++)
      for(size_t i = 0; i < cl0; i += vec_*bc0_)
      for(size_t s = 0; s < vec_; s++)
        iss << format("  cvt.rn.f32.s32 %rc{0}_{1}{2}, %rc{0}_{1}{2};", i, j, vs[s]) << std::endl;
    }

    if(pool_type_ == AvgPool){
      iss << std::endl;
      iss << "  /* Divide by window size (compute average) */" << std::endl;
      iss << "  cvt.rn.f32.s32 %fNfilt, %Nfilt;" << std::endl;
      for(size_t j = 0; j < vect_in; j++)
      for(size_t i = 0; i < cl0; i += vec_*bc0_)
      for(size_t s = 0; s < vec_; s++)
        iss << format("  div.approx.f32 %rc{0}_{1}{2}, %rc{0}_{1}{2}, %fNfilt;", i, j, vs[s]) << std::endl;
    }


    if(in_dtype_ == INT8X4_TYPE){
      for(size_t j = 0; j < vect_in; j++)
      for(size_t i = 0; i < cl0; i += vec_*bc0_)
      for(size_t s = 0; s < vec_; s++)
        iss << format("  div.approx.f32 %rc{0}_{1}{2}, %rc{0}_{1}{2}, %i_scale;", i, j, vs[s]) << std::endl;
    }


    if(out_dtype_ == INT8X4_TYPE){
      for(size_t j = 0; j < vect_in; j++)
      for(size_t i = 0; i < cl0; i += vec_*bc0_)
      for(size_t s = 0; s < vec_; s++){
        iss << format("  mul.f32 %rc{0}_{1}{2}, %rc{0}_{1}{2}, %o_scale;", i, j, vs[s]) << std::endl;
      }

      iss << std::endl;
      iss << "  /* Pack */" << std::endl;
      for(size_t i = 0; i < cl0; i += vec_*bc0_)
      for(size_t s = 0; s < vec_; s++){
        for(size_t j = 0; j < vect_in; j++)
          iss << format("  cvt.rni.sat.s8.f32 %b8_ccvt{1}, %rc{0}_{1}{2};", i, j, vs[s]) << std::endl;
        iss << format("  mov.b32 %rc{0}_0{1}, {{%b8_ccvt0, %b8_ccvt1, %b8_ccvt2, %b8_ccvt3}};", i, vs[s]) << std::endl;
      }
    }


    iss << std::endl;
    iss << "  /* Write back */" << std::endl;
    iss << format("  mul.lo.u32 %offIncdhw, %bid0, {};", cl0) << std::endl;
    iss << format("  mad.lo.u32 %offIncdhw, %id0, {}, %offIncdhw;", vec_) << std::endl;
    for(size_t i = 0; i < cl0; i+=vec_*bc0_)
    for(size_t s = 0; s < vec_; s++){
      iss << format("  add.u32 %offIncdhw{0}, %offIncdhw, {0};", i + s) << std::endl;
      iss << format("  div.u32 %offIncdh{0}, %offIncdhw{0}, %Q;", i + s) << std::endl;
      iss << format("  mad.lo.s32 %offIw{0}, %mQ, %offIncdh{0}, %offIncdhw{0};", i + s) << std::endl;
      iss << format("  div.u32 %offIncd{0}, %offIncdh{0}, %P;", i + s) << std::endl;
      iss << format("  mad.lo.s32 %offIh{0}, %mP, %offIncd{0}, %offIncdh{0};", i + s) << std::endl;
      iss << format("  div.u32 %offInc{0}, %offIncd{0}, %M;", i + s) << std::endl;
      iss << format("  mad.lo.s32 %offId{0}, %mM, %offInc{0}, %offIncd{0};", i + s) << std::endl;
      iss << format("  div.u32 %offIn{0}, %offInc{0}, %C;", i + s) << std::endl;
      iss << format("  mad.lo.s32 %offIc{0}, %mC, %offIn{0}, %offInc{0};", i + s) << std::endl;
      iss << format("  mul.lo.s32 %offIc{0}, %offIc{0}, {1};", i + s, vect_c_in) << std::endl;
      iss << format("  div.s32 %offIc{0}, %offIc{0}, {1};", i + s, vect_c_out) << std::endl;
    }

    for(size_t i = 0; i < cl0; i+= bc0_*vec_)
    for(size_t s = 0; s < vec_; s++){
      iss << format("  setp.lt.s32 %pred{0}, %offIncdhw{0}, %Npix;", i + s) << std::endl;
      iss << format("  mad.lo.s32 %offc0_{0}, %offId{0}, %strideOm, 0;", i + s) << std::endl;
      iss << format("  mad.lo.s32 %offc0_{0}, %offIh{0}, %strideOp, %offc0_{0};", i + s) << std::endl;
      iss << format("  mad.lo.s32 %offc0_{0}, %offIw{0}, %strideOq, %offc0_{0};", i + s) << std::endl;
      iss << format("  mad.lo.s32 %offc0_{0}, %offIn{0}, %strideOn, %offc0_{0};", i + s) << std::endl;
      iss << format("  mad.lo.s32 %offc0_{0}, %offIc{0}, %strideOk, %offc0_{0};", i + s) << std::endl;
      for(size_t j = 0; j < vect_c_in; j+=vect_c_out){
        iss << format("  mad.wide.s32 %pc{0}_{1}, %offc0_{0}, 1, %pc;", i + s, j) << std::endl;
        iss << format("  mad.wide.s32 %pc{0}_{1}, {1}, %strideOk, %pc{0}_{1};", i + s, j) << std::endl;
      }
    }

    bool aligned = (M_*P_*Q_) % vec_ == 0;
    for(size_t j = 0; j < vect_c_in / vect_c_out; j++)
    for(size_t i = 0 ; i < cl0 ; i+=bc0_*vec_)
    for(size_t s = 0; s < vec_; s+=(aligned?vec_:1))
      iss << format("  @%pred{} st.global{}.{} [%pc{}_{}], %rc{}_{}{};", i + s, aligned?vv:"", in_word_type, i + s, j, i, j, aligned?"":vs[s]) << std::endl;
    iss << "}" << std::endl;

    return iss.str();
}


std::vector<unsigned int> Pool::tuning_params() const
{   return {};  }

double Pool::tflops(param_t P, param_t Q, param_t M, param_t K, param_t N, param_t T, param_t R, param_t S, double time)
{ return (double)M*P*Q*K*N*T*R*S/(time*1e3); }

void Pool::enqueue(driver::Kernel& kernel, driver::Stream& stream, driver::Buffer const & I, driver::Buffer& O, float i_scale, float o_scale){
    // Data-type size
    int32_t dtsize = size_of(in_dtype_);

    // I strides
    int32_t strideIw = dtsize;
    int32_t strideIh = W_*strideIw;
    int32_t strideId = H_*strideIh;
    int32_t strideIc = D_*strideId;
    int32_t strideIn = Cin_*strideIc;
    // O strides
    int32_t strideOq = dtsize;
    int32_t strideOp = Q_*strideOq;
    int32_t strideOm = P_*strideOp;
    int32_t strideOk = M_*strideOm;
    int32_t strideOn = Cout_*strideOk;

    // Input information
    int32_t Npix = Cin_*M_*P_*Q_*N_;
    int32_t Nfilt = T_*R_*S_;

    // Constant memory
    driver::Buffer LUT = kernel.module().symbol("_LUT");
    driver::Buffer masks = kernel.module().symbol("_masks");
    stream.write(LUT, false, 0, cLUT.size()*4, cLUT.data());
    stream.write(masks, false, 0, masks_.size()*4, masks_.data());

    // Enqueue
    size_t idx = 0;
    kernel.setArg(idx++, I);
    kernel.setArg(idx++, O);
    kernel.setArg(idx++, Npix);
    kernel.setArg(idx++, Nfilt);
    kernel.setArg(idx++, M_);
    kernel.setArg(idx++, P_);
    kernel.setArg(idx++, Q_);
    kernel.setArg(idx++, Cout_);
    kernel.setArg(idx++, D_);
    kernel.setArg(idx++, H_);
    kernel.setArg(idx++, W_);
    kernel.setArg(idx++, Cin_);
    kernel.setArg(idx++, stride_d_);
    kernel.setArg(idx++, stride_h_);
    kernel.setArg(idx++, stride_w_);
    kernel.setArg(idx++, pad_d_);
    kernel.setArg(idx++, pad_h_);
    kernel.setArg(idx++, pad_w_);
    kernel.setArg(idx++, strideIc);
    kernel.setArg(idx++, strideId);
    kernel.setArg(idx++, strideIh);
    kernel.setArg(idx++, strideIw);
    kernel.setArg(idx++, strideIn);
    kernel.setArg(idx++, strideOk);
    kernel.setArg(idx++, strideOm);
    kernel.setArg(idx++, strideOp);
    kernel.setArg(idx++, strideOq);
    kernel.setArg(idx++, strideOn);
    kernel.setArg(idx++, i_scale);
    kernel.setArg(idx++, o_scale);

    int32_t cl0 = bc0_*cs0_;
    size_t grid0 = ceil(Npix, cl0);
    stream.enqueue(kernel, {grid0, 1, 1}, {bc0_, 1, 1});
}


}
}
