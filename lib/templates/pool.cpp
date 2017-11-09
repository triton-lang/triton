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

Pool::Pool(DType dtype, param_t C, param_t D, param_t H, param_t W, param_t N, param_t M, param_t P, param_t Q, param_t T, param_t R, param_t S, param_t pad_d, param_t pad_h, param_t pad_w, param_t stride_d, param_t stride_h, param_t stride_w,
           param_t vec, param_t bc0, param_t cs0, param_t):
    dtype_(dtype), C_(C), D_(D), H_(H), W_(W), N_(N), M_(M), P_(P), Q_(Q), T_(T), R_(R), S_(S), pad_d_(pad_d), pad_h_(pad_h), pad_w_(pad_w), stride_d_(stride_d), stride_h_(stride_h), stride_w_(stride_w),
    vec_(vec), bc0_(bc0), cs0_(cs0), u_(1){
}

// Validity
void Pool::check_valid(driver::Device const & device, size_t M, param_t* params, uint8_t* valid){
  std::array<int, Nparams> x{0};
  for(size_t m = 0; m < M; ++ m){
    //Parameters
    for(size_t i = 0; i < x.size(); ++i)
      x[i] = params[m*x.size() + i];
    DType dtype = (DType)(x[0]);
    param_t Nfilt = x[2];
    param_t vec = x[3], bc0 = x[4], cs0 = x[5];
    //Test
    bool is_valid =  (4*Nfilt <= device.max_shared_memory())
                  && (bc0 <= device.max_block_dim()[0])
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
    size_t block = u_;
    size_t nlut = T_*R_*S_;
    size_t nthreads = bc0_;
    size_t addr_lut = 0;
    std::string dtype = arith_str(dtype_);
    size_t dtsize = size_of(dtype_);
    size_t size_shmem = nlut*4;

    std::string vv = vec_>1?format(".v{}", vec_):"";
    const char* vs[] = {".x", ".y", ".z", ".w"};
    if(vec_==1)
      vs[0] = "";

    auto lut = [&](){
      iss << "  mov.s32 %trs, %id;" << std::endl;
      iss << format("  setp.lt.u32 %in_bounds, %trs, {};", nlut) << std::endl;
      iss << "  @!%in_bounds bra END_LUT_LOOP;" << std::endl;
      iss << "LUT_LOOP:" << std::endl;
      iss << format("  div.s32 %tr, %trs, {};", S_) << std::endl;
      iss << format("  rem.s32 %s, %trs, {};", S_) << std::endl;
      iss << format("  div.s32 %t, %tr, {};", R_) << std::endl;
      iss << format("  rem.s32 %r, %tr, {};", R_) << std::endl;

      iss << format("  add.s32 %nexttrs, %trs, {};", block) << std::endl;
      iss << format("  div.s32 %nexttr, %nexttrs, {};", S_) << std::endl;
      iss << format("  rem.s32 %nexts, %nexttrs, {};", S_) << std::endl;
      iss << format("  div.s32 %nextt, %nexttr, {};", R_) << std::endl;
      iss << format("  rem.s32 %nextr, %nexttr, {};", R_) << std::endl;

      iss << format("  sub.s32 %tdiff, %nextt, %t;") << std::endl;
      iss << format("  sub.s32 %rdiff, %nextr, %r;") << std::endl;
      iss << format("  sub.s32 %sdiff, %nexts, %s;") << std::endl;

      iss << format("  mul.lo.s32 %inci, %tdiff, %strideId;") << std::endl;
      iss << format("  mad.lo.s32 %inci, %rdiff, %strideIh, %inci;") << std::endl;
      iss << format("  mad.lo.s32 %inci, %sdiff, %strideIw, %inci;") << std::endl;

      iss << "  // Store" << std::endl;
      iss << format("  mad.lo.u32 %writelut, %trs, {}, %shared;", 4) << std::endl;
      iss << format("  st.shared.u32 [%writelut + {}], %inci;", addr_lut) << std::endl;

      iss << "  // Continue loop if necessary" << std::endl;
      iss << format("  add.u32 %trs, %trs, {};", nthreads) << std::endl;
      iss << format("  setp.lt.u32 %in_bounds, %trs, {};", nlut) << std::endl;
      iss << "  @%in_bounds bra LUT_LOOP;" << std::endl;
      iss << std::endl;
      iss << "END_LUT_LOOP:" << std::endl;
      iss << "  bar.sync 0;" << std::endl;
    };

    auto ptr_ldg_i = [&](){
        iss << format("  // I offsets") << std::endl;
        iss << format("  mul.lo.s32 %mM, %M, -1;") << std::endl;
        iss << format("  mul.lo.s32 %mN, %N, -1;") << std::endl;
        iss << format("  mul.lo.s32 %mP, %P, -1;") << std::endl;
        iss << format("  mul.lo.s32 %mQ, %Q, -1;") << std::endl;

        iss << format("  mul.lo.u32 %offIcdhwn, %bid0, {};", cl0) << std::endl;
        iss << format("  mad.lo.u32 %offIcdhwn, %id0, {}, %offIcdhwn;", vec_) << std::endl;
        for(size_t i = 0; i < cl0; i+=vec_*bc0_)
        for(size_t s = 0; s < vec_; s++){
          iss << format("  add.u32 %offIcdhwn{0}, %offIcdhwn, {0};", i + s) << std::endl;
          iss << format("  div.u32 %offIcdhw{0}, %offIcdhwn{0}, %N;", i + s) << std::endl;
          iss << format("  mad.lo.s32 %offIn{0}, %mN, %offIcdhw{0}, %offIcdhwn{0};", i + s) << std::endl;
          iss << format("  div.u32 %offIcdh{0}, %offIcdhw{0}, %Q;", i + s) << std::endl;
          iss << format("  mad.lo.s32 %offIw{0}, %mQ, %offIcdh{0}, %offIcdhw{0};", i + s) << std::endl;
          iss << format("  div.u32 %offIcd{0}, %offIcdh{0}, %P;", i + s) << std::endl;
          iss << format("  mad.lo.s32 %offIh{0}, %mP, %offIcd{0}, %offIcdh{0};", i + s) << std::endl;
          iss << format("  div.u32 %offIc{0}, %offIcd{0}, %M;", i + s) << std::endl;
          iss << format("  mad.lo.s32 %offId{0}, %offIc{0}, %mM, %offIcd{0};", i + s) << std::endl;
        }

        for(size_t i = 0; i < cl0; i += vec_*bc0_)
        for(size_t s = 0; s < vec_; s++){
          iss << format("  mad.wide.u32 %pi{0}, %offIc{0}, %strideIc, %pi;", i + s) << std::endl;
          iss << format("  mad.wide.u32 %pi{0}, %offId{0}, %strideId, %pi{0};", i + s) << std::endl;
          iss << format("  mad.wide.u32 %pi{0}, %offIh{0}, %strideIh, %pi{0};", i + s) << std::endl;
          iss << format("  mad.wide.u32 %pi{0}, %offIw{0}, %strideIw, %pi{0};", i + s) << std::endl;
          iss << format("  mad.wide.u32 %pi{0}, %offIn{0}, %strideIn, %pi{0};", i + s) << std::endl;
        }
    };


    iss << ".entry " << name << "(" << std::endl
        << "            .param .b64 _pi, .param .b64 _pc," << std::endl
        << "            .param .b32 _Npix, .param .b32 _Nfilt," << std::endl
        << "            .param .b32 _M, .param .b32 _P, .param .b32 _Q, .param .b32 _N," << std::endl
        << "            .param .b32 _stride_d, .param .b32 _stride_h, .param .b32 _stride_w, .param .b32 _pad_d, .param .b32 _pad_h, .param .b32 _pad_w, " << std::endl
        << "            .param .b32 _strideIc, .param .b32 _strideId, .param .b32 _strideIh, .param .b32 _strideIw, .param .b32 _strideIn, " << std::endl
        << "            .param .b32 _strideOk, .param .b32 _strideOm, .param .b32 _strideOp, .param .b32 _strideOq, .param .b32 _strideOn)" << std::endl;
    iss << "{" << std::endl;
    iss << "  .reg .pred %in_bounds, %predloop;" << std::endl;
    iss << "  .reg .b32 %id, %id0, %bid0;" << std::endl;
    iss << "  .reg .b32 %trs, %tr, %t, %r, %s, %nexttrs, %nexttr, %nextt, %nextr, %nexts, %tdiff, %rdiff, %sdiff;" << std::endl;
    iss << "  .reg .b32 %Npix, %Nfilt, %M, %P, %Q, %N, %mM, %mP, %mQ, %mN;" << std::endl;
    iss << "  .reg .b32 %pad_d, %pad_h, %pad_w;" << std::endl;
    iss << "  .reg .b32 %stride_d, %stride_h, %stride_w;" << std::endl;
    iss << "  .reg .b32 %strideIc, %strideId, %strideIh, %strideIw, %strideIn;" << std::endl;
    iss << "  .reg .b32 %strideOk, %strideOm, %strideOp, %strideOq, %strideOn;" << std::endl;
    iss << "  .reg .b32 %readlut, %writelut;" << std::endl;
    iss << "  .reg .b32 %inci;" << std::endl;
    iss << "  .reg .b64 %pi, %pc;" << std::endl;
    for(size_t i = 0; i < cl0; i+=vec_*bc0_)
    for(size_t s = 0; s < vec_; s++)
      iss << format("  .reg .b64 %pi{0};", i + s) << std::endl;
    iss << "  .reg .b32 %offIcdhwn;" << std::endl;
    for(size_t i = 0; i < cl0; i+=vec_*bc0_)
    for(size_t s = 0; s < vec_; s++){
      iss << format("  .reg .b32 %offIcdhwn{0}, %offIcdhw{0}, %offIn{0}, %offIcdh{0}, %offIw{0}, %offIcd{0}, %offIh{0}, %offIc{0}, %offId{0};", i + s) << std::endl;
    }
    iss << "  .reg .b32 %TRS;" << std::endl;
    for(size_t i = 0; i < cl0; i+=vec_*bc0_)
      iss << format("  .reg {0}.b32 %rri{1}, %rc{1};", vv, i) << std::endl;
    for(size_t i = 0; i < cl0; i+=vec_*bc0_)
      iss << format("  .reg .b64 %pc{0};", i) << std::endl;
    iss << "  .reg .b32 %offc0;" << std::endl;
    for(size_t i = 0; i < cl0; i+=vec_*bc0_)
        iss << format("  .reg .b32 %offc0_{0};", i) << std::endl;
    iss << format("  .reg .b32 %Npixm<{0}>;", vec_) << std::endl;
    for(size_t i = 0; i < cl0; i += vec_*bc0_)
    for(size_t s = 0; s < vec_; s++)
        iss << format("  .reg .pred %pred{0};", i + s) << std::endl;


    iss << "  // Initialize C" << std::endl;
    for(size_t i = 0; i < cl0; i += vec_*bc0_)
    for(size_t s = 0; s < vec_; s++)
      iss << format("  mov.b32 %rc{0}{1}, 0xff800000;", i, vs[s]) << std::endl;

    iss << std::endl;
    iss << "  // Shared memory" << std::endl;
    iss << format("  .shared .align 16 .b8 _shared[{}];", size_shmem) << std::endl;
    iss << format("  .reg .b64 %shared64;") << std::endl;
    iss << format("  .reg .b32 %shared;") << std::endl;
    iss << format("  mov.u64 %shared64, _shared;") << std::endl;
    iss << format("  cvt.u32.u64 %shared, %shared64;") << std::endl;

    iss << std::endl;
    iss << "  ld.param.u64 %pc, [_pc];" << std::endl;
    iss << "  ld.param.u64 %pi, [_pi];" << std::endl;
    iss << "  ld.param.s32 %Npix, [_Npix];" << std::endl;
    iss << "  ld.param.s32 %Nfilt, [_Nfilt];" << std::endl;
    iss << std::endl;
    iss << "  // Tensor shapes" << std::endl;
    iss << "  ld.param.s32 %M, [_M];" << std::endl;
    iss << "  ld.param.s32 %P, [_P];" << std::endl;
    iss << "  ld.param.s32 %Q, [_Q];" << std::endl;
    iss << "  ld.param.s32 %N, [_N];" << std::endl;
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
    iss << "  // Special registers" << std::endl;
    iss << "  mov.u32 %id0, %tid.x;" << std::endl;
    iss << "  mov.u32 %bid0, %ctaid.x;" << std::endl;
    iss << "  mov.u32 %id, %id0;" << std::endl;
    iss << std::endl;
    iss << "  // Bounds checking" << std::endl;
    for(size_t s = 0; s < vec_; ++s)
      iss << format("  sub.s32 %Npixm{0}, %Npix, {0};", s) << std::endl;
    iss << "  // Look-up table" << std::endl;
    lut();
    iss << std::endl;
    iss << "  // pointers" << std::endl;
    ptr_ldg_i();
    iss << std::endl;
    iss << "  // pooling" << std::endl;
    iss << "  mov.u32 %TRS, %Nfilt;" << std::endl;
    iss << "  mov.u32 %readlut, %shared;" << std::endl;
    iss << "LOOP:" << std::endl;
    for(size_t i = 0; i < cl0; i += vec_*bc0_)
    for(size_t s = 0; s < vec_; s++)
      iss << format("  setp.lt.s32 %pred{}, %offIcdhwn{}, %Npixm{};", i + s, i, s) << std::endl;

    for(size_t i = 0; i < cl0; i += vec_*bc0_)
    for(size_t s = 0; s < vec_; s++)
      iss << format("  @%pred{} ld.global.cg.u32 %rri{}{}, [%pi{}];", i + s, i, vs[s], i + s)  << std::endl;

    for(size_t i = 0; i < cl0; i += vec_*bc0_)
    for(size_t s = 0; s < vec_; s++)
      iss << format("  max.f32 %rc{0}{1}, %rri{0}{1}, %rc{0}{1};", i, vs[s]) << std::endl;

    iss << " // Increment image pcinters" << std::endl;
    iss << format("  ld.shared.u32 %inci, [%readlut + {}];", addr_lut) << std::endl;
    for(size_t i = 0; i < cl0; i += vec_*bc0_)
    for(size_t s = 0; s < vec_; s++)
      iss << format("  mad.wide.u32 %pi{}, %inci, {}, %pi{};", i + s, 1, i + s) << std::endl;
    iss << format("  sub.s32 %TRS, %TRS, 1;") << std::endl;
    iss << format("  add.u32 %readlut, %readlut, 4;") << std::endl;
    iss << "  setp.gt.u32 %predloop, %TRS, 0;" << std::endl;
    iss << "  @%predloop bra.uni LOOP;" << std::endl;

    iss << std::endl;
    iss << "ENDLOOP:" << std::endl;
    iss << "  /* Offsets */" << std::endl;
    iss << format("  mad.lo.s32 %offc0, %bid0, {}, 0;", cl0) << std::endl;
    iss << format("  mad.lo.s32  %offc0, %id0, {}, %offc0;", vec_) << std::endl;
    for(size_t i = 0 ; i < cl0 ; i+= bc0_*vec_)
      iss << format("  add.s32 %offc0_{0}, %offc0, {0};", i) << std::endl;

    iss << "  /* Write back */" << std::endl;
    for(size_t i = 0; i < cl0; i+= bc0_*vec_)
      iss << format("  mad.wide.s32 %pc{0}, %offc0_{0}, %strideOn, %pc;", i) << std::endl;

    for(size_t i = 0; i < cl0; i += vec_*bc0_)
    for(size_t s = 0; s < vec_; s++)
      iss << format("  setp.lt.s32 %pred{}, %offc0_{}, %Npixm{};", i + s, i, s) << std::endl;

    bool aligned = (C_*M_*P_*Q_*N_) % vec_ == 0;
    for(size_t i = 0 ; i < cl0 ; i+=bc0_*vec_){
    for(size_t s = 0; s < vec_; s+=(aligned?vec_:1))
        iss << format("  @%pred{} st.global{}.{} [%pc{} + {}], %rc{}{};", i + s, aligned?vv:"", dtype, i, dtsize*s, i, aligned?"":vs[s]) << std::endl;
    }
    iss << "}" << std::endl;

    return iss.str();
}


std::vector<unsigned int> Pool::tuning_params() const
{   return {};  }

double Pool::tflops(param_t P, param_t Q, param_t M, param_t K, param_t N, param_t T, param_t R, param_t S, double time)
{ return (double)M*P*Q*K*N*T*R*S/(time*1e3); }

void Pool::enqueue(driver::Kernel& kernel, driver::Stream& queue, driver::Buffer const & I, driver::Buffer& O){
    // I strides
    int32_t strideIn = 1;
    int32_t strideIw = N_*strideIn;
    int32_t strideIh = W_*strideIw;
    int32_t strideId = H_*strideIh;
    int32_t strideIc = D_*strideId;
    // O strides
    int32_t strideOn = 1;
    int32_t strideOq = N_*strideOn;
    int32_t strideOp = Q_*strideOq;
    int32_t strideOm = P_*strideOp;
    int32_t strideOk = M_*strideOm;

    // Data-type size
    int32_t dtsize = size_of(dtype_);

    // Input information
    int32_t Npix = C_*M_*P_*Q_*N_;
    int32_t Nfilt = T_*R_*S_;

    kernel.setArg(0, I);
    kernel.setArg(1, O);
    kernel.setArg(2, Npix);
    kernel.setArg(3, Nfilt);
    kernel.setArg(4, M_);
    kernel.setArg(5, P_);
    kernel.setArg(6, Q_);
    kernel.setArg(7, N_);
    kernel.setArg(8, stride_d_);
    kernel.setArg(9, stride_h_);
    kernel.setArg(10, stride_w_);
    kernel.setArg(11, pad_d_);
    kernel.setArg(12, pad_h_);
    kernel.setArg(13, pad_w_);
    kernel.setArg(14, strideIc*dtsize);
    kernel.setArg(15, strideId*dtsize);
    kernel.setArg(16, strideIh*dtsize);
    kernel.setArg(17, strideIw*dtsize);
    kernel.setArg(18, strideIn*dtsize);
    kernel.setArg(19, strideOk*dtsize);
    kernel.setArg(20, strideOm*dtsize);
    kernel.setArg(21, strideOp*dtsize);
    kernel.setArg(22, strideOq*dtsize);
    kernel.setArg(23, strideOn*dtsize);

    int32_t cl0 = bc0_*cs0_;
    size_t grid0 = ceil(Npix, cl0);
    queue.enqueue(kernel, {grid0, 1, 1}, {bc0_, 1, 1});
}


}
}
