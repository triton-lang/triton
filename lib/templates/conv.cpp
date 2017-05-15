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

Conv::Conv(DType dtype, param_t C, param_t H, param_t W, param_t N, param_t K, param_t P, param_t Q, param_t R, param_t S, param_t pad_h, param_t pad_w, param_t stride_h, param_t stride_w,
     param_t vec, param_t bp, param_t bq, param_t bn, param_t bk,  param_t bf_n, param_t ps, param_t qs, param_t ns, param_t ks, param_t crs_l, param_t crs_s, param_t cs, param_t bc, param_t gridc):
  dtype_(dtype), C_(C), H_(H), W_(W), N_(N), K_(K), P_(P), Q_(Q), R_(R), S_(S), pad_h_(pad_h), pad_w_(pad_w), stride_h_(stride_h), stride_w_(stride_w),
  vec_(vec), bp_(bp), bq_(bq), bn_(bn), bk_(bk), bf_n_(bf_n), ps_(ps), qs_(qs), ns_(ns), ks_(ks), crs_l_(crs_l), crs_s_(crs_s), cs_(cs), bc_(bc), gridc_(gridc)
{}

double Conv::tflops(param_t P, param_t Q, param_t K, param_t N, param_t C, param_t R, param_t S, double time)
{ return (double)2*P*Q*K*N*C*R*S/(time*1e3); }

std::string Conv::id() const{
std::ostringstream iss;
  size_t dtvec = (dtype_==HALF_TYPE)?2:1;
  iss << "conv" << dtype_ << "_"
      << N_ % dtvec << "_" << N_ % (dtvec*vec_)<< "_" << vec_
      << "_" << bp_ << "_" << bq_ << "_" << bn_ << "_" << bk_
      << "_" << ps_ << "_" << qs_ << "_" << ns_ << "_" << ks_
      << "_" << crs_l_ << " " << crs_s_ << "_" << cs_ << "_" << bc_ << "_" << gridc_;
  return iss.str();
}

void Conv::check_valid(driver::Device const & device, size_t M, param_t* params, uint8_t* valid){
  std::array<int, 23> x{0};
  for(size_t m = 0; m < M; ++ m){

    //Parameters
    for(size_t i = 0; i < x.size(); ++i)
      x[i] = params[m*x.size() + i];
    DType dtype = (DType)(x[0]);
    //param_t  N = x[1], K = x[2], P = x[3], Q = x[4];
    param_t C = x[5], R = x[6], S = x[7],
          vec = x[8], bp = x[9], bq = x[10], bn = x[11], bk = x[12], bf_n = x[13],
          ps = x[14], qs = x[15], ns = x[16], ks = x[17], crs_l = x[18],
          crs_s = x[19], cs = x[20], bc = x[21], gridc = x[22];

    //std::cout << N << " " << K << " " << P << " " << Q << " " << C << " " << R<< " " << S << " "
    //          << vec << " " << bp << " " << bq << " " << bn << " " << bk << " " << bf_n << " " << ps << " " << qs << " " << ns << " " << ks << " " << crs_l << " " << crs_s << " " << cs << " " << bc << " " << gridc << std::endl;


    //Features
    param_t dtsize = size_of(dtype);
    param_t dtvec = (dtype==HALF_TYPE)?2:1;
    param_t bpqn = bp*bq*bn;
    param_t pqns = ps*qs*ns;
    param_t pqnl = bpqn*pqns;
    param_t pl = ps*bp;
    param_t ql = qs*bq;
    param_t cl = cs*bc;

    param_t nthreads = bn*bk*bp*bq*bc;
    param_t RS = R*S;
    param_t nl = ns*bn;
    param_t kl = ks*bk;

    param_t cd_sharedi = dtsize*pqnl*dtvec;
    param_t cd_sharedf = dtsize*kl*dtvec;
    param_t size_sharedi = cd_sharedi*cl*crs_l;
    param_t size_sharedf = cd_sharedf*cl*crs_l;
    param_t size_lut = 4*RS*(1 + pl*ql); //RS*sizeof(param_t2)
    param_t size_lut_max = 4*pl*ql;
    param_t param_tiles = next_pow2(2*(size_sharedi + size_sharedf));
    param_t size_redc = dtsize*pqnl*kl*(bc==1?0:bc);
    param_t size_shmem = std::max(size_redc, param_tiles + size_lut + size_lut_max);

    param_t bf_crs = crs_l*cl;
    param_t bf_k = nthreads/bf_crs;
    param_t bf_pqn = bf_k;
    param_t bf_pq = bf_pqn / bf_n;

    param_t n_instructions =  pl*ql/bf_pq*nl/bf_n + kl/bf_k + pqns*ks*crs_l/dtvec + pqns*ks*crs_l/(vec*dtvec);
    //Test
    bool is_valid = (bf_n*bf_pq*bf_crs == nthreads)
                  && (bf_k*bf_crs == nthreads)
                  && (crs_l*cl % bf_crs == 0)
                  && (pl*ql % (bf_pq) == 0)
                  && (nl % (bf_n*vec) == 0)
                  && (kl % (bf_k*vec) == 0)
                  && (nthreads % 32 == 0)
                  && (ns % (vec*dtvec) == 0)
                  && (ks % (vec*dtvec) == 0)
                  && (crs_l % crs_s == 0)
                  && (size_shmem <= device.max_shared_memory())
                  && (bn*bp*bq <= device.max_block_dim()[0])
                  && (bk <= device.max_block_dim()[1])
                  && (bc <= device.max_block_dim()[2])
                  && (nthreads <= device.max_threads_per_block())
                  && (n_instructions <= 1024)

//                  && (pl<=P)
//                  && (ql<=Q)
//                  && (nl<=N+vec)
//                  && (kl<=K+vec)
                  && (cl*gridc<=C)
                  && (vec*dtsize <= 16)
                  && (device.compute_capability().first>=6 || dtype != HALF_TYPE)
                  && (gridc==1 || dtype==FLOAT_TYPE);  //Global reduction on fp16x2 and fp64 not always possible
    valid[m] = is_valid;
  }
}


/* Code generation */
std::string Conv::dump(drv::Device const & device, std::string const & name){

  std::stringstream iss;
  size_t dtsize = size_of(dtype_);
  size_t dtvec = (dtype_==HALF_TYPE)?2:1;
  std::string dtype = arith_str(dtype_);
  std::string io_dtype = io_str(dtype_);
  std::string ab_dtype = (dtype_==HALF_TYPE)?"b16":dtype;
  std::string sub_dtype = (dtype_==HALF_TYPE)?"f16":dtype;
  std::string alpha_dtype = (dtype_==DOUBLE_TYPE)?"f64":"f32";
  size_t bpqn = bp_*bq_*bn_;
  size_t pqns = ps_*qs_*ns_;
  size_t pqnl = bpqn*pqns;
  size_t pl = ps_*bp_;
  size_t ql = qs_*bq_;
  size_t cl = cs_*bc_;


  bool check_lut = pad_h_ > 0 || pad_w_ > 0 || stride_h_ > 1 || stride_w_ > 1;

  // helpers
  size_t nthreads = bn_*bk_*bp_*bq_*bc_;
  size_t RS = R_*S_;
  size_t nl = ns_*bn_;
  size_t kl = ks_*bk_;
  size_t cd_sharedi = dtsize*pqnl*dtvec;
  size_t cd_sharedf = dtsize*kl*dtvec;

  size_t size_sharedi = cd_sharedi*cl*crs_l_;
  size_t size_sharedf = cd_sharedf*cl*crs_l_;
  size_t size_lut = 4*RS*(1 + pl*ql); //RS*sizeof(size_t2)
  size_t size_lut_max = 4*pl*ql;
  size_t size_tiles = next_pow2(2*(size_sharedi + size_sharedf));
  size_t size_redc = dtsize*pqnl*kl*(bc_==1?0:bc_);
  size_t size_shmem = std::max(size_redc, size_tiles + size_lut + size_lut_max);
  size_t Bvec = vec_*dtsize;
  size_t addr_i = 0;
  size_t addr_f = size_sharedi;
  size_t addr_lut = size_tiles;
  size_t addr_lut_Frs = addr_lut;
  size_t addr_lut_Ihw = addr_lut + 4*RS;
  size_t double_buffer_off = size_tiles/2;
  // params
  size_t bf_crs = crs_l_*cl;
  size_t bf_k = nthreads/bf_crs;
  size_t bf_pqn = bf_k;
//  bf_n = bp*bq;
  size_t bf_pq = bf_pqn / bf_n_;

  uint8_t is_valid;
  uint32_t params[] = {(uint32_t)size_of(dtype_), N_, K_, P_, Q_, C_, R_, S_,
                     vec_, bp_, bq_, bn_, bk_, bf_n_, ps_, qs_, ns_, ks_, crs_l_, crs_s_, cs_, bc_, gridc_};
  check_valid(device, 1, params, &is_valid);
  if(!is_valid)
    throw invalid_parameters();


  std::string vv = vec_>1?format(".v{}", vec_):"";
  const char* vs[] = {".x", ".y", ".z", ".w"};
  if(vec_==1)
    vs[0] = "";

  auto declare_register_tile = [&](char x, size_t M, size_t N, size_t dtinc){
    for(size_t c = 0; c < cs_; c++){
      for(size_t m = 0 ; m < M ; m+=vec_*dtinc){
        iss << format("  .reg {}.{}", vv, io_dtype);
        for(size_t n = 0 ; n < N ; n++)
          iss << format("{} %r{}{}_{}_{}", n>0?",":"", x, c, m, n);
        iss << ";" << std::endl;
      }
    }
  };

  auto store = [&](size_t n, size_t pqn, size_t k){
    size_t inc = 1;
    std::string ovec = "";
    std::string subvec = ab_dtype;
    if((N_/inc) % dtvec == 0){
      inc *= dtvec;
      if(dtype_==HALF_TYPE)
        subvec = format("b{}", dtsize*8*dtvec);
    }
    if((N_/inc) % vec_ == 0){
      inc *= vec_;
      ovec = vec_>1?format(".v{}", vec_):"";
    }
    for(size_t s = 0; s < vec_*dtvec; s+=inc){
      if(subvec=="b16"){
        if(s%dtvec==0)
          iss << format("  mov.b32 {{%rfh0, %rfh1}}, %ro0_{}_{}{};", pqn, k, vs[s/dtvec]) << std::endl;
        iss << format("  @%pred{} st.global{}.{} [%po + {}], %rfh{};", s, ovec, subvec, dtsize*(bn_*n+s), s%dtvec) << std::endl;
      }
      else
        iss << format("  @%pred{} st.global{}.{} [%po + {}], %ro0_{}_{}{};", s, ovec, subvec, dtsize*(bn_*n+s), pqn, k, (inc/dtvec > 1)?"":vs[s/dtvec]) << std::endl;
    }
  };


  auto ldg = [&](bool last_iter){
      std::vector<std::string> predn(vec_, "%predcrs");
      std::vector<std::string> predk(vec_, "%predcrs");
      iss << format("  cvt.rn.f32.s32 %CRSf, %CRS;") << std::endl;
      iss << format("  mul.f32 %CRSf, %CRSf, %rcpRS;") << std::endl;
      iss << format("  cvt.rzi.s32.f32 %c, %CRSf;") << std::endl;
      iss << format("  vmad.s32.s32.s32 %rs, %c.h0, -%lut_size.h0, %CRS;") << std::endl;
      if(gridc_ > 1)
        iss << format("  add.u32 %c, %c, %offc;") << std::endl;

      iss << format("  mad.lo.u32 %readlut, %rs, {}, %shared;", 4) << std::endl;
      iss << format("  ld.shared.u32 %offFrs, [%readlut + {}];", addr_lut_Frs) << std::endl;

      iss << format("  mad.lo.u32 %readlut, %fidpq, {}, %readlut;", 4*RS) << std::endl;
      for(size_t pq = 0; pq < pl*ql; pq+=bf_pq)
        iss << format("  ld.shared.u32 %offIhw{}, [%readlut + {}];", pq, addr_lut_Ihw + pq*4*RS) << std::endl;

      iss << format("  //Load F: pf[offFrs + offFk + c*KRS]") << std::endl;
      iss << format("  mad.lo.u32 %offF, %c, %KRS, %offFk;") << std::endl;
      iss << format("  add.u32 %offF, %offFrs, %offF;") << std::endl;
      iss << format("  mad.wide.u32 %pf0, %offF, {}, %pf;", dtsize) << std::endl;
      for(size_t k = 0; k < kl; k+=vec_*bf_k){
        for(size_t s = 0; s < (last_iter?vec_:0); ++s){
          iss << format("  setp.lt.and.s32 %pred{}, {}, %boundK, %predcrs;", s, k+s) << std::endl;
          predk[s] = format("%pred{}", s);
        }
        for(size_t s = 0; s < (last_iter?vec_:0); ++s)
         if(dtype_==HALF_TYPE)
            iss << format("  @!{0} and.b16 %rrf{2}{3}, 0, %rrf{2}{3};", predk[s], ab_dtype, k, vs[s]) << std::endl;
          else
            iss << format("  @!{0} mov.{1} %rrf{2}{3}, 0.;", predk[s], ab_dtype, k, vs[s]) << std::endl;
        for(size_t s = 0; s < vec_; ++s)
          iss << format("  @{0} ld.global.nc.{1} %rrf{2}{3}, [%pf0 + {4}];", predk[s], ab_dtype, k, vs[s], (k+s)*dtsize) << std::endl;
        iss << format("  st.shared{1}.{2} [%writef + {4}], %rrf{3};", predk[0], vv, ab_dtype, k, k*dtsize) << std::endl;
      }

      iss << format("  //Load I: pi[offIhw_p_q + offIn + c*HWN]") << std::endl;
      iss << format("  mad.lo.u32 %offI, %c, %HWN, %offIn;") << std::endl;
      for(size_t pq = 0; pq < pl*ql; pq+=bf_pq){
      iss << format("  setp.ne.and.s32 %predi{0}, %offIhw{0}, -1, %predcrs;", pq) << std::endl;
      iss << format("  add.u32 %offI{0}, %offIhw{0}, %offI;", pq) << std::endl;
      iss << format("  mad.wide.u32 %pi{0}, %offI{0}, {1}, %pi;", pq, dtsize) << std::endl;
      }

      for(size_t pq = 0; pq < pl*ql; pq+=bf_pq){
        for(size_t n = 0; n < nl; n+=vec_*bf_n_){
          int pqn = pq*nl + n;
          for(size_t s = 0; s < vec_ ; ++s)
            predn[s] = format("%predi{}", pq);
          for(size_t s = 0; s < (last_iter?vec_:0) ; ++s){
            iss << format("  setp.lt.and.s32 %pred{}, {}, %boundN, %predi{};", s, n+s, pq) << std::endl;
            predn[s] = format("%pred{}", s);
          }
          for(size_t s = 0; s < ((check_lut)?vec_:0); ++s)
            if(dtype_==HALF_TYPE)
              iss << format("  @!{0} and.b16 %rri{2}{3}, 0, %rri{2}{3};", predn[s], ab_dtype, n, vs[s]) << std::endl;
            else
              iss << format("  @!{0} mov.{1} %rri{2}{3}, 0.;", predn[s], ab_dtype, n, vs[s]) << std::endl;
          for(size_t s = 0; s < vec_; ++s)
            iss << format("  @{0} ld.global.nc.{1} %rri{2}{3}, [%pi{4} + {5}];", predn[s], ab_dtype, n, vs[s], pq, (n+s)*dtsize) << std::endl;
          iss << format("  st.shared{0}.{1} [%writei + {2}], %rri{3};", vv, ab_dtype, pqn*dtsize, n) << std::endl;
        }
      }

    };

  auto ptr_sts = [&](char x, int cdx, int off, std::string const & fid0, std::string const & fid1){
    iss << format("  // write{0} = shared + {1} + fid{2}*{3} + fid{4}*{5}", x, off, fid0, Bvec, fid1, cdx) << std::endl;
    iss << format("  add.u32 %write{0}, %shared, {1};", x, off) << std::endl;
    iss << format("  mad.lo.cc.u32  %write{0}, %{1}, {2}, %write{0};", x, fid0, Bvec) << std::endl;
    iss << format("  mad.lo.cc.u32  %write{0}, %{1}, {2}, %write{0};", x, fid1, cdx) << std::endl;
  };

  auto ptr_lds = [&](char x, std::string const & id0, int off){
    iss << format("  // read{0} = shared + {1} + {2}*{3}", x, off, id0, Bvec*dtvec) << std::endl;
    iss << format("  add.u32 %read{0}, %shared, {1};", x, off) << std::endl;
    iss << format("  mad.lo.cc.u32  %read{0}, %{1}, {2}, %read{0};", x, id0, Bvec*dtvec) << std::endl;
  };

  auto lds = [&](char x, size_t nx, size_t crs, size_t cdx, size_t bs){
    for(size_t c = 0; c < cs_; c++)
      for(size_t rx = 0; rx < nx*vec_; rx+=vec_*dtvec){
        iss << format("  ld.shared{0}.{1} %r{2}{3}_{4}_{5}, [%read{2} + {6}];", vv, io_dtype, x, c, rx, crs%crs_s_, rx*bs*dtsize + (c*bc_ + crs*cl)*cdx) << std::endl;
      }
  };

  auto fma = [&](int cc){
    if(dtype_==HALF_TYPE){
      for(size_t c = 0 ; c < cs_ ; ++c)
      for(size_t k = 0; k < ks_; k+=dtvec*vec_)
      for(int kk = vec_-1 ; kk >=0 ; --kk){
        iss << format("  mov.b32 {{%rfh0, %rfh1}}, %rf{}_{}_{}{};", c, k, cc, vs[kk]) << std::endl;
        for(size_t sn = 0 ; sn < dtvec ; ++sn){
          iss << format("  mov.b32  %rf{}_{}_{}{}, {{%rfh{}, %rfh{}}};", c, vec_*((k + kk*dtvec + sn)/vec_), cc, vs[(k + kk*dtvec + sn) % vec_], sn, sn) << std::endl;
        }
      }
    }


    for(size_t c = 0; c < cs_; c++)
      for(size_t i = 0; i < pqns; i+=vec_*dtvec)
        for(size_t k = 0; k < ks_; k+=vec_*dtvec){
          for(size_t sk = 0 ; sk < dtvec ; ++sk)
          for(size_t kk = 0 ; kk < vec_ ; ++kk)
            for(size_t ii = 0 ; ii < vec_ ; ++ii){
              std::string ro = format("%ro{}_{}_{}{}", c, i, k + kk + sk*vec_, vs[ii]);
              std::string ri = format("%ri{}_{}_{}{}", c, i, cc, vs[ii]);
              std::string rf = format("%rf{}_{}_{}{}", c, k + sk*vec_, cc, vs[kk]);
              iss << format("  fma.rn.{0} {1}, {2}, {3}, {1};", dtype, ro, ri, rf) << std::endl;
            }
        }
  };

  auto lut = [&](){
    iss << "  // rs = tid" << std::endl;
    iss << "  mov.u32 %pqrs, %id;" << std::endl;
    iss << "  // extract rs" << std::endl;
    iss << format("  setp.lt.u32 %in_bounds, %pqrs, {};", RS*pl*ql) << std::endl;
    iss << "  @!%in_bounds bra END_LUT_LOOP;" << std::endl;
    iss << "LUT_LOOP:" << std::endl;
    iss << "  // extract rs" << std::endl;
    iss << format("  rem.s32 %rs, %pqrs, {};", RS) <<  std::endl;
    iss << "  // extract incp, incq" << std::endl;
    iss << format("  div.s32 %incpq, %pqrs, {};", RS) <<  std::endl;
    iss << format("  div.s32 %incp, %incpq, {};", ql) << std::endl;
    iss << format("  rem.s32 %incq, %incpq, {};", ql) << std::endl;
    iss << "  // pp = p*stride_h - pad_h" << std::endl;
    iss << "  add.s32 %pp, %base_p, %incp;" << std::endl;
    iss << "  mul.lo.s32 %pp, %pp, %stride_h;" << std::endl;
    iss << "  sub.s32 %pp, %pp, %pad_h;" << std::endl;
    iss << std::endl;
    iss << "  // qq = q*stride_w - pad_w" << std::endl;
    iss << "  add.s32 %qq, %base_q, %incq;" << std::endl;
    iss << "  mul.lo.s32 %qq, %qq, %stride_w;" << std::endl;
    iss << "  sub.s32 %qq, %qq, %pad_w;" << std::endl;
    iss << std::endl;
    iss << format("  div.u32 %r, %rs, {};", S_) << std::endl;
    iss << format("  rem.u32 %s, %rs, {};", S_) << std::endl;
    iss << std::endl;
    iss << "  // (h, w) = (pp + r, qq + s)" << std::endl;
    iss << "  add.s32 %h, %pp, %r;" << std::endl;
    iss << "  add.s32 %w, %qq, %s;" << std::endl;
    iss << std::endl;
    iss << "  // Write LUT" << std::endl;
    iss << format("  mad.lo.u32 %writelut, %rs, {}, %shared;", 4) << std::endl;
    iss << format("  mad.lo.u32 %writelut, %incpq, {}, %writelut;", 4*RS) << std::endl;

    iss << format("  // shared[{} + rs*{}] = offIhw = (w + h*W)*N", addr_lut_Ihw, 4) << std::endl;
    iss << "  mad.lo.s32 %offIhw, %h, %W, %w;" << std::endl;
    iss << "  mul.lo.s32 %offIhw, %offIhw, %N;" << std::endl;
    iss << "  setp.ge.s32 %in_bounds, %h, 0;" << std::endl;
    iss << "  setp.lt.and.s32 %in_bounds, %h, %H, %in_bounds;" << std::endl;
    iss << "  setp.ge.and.s32 %in_bounds, %w, 0, %in_bounds;" << std::endl;
    iss << "  setp.lt.and.s32 %in_bounds, %w, %W, %in_bounds;" << std::endl;
    iss << format("  @%in_bounds st.shared.u32 [%writelut + {}], %offIhw;", addr_lut_Ihw) << std::endl;
    iss << format("  @!%in_bounds st.shared.u32 [%writelut + {}], -1;", addr_lut_Ihw) << std::endl;
    iss << std::endl;
    iss << format("  // shared[{} + rs*{}] = offFrs = rs*K", addr_lut_Frs, 4) << std::endl;
    iss << format("  mad.lo.u32 %writelut, %rs, {}, %shared;", 4) << std::endl;
    iss << "  mul.lo.s32 %offFrs, %rs, %K;" << std::endl;
    iss << "  setp.eq.s32 %pq0, %incpq, 0;" << std::endl;
    iss << format("  @%pq0 st.shared.u32 [%writelut + {}], %offFrs;", addr_lut_Frs) << std::endl;
    iss << std::endl;
    iss << "  // Continue loop if necessary" << std::endl;
    iss << format("  add.u32 %pqrs, %pqrs, {};", bp_*bq_*bn_*bk_) << std::endl;
    iss << format("  setp.lt.u32 %in_bounds, %pqrs, {};", RS*pl*ql) << std::endl;
    iss << "  @%in_bounds bra LUT_LOOP;" << std::endl;
    iss << std::endl;
    iss << "END_LUT_LOOP:" << std::endl;
    iss << "  bar.sync 0;" << std::endl;
  };

  auto cc = device.compute_capability();
  iss << ".version 5.0" << std::endl;
  iss << ".target sm_" << cc.first << cc.second << std::endl;
  iss << ".address_size 64" << std::endl;
  iss << ".entry " << name << "(.param ." << alpha_dtype << " _alpha, .param ." << alpha_dtype << " _beta," << std::endl
      << "            .param .u64 _pi, .param .u64 _pf, .param .u64 _po," << std::endl
      << "            .param .s32 _C, .param .s32 _H, .param .s32 _W, .param .s32 _N, " << std::endl
      << "            .param .s32 _R, .param .s32 _S, .param .s32 _K," << std::endl
      << "            .param .s32 _P, .param .s32 _Q," << std::endl
      << "            .param .s32 _stride_h, .param .s32 _stride_w, .param .s32 _pad_h, .param .s32 _pad_w, " << std::endl
      << "            .param .s32 _HWN, .param .s32 _KRS, .param .s32 _PQN, .param .s32 _PQ, .param .s32 _QN, " << std::endl
      << "            .param .s32 _gridN, .param .s32 _gridQ)" << std::endl;
  iss << "{" << std::endl;
  iss << "  .reg.s32 %H, %pad_h, %stride_h, %W, %pad_w, %stride_w, %N, %K, %boundP, %boundQ, %boundN, %boundK;" << std::endl;
  iss << "  .reg.s32 %base_p, %base_q, %pp, %qq, %h, %w;" << std::endl;
  iss << "  .reg.u32 %c, %pqrs, %incpq, %incp, %incq, %rs, %r, %s, %rlo, %rhi, %slo, %shi, %offlut, %writelut, %readlut, %lut_size, %lut_max, %n_valid;" << std::endl;
  for(size_t p = 0; p < pl; p += bp_)
    for(size_t q = 0; q < ql; q += bq_){
      iss << format("  .reg.u32 %lut_max{}_{};", p, q) << std::endl;
      iss << format("  .reg.pred %predlut{}_{};", p, q) << std::endl;
    }
  iss << "  .reg.pred %rs0, %pq0, %in_bounds, %predkpq, %predp, %predq, %predcrs, %predk, %predloop;" << std::endl;
  iss << format("  .reg.pred %pred<{}>;", vec_*dtvec) << std::endl;
  iss << "  .reg.u32 %id, %idkpqn, %idc, %bc, %gidc, %div, %rem, %offc, %idpqn, %idpq, %idp, %idq, %idn, %idk, %fidpq, %fidp, %fidq, %fidn, %fidpqn_k, %fidcrs;" << std::endl;
  iss << format("  .reg .u32 %writei, %writef, %readi, %readf;") << std::endl;
  iss << "  .reg.u32 %bid0, %bk;" << std::endl;
  iss << "  .reg.s32 %bn, %bp, %bq, %bpq, %bpqn, %p, %q, %PQ, %Q, %P;" << std::endl;
  iss << format("  .reg .u64 %pi, %pf, %pf0;") << std::endl;
  iss << format("  .reg .u32 %offI, %offIbase, %offIn, %offIc, %offIhw, %offF, %offFk, %offFrs, %offFc, %offOk, %offOp, %offOq, %offOn;") << std::endl;
  iss << format("  .reg .s32 %C, %CRS, %HWN, %KRS, %CRSi, %gridN, %gridQ;") << std::endl;
  iss << format("  .reg .u64 %po;") << std::endl;
  iss << format("  .reg .s32 %PQN, %QN;") << std::endl;
  iss << format("  .reg .f32 %rcpRS, %CRSf;") << std::endl;
  iss << format("  .reg .{} %alpha, %beta;", io_dtype) << std::endl;
  if(dtype_==HALF_TYPE){
    iss << format("  .reg .f32 %alpha32;") << std::endl;
    iss << format("  .reg .f32 %beta32;") << std::endl;
    iss << format("  .reg .b16 %alpha16;") << std::endl;
    iss << format("  .reg .b16 %beta16;") << std::endl;
  }

  for(size_t n = 0; n < nl; n+=vec_*bf_n_)
    iss << format("  .reg {}.{} %rri{};", vv, ab_dtype, n) << std::endl;
  for(size_t k = 0; k < kl; k+=vec_*bf_k)
    iss << format("  .reg {}.{} %rrf{};", vv, ab_dtype, k) << std::endl;
  for(size_t pq = 0; pq < pl*ql; pq+=bf_pq){
      iss << format("  .reg .u32 %offIhw{};", pq) << std::endl;
      iss << format("  .reg .u32 %offI{};", pq) << std::endl;
      iss << format("  .reg .pred %predi{};", pq) << std::endl;
      iss << format("  .reg .u64 %pi{};", pq) << std::endl;
    }
  if(dtype_==HALF_TYPE){
    iss << format("  .reg .b16 %rfh0, %rfh1;") << std::endl;
  }
  iss << "  // For O tile" << std::endl;
  declare_register_tile('o', pqns, ks_, dtvec);
  iss << "  // For I tile" << std::endl;
  declare_register_tile('i', pqns, crs_s_, dtvec);
  iss << "  // For F tile" << std::endl;
  declare_register_tile('f', ks_, crs_s_, 1);

  iss << std::endl;
  iss << "  /* Initialize O */" << std::endl;
  for(size_t c = 0; c < cs_; c++)
  for(size_t i = 0 ; i < pqns ; i+=vec_*dtvec)
    for(size_t k = 0; k < ks_ ; ++k)
      for(size_t nn = 0; nn < vec_ ; ++nn)
        if(dtype_==HALF_TYPE)
          iss << format("  and.b32 %ro{1}_{2}_{3}{4}, 0, %ro{1}_{2}_{3}{4};", io_dtype, c, i, k, vs[nn]) << std::endl;
        else
          iss << format("  mov.{} %ro{}_{}_{}{}, 0.;", dtype, c, i, k, vs[nn]) << std::endl;


  iss << "  // Load Param" << std::endl;
  iss << "  ld.param.s32 %P, [_P];" << std::endl;
  iss << "  ld.param.s32 %Q, [_Q];" << std::endl;
  iss << "  ld.param.s32 %PQ, [_PQ];" << std::endl;
  iss << format("  ld.param.s32 %C, [_C];") << std::endl;
  iss << format("  ld.param.s32 %HWN, [_HWN];") << std::endl;
  iss << format("  ld.param.s32 %KRS, [_KRS];") << std::endl;
  iss << format("  ld.param.u64 %po, [_po];") << std::endl;
  iss << format("  ld.param.u64 %pi, [_pi];") << std::endl;
  iss << format("  ld.param.u64 %pf, [_pf];") << std::endl;
  iss << format("  ld.param.s32 %PQN, [_PQN];") << std::endl;
  iss << format("  ld.param.s32 %QN, [_QN];") << std::endl;
  iss << format("  ld.param.s32 %N, [_N];") << std::endl;
  iss << "  ld.param.s32 %pad_h, [_pad_h];" << std::endl;
  iss << "  ld.param.s32 %pad_w, [_pad_w];" << std::endl;
  iss << "  ld.param.s32 %stride_h, [_stride_h];" << std::endl;
  iss << "  ld.param.s32 %stride_w, [_stride_w];" << std::endl;
  iss << "  ld.param.s32 %H, [_H];" << std::endl;
  iss << "  ld.param.s32 %W, [_W];" << std::endl;
  iss << "  ld.param.s32 %N, [_N];" << std::endl;
  iss << "  ld.param.s32 %K, [_K];" << std::endl;
  iss << "  ld.param.s32 %gridN, [_gridN];" << std::endl;
  iss << "  ld.param.s32 %gridQ, [_gridQ];" << std::endl;


  iss << "  // Shared memory" << std::endl;
  iss << format("  .shared .align 16 .b8 _shared[{}];", size_shmem) << std::endl;
  iss << format("  .reg .u64 %shared64;") << std::endl;
  iss << format("  .reg .u32 %shared;") << std::endl;
  iss << format("  mov.u64 %shared64, _shared;") << std::endl;
  iss << format("  cvt.u32.u64 %shared, %shared64;") << std::endl;

  iss << std::endl;
  iss << "  // Thread ID" << std::endl;
  iss << "  mov.u32 %idpqn, %tid.x;" << std::endl;
  iss << format("  mov.u32 %idk, %tid.y;") << std::endl;
  iss << "  mov.u32 %idc, %tid.z;" << std::endl;
  iss << format("  mad.lo.u32 %idkpqn, %idk, {}, %idpqn;", bpqn) << std::endl;
  iss << format("  div.u32 %idpq, %idpqn, {};", bn_) << std::endl;
  iss << format("  rem.u32 %idn, %idpqn, {};", bn_) << std::endl;
  iss << format("  div.u32 %idp, %idpq, {};", bq_) << std::endl;
  iss << format("  rem.u32 %idq, %idpq, {};", bq_) << std::endl;
  iss << format("  mad.lo.u32 %id, %idkpqn, {}, %idc;", bc_) << std::endl;

  iss << std::endl;
  iss << "  // Block ID" << std::endl;
  iss << "  mov.u32 %bpqn, %ctaid.x;" << std::endl;
  iss << "  div.s32 %bpq, %bpqn, %gridN;" << std::endl;
  iss << "  div.s32 %bp, %bpq, %gridQ;" << std::endl;
  iss << "  rem.s32 %bq, %bpq, %gridQ;" << std::endl;
  iss << "  rem.s32 %bn, %bpqn, %gridN;" << std::endl;
  iss << "  mov.u32 %bk, %ctaid.y;" << std::endl;
  iss << "  mov.u32 %bc, %ctaid.z;" << std::endl;

  iss << "  // Pixels" << std::endl;
  iss << format("  mul.lo.u32 %base_p, %bp, {};", pl) << std::endl;
  iss << format("  mul.lo.u32 %base_q, %bq, {};", ql) << std::endl;

  if(gridc_>1){
    iss << format("  // Split") << std::endl;
    iss << format("  div.u32 %div, %C, {};", gridc_) << std::endl;
    iss << format("  rem.u32 %rem, %C, {};", gridc_) << std::endl;
    iss << "  mov.s32 %C, %div;" << std::endl;
    iss << "  mul.lo.u32 %offc, %bc, %div;" << std::endl;
    iss << "  setp.lt.u32 %pred0, %bc, %rem;" << std::endl;
    iss << "  @%pred0 add.s32 %C, %C, 1;" << std::endl;
    iss << "  @%pred0 add.s32 %offc, %bc, %offc;" << std::endl;
    iss << "  @!%pred0 add.s32 %offc, %rem, %offc;" << std::endl;
  }

  iss << std::endl;
  iss << "  /* --------------------------- */" << std::endl;
  iss << "  /* ------ Look-Up Table ------ */" << std::endl;
  iss << "  /* --------------------------- */" << std::endl;
  lut();
  iss << format("  mov.u32 %lut_size, {};", RS) << std::endl;
  iss << "  /* --------------------------- */" << std::endl;

  iss << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  iss << "  /* ------ Make lanes ------ */" << std::endl;
  iss << "  /* ---------------------------- */" << std::endl;
  iss << std::endl;
  iss << format("  div.u32 %fidcrs, %id, {};", bf_k) << std::endl;
  iss << format("  rem.u32 %fidpqn_k, %id, {};", bf_k) << std::endl;
  iss << format("  rem.u32 %fidn, %fidpqn_k, {};", bf_n_) << std::endl;
  iss << format("  div.u32 %fidpq, %fidpqn_k, {};", bf_n_) << std::endl;
  iss << std::endl;
  iss << format("  // STS Lanes") << std::endl;
  ptr_sts('i', cd_sharedi, addr_i, "fidn", "fidcrs");
  iss << format("  mad.lo.cc.u32  %writei, %fidpq, {}, %writei;", nl*dtsize) << std::endl;
  ptr_sts('f', cd_sharedf, addr_f, "fidpqn_k", "fidcrs");

  iss << format("  // LDS lanes") << std::endl;
  ptr_lds('i', "idn", addr_i);
  iss << format("  mad.lo.cc.u32  %readi, %idq, {}, %readi;", nl*dtsize) << std::endl;
  iss << format("  mad.lo.cc.u32  %readi, %idp, {}, %readi;", nl*ql*dtsize) << std::endl;
  iss << format("  mad.lo.cc.u32  %readi, %idc, {}, %readi;", cd_sharedi) << std::endl;
  ptr_lds('f', "idk", addr_f);
  iss << format("  mad.lo.cc.u32  %readf, %idc, {}, %readf;", cd_sharedf) << std::endl;

  iss << format("  mad.lo.s32 %boundN, %bn, -{}, %N;", nl) << std::endl;
  iss << format("  mad.lo.s32 %boundN, %fidn, -{}, %boundN;", vec_) << std::endl;

  iss << format("  mad.lo.s32 %boundK, %bk, -{}, %K;", kl) << std::endl;
  iss << format("  mad.lo.s32 %boundK, %fidpqn_k, -{}, %boundK;", vec_) << std::endl;



  iss << format("  // offIn = bn*{} + fidn*{}", nl, vec_) << std::endl;
  iss << format("  mul.lo.u32 %offIn, %bn, {};", nl) << std::endl;
  iss << format("  mad.lo.u32 %offIn, %fidn, {}, %offIn;", vec_) << std::endl;

  iss << format("  // offFk = bk*{} + fidpqn_k*{}", kl, vec_) << std::endl;
  iss << format("  mul.lo.u32 %offFk, %bk, {};", kl) << std::endl;
  iss << format("  mad.lo.u32 %offFk, %fidpqn_k, {}, %offFk;", vec_) << std::endl;


  iss << format("  // Strides and helpers") << std::endl;
  iss << format("  mul.lo.s32 %CRS, %C, %lut_size;") << std::endl;
  iss << format("  mov.s32 %CRSi, %CRS;") << std::endl;
  iss << format("  sub.s32 %CRS, %CRS, %fidcrs;") << std::endl;
  iss << format("  sub.s32 %CRS, %CRS, 1;") << std::endl;
  iss << format("  cvt.rn.f32.u32 %rcpRS, %lut_size;") << std::endl;
  iss << format("  rcp.rn.f32 %rcpRS, %rcpRS;") << std::endl;

  iss << "bar.sync 0;" << std::endl;
  iss << std::endl;
  iss << " //First load" << std::endl;
  iss << format("  setp.gt.s32 %predloop, %CRSi, {};", crs_l_*cl-1) << std::endl;
  iss << format("  @!%predloop bra LAST_ITER;") << std::endl;

  iss << format("  setp.lt.s32 %predcrs, %fidcrs, %CRSi;") << std::endl;
  ldg(true);
  iss << std::endl;
  iss << " //Main loop" << std::endl;
  iss << "LOOP:" << std::endl;

  iss << "  bar.sync 0;" << std::endl;
  for(size_t c = 0; c < crs_l_; c+=crs_s_){
    //Load I to registers
    for(size_t nc = 0; nc < cs_; ++nc)
    for(size_t cc = 0 ; cc < crs_s_ ; ++cc)
    for(size_t p = 0; p < ps_; p++)
    for(size_t q = 0; q < qs_; q++)
    for(size_t n = 0; n < ns_; n+=vec_*dtvec){
      int pq = p*qs_ + q;
      int pqn = pq*ns_ + n;
      int off = (n*bn_ + q*bq_*nl + p*bp_*nl*ql)*dtsize + (nc*bc_ + (c + cc)*cl)*cd_sharedi;
      iss << format("  ld.shared{0}.{1} %ri{2}_{3}_{4}, [%readi + {5}];", vv, io_dtype, nc, pqn, (c+cc)%crs_s_, off) << std::endl;
    }
    //Load F to registers
    for(size_t cc = 0 ; cc < crs_s_ ; ++cc)
      lds('f', ks_/vec_, c + cc, cd_sharedf, bk_);
    //FFMA
    for(size_t cc = 0 ; cc < crs_s_ ; ++cc)
      fma(cc);
  }
  for(char x: std::vector<char>{'i', 'f'}){
    iss << format("  xor.b32 %write{0}, %write{0}, {1};", x, double_buffer_off) << std::endl;
    iss << format("  xor.b32 %read{0}, %read{0}, {1};", x, double_buffer_off) << std::endl;
  }
  iss << format("  sub.s32 %CRS, %CRS, {};", crs_l_*cl) << std::endl;
  iss << format("  sub.s32 %CRSi, %CRSi, {};", crs_l_*cl) << std::endl;
  iss << format("  setp.lt.s32 %predcrs, %fidcrs, %CRSi;") << std::endl;
  iss << format("  setp.gt.s32 %predloop, %CRSi, {};", crs_l_*cl-1) << std::endl;
  ldg(false);
  iss << format("  @%predloop bra LOOP;") << std::endl;

  iss << "LAST_ITER:" << std::endl;
  iss << format("  setp.eq.s32 %predloop, %CRSi, 0;") << std::endl;
  iss << format("  @%predloop bra ENDLOOP;") << std::endl;
  iss << std::endl;
  iss << "  mov.u32 %CRS, %fidcrs;" << std::endl;
  iss << format("  setp.lt.s32 %predcrs, %fidcrs, %CRSi;") << std::endl;
  ldg(true);
  iss << "  bar.sync 0;" << std::endl;
  iss << "LAST_FMA:" << std::endl;
  for(size_t nc = 0; nc < cs_; ++nc)
  for(size_t p = 0; p < ps_; p++)
  for(size_t q = 0; q < qs_; q++)
  for(size_t n = 0; n < ns_; n+=vec_*dtvec){
    int pq = p*qs_ + q;
    int pqn = pq*ns_ + n;
    int off = (n*bn_ + q*bq_*nl + p*bp_*nl*ql)*dtsize + nc*bc_*cd_sharedi;
    iss << format("  ld.shared{}.{} %ri{}_{}_{}, [%readi + {}];", vv, io_dtype, nc, pqn, 0, off) << std::endl;
  }
  lds('f', ks_/vec_, 0, cd_sharedf, bk_);
  fma(0);
  iss << format("  add.u32 %readi, %readi, {};", cl*cd_sharedi) << std::endl;
  iss << format("  add.u32 %readf, %readf, {};", cl*cd_sharedf) << std::endl;
  iss << format("  sub.s32 %CRSi, %CRSi, {};", cl) << std::endl;
  iss << format("  setp.gt.s32 %predloop, %CRSi, 0;") << std::endl;
  iss << format("  @%predloop bra LAST_FMA;") << std::endl;

  iss << "ENDLOOP:" << std::endl;

  //Reduce in registers
  for(size_t c = 1; c < cs_; ++c)
    for(size_t k = 0; k < ks_; k ++)
    for(size_t pqn = 0; pqn < pqns; pqn += vec_*dtvec)
      for(size_t s = 0; s < vec_; s++)
      iss << format("  add.{0} %ro0_{2}_{3}{4}, %ro{1}_{2}_{3}{4}, %ro0_{2}_{3}{4};", dtype, c, pqn, k, vs[s]) << std::endl;

  if(bc_>1)
  {
    size_t bkpqn = nthreads/bc_;
    iss << ".reg .u32 %readc, %writec, %rid_kpqn, %rid_c;" << std::endl;
    iss << ".reg .pred %predc;" << std::endl;
    for(size_t kpqn = 0; kpqn < pqnl*kl; kpqn += bkpqn)
      iss << format("  .reg .{0} %rrc{1}_0, %rrc{1}_1;", ab_dtype, kpqn) << std::endl;

    iss << format("  mad.lo.cc.u32 %writec, %idc, {}, %shared;", kl*pqnl*dtsize) << std::endl;

    iss << format("  mad.lo.cc.u32 %writec, %idkpqn, {}, %writec;", ks_*pqns*dtsize) << std::endl;

    iss << "  bar.sync 0;" << std::endl;
    for(size_t k = 0; k < ks_; k ++)
    for(size_t pqn = 0; pqn < pqns; pqn += vec_*dtvec)
    for(size_t s = 0; s < vec_; s++){
      iss << format("  st.shared.{} [%writec + {}], %ro0_{}_{}{};", io_dtype, ((pqn+s*dtvec) + k*pqns)*dtsize, pqn, k, vs[s]) << std::endl;
    }
    iss << "  bar.sync 0;" << std::endl;

    iss << std::endl;
    iss << format("  div.u32 %rid_kpqn, %id, {};", bc_) << std::endl;
    iss << format("  rem.u32 %rid_c, %id, {};", bc_) << std::endl;
    iss << format("  mad.lo.cc.u32 %readc, %rid_c, {}, %shared;", kl*pqnl*dtsize) << std::endl;

    iss << format("  mad.lo.cc.u32 %readc, %rid_kpqn, {}, %readc;", dtsize) << std::endl;

    for(size_t c = bc_/2; c > 0; c /=2){
      iss << format("  setp.lt.u32 %predc, %rid_c, {};", c) << std::endl;
      for(size_t kpqn = 0; kpqn < pqnl*kl; kpqn += bkpqn){
        iss << format("  @%predc ld.shared.{} %rrc{}_0, [%readc + {}];", ab_dtype, kpqn, (kpqn)*dtsize) << std::endl;
        iss << format("  @%predc ld.shared.{} %rrc{}_1, [%readc + {}];", ab_dtype, kpqn, (kpqn + c*pqnl*kl)*dtsize) << std::endl;
        iss << format("  @%predc add.{0} %rrc{1}_0, %rrc{1}_0, %rrc{1}_1;", sub_dtype, kpqn) << std::endl;
        iss << format("  @%predc st.shared.{} [%readc + {}], %rrc{}_0;", ab_dtype, kpqn*dtsize, kpqn) << std::endl;
      }
      iss << "  bar.sync 0;" << std::endl;
    }

    iss << format("  mad.lo.cc.u32 %readc, %idkpqn, {}, %shared;", ks_*pqns*dtsize) << std::endl;

    for(size_t k = 0; k < ks_; k ++)
    for(size_t pqn = 0; pqn < pqns; pqn += vec_*dtvec)
    for(size_t s = 0; s < vec_; s++){
      iss << format("  ld.shared.{} %ro0_{}_{}{}, [%readc + {}];", io_dtype, pqn, k, vs[s], ((pqn+s*dtvec) + k*pqns)*dtsize) << std::endl;
    }
  }

  iss << std::endl;
  iss << "SCALE:" << std::endl;
  if(dtype_==HALF_TYPE){
    iss << format("  ld.param.f32 %alpha32, [_alpha];", io_dtype) << std::endl;
    iss << "  cvt.rn.f16.f32 %alpha16, %alpha32;" << std::endl;
    iss << "  mov.b32 %alpha, {%alpha16, %alpha16};" << std::endl;
  }
  else
    iss << format("  ld.param.{} %alpha, [_alpha];", io_dtype) << std::endl;
  for(size_t k = 0; k < ks_ ; k++)
    for(size_t n = 0 ; n < ns_ ; n+=vec_*dtvec)
      for(size_t s = 0; s < vec_; ++s)
        iss << format("  mul.{0} %ro0_{1}_{2}{3}, %ro0_{1}_{2}{3}, %alpha;", dtype, n, k, vs[s]) << std::endl;

  iss << std::endl;
  iss << "STORE_O:" << std::endl;
  iss << "  // Write back" << std::endl;

  iss << "  mov.u32 %idpqn, %tid.x;" << std::endl;
  iss << format("  div.u32 %idpq, %idpqn, {};", bn_) << std::endl;
  iss << format("  rem.u32 %idn, %idpqn, {};", bn_) << std::endl;
  iss << format("  div.u32 %idp, %idpq, {};", bq_) << std::endl;
  iss << format("  rem.u32 %idq, %idpq, {};", bq_) << std::endl;

  iss << format("  mad.lo.cc.u32 %boundN, %fidn, {}, %boundN;", vec_) << std::endl;
  iss << format("  mad.lo.cc.u32 %boundK, %fidpqn_k, {}, %boundK;", vec_) << std::endl;
  iss << format("  mad.lo.cc.u32 %boundN, %idn, -{}, %boundN;", vec_*dtvec) << std::endl;
  iss << format("  mad.lo.cc.u32 %boundK, %idk, -{}, %boundK;", vec_*dtvec) << std::endl;

  iss << format("  mad.lo.cc.u32 %offOk, %bk, {}, 0;", kl) << std::endl;

  iss << format("  mad.lo.cc.u32  %offOk, %idk, {}, %offOk;", vec_*dtvec) << std::endl;

  iss << format("  mad.lo.cc.u32 %offOp, %bp, {}, %idp;", pl) << std::endl;

  iss << format("  mad.lo.cc.u32 %offOq, %bq, {}, %idq;", ql) << std::endl;

  iss << "  sub.s32 %boundP, %P, %offOp;" << std::endl;
  iss << "  sub.s32 %boundQ, %Q, %offOq;" << std::endl;

  iss << format("  mad.lo.cc.u32 %offOn, %bn, {}, 0;", nl) << std::endl;

  iss << format("  mad.lo.cc.u32  %offOn, %idn, {}, %offOn;", vec_*dtvec) << std::endl;

  iss << format("  shl.b32 %N, %N, {};", log2(dtsize)) << std::endl;
  iss << format("  shl.b32 %QN, %QN, {};", log2(dtsize)) << std::endl;
  iss << format("  shl.b32 %PQN, %PQN, {};", log2(dtsize)) << std::endl;

  iss << format("  mad.wide.u32 %po, %offOk, %PQN, %po;") << std::endl;
  iss << format("  mad.wide.u32 %po, %offOp, %QN, %po;") << std::endl;
  iss << format("  mad.wide.u32 %po, %offOq, %N, %po;") << std::endl;
  iss << format("  mad.wide.u32 %po, %offOn, {}, %po;", dtsize) << std::endl;

  iss << format("  setp.eq.s32 %predk, %idc, 0;") << std::endl;
  int inc_k = 0;
  for(size_t k = 0; k < ks_ ; k++){
    iss << format("  setp.lt.and.s32 %predk, {}, %boundK, %predk;", inc_k) << std::endl;
    int step_k = ((k+1)%(vec_*dtvec))?1:(bk_*vec_*dtvec - vec_*dtvec + 1);
    for(size_t p = 0; p < ps_; p++){
      iss << format("  setp.lt.s32 %predp, {}, %boundP;", p*bp_) << std::endl;
      for(size_t q = 0; q < qs_; q++){
        iss << format("  setp.lt.s32 %predq, {}, %boundQ;", q*bq_) << std::endl;
        for(size_t n = 0 ; n < ns_ ; n+=vec_*dtvec){
          int pq = p*qs_ + q;
          int pqn = pq*ns_ + n;
          iss << format("  and.pred %predkpq, %predp, %predq;") << std::endl;
          iss << format("  and.pred %predkpq, %predk, %predkpq;") << std::endl;

          for(size_t s = 0; s < vec_*dtvec; ++s)
            iss << format("  setp.lt.and.s32 %pred{}, {}, %boundN, %predkpq;", s, bn_*n+s) << std::endl;
          if(gridc_!=1)
            for(size_t s = 0; s < vec_; ++s)
              iss << format("  @%pred{} red.add.{} [%po + {}], %ro0_{}_{}{};",  s, dtype, dtsize*(bn_*n + s*dtvec), pqn, k, vs[s]) << std::endl;
          else
            store(n, pqn, k);
        }
        iss << format("  mad.wide.u32 %po, %N, {}, %po;",bq_) << std::endl;
      }
      iss << format("  mad.wide.s32 %po, %N, -{}, %po;", ql) << std::endl;
      iss << format("  mad.wide.u32 %po, %QN, {}, %po;", bp_) << std::endl;
    }
    iss << format("  mad.wide.s32 %po, %QN, -{}, %po;", pl) << std::endl;
    iss << format("  mad.wide.u32 %po, %PQN, {}, %po;", step_k) << std::endl;
    inc_k += step_k;
  }
  iss << "}" << std::endl;
//  std::cout << iss.str() << std::endl;
  return iss.str();
}

void Conv::enqueue(driver::Kernel& kernel, driver::Stream& queue, scalar const & alpha, driver::Buffer const & I, driver::Buffer const & F, scalar const & beta, driver::Buffer& O){
  //Arguments
  int32_t HWN = H_*W_*N_;
  int32_t KRS = K_*R_*S_;
  int32_t PQN = P_*Q_*N_;
  int32_t PQ = P_*Q_;
  int32_t QN = Q_*N_;

  int32_t pl = bp_*ps_, ql = bq_*qs_, nl = bn_*ns_, kl = bk_*ks_;
  size_t gridP = ceil(P_, pl), gridQ = ceil(Q_, ql), gridN = ceil(N_, nl), gridK = ceil(K_, kl);
  DType alpha_dtype = (dtype_==DOUBLE_TYPE)?DOUBLE_TYPE:FLOAT_TYPE;

  kernel.setArg(0, size_of(alpha_dtype), alpha.data());
  kernel.setArg(1, size_of(alpha_dtype), beta.data());
  kernel.setArg(2, I);
  kernel.setArg(3, F);
  kernel.setArg(4, O);
  kernel.setArg(5, C_);
  kernel.setArg(6, H_);
  kernel.setArg(7, W_);
  kernel.setArg(8, N_);
  kernel.setArg(9, R_);
  kernel.setArg(10, S_);
  kernel.setArg(11, K_);
  kernel.setArg(12, P_);
  kernel.setArg(13, Q_);
  kernel.setArg(14, stride_h_);
  kernel.setArg(15, stride_w_);
  kernel.setArg(16, pad_h_);
  kernel.setArg(17, pad_w_);
  kernel.setArg(18, HWN);
  kernel.setArg(19, KRS);
  kernel.setArg(20, PQN);
  kernel.setArg(21, PQ);
  kernel.setArg(22, QN);
  kernel.setArg(23, gridN);
  kernel.setArg(24, gridQ);

  if(gridc_>1)
    O.set_zero(queue);
  queue.enqueue(kernel, {gridP*gridQ*gridN, gridK, gridc_}, {bp_*bq_*bn_, bk_, bc_});
}

}
}
