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
#include "isaac/templates/gemm.h"
#define FMT_HEADER_ONLY
#include "../external/fmt/format.h"

#include "isaac/driver/backend.h"
#include "isaac/driver/module.h"
#include "isaac/driver/error.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/stream.h"
#include "isaac/driver/buffer.h"

#include "isaac/templates/error.hpp"
#include "isaac/scalar.h"

namespace drv = isaac::driver;
using fmt::format;

namespace isaac{
namespace templates{

GEMM::GEMM(DType dtype, IsaacOperation_t AT, IsaacOperation_t BT, param_t M, param_t N, param_t K, param_t offa, param_t lda, param_t offb, param_t ldb, param_t offc, param_t ldc,
     param_t vec, param_t bm, param_t kl, param_t bn, param_t ms, param_t ks, param_t ns, param_t a_bf0, param_t a_bf1, param_t b_bf0, param_t b_bf1, param_t rs, param_t br, param_t gridr):
  dtype_(dtype), AT_(AT), BT_(BT), M_(M), N_(N), K_(K), offa_(offa), lda_(lda), offb_(offb), ldb_(ldb), offc_(offc), ldc_(ldc),
  vec_(vec), bm_(bm), kl_(kl), bn_(bn), ms_(ms), ks_(ks), ns_(ns), a_bf0_(a_bf0), a_bf1_(a_bf1), b_bf0_(b_bf0), b_bf1_(b_bf1),
  rs_(rs), br_(br), gridr_(gridr)
{
  ms = std::min<param_t>(ms, M);
  ns = std::min<param_t>(ns, N);
  vec = std::min<param_t>(vec, M);
  vec = std::min<param_t>(vec, N);
}

double GEMM::tflops(param_t M, param_t N, param_t K, double time)
{ return (double)2*M*N*K/(time*1e3); }

std::string GEMM::id() const{
  std::ostringstream iss;
  size_t dtvec = (dtype_==HALF_TYPE)?2:1;
  iss << "_" << dtype_ << "_"
      << lda_ % dtvec << "_" << ldb_ % dtvec << "_" << ldc_ % dtvec << "_"
      << lda_ % (dtvec*vec_) << "_" << ldb_ % (dtvec*vec_) << "_" << ldc_ % (dtvec*vec_)<< "_"
      << vec_ << "_" << bm_ << "_" << kl_ << "_" << bn_ << "_" << ms_ << "_" << ks_ << "_" << ns_ << "_"
     << a_bf0_ << "_" << a_bf1_ << "_" << b_bf0_ << "_" << b_bf1_ << "_" << rs_ << "_" << br_ << "_" << gridr_;
  return iss.str();
}

void GEMM::check_valid(driver::Device const & device, size_t nkernels, uint32_t* params, uint8_t* valid){
  std::array<int, 20> x{0};
  for(size_t m = 0; m < nkernels; ++ m){
    //Parameters
    for(size_t i = 0; i < x.size(); ++i)
      x[i] = params[m*x.size() + i];
    DType dtype = (DType)(x[0]);
    IsaacOperation_t AT = (IsaacOperation_t)x[1];
    IsaacOperation_t BT = (IsaacOperation_t)x[2];
    param_t M = x[3], N = x[4], K = x[5], vec = x[6], bm = x[7], kl = x[8], bn = x[9],
           ms = x[10], ks = x[11], ns = x[12], a_bf0 = x[13], a_bf1 = x[14], b_bf0 = x[15], b_bf1 = x[16],
           rs = x[17], br = x[18], gridr = x[19];
    //Features
    param_t dtsize = size_of(dtype);
    param_t dtvec = (dtype==HALF_TYPE)?2:1;
    bool A_outer_contig = AT==ISAAC_OP_N;
    bool B_outer_contig = BT==ISAAC_OP_T;
    param_t rl = rs*br;
    param_t ml = bm*ms;
    param_t nl = bn*ns;
    param_t nthreads = bm*bn*br;
    param_t cd_shareda = dtsize*(ml+(A_outer_contig?0:(vec*dtvec)));
    param_t cd_sharedb = dtsize*(nl+(B_outer_contig?0:(vec*dtvec)));
    param_t ncd_shareda = kl*rl;
    param_t ncd_sharedb = kl*rl;
    param_t size_shareda = cd_shareda*ncd_shareda;
    param_t size_sharedb = cd_sharedb*ncd_sharedb;
    param_t size_redc = dtsize*ml*nl*(br==1?0:br);
    param_t size_tiles = 2*(size_shareda + size_sharedb);
    param_t size_shmem = std::max(size_redc, size_tiles);

    param_t npa = (A_outer_contig?(kl*rl):ml) / a_bf1;
    param_t npb = (B_outer_contig?(kl*rl):nl) / b_bf1;
    param_t nra = (A_outer_contig?ml:(kl*rl)) / a_bf0;
    param_t nrb = (B_outer_contig?nl:(kl*rl)) / b_bf0;
    param_t n_instructions =  nra*npa*2 + nrb*npb*2 + ms*ns*kl*rs/dtvec + kl*rs*ms*ns/(vec*dtvec);

    bool overshoot_A = ml > K*M;
    bool overshoot_B = nl > K*N;

    //Test
    bool is_valid =   a_bf0*a_bf1 == nthreads
                    &&  b_bf0*b_bf1 == nthreads
                    &&  (A_outer_contig?(kl*rl):ml) % a_bf1 == 0
                    &&  (A_outer_contig?ml:(kl*rl)) % (a_bf0*vec) == 0
                    &&  (B_outer_contig?(kl*rl):nl) % b_bf1 == 0
                    &&  (B_outer_contig?nl:(kl*rl)) % (b_bf0*vec) == 0

                    &&  ms % (dtvec*vec) == 0
                    &&  ns % (dtvec*vec) == 0
                    &&  kl % ks == 0
                    &&  size_shmem <= device.max_shared_memory()

                    && !overshoot_A
                    && !overshoot_B
//                    &&  nl<=N
//                    &&  kl<=K
//                    &&  ml<=M
                    &&  n_instructions <= 1024 //Doesn't allow more than 1024 instructions in the inner loop
                    &&  bm <= device.max_block_dim()[0]
                    &&  bn <= device.max_block_dim()[1]
                    &&  br <= device.max_block_dim()[2]
                    &&  rl*gridr<=K
                    &&  vec*dtsize <= 16
                    && (device.compute_capability().first>=6 || dtype != HALF_TYPE)
                    && (gridr==1 || dtype==FLOAT_TYPE);  //Global reduction on fp16x2 and fp64 not always possible
    valid[m] = is_valid;
  }
}

/* Code generation */
std::string GEMM::dump(drv::Device const & device, std::string const & name){

  bool A_outer_contig = AT_==ISAAC_OP_N;
  bool B_outer_contig = BT_==ISAAC_OP_T;
  size_t dtsize = size_of(dtype_);
  size_t dtvec = (dtype_==HALF_TYPE)?2:1;
  std::string dtype = arith_str(dtype_);
  std::string io_dtype = io_str(dtype_);
  std::string ab_dtype = (dtype_==HALF_TYPE)?"b16":dtype;
  std::string sub_dtype = (dtype_==HALF_TYPE)?"f16":dtype;

  size_t rl = rs_*br_;
  //vector prefix
  std::string vv = vec_>1?format(".v{}", vec_):"";
  //tile size
  size_t ml = bm_*ms_;
  size_t nl = bn_*ns_;
  //Number of lanes
  size_t npa = (A_outer_contig?(kl_*rl):ml) / a_bf1_;
  size_t npb = (B_outer_contig?(kl_*rl):nl) / b_bf1_;
  size_t nra = (A_outer_contig?ml:(kl_*rl)) / a_bf0_;
  size_t nrb = (B_outer_contig?nl:(kl_*rl)) / b_bf0_;
  //Number of threads
  size_t nthreads = bm_*bn_*br_;
  //Shared memory sizes/offsets
  size_t cd_shareda = dtsize*(ml+(A_outer_contig?0:(vec_*dtvec)));
  size_t cd_sharedb = dtsize*(nl+(B_outer_contig?0:(vec_*dtvec)));
  size_t ncd_shareda = kl_*rl;
  size_t ncd_sharedb = kl_*rl;
  size_t size_shareda = cd_shareda*ncd_shareda;
  size_t size_sharedb = cd_sharedb*ncd_sharedb;
  size_t off_sharedb =  size_shareda;
  size_t size_redc = dtsize*ml*nl*(br_==1?0:br_);
  size_t size_tiles = 2*(size_shareda + size_sharedb);
  size_t size_shmem = std::max(size_redc, size_tiles);
  size_t double_buffer_off = size_tiles/2;
  //Byte stride
  size_t Bvec = vec_*dtsize;
  size_t dtbits = log2(dtsize);
  //Vector suffix
  const char* vs[] = {".x", ".y", ".z", ".w"};
  if(vec_==1)
    vs[0] = "";

  uint8_t is_valid;
  param_t params[] = {dtype_, AT_, BT_, M_, N_, K_, vec_, bm_, kl_, bn_, ms_, ks_, ns_, a_bf0_, a_bf1_, b_bf0_, b_bf1_, rs_, br_, gridr_};
  check_valid(device, 1, params, &is_valid);
  if(!is_valid)
    throw invalid_parameters();

  std::stringstream iss;
  /* Global Loads */
  auto ptr_ldg = [&](char x, size_t /*bf0*/, size_t bf1, size_t npx, char axis, char bound, bool no_trans){
      if(no_trans)
        iss << format("  // p{0} += off{0} + B{0}fid0 + Bbid{1} + (fid1 + offk)*ld{0};", x, axis) << std::endl;
      else
        iss << format("  // p{0} += off{0} + B{0}fid0 + offk + bid{1}*ld{0} + fid1*ld{0};", x, axis) << std::endl;

      //Offset along contiguous dimension
      iss << format("  mad.wide.u32 %p{0}, %off{0}, 1, %p{0};", x) << std::endl;
      iss << format("  mad.wide.u32 %p{0}, %B{0}fid0, 1, %p{0};", x) << std::endl;
      if(no_trans)
        iss << format("  mad.wide.u32 %p{0}, %Bbid{1}, 1, %p{0};", x, axis) << std::endl;
      else if(gridr_>1)
        iss << format("  mad.wide.u32 %p{0}, %offk, {1}, %p{0};", x, dtsize) << std::endl;

      //Offset along non-contiguous dimension
      iss << format("  mov.s32 %off{0}1, %{0}fid1;", x) << std::endl;
      if(no_trans && gridr_>1)
        iss << format("  mad.wide.u32 %p{0}, %offk, %ld{0}, %p{0};", x) << std::endl;
      else if(!no_trans)
        iss << format("  add.s32 %off{0}1, %off{0}1, %bid{1};", x, axis) << std::endl;


      for(size_t rj = 0 ; rj < npx ; ++rj){
          iss << format("  // p{0}{1} = p{0} + (off{0}1 + {2})*ld{0};", x, rj, rj*bf1) << std::endl;
          iss << format("  setp.lt.s32 %pred{0}{1}, {2}, %{3};", x, rj, rj*bf1, (no_trans?'K':bound)) << std::endl;
          iss << format("  @%pred{0}{1} add.s32 %off{0}1_{1}, %off{0}1, {2};", x, rj, rj*bf1) << std::endl;
          iss << format("  @!%pred{0}{1} mov.s32 %off{0}1_{1}, 0;", x, rj, rj*bf1) << std::endl;
          iss << format("  mad.wide.u32 %p{0}{1}, %off{0}1_{1}, %ld{0}, %p{0};", x, rj) << std::endl;
      }

      if(no_trans)
        iss << format("  mul.wide.u32 %stepinc{0}, {1}, %ld{0};", x, kl_*rl) << std::endl;
      else
        iss << format("  cvt.u64.u32 %stepinc{0}, {1};", x, kl_*rl*dtsize) << std::endl;

  };

  auto ldg = [&](char x, size_t bf0, size_t bf1, size_t xl, size_t npx, size_t sharedcd, bool outer_contig, bool check_bounds){
    size_t i, ri, j, rj;
    std::vector<std::string> preds(vec_, "predk");
    std::string bd = (x=='a')?"M":"N";

    if(check_bounds)
        iss << format("  sub.s32 %{0}, %{0}, %{1}fid0;", outer_contig?bd:"K", x) << std::endl;

    for(rj = 0 ; rj < npx ; ++rj){
      if(check_bounds){
        iss << format("  setp.gt.s32 %predk, %{}, %{}fid1;", outer_contig?"K":bd, x) << std::endl;
        iss << format("  sub.s32 %{0}, %{0}, {1};", outer_contig?"K":bd, bf1) << std::endl;
      }
      for(ri = 0, i = 0 ; i < (outer_contig?xl:(kl_*rl)) ; i += bf0*vec_, ri+=vec_){
        if(check_bounds){
          for(size_t s = 0; s < vec_; ++s){
            iss << format("  setp.lt.and.s32 %pred{}, {}, %{}, %predk;",  s, i+s, outer_contig?bd:"K") << std::endl;
            preds[s] = format("pred{}", s);
          }
        }
        for(size_t s = 0; s < (check_bounds?vec_:0); ++s)
          if(dtype_==HALF_TYPE)
            iss << format("  @!%{0} and.b16 %rr{2}{3}_{4}{5}, 0, %rr{2}{3}_{4}{5};", preds[s], ab_dtype, x, ri, rj, vs[s]) << std::endl;
          else
            iss << format("  @!%{0} mov.{1} %rr{2}{3}_{4}{5}, 0.;", preds[s], dtype, x, ri, rj, vs[s]) << std::endl;
        for(size_t s = 0; s < vec_; ++s)
          iss << format("  @%{0} ld.global.nc.{1} %rr{2}{3}_{4}{5}, [%p{2}{4} + {6}];", preds[s], ab_dtype, x, ri, rj, vs[s], (i+s)*dtsize) << std::endl;
      }
    }

    for(rj = 0, j = 0 ; rj < npx ; j += bf1, rj++)
      for(ri = 0, i = 0 ; i < (outer_contig?xl:(kl_*rl)) ; i += bf0*vec_, ri+=vec_)
        if(outer_contig)
          iss << format("  st.shared{0}.{1} [%write{2} + {3}], %rr{2}{4}_{5};", vv, ab_dtype, x, i*dtsize + j*sharedcd, ri, rj) << std::endl;
        else
          for(size_t s = 0 ; s < vec_ ; ++s)
            iss << format("  st.shared.{0} [%write{1} + {2}], %rr{1}{3}_{4}{5};", ab_dtype, x, j*dtsize + (i+s)*sharedcd, ri, rj, vs[s]) << std::endl;
    if(check_bounds){
      iss << format("  add.s32 %{0}, %{0}, {1};", outer_contig?"K":bd, outer_contig?(kl_*rl):xl) << std::endl;
      iss << format("  add.s32 %{0}, %{0}, %{1}fid0;", outer_contig?bd:"K", x) << std::endl;
    }
  };

  auto ptr_sts = [&](char x, size_t cdx, size_t off, bool outer_contiguous){
    std::string id0 = "0", id1 = "1";
    if(!outer_contiguous) std::swap(id0, id1);
    iss << format("  // write{0} = shared + {1} + Bfid{2} + fid{3}*{4}", x, off, id0, id1, cdx) << std::endl;
    iss << format("  mov.u32 %write{}, %shared;", x) << std::endl;
    iss << format("  add.u32 %write{0}, %write{0}, {1};", x, off) << std::endl;
    iss << format("  add.u32 %write{0}, %write{0}, %B{0}fid{1};", x, id0) << std::endl;
    iss << format("  mad.lo.u32  %write{0}, %{0}fid{1}, {2}, %write{0};", x, id1, cdx) << std::endl;
  };

  auto ptr_lds = [&](char x, char axis, size_t off, size_t cd_shared){
    iss << format("  // read{0} = shared + {1} + id{2}*{3}", x, off, axis, Bvec*dtvec) << std::endl;
    iss << format("  mov.u32 %read{}, %shared;", x) << std::endl;
    iss << format("  add.u32 %read{0}, %read{0}, {1};", x, off) << std::endl;
    iss << format("  mad.lo.u32  %read{0}, %id{1}, {2}, %read{0};", x, axis, Bvec*dtvec) << std::endl;
    iss << format("  mad.lo.u32  %read{0}, %idr, {1}, %read{0};", x, cd_shared) << std::endl;

  };

  auto lds = [&](char x, size_t nx, size_t k, size_t cdx, size_t bs){
    for(size_t r = 0; r < rs_; ++r)
    for(size_t rx = 0; rx < nx; rx+=vec_*dtvec){
      iss << format("  ld.shared{0}.{1} %r{2}{3}_{4}_{5}, [%read{2} + {6}];", vv, io_dtype, x, r, rx, k%ks_, rx*bs*dtsize + (r*br_ + k*rl)*cdx) << std::endl;
    }
  };

  auto fma = [&](size_t kk){
    if(dtype_==HALF_TYPE){
      for(size_t r = 0 ; r < rs_ ; ++r)
      for(size_t n = 0; n < ns_; n+=dtvec*vec_)
      for(int nn = vec_-1 ; nn >=0 ; --nn){
        iss << format("  mov.b32 {{%rbh0, %rbh1}}, %rb{}_{}_{}{};", r, n, kk, vs[nn]) << std::endl;
        for(size_t sn = 0 ; sn < dtvec ; ++sn){
          iss << format("  mov.b32  %rb{}_{}_{}{}, {{%rbh{}, %rbh{}}};", r, vec_*((n + nn*dtvec + sn)/vec_), kk, vs[(n + nn*dtvec + sn) % vec_], sn, sn) << std::endl;
        }
      }
    }

    for(size_t r = 0 ; r < rs_ ; ++r)
    for(size_t m = 0; m < ms_; m+=dtvec*vec_)
      for(size_t n = 0; n < ns_; n+=dtvec*vec_){
        for(size_t sn = 0 ; sn < dtvec ; ++sn)
          for(size_t nn = 0 ; nn < vec_ ; ++nn)
            for(size_t mm = 0 ; mm < vec_ ; ++mm){
              std::string rc = format("%rc{}_{}_{}{}", r, m, n + nn + sn*vec_, vs[mm]);
              std::string ra = format("%ra{}_{}_{}{}", r, m, kk, vs[mm]);
              std::string rb = format("%rb{}_{}_{}{}", r, n + sn*vec_, kk, vs[nn]);
              iss << format("  fma.rn.{0} {1}, {2}, {3}, {1};", dtype, rc, ra, rb) << std::endl;
            }
      }
  };

  auto declare_register_tile = [&](char x, size_t M, size_t N, size_t dtinc){
    for(size_t r = 0 ; r < rs_ ; ++r)
      for(size_t m = 0 ; m < M ; m+=dtinc*vec_){
        iss << format("  .reg {}.{}", vv, io_dtype);
        for(size_t n = 0 ; n < N ; n++)
          iss << format("{} %r{}{}_{}_{}", n>0?",":"", x, r, m, n);
        iss << ";" << std::endl;
      }
  };

  auto store = [&](size_t m, size_t n){
    size_t inc = 1;
    std::string cvec = "";
    std::string subvec = ab_dtype;
    if((ldc_/inc) % dtvec == 0){
      inc *= dtvec;
      if(dtype_==HALF_TYPE)
        subvec = format("b{}", dtsize*8*dtvec);
    }
    if((ldc_/inc) % vec_ == 0){
      inc *= vec_;
      cvec = vec_>1?format(".v{}", vec_):"";
    }
    for(size_t s = 0; s < vec_*dtvec; s+=inc){
      if(subvec=="b16"){
        if(s%dtvec==0)
          iss << format("  mov.b32 {{%rbh0, %rbh1}}, %rc0_{}_{}{};", m, n, vs[s/dtvec]) << std::endl;
        iss << format("  @%pred{} st.global{}.{} [%pc + {}], %rbh{};", s, cvec, subvec, dtsize*(bm_*m+s), s%dtvec) << std::endl;
      }
      else
        iss << format("  @%pred{} st.global{}.{} [%pc + {}], %rc0_{}_{}{};", s, cvec, subvec, dtsize*(bm_*m+s), m, n, (inc/dtvec > 1)?"":vs[s/dtvec]) << std::endl;
    }
  };

  auto cc = device.compute_capability();
  iss << ".version 5.0" << std::endl;
  iss << ".target sm_" << cc.first << cc.second << std::endl;
  iss << ".address_size 64" << std::endl;
  iss << ".entry " << name << "(.param .s32 _M, .param .s32 _N, .param .s32 _K," << std::endl;
  iss << "    .param .u64 _pc, .param .s32 _ldc, .param .s32 _offc," << std::endl;
  iss << "    .param ." << ab_dtype << " _alpha, .param .u64 _pa, .param .s32 _lda, .param .s32 _offa," << std::endl;
  iss << "    .param .u64 _pb, .param .s32 _ldb, .param .s32 _offb," << std::endl;
  iss << "    .param ." << ab_dtype << " _beta)" << std::endl;
  iss << "{" << std::endl;

  iss << std::endl;
  iss << "  /* Registers */" << std::endl;
  iss << "  // For C tile" << std::endl;
  declare_register_tile('c', ms_, ns_, dtvec);

  iss << "  // For A tile" << std::endl;
  declare_register_tile('a', ms_, ks_, dtvec);

  iss << "  // For B tile" << std::endl;
  declare_register_tile('b', ns_, ks_, 1);

  iss << "  // Parameters" << std::endl;
  iss << format("  .reg .s32 %M, %N, %K;") << std::endl;
  if(dtype_==HALF_TYPE){
    iss << format("  .reg .b16 %alpha16;") << std::endl;
    iss << format("  .reg .b16 %beta16;") << std::endl;
  }
  iss << format("  .reg .{} %alpha;", io_dtype) << std::endl;
  iss << format("  .reg .{} %beta;", io_dtype) << std::endl;
  for(char x: std::vector<char>{'c', 'a', 'b'}){
    iss << format("  .reg .u64 %p{0};", x) << std::endl;
    iss << format("  .reg .s32 %ld{0}, %off{0};", x) << std::endl;
  }
  iss << "  // IDs" << std::endl;
  iss << format("  .reg .u32 %bid0, %bid1;") << std::endl;
  iss << format("  .reg .u32 %idmn, %id;") << std::endl;
  iss << format("  .reg .u32 %afid0, %afid1, %bfid0, %bfid1;") << std::endl;
  iss << format("  .reg .u32 %Bbid0, %Bbid1, %Bafid0, %Bafid1, %Bbfid0, %Bbfid1;") << std::endl;
  iss << format("  .reg .u32 %idu, %id0, %id1;") << std::endl;
  iss << "  // Lanes in global memory" << std::endl;
  iss << format("  .reg .u64 %pa<{0}>;", npa) << std::endl;
  iss << format("  .reg .u64 %pb<{0}>;", npb) << std::endl;
  iss << format("  .reg .pred %pred<{0}>;", vec_*dtvec) << std::endl;
  iss << format("  .reg .pred %predk, %predn;") << std::endl;
  iss << format("  .reg .u64 %stepinca, %stepincb;") << std::endl;
  iss << "  // Lanes in shared memory" << std::endl;
  iss << format("  .reg .u32 %writea, %writeb;") << std::endl;
  iss << format("  .reg .u32 %reada, %readb;") << std::endl;
  iss << format("  .reg .u32 %toff;") << std::endl;
  iss << format("  .reg .u64 %btoff;") << std::endl;
  iss << format("  .reg .u32 %offc0, %offc1;") << std::endl;
  iss << format("  .reg .u32 %offa1, %offb1;") << std::endl;
  for(size_t rj = 0 ; rj < npa ; ++rj){
      iss << format("  .reg .pred %preda{};", rj) << std::endl;
      iss << format("  .reg .u32 %offa1_{};", rj) << std::endl;
  }
  for(size_t rj = 0 ; rj < npb ; ++rj){
      iss << format("  .reg .pred %predb{};", rj) << std::endl;
      iss << format("  .reg .u32 %offb1_{};", rj) << std::endl;
  }

  iss << format("  .reg .u32 %div, %rem, %idr, %bidr, %offk;") << std::endl;
  iss << ".reg .pred %predr;" << std::endl;

  if(dtype_==HALF_TYPE){
    iss << format("  .reg .b16 %rbh0, %rbh1;") << std::endl;
  }

  for(size_t ri = 0 ; ri < nra ; ri+=vec_)
    for(size_t rj = 0 ; rj < npa ; rj++){
      iss << format("  .reg {}.{} %rra{}_{};", vv, ab_dtype, ri, rj) << std::endl;
    }

  for(size_t ri = 0 ; ri < nrb ; ri+=vec_)
    for(size_t rj = 0 ; rj < npb ; rj++){
      iss << format("  .reg {}.{} %rrb{}_{};", vv, ab_dtype, ri, rj) << std::endl;
    }

  iss << std::endl;
  iss << "  /* Shared */" << std::endl;
  iss << format("  .reg .u64 %shared64;") << std::endl;
  iss << "  .reg .u32 %shared;" << std::endl;
  iss << format("  .shared .align 16 .b8 _shared[{}];", size_shmem) << std::endl;
  iss << format("  mov.u64 %shared64, _shared;") << std::endl;
  iss << format("  cvt.u32.u64 %shared, %shared64;") << std::endl;
  iss << std::endl;
  iss << "  /* Initialize C */" << std::endl;
  for(size_t r = 0 ; r < rs_ ; ++r)
  for(size_t m = 0 ; m < ms_ ; m+=dtvec*vec_)
    for(size_t n = 0; n < ns_ ; ++n)
      for(size_t mm = 0; mm < vec_ ; ++mm)
        if(dtype_==HALF_TYPE)
          iss << format("  and.b32 %rc{1}_{2}_{3}{4}, 0, %rc{1}_{2}_{3}{4};", io_dtype, r, m, n, vs[mm]) << std::endl;
        else
          iss << format("  mov.{} %rc{}_{}_{}{}, 0.;", dtype, r, m, n, vs[mm]) << std::endl;



  iss << std::endl;
  iss << "  /* Load parameters */" << std::endl;
  for(char x: std::vector<char>{'a', 'b', 'c'}){
    iss << format("  ld.param.u64 %p{0}, [_p{0}];", x) << std::endl;
    iss << format("  ld.param.s32 %off{0}, [_off{0}];", x) << std::endl;
    iss << format("  ld.param.s32 %ld{0}, [_ld{0}];", x) << std::endl;
  }
  for(char x: std::vector<char>{'M','N','K'})
    iss << format("  ld.param.u32 %{0}, [_{0}];", x) << std::endl;

  iss << std::endl;
  iss << "  /* Block ID */" << std::endl;
  iss << format("  mov.u32 %bid0, %ctaid.x;") << std::endl;
  iss << format("  mov.u32 %bid1, %ctaid.y;") << std::endl;
  iss << format("  mov.u32 %bidr, %ctaid.z;") << std::endl;

  iss << std::endl;
  iss << "  /* Adjust IDs and bounds */" << std::endl;
  iss << format("  mul.lo.u32 %bid0, %bid0, {};", ml) << std::endl;
  iss << format("  mul.lo.u32 %bid1, %bid1, {};", nl) << std::endl;

  iss << std::endl;
  iss << "  /* Thread ID */" << std::endl;
  iss << format("  mov.u32 %id0, %tid.x;") << std::endl;
  iss << format("  mov.u32 %id1, %tid.y;") << std::endl;
  iss << format("  mov.u32 %idr, %tid.z;") << std::endl;

  if(gridr_ > 1){
    iss << format("  // Split") << std::endl;
    iss << format("  div.u32 %div, %K, {};", gridr_) << std::endl;
    iss << format("  rem.u32 %rem, %K, {};", gridr_) << std::endl;
    iss << "  mov.s32 %K, %div;" << std::endl;
    iss << "  mul.lo.u32 %offk, %bidr, %div;" << std::endl;
    iss << "  setp.lt.u32 %pred0, %bidr, %rem;" << std::endl;
    iss << "  @%pred0 add.s32 %K, %K, 1;" << std::endl;
    iss << "  @%pred0 add.s32 %offk, %bidr, %offk;" << std::endl;
    iss << "  @!%pred0 add.s32 %offk, %rem, %offk;" << std::endl;
  }

  iss << std::endl;
  iss << "  /* Fetch ID */" << std::endl;
  iss << format("  mad.lo.u32 %idmn, %id1, {}, %id0;", bm_) << std::endl;

  iss << format("  mad.lo.u32 %id, %idmn, {}, %idr;", br_) << std::endl;


  iss << format("  div.u32 %afid1, %id, {};", a_bf0_) << std::endl;
  iss << format("  rem.u32 %afid0, %id, {};", a_bf0_) << std::endl;
  iss << format("  div.u32 %bfid1, %id, {};", b_bf0_) << std::endl;
  iss << format("  rem.u32 %bfid0, %id, {};", b_bf0_) << std::endl;
  iss << format("  mul.lo.u32 %afid0, %afid0, {};", vec_) << std::endl;
  iss << format("  mul.lo.u32 %bfid0, %bfid0, {};", vec_) << std::endl;
  iss << std::endl;
  iss << "  /* Scale by dtype size */" << std::endl;
  iss << format("  shl.b32 %Bafid0, %afid0, {};", dtbits) << std::endl;
  iss << format("  shl.b32 %Bafid1, %afid1, {};", dtbits) << std::endl;
  iss << format("  shl.b32 %Bbfid0, %bfid0, {};", dtbits) << std::endl;
  iss << format("  shl.b32 %Bbfid1, %bfid1, {};", dtbits) << std::endl;
  iss << format("  shl.b32 %Bbid0, %bid0, {};", dtbits) << std::endl;
  iss << format("  shl.b32 %Bbid1, %bid1, {};", dtbits) << std::endl;
  for(char x: std::vector<char>{'a', 'b'}){
    iss << format("  shl.b32 %off{0}, %off{0}, {1};", x, dtbits) << std::endl;
    iss << format("  shl.b32 %ld{0}, %ld{0}, {1};", x, dtbits) << std::endl;
  }

  iss << "  // Adjust bounds" << std::endl;
  iss << "  sub.s32 %M, %M, %bid0;" << std::endl;
  iss << "  sub.s32 %N, %N, %bid1;" << std::endl;
  iss << format("  sub.s32 %M, %M, %{};", A_outer_contig?"afid0":"afid1") << std::endl;
  iss << format("  sub.s32 %N, %N, %{};", B_outer_contig?"bfid0":"bfid1") << std::endl;

  iss << std::endl;
  iss << "  /* LDG Lanes */" << std::endl;
  iss << "  // Lane A" << std::endl;
  ptr_ldg('a', a_bf0_, a_bf1_, npa, '0', 'M', A_outer_contig);
  iss << "  // Lane B" << std::endl;
  ptr_ldg('b', b_bf0_, b_bf1_, npb, '1', 'N', B_outer_contig);
  iss << format("  add.s32 %M, %M, %{};", A_outer_contig?"afid0":"afid1") << std::endl;
  iss << format("  add.s32 %N, %N, %{};", B_outer_contig?"bfid0":"bfid1") << std::endl;

  iss << std::endl;
  iss << "  /* STS Lanes */" << std::endl;
  iss << "  // Lane A" << std::endl;
  ptr_sts('a', cd_shareda, 0, A_outer_contig);
  iss << "  // Lane B" << std::endl;
  ptr_sts('b', cd_sharedb, off_sharedb, B_outer_contig);

  iss << std::endl;
  iss << "  /* LDS Lanes */" << std::endl;
  iss << "  // Lane A" << std::endl;
  ptr_lds('a', '0', 0, cd_shareda);
  iss << "  // Lane B" << std::endl;
  ptr_lds('b', '1', off_sharedb, cd_sharedb);

  iss << "  .reg.s32 %switch;" << std::endl;
  iss << format("  mov.s32 %switch, {};", double_buffer_off) << std::endl;

  iss << std::endl;
  iss << format("  setp.lt.s32 %predk, %K, {};", kl_*rl) << std::endl;
  iss << format("  @%predk bra LAST_ITER;") << std::endl;
  iss << format("  setp.gt.s32 %predk, %K, {};", kl_*rl-1) << std::endl;
  ldg('a', a_bf0_, a_bf1_, ml, npa, cd_shareda, A_outer_contig, false);
  ldg('b', b_bf0_, b_bf1_, nl, npb, cd_sharedb, B_outer_contig, false);
  iss << std::endl;
  iss << "LOOP:" << std::endl;
  iss << "  bar.sync 0;" << std::endl;
  for(size_t k = 0; k < kl_; k+=ks_){
    for(size_t kk = 0 ; kk < ks_ ; ++kk){
        lds('a', ms_, k + kk, cd_shareda, bm_);
        lds('b', ns_, k + kk, cd_sharedb, bn_);
    }
    for(size_t kk = 0 ; kk < ks_ ; ++kk)
      fma(kk);
  }
  for(size_t rj = 0 ; rj < npa ; ++rj)
    iss << format("  add.u64 %pa{0}, %stepinca, %pa{0};", rj) << std::endl;
  for(size_t rj = 0 ; rj < npb ; ++rj)
    iss << format("  add.u64 %pb{0}, %stepincb, %pb{0};", rj) << std::endl;
  iss << format("  sub.s32 %K, %K, {};", kl_*rl) << std::endl;
  if((double_buffer_off & (double_buffer_off-1)) == 0){
    for(char x: std::vector<char>{'a', 'b'}){
      iss << format("  xor.b32 %write{0}, {1}, %write{0};", x, double_buffer_off) << std::endl;
      iss << format("  xor.b32 %read{0},  {1}, %read{0};", x, double_buffer_off) << std::endl;
    }
  }
  else{
    for(char x: std::vector<char>{'a', 'b'}){
      iss << format("  add.s32 %write{0}, %switch, %write{0};", x) << std::endl;
      iss << format("  add.s32 %read{0},  %switch, %read{0};", x) << std::endl;
    }
    iss << "  mul.lo.s32 %switch, %switch, -1;" << std::endl;
  }
  iss << format("  setp.gt.s32 %predk, %K, {};", kl_*rl) << std::endl;
  ldg('a', a_bf0_, a_bf1_, ml, npa, cd_shareda, A_outer_contig, false);
  ldg('b', b_bf0_, b_bf1_, nl, npb, cd_sharedb, B_outer_contig, false);
  iss << format("  @%predk bra.uni LOOP;") << std::endl;
  iss << std::endl;
  iss << format("  setp.eq.s32 %predk, %K, 0;") << std::endl;
  iss << format("  @%predk bra.uni ENDLOOP;") << std::endl;

  iss << std::endl;
  iss << "LAST_ITER:" << std::endl;
  ldg('a', a_bf0_, a_bf1_, ml, npa, cd_shareda, A_outer_contig, true);
  ldg('b', b_bf0_, b_bf1_, nl, npb, cd_sharedb, B_outer_contig, true);
  iss << "  bar.sync 0;" << std::endl;
  iss << "LAST_FMA:" << std::endl;
  lds('a', ms_, 0, cd_shareda, bm_);
  lds('b', ns_, 0, cd_sharedb, bn_);
  fma(0);
  iss << format("  add.u32 %reada, %reada, {};", rl*cd_shareda) << std::endl;
  iss << format("  add.u32 %readb, %readb, {};", rl*cd_sharedb) << std::endl;
  iss << format("  sub.s32 %K, %K, {};", rl) << std::endl;
  iss << format("  setp.gt.s32 %predk, %K, 0;") << std::endl;
  iss << format("  @%predk bra LAST_FMA;") << std::endl;

  iss << "ENDLOOP:" << std::endl;
  //Reduce in registers
  for(size_t r = 1; r < rs_; ++r)
  for(size_t m = 0 ; m < ms_ ; m+=dtvec*vec_)
  for(size_t n = 0; n < ns_ ; n++)
  for(size_t s = 0; s < vec_; ++s)
    iss << format("  add.{0} %rc0_{2}_{3}{4}, %rc{1}_{2}_{3}{4}, %rc0_{2}_{3}{4};", dtype, r, m, n, vs[s]) << std::endl;

  if(br_>1)
  {
    size_t bmn = nthreads/br_;
    iss << ".reg .u32 %readk, %writek, %rid_mn, %rid_k;" << std::endl;
    for(size_t mn = 0; mn < ml*nl; mn += bmn)
      iss << format("  .reg .{0} %rrk{1}_0, %rrk{1}_1;", ab_dtype, mn) << std::endl;

    iss << format("  mad.lo.u32 %writek, %idr, {}, %shared;", ml*nl*dtsize) << std::endl;
    iss << format("  mad.lo.u32 %writek, %idmn, {}, %writek;", ms_*ns_*dtsize) << std::endl;

    iss << "  bar.sync 0;" << std::endl;
    for(size_t n = 0; n < ns_; n ++)
    for(size_t m = 0; m < ms_; m += vec_*dtvec)
    for(size_t s = 0; s < vec_; s++){
      size_t mn = m + n*ms_;
      iss << format("  st.shared.{} [%writek + {}], %rc0_{}_{}{};", io_dtype, (mn + s*dtvec)*dtsize, m, n, vs[s]) << std::endl;
    }
    iss << "  bar.sync 0;" << std::endl;

    iss << std::endl;
    iss << format("  div.u32 %rid_mn, %id, {};", br_) << std::endl;
    iss << format("  rem.u32 %rid_k, %id, {};", br_) << std::endl;
    iss << format("  mad.lo.u32 %readk, %rid_k, {}, %shared;", ml*nl*dtsize) << std::endl;
    iss << format("  mad.lo.u32 %readk, %rid_mn, {}, %readk;", dtsize) << std::endl;
    for(size_t c = br_/2; c > 0; c /=2){
      iss << format("  setp.lt.u32 %predr, %rid_k, {};", c) << std::endl;
      for(size_t mn = 0; mn < ml*nl; mn += bmn){
        iss << format("  @%predr ld.shared.{} %rrk{}_0, [%readk + {}];", ab_dtype, mn, (mn)*dtsize) << std::endl;
        iss << format("  @%predr ld.shared.{} %rrk{}_1, [%readk + {}];", ab_dtype, mn, (mn + c*ml*nl)*dtsize) << std::endl;
        iss << format("  @%predr add.{0} %rrk{1}_0, %rrk{1}_0, %rrk{1}_1;", sub_dtype, mn) << std::endl;
        iss << format("  @%predr st.shared.{} [%readk + {}], %rrk{}_0;", ab_dtype, mn*dtsize, mn) << std::endl;
      }
      iss << "  bar.sync 0;" << std::endl;
    }


    iss << format("  mad.lo.u32 %readk, %idmn, {}, %shared;", ms_*ns_*dtsize) << std::endl;
    for(size_t n = 0; n < ns_; n ++)
    for(size_t m = 0; m < ms_; m += vec_*dtvec)
    for(size_t s = 0; s < vec_; s++){
      iss << format("  ld.shared.{} %rc0_{}_{}{}, [%readk + {}];", io_dtype, m, n, vs[s], ((m+s*dtvec) + n*ms_)*dtsize) << std::endl;
    }
  }

  iss << "SCALE:" << std::endl;
  if(dtype_==HALF_TYPE){
    iss << format("  ld.param.b16 %alpha16, [_alpha];", io_dtype) << std::endl;
    iss << "  mov.b32 %alpha, {%alpha16, %alpha16};" << std::endl;
  }
  else
    iss << format("  ld.param.{} %alpha, [_alpha];", io_dtype) << std::endl;
  for(size_t r = 0; r < rs_; ++r)
  for(size_t m = 0 ; m < ms_ ; m+=vec_*dtvec)
  for(size_t n = 0; n < ns_ ; n++)
  for(size_t s = 0; s < vec_; ++s)
    iss << format("  mul.{0} %rc{1}_{2}_{3}{4}, %rc{1}_{2}_{3}{4}, %alpha;", dtype, r, m, n, vs[s]) << std::endl;

  iss << std::endl;
  iss << "STORE_C:" << std::endl;
  iss << format("  mad.lo.s32 %M, %id0, -{}, %M;", vec_*dtvec) << std::endl;
  iss << format("  mad.lo.s32 %N, %id1, -{}, %N;", vec_*dtvec) << std::endl;

  iss << "  /* Write back result */" << std::endl;
  iss << format("  shl.b32 %offc, %offc, {};", dtbits) << std::endl;
  iss << format("  shl.b32 %ldc, %ldc, {};", dtbits) << std::endl;
  // C += (bid0 + id0*vec) + (bid1 + id1*vec)*ldc
  iss << format("  mad.lo.u32 %offc0, %id0, {}, %Bbid0;", dtvec*Bvec) << std::endl;
  iss << format("  mad.lo.u32 %offc1, %id1, {}, %bid1;", dtvec*vec_) << std::endl;
  iss << format("  mad.lo.u32 %toff, %offc1, %ldc, %offc0;") << std::endl;
  iss << format("  cvt.u64.u32 %btoff, %toff;") << std::endl;
  iss << format("  add.u64 %pc, %btoff, %pc;") << std::endl;
  iss << "  // Write back" << std::endl;
  iss << format("  setp.eq.s32 %predr, %idr, 0;") << std::endl;
  size_t bn = 0;
  for(size_t n = 0; n < ns_ ; n++){
    size_t inc = ((n+1)%(vec_*dtvec))?1:(bn_*vec_*dtvec - vec_*dtvec + 1);
    if(gridr_>1)
      iss << format("  setp.lt.and.s32 %predn, {}, %N, %predr;", bn) << std::endl;
    else
      iss << format("  setp.lt.s32 %predn, {}, %N;", bn) << std::endl;
    for(size_t m = 0 ; m < ms_ ; m+=vec_*dtvec){
      for(size_t s = 0; s < vec_*dtvec; ++s)
        iss << format("  setp.lt.and.s32 %pred{}, {}, %M, %predn;", s, bm_*m+s) << std::endl;
      if(gridr_!=1)
        for(size_t s = 0; s < vec_; ++s)
          iss << format("  @%pred{} red.add.{} [%pc + {}], %rc0_{}_{}{};", s, dtype, dtsize*(bm_*m+s*dtvec), m, n, vs[s]) << std::endl;
      else{
        store(m, n);
      }
    }
    iss << format("  mad.wide.u32 %pc, %ldc, {}, %pc;", inc) << std::endl;
    bn += inc;
  }
  iss << "}" << std::endl;
//  std::cout << bm_ << " " << bn_ << " " << ms_ << " " << ns_ << std::endl;
  return iss.str();
}

void GEMM::enqueue(driver::Kernel &gemm, driver::Stream &queue, const scalar& alpha, const driver::Buffer &A, const driver::Buffer &B, const scalar& beta, driver::Buffer &C)
{
    //Arguments
    gemm.setArg(0, M_);
    gemm.setArg(1, N_);
    gemm.setArg(2, K_);
    gemm.setArg(3, C);
    gemm.setArg(4, ldc_);
    gemm.setArg(5, offc_);
    gemm.setArg(6, size_of(dtype_), alpha.data());
    gemm.setArg(7, A);
    gemm.setArg(8, lda_);
    gemm.setArg(9, offa_);
    gemm.setArg(10, B);
    gemm.setArg(11, ldb_);
    gemm.setArg(12, offb_);
    gemm.setArg(13, size_of(dtype_), beta.data());
    //Grid-Block
    int32_t ml = bm_*ms_, nl = bn_*ns_;
    size_t gridM = ceil(M_, ml), gridN = ceil(N_, nl);
    //Launch
    if(gridr_>1)
      C.set_zero(queue);
    queue.enqueue(gemm, {gridM, gridN, gridr_}, {bm_, bn_, br_});
}

}
}
