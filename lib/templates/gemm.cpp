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

const std::string GEMM::id = "gemm";
const size_t GEMM::Nshapes = 6;
const size_t GEMM::Ntune = 14;
const size_t GEMM::Nparams = Nshapes + Ntune;

struct io_conf{
  io_conf(size_t ld, size_t vec, size_t dtsize, bool vectorize = true): inc(1), word_size(8*dtsize), num_words(1){
    if(vectorize && ld % (vec) == 0){
      inc *= vec;
      num_words *= vec;
    }
    num_packed = word_size/(8*dtsize);
    suffix = format(".b{}", word_size);
    if(num_words > 1) suffix = format(".v{}", num_words) + suffix;
  }

  std::string suffix;
  size_t num_packed;
  size_t inc;
  size_t word_size;
  size_t num_words;
};

GEMM::GEMM(DType dtype, IsaacOperation_t AT, IsaacOperation_t BT, param_t M, param_t N, param_t K, param_t offa, param_t lda, param_t offb, param_t ldb, param_t offc, param_t ldc,
     param_t vec, param_t bc0, param_t u, param_t bc1, param_t cs0, param_t, param_t cs1, param_t ba0, param_t ba1, param_t bb0, param_t bb1, param_t zs, param_t bz, param_t gridz):
  dtype_(dtype), AT_(AT), BT_(BT), M_(M), N_(N), K_(K), offa_(offa), lda_(lda), offb_(offb), ldb_(ldb), offc_(offc), ldc_(ldc),
  vec_(vec), bc0_(bc0), bc1_(bc1), cs0_(cs0), cs1_(cs1), u_(u), us_(u), ba0_(ba0), ba1_(ba1), bb0_(bb0), bb1_(bb1),
  zs_(zs), bz_(bz), gridz_(gridz), stn_(8)
{}

std::vector<param_t> GEMM::tuning_params() const
{ return {vec_, bc0_, bc1_, cs0_, cs1_, u_, ba0_, ba1_, bb0_, bb1_, zs_, bz_, gridz_}; }

double GEMM::tflops(param_t M, param_t N, param_t K, double time)
{ return (double)2*M*N*K/(time*1e3); }


void GEMM::check_valid(driver::Device const & device, size_t nkernels, uint32_t* params, uint8_t* valid){
  std::array<int, 20> x{0};
  for(size_t m = 0; m < nkernels; ++ m){
    //Parameters
    for(size_t i = 0; i < x.size(); ++i)
      x[i] = params[m*x.size() + i];
    DType dtype = (DType)(x[0]);
    IsaacOperation_t AT = (IsaacOperation_t)x[1];
    IsaacOperation_t BT = (IsaacOperation_t)x[2];
    param_t M = x[3], N = x[4], vec = x[6], bm = x[7], kl = x[8], bn = x[9],
           ms = x[10], ks = x[11], ns = x[12], a_bf0 = x[13], a_bf1 = x[14], b_bf0 = x[15], b_bf1 = x[16],
           rs = x[17], br = x[18], gridr = x[19];
    //Features
    param_t dtsize = size_of(dtype);
    bool A_outer_contig = AT==ISAAC_OP_N;
    bool B_outer_contig = BT==ISAAC_OP_T;
    param_t rl = rs*br;
    param_t ml = bm*ms;
    param_t nl = bn*ns;
    param_t gridM = ceil(M, ml), gridN = ceil(N, nl);
    param_t nthreads = bm*bn*br;
    param_t cd_shareda = dtsize*(ml+(A_outer_contig?0:(vec)));
    param_t cd_sharedb = dtsize*(nl+(B_outer_contig?0:(vec)));
    param_t ncd_shareda = kl*rl;
    param_t ncd_sharedb = kl*rl;
    param_t size_shareda = cd_shareda*ncd_shareda;
    param_t size_sharedb = cd_sharedb*ncd_sharedb;
    param_t size_redc = dtsize*ml*nl*(br==1?0:br);
    param_t size_tiles = 2*(size_shareda + size_sharedb);
    param_t size_unswizzle = dtsize*ml*bn;
    size_t size_shmem = std::max(size_redc, std::max(size_tiles, size_unswizzle));
    param_t npa = (A_outer_contig?(kl*rl):ml) / a_bf1;
    param_t npb = (B_outer_contig?(kl*rl):nl) / b_bf1;
    param_t nra = (A_outer_contig?ml:(kl*rl)) / a_bf0;
    param_t nrb = (B_outer_contig?nl:(kl*rl)) / b_bf0;
    param_t n_instructions =  nra*npa*2 + nrb*npb*2 + ms*ns*kl*rs + kl*rs*ms*ns/(vec);
    //Test
    bool is_valid =   a_bf0*a_bf1 == nthreads
                    &&  b_bf0*b_bf1 == nthreads
                    &&  (A_outer_contig?(kl*rl):ml) % a_bf1 == 0
                    &&  (A_outer_contig?ml:(kl*rl)) % (a_bf0*vec) == 0
                    &&  (B_outer_contig?(kl*rl):nl) % b_bf1 == 0
                    &&  (B_outer_contig?nl:(kl*rl)) % (b_bf0*vec) == 0

                    && (nthreads % 32 == 0)
                    &&  ms % (vec) == 0
                    &&  ns % (vec) == 0
                    &&  kl % ks == 0
                    &&  size_shmem <= device.max_shared_memory()
                    &&  (gridr == 1 || gridM*gridN < 64*64)
                    &&  n_instructions <= 1024 //Doesn't allow more than 1024 instructions in the inner loop
                    &&  bm <= device.max_block_dim()[0]
                    &&  bn <= device.max_block_dim()[1]
                    &&  br <= device.max_block_dim()[2]

                    &&  vec*dtsize <= 16;
    valid[m] = is_valid;
  }
}

/* Code generation */
std::string GEMM::dump(drv::Device const & device, std::string const & name){
//  static const param_t warp_size = 32;
  bool A_outer_contig = AT_==ISAAC_OP_N;
  bool B_outer_contig = BT_==ISAAC_OP_T;
  size_t dtsize = size_of(dtype_);
  std::string dtype = arith_str(dtype_);
  std::string io_dtype = io_str(dtype_);
  std::string ab_dtype = dtype;
  std::string sub_dtype = dtype;


  size_t rl = zs_*bz_;
  size_t block = u_*rl;
  //vector prefix
  std::string vv = vec_>1?format(".v{}", vec_):"";
  //tile size
  size_t cl0 = bc0_*cs0_;
  size_t cl1 = bc1_*cs1_;
  //Number of lanes
  size_t npa = (A_outer_contig?block:cl0) / ba1_;
  size_t npb = (B_outer_contig?block:cl1) / bb1_;
  size_t nra = (A_outer_contig?cl0:block) / ba0_;
  size_t nrb = (B_outer_contig?cl1:block) / bb0_;
  std::string As0 = A_outer_contig?"M":"K";
  std::string As1 = A_outer_contig?"K":"M";
  std::string Bs0 = B_outer_contig?"N":"K";
  std::string Bs1 = B_outer_contig?"K":"N";
  //Number of warps
//  size_t nwarp0 = ceil(bm_, warp_size);
//  size_t nwarp1 = ceil(bm_*bn_, nw0*warp_size);
//  size_t nwarp2 = ceil(bm_*bn_*br_, nw0*nw1*warp_size);

  //Number of threads
  size_t nthreads = bc0_*bc1_*bz_;
  //Shared memory sizes/offsets
  size_t cd_shareda = dtsize*(cl0+(A_outer_contig?0:(vec_)));
  size_t cd_sharedb = dtsize*(cl1+(B_outer_contig?0:(vec_)));
  size_t ncd_shareda = block;
  size_t ncd_sharedb = block;
  size_t size_shareda = cd_shareda*ncd_shareda;
  size_t size_sharedb = cd_sharedb*ncd_sharedb;
  size_t off_sharedb =  size_shareda;
  size_t size_redc = dtsize*cl0*cl1*(bz_==1?0:bz_);
  size_t size_tiles = 2*(size_shareda + size_sharedb);
  size_t size_unswizzle = dtsize*cl0*bc1_;
  param_t size_shmem = std::max(size_redc, std::max(size_tiles, size_unswizzle));
  size_t double_buffer_off = size_tiles/2;
  //Byte stride
  size_t Bvec = vec_*dtsize;
  size_t dtbits = log2(dtsize);
  //Vector suffix
  const char* vs[] = {".x", ".y", ".z", ".w"};
  if(vec_==1)
    vs[0] = "";
  //Load-Store alignments
  io_conf Cio(ldc_, vec_, dtsize, false);
  io_conf Aio(lda_, vec_, dtsize, false);
  io_conf Bio(ldb_, vec_, dtsize, false);

  uint8_t is_valid;
  param_t params[] = {dtype_, AT_, BT_, M_, N_, K_, vec_, bc0_, u_, bc1_, cs0_, us_, cs1_, ba0_, ba1_, bb0_, bb1_, zs_, bz_, gridz_};
  check_valid(device, 1, params, &is_valid);
  if(!is_valid)
    throw invalid_parameters();

  std::stringstream iss;
  /* Global Loads */
  auto ptr_ldg = [&](char x, size_t /*bf0*/, size_t bf1, size_t npx, char bound, char axis, bool no_trans){
      if(no_trans)
        iss << format("  // p{0} += off{0} + B{0}fid0 + Bbid{1} + (fid1 + offk)*ld{0};", x, axis) << std::endl;
      else
        iss << format("  // p{0} += off{0} + B{0}fid0 + offk + bid{1}*ld{0} + fid1*ld{0};", x, axis) << std::endl;

      //General offset
      iss << format("  mad.wide.u32 %p{0}, %off{0}, 1, %p{0};", x) << std::endl;
      if(gridz_>1)
        iss << format("  mad.wide.u32 %p{0}, %offk, {1}, %p{0};", x, no_trans?format("%ld{}", x):format("{}",dtsize)) << std::endl;

      //Offset along contiguous dimension
      iss << format("  mov.u32 %off{0}0, %{0}fid0;", x) << std::endl;
      if(no_trans)
        iss << format("  add.u32 %off{0}0, %bid{1}, %off{0}0;", x, axis) << std::endl;
      iss << format("  setp.lt.s32 %pred{0}0, %off{0}0, %{1};", x, (no_trans?bound:'K')) << std::endl;
      iss << format("  @%pred{0}0 mad.wide.u32 %p{0}, %off{0}0, {1}, %p{0};", x, dtsize) << std::endl;

      //Offset along non-contiguous dimension
      iss << format("  mov.s32 %off{0}1, %{0}fid1;", x) << std::endl;
      if(!no_trans)
        iss << format("  add.s32 %off{0}1, %off{0}1, %bid{1};", x, axis) << std::endl;
      for(size_t rj = 0 ; rj < npx ; ++rj)
          iss << format("  mov.u64 %p{0}{1}, %p{0};", x, rj) << std::endl;
      for(size_t rj = 0 ; rj < npx ; ++rj)
          iss << format("  add.s32 %off{0}1_{1}, %off{0}1, {2};", x, rj, rj*bf1) << std::endl;
      for(size_t rj = 0 ; rj < npx ; ++rj)
          iss << format("  setp.lt.s32 %pred{0}{1}, %off{0}1_{1}, %{3};", x, rj, rj*bf1, (no_trans?'K':bound)) << std::endl;
      for(size_t rj = 0 ; rj < npx ; ++rj)
          iss << format("  @%pred{0}{1} mad.wide.u32 %p{0}{1}, %off{0}1_{1}, %ld{0}, %p{0};", x, rj) << std::endl;

      //Step increment
      if(no_trans)
        iss << format("  mul.wide.u32 %stepinc{0}, {1}, %ld{0};", x, block) << std::endl;
      else
        iss << format("  cvt.u64.u32 %stepinc{0}, {1};", x, block*dtsize) << std::endl;
  };

  auto ldg = [&](char x, io_conf const & conf, size_t bf0, size_t bf1, size_t nrx, size_t npx, size_t sharedcd, std::string const & Xs0, std::string const & Xs1, bool outer_contig, bool check_bounds){
    size_t i, ri, j, rj;
    std::vector<std::string> preds(vec_, "predk");
    for(size_t s = 0; s < (check_bounds?vec_:0); s+=conf.inc)
      preds[s] = format("pred{}", s);

    // Load from global
    for(rj = 0 ; rj < npx ; ++rj){
      if(check_bounds)
        iss << format("  setp.lt.s32 %predk, %off{}1_{}, %{};", x, rj, Xs1) << std::endl;
      for(ri = 0, i = 0 ; ri < nrx ; i += bf0*conf.num_packed*vec_, ri+=conf.num_packed*vec_){
        for(size_t s = 0; s < (check_bounds?conf.num_packed*vec_:0); s+=conf.inc)
          iss << format("  setp.lt.and.s32 %pred{0}, %off{1}0_{2}, %{3}m{4}, %predk;",  s, x, ri, Xs0, s) << std::endl;
        for(size_t s = 0; s < (check_bounds?conf.num_packed*vec_:0); s+=conf.inc){
          iss << format("  @!%{0} mov.b{1} %rr{2}{3}_{4}{5}, 0x0;", preds[s], conf.word_size, x, ri + vec_*(s/conf.num_packed/vec_), rj, vs[s/conf.num_packed % vec_]) << std::endl;
        }for(size_t s = 0; s < conf.num_packed*vec_; s+=conf.inc){
          iss << format("  @%{0} ld.global.nc{1} %rr{2}{3}_{4}{5}, [%p{2}{4} + {6}];", preds[s], conf.suffix, x, ri + vec_*(s/conf.num_packed/vec_), rj, (conf.num_words>1)?"":vs[s/conf.num_packed % vec_], (i+s)*dtsize) << std::endl;
        }
      }
    }

    // Store to shared
    for(rj = 0, j = 0 ; rj < npx ; j += bf1, rj++){
      for(ri = 0, i = 0 ; ri < nrx ; i += bf0*conf.num_packed*vec_, ri+=conf.num_packed*vec_)
        if(outer_contig)
          iss << format("  st.shared{0}.b{1} [%write{2} + {3}], %rr{2}{4}_{5};", vv, conf.word_size, x, i*dtsize + j*sharedcd, ri, rj) << std::endl;
        else
          for(size_t s = 0 ; s < vec_ ; ++s){
            if(conf.num_packed > 1){
              iss << format("  mov.b32 {{%rbh0, %rbh1}}, %rr{}{}_{}{};", x, ri, rj, vs[s]) << std::endl;
              for(size_t ss = 0; ss < conf.num_packed; ++ss)
                iss << format("  st.shared.{0} [%write{1} + {2}], %rbh{3};", ab_dtype, x, j*dtsize + (i + s*conf.num_packed + ss)*sharedcd, ss) << std::endl;
            }
            else
              iss << format("  st.shared.{0} [%write{1} + {2}], %rr{1}{3}_{4}{5};", ab_dtype, x, j*dtsize + (i+s)*sharedcd, ri, rj, vs[s]) << std::endl;
          }
    }

  };

  auto ptr_sts = [&](char x, size_t cdx, size_t off, bool outer_contiguous){
    std::string id0 = "0", id1 = "1";
    if(!outer_contiguous) std::swap(id0, id1);
    iss << format("  // write{0} = shared + {1} + Bfid{2} + fid{3}*{4}", x, off, id0, id1, cdx) << std::endl;
    iss << format("  mov.u32 %write{}, %shared;", x) << std::endl;
    iss << format("  add.u32 %write{0}, %write{0}, {1};", x, off) << std::endl;
    iss << format("  mad.lo.u32 %write{0}, %{0}fid{1}, {2},  %write{0};", x, id0, dtsize) << std::endl;
    iss << format("  mad.lo.u32 %write{0}, %{0}fid{1}, {2}, %write{0};", x, id1, cdx) << std::endl;
  };

  auto ptr_lds = [&](char x, char axis, size_t off, size_t cd_shared){
    iss << format("  // read{0} = shared + {1} + id{2}*{3}", x, off, axis, Bvec) << std::endl;
    iss << format("  mov.u32 %read{}, %shared;", x) << std::endl;
    iss << format("  add.u32 %read{0}, %read{0}, {1};", x, off) << std::endl;
    iss << format("  mad.lo.u32  %read{0}, %id{1}, {2}, %read{0};", x, axis, Bvec) << std::endl;
    iss << format("  mad.lo.u32  %read{0}, %idz, {1}, %read{0};", x, cd_shared) << std::endl;

  };

  auto lds = [&](char x, size_t nx, size_t k, size_t cdx, size_t bs){
    for(size_t r = 0; r < zs_; ++r)
    for(size_t rx = 0; rx < nx; rx+=vec_){
      iss << format("  ld.shared{0}.{1} %r{2}{3}_{4}_{5}, [%read{2} + {6}];", vv, io_dtype, x, r, rx, k%us_, rx*bs*dtsize + (r*bz_ + k*rl)*cdx) << std::endl;
    }
  };

  auto fma = [&](size_t kk){

    for(size_t r = 0 ; r < zs_ ; ++r)
    for(size_t m = 0; m < cs0_; m+=vec_)
      for(size_t n = 0; n < cs1_; n+=vec_){
          for(size_t nn = 0 ; nn < vec_ ; ++nn)
            for(size_t mm = 0 ; mm < vec_ ; ++mm){
              std::string rc = format("%rc{}_{}_{}{}", r, m, n + nn, vs[mm]);
              std::string ra = format("%ra{}_{}_{}{}", r, m, kk, vs[mm]);
              std::string rb = format("%rb{}_{}_{}{}", r, n, kk, vs[nn]);
              iss << format("  fma.rn.{0} {1}, {2}, {3}, {1};", dtype, rc, ra, rb) << std::endl;
            }
      }
  };

  auto declare_register_tile = [&](char x, size_t M, size_t N, size_t dtinc){
    for(size_t r = 0 ; r < zs_ ; ++r)
      for(size_t m = 0 ; m < M ; m+=dtinc*vec_){
        iss << format("  .reg {}.{}", vv, io_dtype);
        for(size_t n = 0 ; n < N ; n++)
          iss << format("{} %r{}{}_{}_{}", n>0?",":"", x, r, m, n);
        iss << ";" << std::endl;
      }
  };

  iss << std::endl;
  iss << format(".func store_col(.reg .b64 %pc, .reg .b32 %Cs0, .reg .b32 %offc0, .reg .b32 %readc, .reg .b32 %writec, .reg .{0} %beta, .reg .b32 %offc1, .reg .b32 %Cs1, .reg .b32 %idz, .reg .b32 %bidz, .param .{0} _rc[{1}])", io_dtype, cs0_) << std::endl;
  iss << "{" << std::endl;
  iss << "  .reg .b16 %rbh0, %rbh1;" << std::endl;
  iss << format("  .reg .pred %predc<{}>, %predn, %predbeta0, %predz, %predbidze0;", cs0_) << std::endl;
  iss << format("  .reg .b32 %offc0_<{}>;", cs0_) << std::endl;
  iss << format("  .reg .{0} %ccol<{1}>, %dcol<{1}>;", io_dtype, cs0_) << std::endl;
  for(size_t m = 0; m < cs0_; m+=vec_)
    iss << format("  .reg {}.{} %rc{};", vv, io_dtype, m) << std::endl;
  for(size_t m = 0; m < cs0_; m+=vec_)
    iss << format("  ld.param{}.{} %rc{}, [_rc + {}];", vv, io_dtype, m, m*dtsize) << std::endl;


  for(size_t m = 0; m < cs0_; m++)
    iss << format("  add.u32 %offc0_{}, %offc0, {};", m, m*bc0_) << std::endl;

  if(gridz_ > 1){
    iss << format("  setp.eq.u32 %predbidze0, %bidz, 0;") << std::endl;
    iss << format("  setp.eq.or.{} %predbeta0, %beta, 0, %predbidze0;", io_dtype) << std::endl;
  }
  else
    iss << format("  setp.eq.{} %predbeta0, %beta, 0;", io_dtype) << std::endl;

  iss << format("  setp.eq.s32 %predz, %idz, 0;") << std::endl;
  iss << format("  setp.lt.and.s32 %predn, %offc1, %Cs1, %predz;") << std::endl;
  for(size_t m = 0; m < cs0_; m++)
    iss << format("  setp.lt.and.s32 %predc{0}, %offc0_{0}, %Cs0, %predn;", m) << std::endl;

  iss << "  bar.sync 0;" << std::endl;
  for(size_t m = 0 ; m < cs0_; m+=vec_)
    iss << format("  @%predn st.shared{}.{} [%writec + {}], %rc{};", vv, io_dtype, m*bc0_*dtsize, m) << std::endl;
  iss << "  bar.sync 0;" << std::endl;
  for(size_t m = 0 ; m < cs0_; m++)
    iss << format("  @%predn ld.shared.{} %ccol{}, [%readc + {}];", ab_dtype, m, m*bc0_*dtsize) << std::endl;

  iss << "  @%predbeta0 bra.uni BETA_DONE;" << std::endl;
  iss << "HANDLE_BETA:" << std::endl;
  for(size_t m = 0 ; m < cs0_; m++)
    iss << format("  @%predc{} ld.global.{} %dcol{}, [%pc + {}];", m, ab_dtype, m,  m*bc0_*dtsize) << std::endl;

  for(size_t m = 0 ; m < cs0_ ; m++)
    iss << format("  fma.rn.{0} %ccol{1}, %dcol{1}, %beta, %ccol{1};", dtype, m) << std::endl;

  iss << "BETA_DONE:" << std::endl;
  for(size_t m = 0 ; m < cs0_; m++)
    iss << format("  @%predc{} st.global.{} [%pc + {}], %ccol{};", m, ab_dtype, m*bc0_*dtsize, m) << std::endl;
  iss << "}" << std::endl;

  iss << "// Launch with CTA sizes = (" << bc0_ << ", " << bc1_ << ", " << bz_ << ")" << std::endl;
  iss << "// Launch with GRID sizes = (ceil(M, " << cl0 << "), ceil(N, " << cl1 << "), " << gridz_ << ")" << std::endl;
  iss << ".entry " << name << "(.param .b32 _M, .param .b32 _N, .param .b32 _K," << std::endl;
  iss << "    .param .b64 _pc, .param .b32 _ldc, .param .b32 _offc," << std::endl;
  iss << "    .param ." << ab_dtype << " _alpha, .param .b64 _pa, .param .b32 _lda, .param .b32 _offa," << std::endl;
  iss << "    .param .b64 _pb, .param .b32 _ldb, .param .b32 _offb," << std::endl;
  iss << "    .param ." << ab_dtype << " _beta," << std::endl;
  iss << "    .param .b32 _bound, .param .b64 _plock)" << std::endl;
  iss << "{" << std::endl;
  iss << std::endl;
  iss << "  .reg .b32 %bound;" << std::endl;
  iss << "  /* Registers */" << std::endl;
  iss << "  // For C tile" << std::endl;
  declare_register_tile('d', cs0_, cs1_, 1);
  declare_register_tile('c', cs0_, cs1_, 1);

  iss << "  // For A tile" << std::endl;
  declare_register_tile('a', cs0_, us_, 1);

  iss << "  // For B tile" << std::endl;
  declare_register_tile('b', cs1_, us_, 1);

  iss << "  // Parameters" << std::endl;
  iss << format("  .reg .b32 %M, %N, %K;") << std::endl;
  for(size_t s = 0 ; s < vec_ ; ++s){
    iss << format("  .reg .b32 %Mm{};", s);
    iss << format("  .reg .b32 %Nm{};", s);
    if(Bs0 == "K" || As0 == "K")
      iss << format("  .reg .b32 %Km{};", s);
  }

  iss << format("  .reg .{} %alpha;", io_dtype) << std::endl;
  iss << format("  .reg .{} %beta;", io_dtype) << std::endl;
  for(char x: std::vector<char>{'c', 'a', 'b'}){
    iss << format("  .reg .b64 %p{0};", x) << std::endl;
    iss << format("  .reg .b32 %ld{0}, %off{0};", x) << std::endl;
  }
  iss << "  // IDs" << std::endl;
  iss << format("  .reg .b32 %bid0, %bid1;") << std::endl;
  iss << format("  .reg .b32 %idmn, %id;") << std::endl;
  iss << format("  .reg .b32 %afid0, %afid1, %bfid0, %bfid1;") << std::endl;
  iss << format("  .reg .b32 %idu, %id0, %id1;") << std::endl;
  iss << format("  .reg .b32 %warpid, %warpid0, %warpid1, %warpid2, %warpid12;") << std::endl;
  iss << "  // Lanes in global memory" << std::endl;
  iss << format("  .reg .b64 %pa<{0}>;", npa) << std::endl;
  iss << format("  .reg .b64 %pb<{0}>;", npb) << std::endl;
  iss << format("  .reg .pred %pred<{0}>;", vec_) << std::endl;
  iss << format("  .reg .pred %predk;") << std::endl;
  iss << format("  .reg .b64 %stepinca, %stepincb;") << std::endl;
  iss << "  // Lanes in shared memory" << std::endl;
  iss << format("  .reg .b32 %writea, %writeb, %writec;") << std::endl;
  iss << format("  .reg .b32 %reada, %readb, %readc;") << std::endl;
  iss << format("  .reg .b32 %toff;") << std::endl;
  iss << format("  .reg .b64 %btoff;") << std::endl;
  iss << format("  .reg .b64 %plock;") << std::endl;

  iss << format("  .reg .b32 %offc0, %offc1;") << std::endl;
  for(size_t ri = 0; ri < cs0_; ri++)
    iss << format("  .reg.b32 %offc0_{};", ri) << std::endl;
  for(size_t rj = 0; rj < cs1_; rj+=vec_)
    iss << format("  .reg.b32 %offc1_{};", rj) << std::endl;

  iss << format("  .reg .b32 %offa0, %offa1;") << std::endl;
  for(size_t ri = 0 ; ri < nra; ++ri)
      iss << format("  .reg .b32 %offa0_{};", ri) << std::endl;
  for(size_t rj = 0 ; rj < npa; ++rj)
      iss << format("  .reg .b32 %offa1_{};", rj) << std::endl;

  iss << format("  .reg .b32 %offb0, %offb1;") << std::endl;
  for(size_t ri = 0 ; ri < nrb ; ++ri)
      iss << format("  .reg .b32 %offb0_{};", ri) << std::endl;
  for(size_t rj = 0 ; rj < npb ; ++rj)
      iss << format("  .reg .b32 %offb1_{};", rj) << std::endl;

  for(size_t rj = 0 ; rj < std::max<param_t>(npa, vec_) ; ++rj)
      iss << format("  .reg .pred %preda{};", rj) << std::endl;
  for(size_t rj = 0 ; rj < std::max<param_t>(npb, vec_) ; ++rj)
      iss << format("  .reg .pred %predb{};", rj) << std::endl;

  iss << format("  .reg .b32 %div, %rem, %idz, %bidz, %offk;") << std::endl;
  iss << "  .reg .pred %predr;" << std::endl;

  for(size_t ri = 0 ; ri < nra ; ri+=vec_)
    for(size_t rj = 0 ; rj < npa ; rj++){
      iss << format("  .reg {}.b{} %rra{}_{};", vv, Aio.word_size, ri, rj) << std::endl;
    }

  for(size_t ri = 0 ; ri < nrb ; ri+=vec_)
    for(size_t rj = 0 ; rj < npb ; rj++){
      iss << format("  .reg {}.b{} %rrb{}_{};", vv, Bio.word_size, ri, rj) << std::endl;
    }

  iss << format("  .reg.b32 %bid, %stm, %stn, %nctaid0, %nctaid1, %npercol, %nperblock, %sbid0, %sbid1;") << std::endl;
  iss << "  .reg .b32 %switch;" << std::endl;


  iss << std::endl;
  iss << "  /* Shared */" << std::endl;
  iss << format("  .reg .b64 %shared64;") << std::endl;
  iss << "  .reg .b32 %shared;" << std::endl;
  iss << format("  .shared .align 16 .b8 _shared[{}];", size_shmem) << std::endl;
  iss << format("  mov.u64 %shared64, _shared;") << std::endl;
  iss << format("  cvt.u32.u64 %shared, %shared64;") << std::endl;
  iss << std::endl;
  iss << "  /* Initialize C */" << std::endl;
  for(size_t r = 0 ; r < zs_ ; ++r)
  for(size_t m = 0 ; m < cs0_ ; m+=vec_)
  for(size_t n = 0; n < cs1_ ; ++n)
  for(size_t mm = 0; mm < vec_ ; ++mm)
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
  iss << format("  ld.param.u64 %plock, [_plock];") << std::endl;
  iss << std::endl;
  iss << "  /* Block ID */" << std::endl;
  iss << format("  mov.u32 %nctaid0, %nctaid.x;") << std::endl;
  iss << format("  mov.u32 %nctaid1, %nctaid.y;") << std::endl;
  iss << format("  mov.u32 %bid0, %ctaid.x;") << std::endl;
  iss << format("  mov.u32 %bid1, %ctaid.y;") << std::endl;
  iss << format("  mov.u32 %bidz, %ctaid.z;") << std::endl;
  iss << format("  mad.lo.u32 %bid, %bid1, %nctaid0, %bid0;") << std::endl;


  iss << "  // Index of supertile" << std::endl;
  iss << format("  // npercol =  (nctaid0*min({}, nctaid1)", stn_) << std::endl;
  iss << format("  min.u32 %npercol, {}, %nctaid1;", stn_) << std::endl;
  iss << format("  mul.lo.u32 %npercol, %nctaid0, %npercol;") << std::endl;

  iss << format("  // sbid1 = bid / npercol)") << std::endl;
  iss << format("  div.u32 %sbid1, %bid, %npercol;") << std::endl;
  iss << format("  vmad.u32.u32.u32 %stn, %sbid1, -{}, %nctaid1;", stn_) << std::endl;
  iss << format("  min.u32 %stn, %stn, {};", stn_) << std::endl;

  iss << format("  // bid0 = (bid % npercol)/stn") << std::endl;
  iss << format("  rem.u32 %bid0, %bid, %npercol;") << std::endl;
  iss << format("  div.u32 %bid0, %bid0, %stn;") << std::endl;

  iss << "  // Index within a supertile" << std::endl;
  iss << format("  rem.u32 %bid1, %bid, %stn;") << std::endl;
  iss << format("  mad.lo.u32 %bid1, %sbid1, {}, %bid1;", stn_) << std::endl;

  iss << std::endl;
  iss << "  /* Adjust IDs and bounds */" << std::endl;
  iss << format("  mul.lo.u32 %bid0, %bid0, {};", cl0) << std::endl;
  iss << format("  mul.lo.u32 %bid1, %bid1, {};", cl1) << std::endl;

  iss << std::endl;
  iss << "  /* Thread ID */" << std::endl;
  iss << format("  mov.u32 %id0, %tid.x;") << std::endl;
  iss << format("  mov.u32 %id1, %tid.y;") << std::endl;
  iss << format("  mov.u32 %idz, %tid.z;") << std::endl;
  iss << format("  mad.lo.u32 %idmn, %id1, {}, %id0;", bc0_) << std::endl;
  iss << format("  mad.lo.u32 %id, %idmn, {}, %idz;", bz_) << std::endl;

  if(gridz_ > 1){
    iss << format("  // Split") << std::endl;
    iss << format("  div.u32 %div, %K, {};", gridz_) << std::endl;
    iss << format("  rem.u32 %rem, %K, {};", gridz_) << std::endl;
    iss << "  mov.s32 %K, %div;" << std::endl;
    iss << "  mul.lo.u32 %offk, %bidz, %div;" << std::endl;
    iss << "  setp.lt.u32 %pred0, %bidz, %rem;" << std::endl;
    iss << "  @%pred0 add.s32 %K, %K, 1;" << std::endl;
    iss << "  @%pred0 add.s32 %offk, %bidz, %offk;" << std::endl;
    iss << "  @!%pred0 add.s32 %offk, %rem, %offk;" << std::endl;
  }

  iss << std::endl;
  iss << "  /* Fetch ID */" << std::endl;
  iss << format("  div.u32 %afid1, %id, {};", ba0_) << std::endl;
  iss << format("  rem.u32 %afid0, %id, {};", ba0_) << std::endl;
  iss << format("  div.u32 %bfid1, %id, {};", bb0_) << std::endl;
  iss << format("  rem.u32 %bfid0, %id, {};", bb0_) << std::endl;
  iss << format("  mul.lo.u32 %afid0, %afid0, {};", Aio.num_packed*vec_) << std::endl;
  iss << format("  mul.lo.u32 %bfid0, %bfid0, {};", Bio.num_packed*vec_) << std::endl;
  iss << std::endl;
  iss << "  /* Scale by dtype size */" << std::endl;
  for(char x: std::vector<char>{'a', 'b'}){
    iss << format("  shl.b32 %off{0}, %off{0}, {1};", x, dtbits) << std::endl;
    iss << format("  shl.b32 %ld{0}, %ld{0}, {1};", x, dtbits) << std::endl;
  }

  iss << std::endl;
  iss << "  /* LDG Lanes */" << std::endl;
  iss << "  // Lane A" << std::endl;
  ptr_ldg('a', ba0_, ba1_, npa, 'M', '0', A_outer_contig);
  iss << "  // Lane B" << std::endl;
  ptr_ldg('b', bb0_, bb1_, npb, 'N', '1', B_outer_contig);

  for(size_t ri = 0; ri < nra; ri+=Aio.num_packed*vec_)
    iss << format("  add.u32 %offa0_{}, %offa0, {};", ri, ri*ba0_) << std::endl;
  for(size_t ri = 0; ri < nrb; ri+=Bio.num_packed*vec_)
    iss << format("  add.u32 %offb0_{}, %offb0, {};", ri, ri*bb0_) << std::endl;


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

  iss << std::endl;
  iss << "  /* Main Loop */" << std::endl;
  iss << format("  ld.param.b32 %bound, [_bound];") << std::endl;
  iss << format("  mov.s32 %switch, {};", double_buffer_off) << std::endl;
  iss << format("  setp.gt.s32 %predk, %K, %bound;") << std::endl;
  iss << format("  @!%predk bra LAST_ITER;") << std::endl;
  ldg('a', Aio, ba0_, ba1_, nra, npa, cd_shareda, As0, As1, A_outer_contig, false);
  ldg('b', Bio, bb0_, bb1_, nrb, npb, cd_sharedb, Bs0, Bs1, B_outer_contig, false);
  iss << std::endl;

  iss << "LOOP:" << std::endl;
  iss << "  bar.sync 0;" << std::endl;
  for(size_t k = 0; k < u_; k+=us_){
    for(size_t kk = 0 ; kk < us_ ; ++kk){
        lds('a', cs0_, k + kk, cd_shareda, bc0_);
        lds('b', cs1_, k + kk, cd_sharedb, bc1_);
    }
    for(size_t kk = 0 ; kk < us_ ; ++kk)
      fma(kk);
  }
  for(size_t rj = 0 ; rj < npa ; ++rj)
    iss << format("  add.u64 %pa{0}, %stepinca, %pa{0};", rj) << std::endl;
  for(size_t rj = 0 ; rj < npb ; ++rj)
    iss << format("  add.u64 %pb{0}, %stepincb, %pb{0};", rj) << std::endl;
  iss << format("  sub.s32 %K, %K, {};", block) << std::endl;
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
  iss << format("  setp.gt.s32 %predk, %K, %bound;") << std::endl;
  ldg('a', Aio, ba0_, ba1_, nra, npa, cd_shareda, As0, As1, A_outer_contig, false);
  ldg('b', Bio, bb0_, bb1_, nrb, npb, cd_sharedb, Bs0, Bs1, B_outer_contig, false);
  iss << format("  @%predk bra.uni LOOP;") << std::endl;
  iss << std::endl;
  iss << format("  setp.le.s32 %predk, %K, 0;") << std::endl;
  iss << format("  @%predk bra.uni ENDLOOP;") << std::endl;
  iss << "LAST_ITER:" << std::endl;
  for(size_t s = 0 ; s < vec_ ; ++s)
    iss << format("  sub.s32 %{0}m{1}, %{0}, {1};", As0, s) << std::endl;
  for(size_t s = 0 ; s < vec_ ; ++s)
    iss << format("  sub.s32 %{0}m{1}, %{0}, {1};", Bs0, s) << std::endl;
  ldg('a', Aio, ba0_, ba1_, nra, npa, cd_shareda, As0, As1, A_outer_contig, true);
  ldg('b', Bio, bb0_, bb1_, nrb, npb, cd_sharedb, Bs0, Bs1, B_outer_contig, true);
  iss << "  bar.sync 0;" << std::endl;
  iss << format("  setp.gt.s32 %predk, %K, 0;") << std::endl;
  iss << format("  @%predk bra LOOP;") << std::endl;

  iss << std::endl;
  iss << "ENDLOOP:" << std::endl;
  //Reduce in registers
  for(size_t r = 1; r < zs_; ++r)
  for(size_t m = 0 ; m < cs0_ ; m+=vec_)
  for(size_t n = 0; n < cs1_ ; n++)
  for(size_t s = 0; s < vec_; ++s)
    iss << format("  add.{0} %rc0_{2}_{3}{4}, %rc{1}_{2}_{3}{4}, %rc0_{2}_{3}{4};", dtype, r, m, n, vs[s]) << std::endl;

  if(bz_>1)
  {
    size_t bmn = nthreads/bz_;
    iss << ".reg .u32 %readk, %writek, %rid_mn, %rid_k;" << std::endl;
    for(size_t mn = 0; mn < cl0*cl1; mn += bmn)
      iss << format("  .reg .{0} %rrk{1}_0, %rrk{1}_1;", ab_dtype, mn) << std::endl;

    iss << format("  mad.lo.u32 %writek, %idz, {}, %shared;", cl0*cl1*dtsize) << std::endl;
    iss << format("  mad.lo.u32 %writek, %idmn, {}, %writek;", cs0_*cs1_*dtsize) << std::endl;

    iss << "  bar.sync 0;" << std::endl;
    for(size_t n = 0; n < cs1_; n ++)
    for(size_t m = 0; m < cs0_; m += vec_)
    for(size_t s = 0; s < vec_; s++){
      size_t mn = m + n*cs0_;
      iss << format("  st.shared.{} [%writek + {}], %rc0_{}_{}{};", io_dtype, (mn + s)*dtsize, m, n, vs[s]) << std::endl;
    }
    iss << "  bar.sync 0;" << std::endl;

    iss << std::endl;
    iss << format("  div.u32 %rid_mn, %id, {};", bz_) << std::endl;
    iss << format("  rem.u32 %rid_k, %id, {};", bz_) << std::endl;
    iss << format("  mad.lo.u32 %readk, %rid_k, {}, %shared;", cl0*cl1*dtsize) << std::endl;
    iss << format("  mad.lo.u32 %readk, %rid_mn, {}, %readk;", dtsize) << std::endl;
    for(size_t c = bz_/2; c > 0; c /=2){
      iss << format("  setp.lt.u32 %predr, %rid_k, {};", c) << std::endl;
      for(size_t mn = 0; mn < cl0*cl1; mn += bmn){
        iss << format("  @%predr ld.shared.{} %rrk{}_0, [%readk + {}];", ab_dtype, mn, (mn)*dtsize) << std::endl;
        iss << format("  @%predr ld.shared.{} %rrk{}_1, [%readk + {}];", ab_dtype, mn, (mn + c*cl0*cl1)*dtsize) << std::endl;
        iss << format("  @%predr add.{0} %rrk{1}_0, %rrk{1}_0, %rrk{1}_1;", sub_dtype, mn) << std::endl;
        iss << format("  @%predr st.shared.{} [%readk + {}], %rrk{}_0;", ab_dtype, mn*dtsize, mn) << std::endl;
      }
      iss << "  bar.sync 0;" << std::endl;
    }


    iss << format("  mad.lo.u32 %readk, %idmn, {}, %shared;", cs0_*cs1_*dtsize) << std::endl;
    for(size_t n = 0; n < cs1_; n ++)
    for(size_t m = 0; m < cs0_; m += vec_)
    for(size_t s = 0; s < vec_; s++){
      iss << format("  ld.shared.{} %rc0_{}_{}{}, [%readk + {}];", io_dtype, m, n, vs[s], ((m+s) + n*cs0_)*dtsize) << std::endl;
    }
  }

  iss << "SCALE:" << std::endl;
  iss << format("  ld.param.{} %alpha, [_alpha];", io_dtype) << std::endl;
  iss << format("  ld.param.{} %beta, [_beta];", io_dtype) << std::endl;

  for(size_t r = 0; r < zs_; ++r)
  for(size_t m = 0 ; m < cs0_ ; m+=vec_)
  for(size_t n = 0; n < cs1_ ; n++)
  for(size_t s = 0; s < vec_; ++s)
    iss << format("  mul.{0} %rc{1}_{2}_{3}{4}, %rc{1}_{2}_{3}{4}, %alpha;", dtype, r, m, n, vs[s]) << std::endl;

  iss << std::endl;
  iss << "STORE_C:" << std::endl;
  iss << format("  .reg .pred %predc1_<{}>;", cs1_) << std::endl;
  iss << format("  .reg .pred %predbidze0, %predbeta0;") << std::endl;
  iss << format("  .reg .pred %spin;") << std::endl;
  iss << format("  .reg .u32 %lock;") << std::endl;
  iss << format("  .reg .u64 %pc<{}>;", cs1_) << std::endl;

  iss << "  // Coalescing lanes" << std::endl;
  iss << format("  mad.lo.u32 %writec, %id0, {}, %shared;", vec_*dtsize) << std::endl;
  iss << format("  mad.lo.u32 %writec, %id1, {}, %writec;", cl0*dtsize) << std::endl;

  iss << format("  mad.lo.u32 %readc, %id0, {}, %shared;", dtsize) << std::endl;
  iss << format("  mad.lo.u32 %readc, %id1, {}, %readc;", cl0*dtsize) << std::endl;

  iss << format("  mad.lo.u32 %offc0, %id0, {}, %bid0;", 1) << std::endl;
  iss << format("  mad.lo.u32 %offc1, %id1, {}, %bid1;", vec_) << std::endl;
  iss << format("  shl.b32 %ldc, %ldc, {};", dtbits) << std::endl;
  iss << format("  mad.wide.u32 %pc, %offc, {}, %pc;", dtsize) << std::endl;
  iss << format("  mad.wide.u32 %pc, %offc0, {}, %pc;", dtsize) << std::endl;
  iss << format("  mad.wide.u32 %pc, %offc1, %ldc, %pc;", dtsize) << std::endl;

  for(size_t n = 0; n < cs1_; n += vec_)
    iss << format("  mad.wide.u32 %pc{}, {}, %ldc, %pc;", n, n*bc1_) << std::endl;
  for(size_t n = 0; n < cs1_; n ++)
    iss << format("  mad.wide.u32 %pc{}, {}, %ldc, %pc{};", n, n % (vec_), n / (vec_) * vec_) << std::endl;

  for(size_t s = 0 ; s < vec_ ; ++s)
    iss << format("  sub.s32 %Nm{0}, %N, {0};", s) << std::endl;
  for(size_t m = 0; m < cs0_; m++)
    iss << format("  add.u32 %offc0_{}, %offc0, {};", m, m*bc0_) << std::endl;
  for(size_t n = 0; n < cs1_; n+=vec_)
    iss << format("  add.u32 %offc1_{}, %offc1, {};", n, n*bc1_) << std::endl;


  if(gridz_ > 1){
      iss << "  mad.wide.u32 %plock, %bid, 4, %plock;" << std::endl;
      iss << "  mov.u32 %lock, 0;" << std::endl;
      iss << "SPIN: " << std::endl;
      iss << "  setp.ne.u32 %spin, %lock, %bidz;" << std::endl;
      iss << "  @%spin ld.global.cg.b32 %lock, [%plock];" << std::endl;
      iss << "  @%spin bra.uni SPIN;" << std::endl;

      iss << format("  setp.eq.u32 %predbidze0, %bidz, 0;") << std::endl;
      iss << format("  @!%predbidze0 mov.{} %beta, 1.;", dtype) << std::endl;
  }


  iss << format("  .param .{} _rc[{}];", io_dtype, cs0_) << std::endl;
  for(size_t n = 0; n < cs1_; ++n){
    for(size_t m = 0; m < cs0_; m+=vec_)
      iss << format("  st.param{}.{} [_rc + {}], %rc0_{}_{};", vv, io_dtype, m*dtsize, m, n) << std::endl;
    iss << format("  call.uni store_col, (%pc{}, %M, %offc0, %readc, %writec, %beta,  %offc1_{}, %Nm{}, %idz, %bidz, _rc);", n, n/(vec_)*vec_, n%(vec_)) << std::endl;
  }

  if(gridz_>1){
    iss << "  setp.eq.u32 %predr, %id, 0;" << std::endl;
    iss << "  add.u32 %bidz, %bidz, 1;" << std::endl;
    iss << "  bar.sync 0;" << std::endl;
    iss << "  @%predr membar.gl;" << std::endl;
    iss << "  @%predr st.global.cg.u32 [%plock], %bidz;" << std::endl;
  }
  iss << "}" << std::endl;


  return iss.str();
}

void GEMM::enqueue(driver::Kernel &gemm, driver::Stream &queue, const scalar& alpha, const driver::Buffer &A, const driver::Buffer &B, const scalar& beta, driver::Buffer &C)
{
  //Grid-Block
  int32_t ml = bc0_*cs0_, nl = bc1_*cs1_, rl = bz_*zs_;
  size_t gridM = ceil(M_, ml), gridN = ceil(N_, nl);
  size_t lasti = (gridM - 1)*ml + ml - 1;
  size_t lastk = u_*rl - 1;
  size_t lastj = (gridN - 1)*nl + nl - 1;
  size_t last_safe_a = (AT_==ISAAC_OP_N)?(M_*K_ - 1 - lasti)/M_ - lastk : M_*K_ - 1 - lasti*K_ - lastk;
  size_t last_safe_b = (BT_==ISAAC_OP_T)?(N_*K_ - 1 - lastj)/N_ - lastk : N_*K_ - 1 - lastj*K_ - lastk;
  int32_t bound = std::max<int32_t>(1, std::max(K_ - last_safe_a, K_ - last_safe_b));
  static std::map<driver::Stream, driver::Buffer> locks_pool;
  if(locks_pool.find(queue)==locks_pool.end())
    locks_pool.insert(std::make_pair(queue, driver::Buffer(queue.context(), 64*64*4)));
  driver::Buffer& locks = locks_pool.at(queue);

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
  gemm.setArg(14, bound);
  gemm.setArg(15, locks);

//  std::cout << gridM << " " << gridN << " " << std::endl;
  //Launch
  if(gridz_ > 1)
    locks.set_zero(queue, gridM*gridN*4);
  queue.enqueue(gemm, {gridM, gridN, gridz_}, {bc0_, bc1_, bz_});
}

}
}
