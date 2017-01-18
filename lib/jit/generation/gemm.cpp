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

#include "isaac/array.h"
#include "isaac/driver/dispatch.h"
#include "isaac/jit/syntax/expression/preset.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/jit/generation/gemm.h"
#include "isaac/jit/generation/engine/keywords.h"
#include "isaac/exception/api.h"
#include "tools/arguments.hpp"
#include "tools/vector_types.hpp"


#include <string>
#include "isaac/tools/cpp/align.hpp"

namespace isaac
{
namespace templates
{

std::vector<int_t> infos(expression_tree const & tree, symbolic::preset::gemm::args& arguments, char A_trans)
{
  expression_tree::data_type const & array = tree.data();
  std::size_t root = tree.root();
  arguments = symbolic::preset::gemm::check(array, root);
  int_t M = arguments.C->shape[0];
  int_t N = arguments.C->shape[1];
  int_t K = (A_trans=='T')?arguments.A->shape[0]:arguments.A->shape[1];
  return {M, N, K};
}

/* ------------------ CUBLAS ------------------ */
cublas_gemm::cublas_gemm(char A_trans, char B_trans): A_trans_(A_trans), B_trans_(B_trans), init_(true)
{ }

int cublas_gemm::is_invalid(expression_tree const  &, driver::Device const & device) const
{ return (init_ && device.backend()==driver::CUDA)?0:-1; }

std::vector<int_t> cublas_gemm::input_sizes(expression_tree const & expressions) const
{
  symbolic::preset::gemm::args dummy;
  return infos((expression_tree&)expressions, dummy, A_trans_);
}

expression_type cublas_gemm::type() const
{
  if(A_trans_=='N' && B_trans_=='N')
    return GEMM_NN;
  else if(A_trans_=='T' && B_trans_=='N')
    return GEMM_TN;
  else if(A_trans_=='N' && B_trans_=='T')
    return GEMM_NT;
  else
    return GEMM_TT;
}

void cublas_gemm::enqueue(driver::CommandQueue & queue, driver::Program const &, std::string const &, runtime::execution_handler const & control)
{
  namespace drv = driver;
  //Get GEMM info
  symbolic::preset::gemm::args args;
  std::vector<int_t> MNK = infos(control.x(), args, A_trans_);
  int_t M = MNK[0], N = MNK[1], K = MNK[2];
  CUdeviceptr cuA = args.A->array.handle.cu;
  CUdeviceptr cuB = args.B->array.handle.cu;
  CUdeviceptr cuC = args.C->array.handle.cu;
  runtime::execution_options_type const & opt = control.execution_options();
  auto cuT = [](char xt) { return (xt=='N')?CUBLAS_OP_N:CUBLAS_OP_T; };
  int offA = args.A->array.start, offB = args.B->array.start, offC = args.C->array.start;
  cublasHandle_t h = drv::dispatch::cublasHandle(queue.context());
  //Set new stream
  cudaStream_t bkp;
  drv::Event event(drv::CUDA);
  drv::dispatch::cublasGetStream_v2(h,&bkp);
  drv::dispatch::cublasSetStream_v2(h,(cudaStream_t)queue.handle().cu());
  values_holder alpha = args.alpha.values();
  values_holder beta = args.beta.values();
  if(opt.events)
    drv::check(drv::dispatch::cuEventRecord(event.handle().cu().first, queue.handle().cu()));
  if(args.C->dtype==FLOAT_TYPE)
    drv::dispatch::cublasSgemm_v2(h,cuT(A_trans_), cuT(B_trans_), M, N, K, &alpha.float32, (float*)cuA + offA , args.A->ld[1], (float*)cuB + offB, args.B->ld[1], &beta.float32, (float*)cuC + offC, args.C->ld[1]);
  else
    drv::dispatch::cublasDgemm_v2(h,cuT(A_trans_), cuT(B_trans_), M, N, K, &alpha.float64, (double*)cuA + offA, args.A->ld[1], (double*)cuB + offB, args.B->ld[1], &beta.float64, (double*)cuC + offC, args.C->ld[1]);
  if(opt.events){
    drv::check(drv::dispatch::cuEventRecord(event.handle().cu().second, queue.handle().cu()));
    opt.events->push_back(event);
  }
  //Revert old stream
  drv::dispatch::cublasSetStream_v2(h,bkp);
}


/* -------------------------------------------- */
unsigned int gemm::lmem_usage(expression_tree const & expression) const
{
  unsigned int N = 0;
  size_t llda = (A_trans_=='N')?mL_:kL_+vwidth_;
  size_t lnda = (A_trans_=='N')?kL_:mL_;
  size_t lldb = (B_trans_=='T')?nL_:kL_+vwidth_;
  size_t lndb = (B_trans_=='T')?kL_:nL_;
  N += llda*lnda;
  N += lldb*lndb;
  return N*size_of(expression.dtype());
}

expression_type gemm::type() const
{
  if(A_trans_=='N' && B_trans_=='N')
    return GEMM_NN;
  else if(A_trans_=='T' && B_trans_=='N')
    return GEMM_TN;
  else if(A_trans_=='N' && B_trans_=='T')
    return GEMM_NT;
  else
    return GEMM_TT;
}


unsigned int gemm::registers_usage(expression_tree const & expression) const
{
  unsigned int N = mS_ * nS_ + mS_ * kS_ + kS_ * nS_;
  return N*size_of(expression.dtype());
}

unsigned int gemm::temporary_workspace(expression_tree const & expressions) const
{
  std::vector<int_t> MNK = input_sizes(expressions);
  int_t M = MNK[0]; int_t N = MNK[1];
  if(depth_ > 1)
    return M*N*depth_;
  return 0;
}

int gemm::is_invalid_impl(driver::Device const &, expression_tree const &) const
{
  if ((mS_ % vwidth_) > 0 || (nS_ % vwidth_) > 0)
    return TEMPLATE_MS_NS_MUST_BE_SIMD_WIDTH_MULTIPLE;

  if(mL_ > 256 || nL_ > 256)
    return TEMPLATE_BLOCK_SIZE_TOO_LARGE;

  if ( kS_ % kL_ == 0)
    return TEMPLATE_KS_MUST_BE_SMALLER_THAN_KL;

  if ((lf0_*lf1_) !=(ls0_*ls1_))
    return TEMPLATE_LOCAL_FETCH_PRODUCT_MUST_MATCH_LOCAL_SIZE_PRODUCT;

  {
    unsigned int bound1 = (A_trans_=='N')?kL_:mL_;
    unsigned int bound0 = (A_trans_=='N')?mL_:kL_;

    if (lf1_>0 && (bound1 % lf1_)> 0)
      return A_trans_=='N'?TEMPLATE_LOCAL_FETCH_1_MUST_BE_KL_MULTIPLE:TEMPLATE_LOCAL_FETCH_1_MUST_BE_ML_MULTIPLE;

    if (lf0_>0 && (bound0 % (lf0_*vwidth_)) > 0)
      return A_trans_=='N'?TEMPLATE_LOCAL_FETCH_0_MUST_BE_NL_MULTIPLE:TEMPLATE_LOCAL_FETCH_0_MUST_BE_KL_MULTIPLE;

  }

  {
    unsigned int bound1 = (B_trans_=='T')?kL_:nL_;
    unsigned int bound0 = (B_trans_=='T')?nL_:kL_;

    if (lf1_>0 && (bound1 % lf1_)> 0)
      return B_trans_=='T'?TEMPLATE_LOCAL_FETCH_1_MUST_BE_KL_MULTIPLE:TEMPLATE_LOCAL_FETCH_1_MUST_BE_ML_MULTIPLE;

    if (lf0_>0 && (bound0 % (lf0_*vwidth_)) > 0)
      return B_trans_=='T'?TEMPLATE_LOCAL_FETCH_1_MUST_BE_KL_MULTIPLE:TEMPLATE_LOCAL_FETCH_1_MUST_BE_ML_MULTIPLE;

  }

  return TEMPLATE_VALID;
}

std::string gemm::generate_impl(std::string const & suffix, expression_tree const & tree, driver::Device const & device, symbolic::symbols_table const &) const
{
  using std::string;
  using tools::to_string;

  driver::backend_type backend = device.backend();
  bool has_depth = depth_ > 1;
#define VLOAD(offset, ptr) vload(vwidth_, sdtype, offset, ptr, "1", backend, true)
#define VLOAD_MISALIGNED(offset, ptr) vload(vwidth_, sdtype, offset, ptr, "1", backend, false)
#define VSTORE_LDSA(value, offset, ptr) vstore(vwidth_, sdtype, value, offset, ptr, "1", backend, llda%vwidth_==0)
#define VSTORE_LDSB(value, offset, ptr) vstore(vwidth_, sdtype, value, offset, ptr, "1", backend, lldb%vwidth_==0)

  symbolic::preset::gemm::args args;
  infos(tree, args, A_trans_);
  std::string ASTRIDE1 = (args.A->ld[0] > 1)?"*Astride1":"";
  std::string BSTRIDE1 = (args.B->ld[0] > 1)?"*Bstride1":"";
  std::string CSTRIDE1 = (args.C->ld[0] > 1)?"*Cstride1":"";

  //////////////////
  /// INIT
  /// //////////////
  kernel_generation_stream stream(backend);
  numeric_type dtype = tree.dtype();
  std::string sdtype = to_string(dtype);
  std::string vdtype = append_width(sdtype, vwidth_);

  //////////////////
  /// DECLARATIONS
  /// //////////////
  std::string gemm_name = "gemm";
  std::string reduce_name = "reduce";

  gemm_name += suffix;
  reduce_name += suffix;

  switch(backend)
  {
  case driver::OPENCL:
    stream << " __attribute__((reqd_work_group_size(" << ls0_ << "," << ls1_ << ",1)))" << std::endl;
    break;
  default:
    break;
  }

  stream << "$KERNEL void gemm" << suffix << "($SIZE_T M, $SIZE_T N, $SIZE_T K, "
         << "$GLOBAL " << sdtype << "* C, $SIZE_T ldc, $SIZE_T offc, $SIZE_T Cstride1, "
         << sdtype << " alpha,"
         << "$GLOBAL " << sdtype << "* A, $SIZE_T lda, $SIZE_T offa, $SIZE_T Astride1,"
         << "$GLOBAL " << sdtype << "* B, $SIZE_T ldb, $SIZE_T offb, $SIZE_T Bstride1,"
         << sdtype << " beta)"
         << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  ///Declare
  stream << "//blocks" << std::endl;
  stream << sdtype << " rC[" << mS_ << "][" << nS_ << "] = {{0}};" << std::endl;
  stream << vdtype << " rA[" << kS_ << "][" << mS_/vwidth_ << "];" << std::endl;
  stream << vdtype << " rB[" << kS_ << "][" << nS_/vwidth_ << "];" << std::endl;
  stream << std::endl;

  stream << "//pointers" << std::endl;
  size_t llda = (A_trans_=='N')?mL_:kL_+vwidth_;
  size_t lnda = (A_trans_=='N')?kL_:mL_;
  size_t lldb = (B_trans_=='T')?nL_:kL_+vwidth_;
  size_t lndb = (B_trans_=='T')?kL_:nL_;
  stream << "$LOCAL " << sdtype << " lA[" << llda*lnda << "];" << std::endl;
  stream << "$LOCAL " << sdtype << " lB[" << lldb*lndb << "];" << std::endl;
  unsigned int npA = mL_/(A_trans_=='N'?lf0_*vwidth_:lf1_);
  unsigned int npB = nL_/(B_trans_=='T'?lf0_*vwidth_:lf1_);
  stream << "$GLOBAL " << sdtype << "* Ai[" << npA << "];" << std::endl;
  stream << "$GLOBAL " << sdtype << "* Bi[" << npB << "];" << std::endl;
  stream << std::endl;

  stream << "//identifiers" << std::endl;
  stream << "int2 idT;" << std::endl;
  stream << "int idt;" << std::endl;
  if(has_depth)
    stream << "int gidz, div, offz;" << std::endl;
  stream << "uint4 ids;" << std::endl;
  stream << "ids.x = $GROUP_IDX_0;" << std::endl;
  stream << "ids.y = $GROUP_IDX_1;" << std::endl;
  stream << "ids.z = $LOCAL_IDX_0;" << std::endl;
  stream << "ids.w = $LOCAL_IDX_1;" << std::endl;
  stream << std::endl;

  stream << "//offsets" << std::endl;
  stream << "A += offa;" << std::endl;
  stream << "B += offb;" << std::endl;
  stream << "C += offc;" << std::endl;

  if(has_depth)
  {
    stream << "gidz = $GROUP_IDX_2;" << std::endl;
    stream << "div = (K+" << depth_-1 << ")/" << depth_ << ";" << std::endl;
    stream << "offz = div*gidz;" << std::endl;
    stream << "K = max(0, min(K - div*gidz, ($SIZE_T)div));" << std::endl;
  }

  stream << "idt = " << ls0_ << "*ids.w + ids.z;" << std::endl;
  stream << "idT.y = idt/" << lf0_ << ";" << std::endl;
  stream << "idT.x = idt - " << lf0_ << "*idT.y;" << std::endl;
  stream << std::endl;

  stream << "//Adjust pointers and bounds per work-item" << std::endl;
  stream << "ids.x *= " << mL_ << ";" << std::endl;
  stream << "ids.y *= " << nL_ << ";" << std::endl;
  stream << "idT.x *= " << vwidth_ << ";" << std::endl;

  stream << "M -= ids.x;" << std::endl;
  if(A_trans_=='N')
    stream << "M -= idT.x;" << std::endl;
  else
    stream << "M -= idT.y;" << std::endl;

  stream << "N -= ids.y;" << std::endl;
  if(B_trans_=='T')
    stream << "N -= idT.x;" << std::endl;
  else
    stream << "N -= idT.y;" << std::endl;

  if (A_trans_=='N')
  {
    stream << "A += ids.x" << ASTRIDE1 << ";" << std::endl;
    stream << "A += idT.y*lda;" << std::endl;
    if(has_depth)
      stream << "A += offz*lda;" << std::endl;

  }
  else
  {
    stream << "A += ids.x*lda;" << std::endl;
    stream << "A += idT.x" << ASTRIDE1 << ";" << std::endl;
    if(has_depth)
      stream << "A += offz;" << std::endl;
  }

  if(B_trans_=='T')
  {
    stream << "B += ids.y" << BSTRIDE1 << ";" << std::endl;
    stream << "B += idT.y*ldb;" << std::endl;
    if(has_depth)
      stream << "B += offz*ldb;" << std::endl;
  }
  else
  {
    stream << "B += ids.y*ldb;" << std::endl;
    stream << "B += idT.x" << BSTRIDE1 << ";" << std::endl;
    if(has_depth)
      stream << "B += offz;" << std::endl;
  }

  stream << "#pragma unroll" << std::endl;
  stream << "for(int i = 0 ; i < " << npA << " ; ++i){" << std::endl;
  stream.inc_tab();
  stream << "Ai[i] = A;" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  stream << std::endl;

  stream << "#pragma unroll" << std::endl;
  stream << "for(int i = 0 ; i < " << npB << " ; ++i){" << std::endl;
  stream.inc_tab();
  stream << "Bi[i] = B;" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  stream << std::endl;

  for(unsigned int i = 0 ; i < npA ; i++ )
    if (A_trans_=='N')
      stream << "Ai[" << i << "] += " << Select(backend, to_string(i*lf0_*vwidth_) + " < M", "(int)((idT.x + " + to_string(i*lf0_*vwidth_) + ")" + ASTRIDE1 + ")", "0") << ";" << std::endl;
    else
      stream << "Ai[" << i << "] += " << Select(backend, to_string(i*lf1_) + " < M", "(int)((idT.y + " + to_string(i*lf1_) + ")*lda)", "0") << ";" << std::endl;

  for(unsigned int i = 0 ; i < npB ; i++ )
    if (B_trans_=='T')
      stream << "Bi[" << i << "] += " << Select(backend, to_string(i*lf0_*vwidth_) + " < N", "(int)((idT.x + " + to_string(i*lf0_*vwidth_) + ")" + BSTRIDE1 + ")", "0") << ";" << std::endl;
    else
      stream << "Bi[" << i << "] += " << Select(backend, to_string(i*lf1_) + " < N", "(int)((idT.y + " + to_string(i*lf1_) + ")*ldb)", "0") << ";" << std::endl;

  stream << std::endl;
  stream << "//Outer loop" << std::endl;
  stream << "while(K >=" << kL_ << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();


  auto fetch_to_lds = [&](bool last_iteration)
  {
    stream << "$LOCAL_BARRIER;" << std::endl;
    stream << "$LOCAL_PTR " << sdtype << "* ldsA = lA + idT.y*" << llda << " + idT.x;" << std::endl;
    stream << "$LOCAL_PTR " << sdtype << "* ldsB = lB + idT.y*" << lldb << " + idT.x;" << std::endl;

    stream << "//Fetch A to local memory" << std::endl;
    if (A_trans_=='N')
    {
      for(unsigned int k = 0; k < kL_; k += lf1_)
        for(unsigned int m = 0; m < mL_; m += lf0_*vwidth_)
        {
          std::string mm = to_string(m/(vwidth_*lf0_));
          std::string kk = to_string(k);
          if(last_iteration)
            for(unsigned int s = 0 ; s < vwidth_ ; ++s)
              stream << "ldsA[" << k*llda + m + s << "] = (condy" << k << " && " << s << "< M)? Ai[" << mm << "][" << k << "*lda + " << s << "] : 0;" << std::endl;
          else
            stream << VSTORE_LDSA(VLOAD_MISALIGNED("0" ,"&Ai[" + mm +"][" + kk + "*lda]"), "0", "ldsA + " + to_string(k*llda+m)) << ";" << std::endl;
        }
    }
    else
    {
      for(unsigned int k = 0; k < kL_; k += lf0_*vwidth_)
        for(unsigned int m = 0; m < mL_; m += lf1_)
        {
          std::string mm = to_string(m/lf1_);
          std::string kk = to_string(k);
          if(last_iteration)
            for(unsigned int s = 0 ; s < vwidth_ ; ++s)
              stream << "ldsA[" << m*llda + k + s << "] = condx" << k + s << "? Ai[" << mm << "][" << k + s << ASTRIDE1 << "] : 0;" << std::endl;

          else
            stream << VSTORE_LDSA(VLOAD_MISALIGNED("0", "&Ai[" + mm + "][" + kk + ASTRIDE1 + "]"), "0", "ldsA + " + to_string(m*llda+k)) << ";" << std::endl;
        }
    }

    stream << "//Fetch B to local memory" << std::endl;
    if (B_trans_=='T')
    {
      for(unsigned int k = 0; k < kL_; k += lf1_)
        for(unsigned int n = 0; n < nL_; n += lf0_*vwidth_)
        {
          std::string nn = to_string(n/(vwidth_*lf0_));
          std::string kk = to_string(k);
          if(last_iteration)
            for(unsigned int s = 0 ; s < vwidth_ ; ++s)
              stream << "ldsB[" << k*lldb + n + s << "] = (condy" << k << " && " << s << "< N)? Bi[" <<  nn << "][" << kk << "*ldb +" << s << "] : 0;" << std::endl;
          else
            stream << VSTORE_LDSB(VLOAD_MISALIGNED("0" ,"&Bi[" + nn +"][" + kk + "*ldb]"), "0", "ldsB + " + to_string(k*lldb+n)) << ";" << std::endl;
        }
    }
    else
    {
      for(unsigned int k = 0; k < kL_; k += lf0_*vwidth_)
        for(unsigned int n = 0; n < nL_; n += lf1_)
        {
          std::string nn = to_string(n/lf1_);
          std::string kk = to_string(k);
          if(last_iteration)
            for(unsigned int s = 0 ; s < vwidth_ ; ++s)
              stream << "ldsB[" << n*lldb + k + s << "] = condx" << k + s << "? Bi[" << nn << "][" << k + s << BSTRIDE1 << "] : 0;" << std::endl;

          else
            stream << VSTORE_LDSB(VLOAD_MISALIGNED("0", "&Bi[" + nn + "][" + kk + BSTRIDE1 + "]"), "0", "ldsB + " + to_string(n*lldb+k)) << ";" << std::endl;
        }
    }

    if(A_trans_=='N')
      stream << "ldsA = lA + ids.z*" << vwidth_ << ";" << std::endl;
    else
      stream << "ldsA = lA + ids.z*" << llda*vwidth_ << ";" << std::endl;

    if(B_trans_=='T')
      stream << "ldsB = lB + ids.w*" << vwidth_ << ";" << std::endl;
    else
      stream << "ldsB = lB + ids.w*" << lldb*vwidth_ << ";" << std::endl;

    stream << "$LOCAL_BARRIER;" << std::endl;
    std::string bound = last_iteration?"K":tools::to_string(kL_);
    size_t ks = last_iteration?1:kS_;
    stream << "//Inner loop" << std::endl;
    stream << "for(unsigned int k = 0; k < " << bound << "; k+=" << ks << "){" << std::endl;
    stream.inc_tab();

    stream << "//Fetch A to registers" << std::endl;
    stream << "#pragma unroll" << std::endl;
    stream << "for(unsigned int kk = 0; kk < " << ks << "; kk++)" << std::endl;
    stream << "#pragma unroll " << mS_/vwidth_ << std::endl;
    stream << "for(unsigned int mm = 0; mm < " << mS_/vwidth_ << "; mm++)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    if(A_trans_=='N')
      stream << "rA[kk][mm] = "  << VLOAD("0", "ldsA + k*" + to_string(llda) + " + mm*" + to_string(ls0_*vwidth_) + "+ kk*" + to_string(llda)) << ";" << std::endl;
    else
    {
      if(vwidth_==1)
        stream << "rA[kk][mm] = ldsA[k + mm*" << ls0_*llda <<  "+ kk"  << "];" << std::endl;
      else
        for(unsigned int s = 0 ; s < vwidth_ ; ++s)
          stream << access_vector_type("rA[kk][mm]", s) << " = ldsA[k + (mm*" << vwidth_*ls0_ << " + " << s << ")*" << llda <<  "+ kk];" << std::endl;
    }

    stream.dec_tab();
    stream << "}" << std::endl;

    stream << "//Fetch B to registers" << std::endl;
    stream << "#pragma unroll " << ks << std::endl;
    stream << "for(unsigned int kk = 0; kk < " << ks << "; kk++)" << std::endl;
    stream << "#pragma unroll " << nS_/vwidth_ << std::endl;
    stream << "for(unsigned int nn = 0; nn < " << nS_/vwidth_ << "; nn++)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    if(B_trans_=='T')
      stream << "rB[kk][nn] = " << VLOAD("0", "ldsB + k*" + to_string(lldb) + " + nn*" + to_string(ls1_*vwidth_)  + "+ kk*" + to_string(lldb)) << ";" << std::endl;
    else
    {
      if(vwidth_==1)
        stream << "rB[kk][nn] = ldsB[k"  << " + nn*" << ls1_*lldb <<  "+ kk"  << "];" << std::endl;
      else
        for(unsigned int s = 0 ; s < vwidth_ ; ++s)
          stream << access_vector_type("rB[kk][nn]", s) << " = ldsB[k"  << " + (nn*" << vwidth_*ls1_ << " + " << s << ")*" << lldb <<  "+ kk];" << std::endl;
    }
    stream.dec_tab();
    stream << "}" << std::endl;

    stream << "//FMA computations" << std::endl;
    stream << "#pragma unroll" << std::endl;
    stream << "for(unsigned int kk = 0 ; kk < " << ks << "; ++kk){" << std::endl;
    stream.inc_tab();
    for(unsigned int nn=0; nn < nS_; ++nn)
      for(unsigned int mm=0; mm < mS_; ++mm){
        string res_str, lhs_str, rhs_str;
        res_str = "rC[" + to_string(mm) + "][" + to_string(nn) + "]";
        if (vwidth_==1)
          lhs_str = "rA[kk][" + to_string(mm) + "]";
        else
          lhs_str = access_vector_type("rA[kk][" + to_string(mm/vwidth_) + "]", mm%vwidth_);
        if (vwidth_==1)
          rhs_str = "rB[kk]["+to_string(nn)+"]";
        else
          rhs_str = access_vector_type("rB[kk]["+to_string(nn/vwidth_)+"]", nn%vwidth_);
        stream << res_str << "= $MAD(" << lhs_str << "," << rhs_str << "," << res_str << ");" << std::endl;
      }
    stream.dec_tab();
    stream << "}" << std::endl;
    stream.dec_tab();
    stream << "}" << std::endl;
    stream << "K -= " << kL_ << ";" << std::endl;

    //Increment A pointers to global memory
    if (A_trans_=='N')
      for(unsigned int i = 0 ; i < npA ; ++i)
        stream << "Ai[" << i << "] += "  << kL_ << "*lda;" << std::endl;
    else
      for(unsigned int i = 0 ; i < npA ; ++i)
        stream << "Ai[" << i << "] += "  << kL_ << ASTRIDE1 << ";" << std::endl;

    //Increment B pointers to global memory
    if (B_trans_=='T')
      for(unsigned int i = 0 ; i < npB ; ++i)
        stream << "Bi[" << i << "] += " << kL_ << "*ldb;" << std::endl;
    else
      for(unsigned int i = 0 ; i < npB ; ++i)
        stream << "Bi[" << i << "] += " << kL_ << BSTRIDE1 << ";" << std::endl;
  };
  fetch_to_lds(false);
  stream.dec_tab();
  stream << "}" << std::endl;


  if(A_trans_=='N' || B_trans_=='T')
  {
    stream << "int Ky = K - idT.y;" << std::endl;
    for(unsigned int k = 0; k < kL_; k += lf1_)
      stream << "int condy" << k << " = " << k << " < Ky;" << std::endl;
  }

  if(A_trans_=='T' || B_trans_=='N')
  {
    stream << "int Kx = K - idT.x;" << std::endl;
    for(unsigned int k = 0 ; k < kL_ ; k += lf0_*vwidth_)
      for(unsigned int s = 0 ; s < vwidth_ ; ++s)
        stream << "int condx" << k + s << " = " << k + s << " < Kx;" << std::endl;
  }
  fetch_to_lds(true);

  stream << "//Write back C" << std::endl;
  stream << "M += ids.x;" << std::endl;
  if(A_trans_=='N')
    stream << "M += idT.x;" << std::endl;
  else
    stream << "M += idT.y;" << std::endl;

  if(B_trans_=='T')
    stream << "N += idT.x;" << std::endl;
  else
    stream << "N += idT.y;" << std::endl;
  stream << "N += ids.y;" << std::endl;

  stream << "C += ids.x" << CSTRIDE1 << ";" << std::endl;
  stream << "C += ids.z*" << vwidth_ << CSTRIDE1 << ";" << std::endl;
  stream << "C += ids.y*ldc;" << std::endl;
  stream << "C += ids.w*" << vwidth_ << "*ldc;" << std::endl;
  if(has_depth)
    stream << "C += gidz*ldc*N;" << std::endl;

  stream << "M -= ids.x;" << std::endl;
  stream << "M -= ids.z*" << vwidth_ << ";" << std::endl;

  stream << "N -= ids.y;" << std::endl;
  stream << "N -= ids.w*" << vwidth_ <<  ";" << std::endl;

  for(unsigned int n=0; n < nS_; ++n)
  {
    string Cj = to_string((n/vwidth_)*(ls1_*vwidth_) + n%vwidth_);
    stream << "if(" << Cj << " >= N) return;" << std::endl;
    for(unsigned int m=0; m < mS_; ++m)
      stream << "rC[" << m << "][" << n << "] *= alpha;" << std::endl;
    for(unsigned int m=0; m < mS_; ++m)
    {
      string Ci = to_string((m/vwidth_)*(ls0_*vwidth_) + m%vwidth_);
      stream << "if(" << Ci << "< M) ";
      if(has_depth)
        stream << "C[" << Ci << CSTRIDE1 << "] = rC[" << m << "][" << n << "];" << std::endl;
      else
        stream << "C[" << Ci << CSTRIDE1 << "] = rC[" << m << "][" << n << "] + ((beta != (" << sdtype << ")0)?(beta*" << "C[" << Ci << CSTRIDE1 << "]):0);" << std::endl;
    }
    if((n+1)%vwidth_==0){
      stream << "C += ldc*" << ls1_*vwidth_ - vwidth_ + 1 << ";" << std::endl;
    }
    else{
      stream << "C += ldc;" << std::endl;
    }

  }

  stream.dec_tab();
  stream << "}" << std::endl;

  if(has_depth)
  {
    stream << "$KERNEL void reduce" << suffix << "($SIZE_T M, $SIZE_T N, $SIZE_T D, "
           << "$GLOBAL " << sdtype << "* Z, $SIZE_T Zld,"
           << "$GLOBAL " << sdtype << "* C, $SIZE_T ldc, $SIZE_T Cstart, $SIZE_T Cstride,"
           << sdtype << " beta)"
           << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();

    stream << "C += Cstart;" << std::endl;
    stream << "for(unsigned int i = $GLOBAL_IDX_0 ;  i < M ;  i += $GLOBAL_SIZE_0)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    stream << "for(unsigned int j = $GLOBAL_IDX_1 ;  j < N ;  j += $GLOBAL_SIZE_1)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    stream << sdtype << " acc = 0;" << std::endl;
    stream << "for(unsigned int k = 0 ;  k < D ;  k++)" << std::endl;
    stream.inc_tab();
    stream << "acc += Z[i + j*Zld + k*Zld*N];" << std::endl;
    stream.dec_tab();
    stream << "C[i*Cstride + j*ldc] = acc + ((beta != (" << sdtype << ")0)?(beta*C[i*Cstride + j*ldc]):0);" << std::endl;
    stream.dec_tab();
    stream << "}" << std::endl;
    stream.dec_tab();
    stream << "}" << std::endl;

    stream.dec_tab();
    stream << "}" << std::endl;
  }

  return stream.str();

#undef VLOAD
#undef VST0RE
}

void gemm::enqueue_block(driver::CommandQueue & queue, int_t M, int_t N, int_t K,
                         expression_tree::node const & A, expression_tree::node const & B, expression_tree::node const & C,
                         value_scalar const & alpha, value_scalar const & beta,
                         driver::Program const & program, std::string const & suffix, runtime::execution_options_type const & options)
{
  using tools::align;

  if(M==0 || N==0 || K==0)
    return;

  driver::backend_type backend = queue.context().backend();

  std::string gemm_name = "gemm";
  std::string reduce_name = "reduce";

  gemm_name += suffix;
  reduce_name += suffix;

  driver::Kernel gemm(program, gemm_name.c_str());
  driver::NDRange local(ls0_, ls1_, 1);
  driver::NDRange global(align(align(M,mS_)/mS_, ls0_), align(align(N,nS_)/nS_, ls1_), depth_);

  unsigned int current_arg = 0;

  driver::Buffer& workspace = driver::backend::workspaces::get(options.queue(queue.context()));
  gemm.setSizeArg(current_arg++, M);
  gemm.setSizeArg(current_arg++, N);
  gemm.setSizeArg(current_arg++, K);
  if(depth_==1)
  {
    if(backend==driver::OPENCL)
      gemm.setArg(current_arg++, C.array.handle.cl);
    else
      gemm.setArg(current_arg++, C.array.handle.cu);
    gemm.setSizeArg(current_arg++, C.ld[1]);
    gemm.setSizeArg(current_arg++, C.array.start);
    gemm.setSizeArg(current_arg++, C.ld[0]);
  }
  else
  {
    gemm.setArg(current_arg++, workspace);
    gemm.setSizeArg(current_arg++, M);
    gemm.setSizeArg(current_arg++, 0);
    gemm.setSizeArg(current_arg++, 1);
  }


  gemm.setArg(current_arg++, alpha);
  if(backend==driver::OPENCL)
    gemm.setArg(current_arg++, A.array.handle.cl);
  else
    gemm.setArg(current_arg++, A.array.handle.cu);
  gemm.setSizeArg(current_arg++, A.ld[1]);
  gemm.setSizeArg(current_arg++, A.array.start);
  gemm.setSizeArg(current_arg++, A.ld[0]);

  if(backend==driver::OPENCL)
    gemm.setArg(current_arg++, B.array.handle.cl);
  else
    gemm.setArg(current_arg++, B.array.handle.cu);
  gemm.setSizeArg(current_arg++, B.ld[1]);
  gemm.setSizeArg(current_arg++, B.array.start);
  gemm.setSizeArg(current_arg++, B.ld[0]);

  gemm.setArg(current_arg++, beta);
  options.enqueue(program.context(), gemm, global, local);

  if(depth_ > 1)
  {
    unsigned int current_arg = 0;
    driver::Kernel reduce(program, reduce_name.c_str());
    driver::NDRange local(ls0_, ls1_);
    driver::NDRange global(align(M, ls0_), align(N, ls1_));
    reduce.setSizeArg(current_arg++, M);
    reduce.setSizeArg(current_arg++, N);
    reduce.setSizeArg(current_arg++, depth_);
    reduce.setArg(current_arg++, workspace);
    reduce.setSizeArg(current_arg++, M);
    if(backend==driver::OPENCL)
      reduce.setArg(current_arg++, C.array.handle.cl);
    else
      reduce.setArg(current_arg++, C.array.handle.cu);
    reduce.setSizeArg(current_arg++, C.ld[1]);
    reduce.setSizeArg(current_arg++, C.array.start);
    reduce.setSizeArg(current_arg++, C.ld[0]);
    reduce.setArg(current_arg++, beta);
    options.enqueue(program.context(), reduce, global, local);
  }

}

gemm::gemm(unsigned int vwidth
           ,int_t ls0, int_t kL, int_t ls1, int_t D
           ,int_t ms, int_t ks, int_t ns
           ,int_t lf0, int_t lf1, char A_trans, char B_trans) :
  parameterized_base(vwidth, ls0, ls1), mL_(ms*ls0), kL_(kL), nL_(ns*ls1), depth_(D), mS_(ms), kS_(ks)
                                     , nS_(ns), lf0_(lf0), lf1_(lf1), A_trans_(A_trans), B_trans_(B_trans)
{
  if(A_trans_=='N' && B_trans_=='N') type_ = GEMM_NN;
  else if(A_trans_=='T' && B_trans_=='N') type_ = GEMM_TN;
  else if(A_trans_=='N' && B_trans_=='T') type_ = GEMM_NT;
  else if(A_trans_=='T' && B_trans_=='T') type_ = GEMM_TT;
  else throw;
}

std::vector<int_t> gemm::input_sizes(expression_tree const & expressions) const
{
  symbolic::preset::gemm::args dummy;
  return infos((expression_tree&)expressions, dummy, A_trans_);
}

void gemm::enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, runtime::execution_handler const & control)
{
  expression_tree const & expressions = control.x();
  symbolic::preset::gemm::args args;
  std::vector<int_t> MNK = infos(expressions, args, A_trans_);
  int_t M = MNK[0];
  int_t N = MNK[1];
  int_t K = MNK[2];
  //Skip if empty
  if(M==0 || N == 0 || K ==0)
    return;
  //Enqueue
  runtime::execution_options_type const & options = control.execution_options();
  enqueue_block(queue,  M, N, K, *args.A, *args.B, *args.C, args.alpha, args.beta, program, suffix, options);
}

//
gemm_nn::gemm_nn(unsigned int vwidth
                 , int_t ls0, int_t KL, int_t ls1, int_t D
                 , int_t ms, int_t ks, int_t ns
                 , int_t lf0, int_t lf1) :
  gemm(vwidth, ls0, KL, ls1, D, ms, ks, ns, lf0, lf1, 'N', 'N')
{
}

//
gemm_tn::gemm_tn(unsigned int vwidth
                 , int_t ls0, int_t KL, int_t ls1, int_t D
                 , int_t ms, int_t ks, int_t ns
                 , int_t lf0, int_t lf1) :
  gemm(vwidth, ls0, KL, ls1, D, ms, ks, ns, lf0, lf1, 'T', 'N')
{ }

//
gemm_nt::gemm_nt(unsigned int vwidth
                 , int_t ls0, int_t KL, int_t ls1, int_t D
                 , int_t ms, int_t ks, int_t ns
                 , int_t lf0, int_t lf1) :
  gemm(vwidth, ls0, KL, ls1, D, ms, ks, ns, lf0, lf1, 'N', 'T')
{ }

//
gemm_tt::gemm_tt(unsigned int vwidth
                 , int_t ls0, int_t KL, int_t ls1, int_t D
                 , int_t ms, int_t ks, int_t ns
                 , int_t lf0, int_t lf1) :
  gemm(vwidth, ls0, KL, ls1, D, ms, ks, ns, lf0, lf1, 'T', 'T')
{ }

}
}

