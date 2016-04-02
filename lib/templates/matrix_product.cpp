/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */

#include "isaac/array.h"
#include "isaac/symbolic/expression/preset.h"
#include "isaac/symbolic/engine/process.h"
#include "isaac/templates/matrix_product.h"
#include "isaac/templates/engine/keywords.h"
#include "isaac/exception/api.h"
#include "tools/arguments.hpp"
#include "tools/vector_types.hpp"


#include <string>
#include "isaac/tools/cpp/align.hpp"

namespace isaac
{
namespace templates
{

matrix_product_parameters::matrix_product_parameters(unsigned int simd_width
                          , unsigned int local_size_0, unsigned int KL, unsigned int local_size_1, unsigned int D
                          , unsigned int ms, unsigned int ks, unsigned int ns
                          , fetching_policy_type A_fetching_policy, fetching_policy_type B_fetching_policy
                          , unsigned int local_fetch_0, unsigned int local_fetch_1): base::parameters_type(simd_width, local_size_0, local_size_1, 1),
  kL(KL), depth(D), mS(ms), kS(ks), nS(ns), A_fetching_policy(A_fetching_policy), B_fetching_policy(B_fetching_policy),
  local_fetch_0(local_fetch_0), local_fetch_1(local_fetch_1),
  mL(ms*local_size_0), nL(ns*local_size_1)
{
}


  unsigned int matrix_product::lmem_usage(expression_tree const & expression) const
  {
    unsigned int N = 0;
    N += p_.kL * p_.mL;
    N += p_.nL * p_.kL;
    return N*size_of(expression.dtype());
  }

  unsigned int matrix_product::registers_usage(expression_tree const & expression) const
  {
    unsigned int N = p_.mS * p_.nS + p_.mS * p_.kS + p_.kS * p_.nS;
    return N*size_of(expression.dtype());
  }

  unsigned int matrix_product::temporary_workspace(expression_tree const & expressions) const
  {
      std::vector<int_t> MNK = input_sizes(expressions);
      int_t M = MNK[0]; int_t N = MNK[1];
      if(p_.depth > 1)
        return M*N*p_.depth;
      return 0;
  }

  int matrix_product::is_invalid_impl(driver::Device const &, expression_tree const &) const
  {
//    if(device.vendor()==driver::Device::Vendor::NVIDIA && p_.simd_width > 1)
//      return TEMPLATE_INVALID_SIMD_WIDTH;

    if(p_.A_fetching_policy!=FETCH_FROM_LOCAL || p_.B_fetching_policy!=FETCH_FROM_LOCAL)
      return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;

    if ((p_.mS % p_.simd_width) > 0 || (p_.nS % p_.simd_width) > 0)
      return TEMPLATE_MS_NS_MUST_BE_SIMD_WIDTH_MULTIPLE;

    if(p_.mL > 256 || p_.nL > 256)
       return TEMPLATE_BLOCK_SIZE_TOO_LARGE;

    if ( p_.kS % p_.kL == 0)
      return TEMPLATE_KS_MUST_BE_SMALLER_THAN_KL;

    if (p_.A_fetching_policy==FETCH_FROM_LOCAL || p_.B_fetching_policy==FETCH_FROM_LOCAL){
      if ((p_.local_fetch_0*p_.local_fetch_1) !=(p_.local_size_0*p_.local_size_1))
        return TEMPLATE_LOCAL_FETCH_PRODUCT_MUST_MATCH_LOCAL_SIZE_PRODUCT;
    }

    if (p_.A_fetching_policy==FETCH_FROM_LOCAL)
    {
      unsigned int bound1 = (A_trans_=='N')?p_.kL:p_.mL;
      unsigned int bound0 = (A_trans_=='N')?p_.mL:p_.kL;

      if (p_.local_fetch_1>0 && (bound1 % p_.local_fetch_1)> 0)
        return A_trans_=='N'?TEMPLATE_LOCAL_FETCH_1_MUST_BE_KL_MULTIPLE:TEMPLATE_LOCAL_FETCH_1_MUST_BE_ML_MULTIPLE;

      if (p_.local_fetch_0>0 && (bound0 % (p_.local_fetch_0*p_.simd_width)) > 0)
        return A_trans_=='N'?TEMPLATE_LOCAL_FETCH_0_MUST_BE_NL_MULTIPLE:TEMPLATE_LOCAL_FETCH_0_MUST_BE_KL_MULTIPLE;

    }
    if (p_.B_fetching_policy==FETCH_FROM_LOCAL)
    {
      unsigned int bound1 = (B_trans_=='T')?p_.kL:p_.nL;
      unsigned int bound0 = (B_trans_=='T')?p_.nL:p_.kL;

      if (p_.local_fetch_1>0 && (bound1 % p_.local_fetch_1)> 0)
        return B_trans_=='T'?TEMPLATE_LOCAL_FETCH_1_MUST_BE_KL_MULTIPLE:TEMPLATE_LOCAL_FETCH_1_MUST_BE_ML_MULTIPLE;

      if (p_.local_fetch_0>0 && (bound0 % (p_.local_fetch_0*p_.simd_width)) > 0)
        return B_trans_=='T'?TEMPLATE_LOCAL_FETCH_1_MUST_BE_KL_MULTIPLE:TEMPLATE_LOCAL_FETCH_1_MUST_BE_ML_MULTIPLE;

    }

    return TEMPLATE_VALID;
  }

  std::string matrix_product::generate_impl(std::string const & suffix, expression_tree const & tree, driver::Device const & device, symbolic::symbols_table const &) const
  {
    using std::string;
    using tools::to_string;

    driver::backend_type backend = device.backend();
    bool has_depth = p_.depth > 1;
#define VLOAD(offset, ptr) vload(p_.simd_width, sdtype, offset, ptr, "1", backend, true)
#define VLOAD_MISALIGNED(offset, ptr) vload(p_.simd_width, sdtype, offset, ptr, "1", backend, false)
#define VSTORE(value, offset, ptr) vstore(p_.simd_width, sdtype, value, offset, ptr, "1", backend)
#define ASTRIDE1 string(check_bounds_?"*Astride1":"")
#define BSTRIDE1 string(check_bounds_?"*Bstride1":"")
#define CSTRIDE1 string(check_bounds_?"*Cstride1":"")



    //////////////////
    /// INIT
    /// //////////////
    kernel_generation_stream stream(backend);
    numeric_type dtype = tree.dtype();
    std::string sdtype = to_string(dtype);
    std::string vdtype = append_width(sdtype, p_.simd_width);

    //////////////////
    /// DECLARATIONS
    /// //////////////
    std::string matrix_product_name = "matrix_product";
    std::string reduce_name = "reduce";

    matrix_product_name += suffix;
    reduce_name += suffix;

    switch(backend)
    {
      case driver::CUDA:
        stream << "#include  \"helper_math.h\"" << std::endl; break;
      case driver::OPENCL:
        stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl; break;
    }

    stream << "$KERNEL void matrix_product" << suffix << "($SIZE_T M, $SIZE_T N, $SIZE_T K, "
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
    stream << sdtype << " rC[" << p_.mS << "][" << p_.nS << "] = {{0}};" << std::endl;
    stream << vdtype << " rA[" << p_.kS << "][" << p_.mS/p_.simd_width << "];" << std::endl;
    stream << vdtype << " rB[" << p_.kS << "][" << p_.nS/p_.simd_width << "];" << std::endl;
    stream << std::endl;

    stream << "//pointers" << std::endl;
    size_t llda = (A_trans_=='N')?p_.mL:p_.kL;
    size_t lldb = (B_trans_=='T')?p_.nL:p_.kL;
    stream << "$LOCAL " << sdtype << " lA[" << p_.kL*p_.mL << "];" << std::endl;
    stream << "$LOCAL " << sdtype << " lB[" << p_.kL*p_.nL << "];" << std::endl;
    unsigned int npA = p_.mL/(A_trans_=='N'?p_.local_fetch_0*p_.simd_width:p_.local_fetch_1);
    unsigned int npB = p_.nL/(B_trans_=='T'?p_.local_fetch_0*p_.simd_width:p_.local_fetch_1);
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
      stream << "div = (K+" << p_.depth-1 << ")/" << p_.depth << ";" << std::endl;
      stream << "offz = div*gidz;" << std::endl;
      stream << "K = min(K - div*gidz, ($SIZE_T)div);" << std::endl;
    }

    stream << "idt = " << p_.local_size_0 << "*ids.w + ids.z;" << std::endl;
    stream << "idT.y = idt/" << p_.local_fetch_0 << ";" << std::endl;
    stream << "idT.x = idt - " << p_.local_fetch_0 << "*idT.y;" << std::endl;
    stream << std::endl;

    stream << "//Adjust pointers and bounds per work-item" << std::endl;
    stream << "ids.x *= " << p_.mL << ";" << std::endl;
    stream << "ids.y *= " << p_.nL << ";" << std::endl;
    stream << "idT.x *= " << p_.simd_width << ";" << std::endl;

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
          stream << "Ai[" << i << "] += " << Select(backend, to_string(i*p_.local_fetch_0*p_.simd_width) + " < M", "(int)((idT.x + " + to_string(i*p_.local_fetch_0*p_.simd_width) + ")" + ASTRIDE1 + ")", "0") << ";" << std::endl;
        else
          stream << "Ai[" << i << "] += " << Select(backend, to_string(i*p_.local_fetch_1) + " < M", "(int)((idT.y + " + to_string(i*p_.local_fetch_1) + ")*lda)", "0") << ";" << std::endl;

    for(unsigned int i = 0 ; i < npB ; i++ )
        if (B_trans_=='T')
            stream << "Bi[" << i << "] += " << Select(backend, to_string(i*p_.local_fetch_0*p_.simd_width) + " < N", "(int)((idT.x + " + to_string(i*p_.local_fetch_0*p_.simd_width) + ")" + BSTRIDE1 + ")", "0") << ";" << std::endl;
        else
            stream << "Bi[" << i << "] += " << Select(backend, to_string(i*p_.local_fetch_1) + " < N", "(int)((idT.y + " + to_string(i*p_.local_fetch_1) + ")*ldb)", "0") << ";" << std::endl;

    stream << std::endl;
    stream << "//Outer loop" << std::endl;
    stream << "while(K >=" << p_.kL << ")" << std::endl;
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
          for(unsigned int k = 0; k < p_.kL; k += p_.local_fetch_1)
            for(unsigned int m = 0; m < p_.mL; m += p_.local_fetch_0*p_.simd_width)
            {
              std::string mm = to_string(m/(p_.simd_width*p_.local_fetch_0));
              std::string kk = to_string(k);
              if(last_iteration)
                  for(unsigned int s = 0 ; s < p_.simd_width ; ++s)
                      stream << "ldsA[" << k*llda + m + s << "] = (condy" << k << " && " << s << "< M)? Ai[" << mm << "][" << k << "*lda + " << s << "] : 0;" << std::endl;
              else
                stream << VSTORE(VLOAD_MISALIGNED("0" ,"&Ai[" + mm +"][" + kk + "*lda]"), "0", "ldsA + " + to_string(k*llda+m)) << ";" << std::endl;
            }
        }
        else
        {
            for(unsigned int k = 0; k < p_.kL; k += p_.local_fetch_0*p_.simd_width)
            for(unsigned int m = 0; m < p_.mL; m += p_.local_fetch_1)
              {
                std::string mm = to_string(m/p_.local_fetch_1);
                std::string kk = to_string(k);
                if(last_iteration)
                    for(unsigned int s = 0 ; s < p_.simd_width ; ++s)
                        stream << "ldsA[" << m*llda + k + s << "] = condx" << k + s << "? Ai[" << mm << "][" << k + s << ASTRIDE1 << "] : 0;" << std::endl;

                else
                    stream << VSTORE(VLOAD_MISALIGNED("0", "&Ai[" + mm + "][" + kk + ASTRIDE1 + "]"), "0", "ldsA + " + to_string(m*llda+k)) << ";" << std::endl;
              }
        }

        stream << "//Fetch B to local memory" << std::endl;
        if (B_trans_=='T')
        {
          for(unsigned int k = 0; k < p_.kL; k += p_.local_fetch_1)
            for(unsigned int n = 0; n < p_.nL; n += p_.local_fetch_0*p_.simd_width)
            {
              std::string nn = to_string(n/(p_.simd_width*p_.local_fetch_0));
              std::string kk = to_string(k);
              if(last_iteration)
                  for(unsigned int s = 0 ; s < p_.simd_width ; ++s)
                      stream << "ldsB[" << k*lldb + n + s << "] = (condy" << k << " && " << s << "< N)? Bi[" <<  nn << "][" << kk << "*ldb +" << s << "] : 0;" << std::endl;
              else
                stream << VSTORE(VLOAD_MISALIGNED("0" ,"&Bi[" + nn +"][" + kk + "*ldb]"), "0", "ldsB + " + to_string(k*lldb+n)) << ";" << std::endl;
            }
        }
        else
        {
          for(unsigned int k = 0; k < p_.kL; k += p_.local_fetch_0*p_.simd_width)
            for(unsigned int n = 0; n < p_.nL; n += p_.local_fetch_1)
            {
              std::string nn = to_string(n/p_.local_fetch_1);
              std::string kk = to_string(k);
              if(last_iteration)
                  for(unsigned int s = 0 ; s < p_.simd_width ; ++s)
                      stream << "ldsB[" << n*lldb + k + s << "] = condx" << k + s << "? Bi[" << nn << "][" << k + s << BSTRIDE1 << "] : 0;" << std::endl;

              else
                  stream << VSTORE(VLOAD_MISALIGNED("0", "&Bi[" + nn + "][" + kk + BSTRIDE1 + "]"), "0", "ldsB + " + to_string(n*lldb+k)) << ";" << std::endl;
            }
        }

        if(A_trans_=='N')
            stream << "ldsA = lA + ids.z*" << p_.simd_width << ";" << std::endl;
        else
            stream << "ldsA = lA + ids.z*" << llda*p_.simd_width << ";" << std::endl;

        if(B_trans_=='T')
            stream << "ldsB = lB + ids.w*" << p_.simd_width << ";" << std::endl;
        else
            stream << "ldsB = lB + ids.w*" << lldb*p_.simd_width << ";" << std::endl;

        stream << "$LOCAL_BARRIER;" << std::endl;

        stream << "//Inner loop" << std::endl;
        stream << "for(unsigned int k = 0; k < " << p_.kL << "; k+=" << p_.kS << "){" << std::endl;
        stream.inc_tab();

        stream << "//Fetch A to registers" << std::endl;
        stream << "#pragma unroll" << std::endl;
        stream << "for(unsigned int kk = 0; kk < " << p_.kS << "; kk++)" << std::endl;
        stream << "#pragma unroll " << p_.mS/p_.simd_width << std::endl;
        stream << "for(unsigned int mm = 0; mm < " << p_.mS/p_.simd_width << "; mm++)" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();
        if(A_trans_=='N')
            stream << "rA[kk][mm] = "  << VLOAD("0", "ldsA + k*" + to_string(llda) + " + mm*" + to_string(p_.local_size_0*p_.simd_width) + "+ kk*" + to_string(llda)) << ";" << std::endl;
        else
        {
            if(p_.simd_width==1)
                stream << "rA[kk][mm] = ldsA[k + mm*" << p_.local_size_0*llda <<  "+ kk"  << "];" << std::endl;
            else
                for(unsigned int s = 0 ; s < p_.simd_width ; ++s)
                    stream << access_vector_type("rA[kk][mm]", s) << " = ldsA[k + (mm*" << p_.simd_width*p_.local_size_0 << " + " << s << ")*" << llda <<  "+ kk];" << std::endl;
        }

        stream.dec_tab();
        stream << "}" << std::endl;

        stream << "//Fetch B to registers" << std::endl;
        stream << "#pragma unroll " << p_.kS << std::endl;
        stream << "for(unsigned int kk = 0; kk < " << p_.kS << "; kk++)" << std::endl;
        stream << "#pragma unroll " << p_.nS/p_.simd_width << std::endl;
        stream << "for(unsigned int nn = 0; nn < " << p_.nS/p_.simd_width << "; nn++)" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();
        if(B_trans_=='T')
            stream << "rB[kk][nn] = " << VLOAD("0", "ldsB + k*" + to_string(lldb) + " + nn*" + to_string(p_.local_size_1*p_.simd_width)  + "+ kk*" + to_string(lldb)) << ";" << std::endl;
        else
        {
            if(p_.simd_width==1)
                stream << "rB[kk][nn] = ldsB[k"  << " + nn*" << p_.local_size_1*lldb <<  "+ kk"  << "];" << std::endl;
            else
                for(unsigned int s = 0 ; s < p_.simd_width ; ++s)
                    stream << access_vector_type("rB[kk][nn]", s) << " = ldsB[k"  << " + (nn*" << p_.simd_width*p_.local_size_1 << " + " << s << ")*" << lldb <<  "+ kk];" << std::endl;
        }
        stream.dec_tab();
        stream << "}" << std::endl;

        stream << "//FMA computations" << std::endl;
        for(unsigned int kk=0 ; kk < p_.kS; ++kk)
        for(unsigned int nn=0; nn < p_.nS; ++nn)
        for(unsigned int mm=0; mm < p_.mS; ++mm){
          string res_str, lhs_str, rhs_str;
          res_str = "rC[" + to_string(mm) + "][" + to_string(nn) + "]";
          if (p_.simd_width==1)
            lhs_str = "rA[" + to_string(kk) + "][" + to_string(mm) + "]";
          else
            lhs_str = access_vector_type("rA[" + to_string(kk) + "][" + to_string(mm/p_.simd_width) + "]", mm%p_.simd_width);
          if (p_.simd_width==1)
            rhs_str = "rB[" + to_string(kk) + "]["+to_string(nn)+"]";
          else
            rhs_str = access_vector_type("rB[" + to_string(kk) + "]["+to_string(nn/p_.simd_width)+"]", nn%p_.simd_width);
          stream << res_str << "= $MAD(" << lhs_str << "," << rhs_str << "," << res_str << ");" << std::endl;
        }

        stream.dec_tab();
        stream << "}" << std::endl;
        stream << "K -= " << p_.kL << ";" << std::endl;

        //Increment A pointers to global memory
        if (A_trans_=='N')
          for(unsigned int i = 0 ; i < npA ; ++i)
              stream << "Ai[" << i << "] += "  << p_.kL << "*lda;" << std::endl;
        else
          for(unsigned int i = 0 ; i < npA ; ++i)
              stream << "Ai[" << i << "] += "  << p_.kL << ASTRIDE1 << ";" << std::endl;

        //Increment B pointers to global memory
        if (B_trans_=='T')
          for(unsigned int i = 0 ; i < npB ; ++i)
              stream << "Bi[" << i << "] += " << p_.kL << "*ldb;" << std::endl;
        else
          for(unsigned int i = 0 ; i < npB ; ++i)
              stream << "Bi[" << i << "] += " << p_.kL << BSTRIDE1 << ";" << std::endl;
    };
    fetch_to_lds(false);
    stream.dec_tab();
    stream << "}" << std::endl;


    if(A_trans_=='N' || B_trans_=='T')
    {
        stream << "int Ky = K - idT.y;" << std::endl;
        for(unsigned int k = 0; k < p_.kL; k += p_.local_fetch_1)
            stream << "int condy" << k << " = " << k << " < Ky;" << std::endl;
    }

    if(A_trans_=='T' || B_trans_=='N')
    {
        stream << "int Kx = K - idT.x;" << std::endl;
        for(unsigned int k = 0 ; k < p_.kL ; k += p_.local_fetch_0*p_.simd_width)
            for(unsigned int s = 0 ; s < p_.simd_width ; ++s)
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
    stream << "C += ids.z*" << p_.simd_width << CSTRIDE1 << ";" << std::endl;
    stream << "C += ids.y*ldc;" << std::endl;
    stream << "C += ids.w*" << p_.simd_width << "*ldc;" << std::endl;
    if(has_depth)
        stream << "C += gidz*ldc*N;" << std::endl;

    stream << "M -= ids.x;" << std::endl;
    stream << "M -= ids.z*" << p_.simd_width << ";" << std::endl;

    stream << "N -= ids.y;" << std::endl;
    stream << "N -= ids.w*" << p_.simd_width <<  ";" << std::endl;

    for(unsigned int n=0; n < p_.nS; ++n)
    {
        string Cj = to_string((n/p_.simd_width)*(p_.local_size_1*p_.simd_width) + n%p_.simd_width);
        stream << "if(" << Cj << " >= N) return;" << std::endl;
        for(unsigned int m=0; m < p_.mS; ++m)
            stream << "rC[" << m << "][" << n << "] *= alpha;" << std::endl;
        for(unsigned int m=0; m < p_.mS; ++m)
        {
            string Ci = to_string((m/p_.simd_width)*(p_.local_size_0*p_.simd_width) + m%p_.simd_width);
            stream << "if(" << Ci << "< M) ";
            if(has_depth)
                stream << "C[" << Ci << CSTRIDE1 << "] = rC[" << m << "][" << n << "];" << std::endl;
            else
                stream << "C[" << Ci << CSTRIDE1 << "] = rC[" << m << "][" << n << "] + ((beta != (" << sdtype << ")0)?(beta*" << "C[" << Ci << CSTRIDE1 << "]):0);" << std::endl;
        }
        if((n+1)%p_.simd_width==0){
            stream << "C += ldc*" << p_.local_size_1*p_.simd_width - p_.simd_width + 1 << ";" << std::endl;
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
      stream << "C[i*Cstride + j*ldc] = acc + beta*C[i*Cstride + j*ldc];" << std::endl;
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

  void matrix_product::enqueue_block(driver::CommandQueue & queue, int_t M, int_t N, int_t K,
                     expression_tree::node const & A, expression_tree::node const & B, expression_tree::node const & C,
                     value_scalar const & alpha, value_scalar const & beta,
                     driver::Program const & program, std::string const & suffix, execution_options_type const & options)
  {
    using tools::align;

    if(M==0 || N==0 || K==0)
      return;

    driver::backend_type backend = queue.context().backend();

    std::string matrix_product_name = "matrix_product";
    std::string reduce_name = "reduce";

    matrix_product_name += suffix;
    reduce_name += suffix;

    driver::Kernel matrix_product(program, matrix_product_name.c_str());
    driver::NDRange local(p_.local_size_0, p_.local_size_1, 1);
    driver::NDRange global(align(align(M,p_.mS)/p_.mS, p_.local_size_0), align(align(N,p_.nS)/p_.nS, p_.local_size_1), p_.depth);

    unsigned int current_arg = 0;

    driver::Buffer& workspace = driver::backend::workspaces::get(options.queue(queue.context()));
    matrix_product.setSizeArg(current_arg++, M);
    matrix_product.setSizeArg(current_arg++, N);
    matrix_product.setSizeArg(current_arg++, K);
    if(p_.depth==1)
    {
        if(backend==driver::OPENCL)
          matrix_product.setArg(current_arg++, C.array.handle.cl);
        else
          matrix_product.setArg(current_arg++, C.array.handle.cu);
        matrix_product.setSizeArg(current_arg++, C.ld[1]);
        matrix_product.setSizeArg(current_arg++, C.array.start);
        matrix_product.setSizeArg(current_arg++, C.ld[0]);
    }
    else
    {
        matrix_product.setArg(current_arg++, workspace);
        matrix_product.setSizeArg(current_arg++, M);
        matrix_product.setSizeArg(current_arg++, 0);
        matrix_product.setSizeArg(current_arg++, 1);
    }


    matrix_product.setArg(current_arg++, alpha);
    if(backend==driver::OPENCL)
      matrix_product.setArg(current_arg++, A.array.handle.cl);
    else
      matrix_product.setArg(current_arg++, A.array.handle.cu);
    matrix_product.setSizeArg(current_arg++, A.ld[1]);
    matrix_product.setSizeArg(current_arg++, A.array.start);
    matrix_product.setSizeArg(current_arg++, A.ld[0]);

    if(backend==driver::OPENCL)
      matrix_product.setArg(current_arg++, B.array.handle.cl);
    else
      matrix_product.setArg(current_arg++, B.array.handle.cu);
    matrix_product.setSizeArg(current_arg++, B.ld[1]);
    matrix_product.setSizeArg(current_arg++, B.array.start);
    matrix_product.setSizeArg(current_arg++, B.ld[0]);

    matrix_product.setArg(current_arg++, beta);
    options.enqueue(program.context(), matrix_product, global, local);

    if(p_.depth > 1)
    {
      unsigned int current_arg = 0;
      driver::Kernel reduce(program, reduce_name.c_str());
      driver::NDRange local(p_.local_size_0, p_.local_size_1);
      driver::NDRange global(align(M, p_.local_size_0), align(N, p_.local_size_1));
      reduce.setSizeArg(current_arg++, M);
      reduce.setSizeArg(current_arg++, N);
      reduce.setSizeArg(current_arg++, p_.depth);
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

  std::vector<int_t> matrix_product::infos(expression_tree const & tree, symbolic::preset::matrix_product::args& arguments) const
  {
    expression_tree::data_type const & array = tree.data();
    std::size_t root = tree.root();
    arguments = symbolic::preset::matrix_product::check(array, root);
    int_t M = arguments.C->shape[0];
    int_t N = arguments.C->shape[1];
    int_t K = (A_trans_=='T')?arguments.A->shape[0]:arguments.A->shape[1];
    return {M, N, K};
  }

  matrix_product::matrix_product(matrix_product_parameters const & parameters, bool check_bounds, char A_trans, char B_trans) : base_impl<matrix_product, matrix_product_parameters>(parameters, FUSE_INDEPENDENT), A_trans_(A_trans), B_trans_(B_trans), check_bounds_(check_bounds)
  {
    if(A_trans_=='N' && B_trans_=='N') type_ = MATRIX_PRODUCT_NN;
    else if(A_trans_=='T' && B_trans_=='N') type_ = MATRIX_PRODUCT_TN;
    else if(A_trans_=='N' && B_trans_=='T') type_ = MATRIX_PRODUCT_NT;
    else if(A_trans_=='T' && B_trans_=='T') type_ = MATRIX_PRODUCT_TT;
    else throw;
  }

  std::vector<int_t> matrix_product::input_sizes(expression_tree const & expressions) const
  {
    symbolic::preset::matrix_product::args dummy;
    return infos((expression_tree&)expressions, dummy);
  }

  void matrix_product::enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, base & fallback_base, execution_handler const & control)
  {
    using namespace tools;

    matrix_product & fallback = (matrix_product&)fallback_base;
    expression_tree const & expressions = control.x();


    symbolic::preset::matrix_product::args args;
    std::vector<int_t> MNK = infos(expressions, args);
    int_t M = MNK[0];
    int_t N = MNK[1];
    int_t K = MNK[2];
    //Skip if empty
    if(M==0 || N == 0 || K ==0)
      return;
    //Enqueue
    execution_options_type const & options = control.execution_options();
    if (args.A->ld[0] > 1 || args.B->ld[0] > 1 || args.C->ld[0] > 1)
      fallback.enqueue_block(queue, M, N, K, *args.A, *args.B, *args.C, args.alpha, args.beta, program, "fallback", options);
    else
      enqueue_block(queue,  M, N, K, *args.A, *args.B, *args.C, args.alpha, args.beta, program, suffix, options);
  }

  //
  matrix_product_nn::matrix_product_nn(unsigned int simd
                           , int_t ls0, int_t KL, int_t ls1, int_t D
                           , int_t ms, int_t ks, int_t ns
                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                           , int_t lfetch0, int_t lfetch1, bool check_bound) :
    matrix_product(matrix_product_parameters(simd, ls0, KL, ls1, D, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1), check_bound, 'N', 'N')
  {
  }

  //
  matrix_product_tn::matrix_product_tn(unsigned int simd
                           , int_t ls0, int_t KL, int_t ls1, int_t D
                           , int_t ms, int_t ks, int_t ns
                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                           , int_t lfetch0, int_t lfetch1, bool check_bound) :
    matrix_product(matrix_product_parameters(simd, ls0, KL, ls1, D, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1), check_bound, 'T', 'N')
  { }

  //
  matrix_product_nt::matrix_product_nt(unsigned int simd
                           , int_t ls0, int_t KL, int_t ls1, int_t D
                           , int_t ms, int_t ks, int_t ns
                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                           , int_t lfetch0, int_t lfetch1, bool check_bound) :
    matrix_product(matrix_product_parameters(simd, ls0, KL, ls1, D, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1), check_bound, 'N', 'T')
  { }

  //
  matrix_product_tt::matrix_product_tt(unsigned int simd
                           , int_t ls0, int_t KL, int_t ls1, int_t D
                           , int_t ms, int_t ks, int_t ns
                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                           , int_t lfetch0, int_t lfetch1, bool check_bound) :
    matrix_product(matrix_product_parameters(simd, ls0, KL, ls1, D, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1), check_bound, 'T', 'T')
  { }

}
}

