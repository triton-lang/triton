#include "isaac/array.h"
#include "isaac/backend/templates/gemm.h"
#include "isaac/backend/keywords.h"
#include "isaac/model/model.h"
#include "isaac/symbolic/preset.h"
#include "isaac/exception/operation_not_supported.h"
#include "isaac/tools/make_vector.hpp"
#include "isaac/tools/to_string.hpp"
#include "isaac/tools/miscellaneous.hpp"

namespace isaac
{
namespace templates
{

gemm_parameters::gemm_parameters(unsigned int simd_width
                          , unsigned int local_size_0, unsigned int KL, unsigned int local_size_1, unsigned int D
                          , unsigned int ms, unsigned int ks, unsigned int ns
                          , fetching_policy_type A_fetching_policy, fetching_policy_type B_fetching_policy
                          , unsigned int local_fetch_0, unsigned int local_fetch_1): base::parameters_type(simd_width, local_size_0, local_size_1, 1),
  kL(KL), depth(D), mS(ms), kS(ks), nS(ns), A_fetching_policy(A_fetching_policy), B_fetching_policy(B_fetching_policy),
  local_fetch_0(local_fetch_0), local_fetch_1(local_fetch_1),
  mL(ms*local_size_0), nL(ns*local_size_1){}


  unsigned int gemm::lmem_usage(expressions_tuple const & expressions) const
  {
    isaac::array_expression const & array_expression = (*expressions.data().front());
    numeric_type numeric_t = lhs_most(array_expression.tree(), array_expression.root()).lhs.dtype;

    unsigned int N = 0;
    N += p_.kL * p_.mL;
    N += p_.nL * p_.kL;
    return N*size_of(numeric_t);
  }

  unsigned int gemm::registers_usage(expressions_tuple const & expressions) const
  {
    isaac::array_expression const & array_expression = (*expressions.data().front());
    numeric_type numeric_t = lhs_most(array_expression.tree(), array_expression.root()).lhs.dtype;

    unsigned int N = p_.mS * p_.nS + p_.mS * p_.kS + p_.kS * p_.nS;
    return N*size_of(numeric_t);
  }

  int gemm::is_invalid_impl(driver::Device const &, expressions_tuple const & expressions) const
  {
    std::vector<int_t> MNK = input_sizes(expressions);
//    int_t M = MNK[0]; int_t N = MNK[1];

    if(p_.A_fetching_policy!=FETCH_FROM_LOCAL || p_.B_fetching_policy!=FETCH_FROM_LOCAL)
      throw operation_not_supported_exception("Only local memory is supported for GEMM");

//    if(p_.depth > 1 && M*N*p_.depth > 2e6)
//      throw operation_not_supported_exception("This would necessitate a temporary larger than 1MB");

    if ((p_.mS % p_.simd_width) > 0 || (p_.nS % p_.simd_width) > 0)
      return TEMPLATE_MS_NS_MUST_BE_SIMD_WIDTH_MULTIPLE;

    if(p_.mL > 256 || p_.nL > 256)
       return 1;

    if ( p_.kS % p_.kL == 0)
      return TEMPLATE_KS_MUST_BE_SMALLER_THAN_KL;

    if (p_.A_fetching_policy==FETCH_FROM_LOCAL || p_.B_fetching_policy==FETCH_FROM_LOCAL)
    {
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

  std::string gemm::generate_impl(const char * suffix, expressions_tuple const & expressions, driver::Device const & device, std::vector<mapping_type> const &) const
  {
    using std::string;
    using tools::to_string;

    driver::backend_type backend = device.backend();
    bool has_depth = p_.depth > 1;
#define VLOAD(offset, ptr) vload(p_.simd_width, sdtype, offset, ptr, backend)
#define VSTORE(value, offset, ptr) vstore(p_.simd_width, sdtype, value, offset, ptr, backend)
#define ASTRIDE1 string(check_bounds_?"*Astride1":"")
#define BSTRIDE1 string(check_bounds_?"*Bstride1":"")
#define CSTRIDE1 string(check_bounds_?"*Cstride1":"")



    //////////////////
    /// INIT
    /// //////////////
    kernel_generation_stream stream;
    array_expression const & st = (*expressions.data().front());
    numeric_type dtype = lhs_most(st.tree(), st.root()).lhs.dtype;
    std::string sdtype = numeric_type_to_string(dtype);
    std::string vdtype = append_width(sdtype, p_.simd_width);
    std::string _size_t = size_type(device);
    std::string vint = append_width("int", p_.simd_width);

    //////////////////
    /// DECLARATIONS
    /// //////////////
    char gemm_name[32] = {"gemm"};
    char reduce_name[32] = {"reduce"};
    strcat(gemm_name, suffix);
    strcat(reduce_name, suffix);

    switch(backend)
    {
  #ifdef ISAAC_WITH_CUDA
      case driver::CUDA: stream << "#include  \"helper_math.h\"" << std::endl; break;
  #endif
      case driver::OPENCL: stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl; break;
    }

    stream << KernelPrefix(backend) << " void " << gemm_name << "(" << _size_t << " M, " << _size_t << " N, " << _size_t << " K, "
                               << Global(backend) << " " << sdtype << "* C, "  << _size_t << " ldc," << _size_t << " offc," << _size_t << " Cstride1, "
                               << sdtype << " alpha,"
                               << Global(backend) << " " << sdtype << "* A, "  << _size_t << " lda," << _size_t << " offa," << _size_t << " Astride1,"
                               << Global(backend) << " " << sdtype << "* B, "  << _size_t << " ldb," << _size_t << " offb," << _size_t << " Bstride1,"
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
    stream << Local(backend) << " " << sdtype << " lA[" << p_.kL*p_.mL << "];" << std::endl;
    stream << Local(backend) << " " << sdtype << " lB[" << p_.kL*p_.nL << "];" << std::endl;
    unsigned int npA = p_.mL/(A_trans_=='N'?p_.local_fetch_0*p_.simd_width:p_.local_fetch_1);
    unsigned int npB = p_.nL/(B_trans_=='T'?p_.local_fetch_0*p_.simd_width:p_.local_fetch_1);
    stream << "__global " << sdtype << "* Ai[" << npA << "];" << std::endl;
    stream << "__global " << sdtype << "* Bi[" << npB << "];" << std::endl;
    stream << std::endl;

    stream << "//identifiers" << std::endl;
    stream << "int2 idT;" << std::endl;
    stream << "int idt;" << std::endl;
    if(has_depth)
        stream << "int gidz, div, offz;" << std::endl;
    stream << "int4 ids = (int4)(" << GroupIdx0(backend) << "," << GroupIdx1(backend) << "," << LocalIdx0(backend) << "," << LocalIdx1(backend) << ");" << std::endl;
    stream << std::endl;

    stream << "//offsets" << std::endl;
    stream << "A += offa;" << std::endl;
    stream << "B += offb;" << std::endl;
    stream << "C += offc;" << std::endl;

    if(has_depth)
    {
      stream << "gidz = " << GroupIdx2(backend) << ";" << std::endl;
      stream << "div = (K+" << p_.depth-1 << ")/" << p_.depth << ";" << std::endl;
      stream << "offz = div*gidz;" << std::endl;
      stream << "K = min(K - div*gidz, div);" << std::endl;
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
          stream << "Ai[" << i << "] += select(0, (int)((idT.x + " << i*p_.local_fetch_0*p_.simd_width << ")" << ASTRIDE1 << "), " << i*p_.local_fetch_0*p_.simd_width << " < M);" << std::endl;
        else
          stream << "Ai[" << i << "] += select(0, (int)((idT.y + " << i*p_.local_fetch_1 << ")*lda), " << i*p_.local_fetch_1 << " < M);" << std::endl;

    for(unsigned int i = 0 ; i < npB ; i++ )
        if (B_trans_=='T')
            stream << "Bi[" << i << "] += select(0, (int)((idT.x + " << i*p_.local_fetch_0*p_.simd_width << ")" << BSTRIDE1 << "), " << i*p_.local_fetch_0*p_.simd_width << " < N);" << std::endl;
        else
            stream << "Bi[" << i << "] += select(0, (int)((idT.y + " << i*p_.local_fetch_1 << ")*ldb), " << i*p_.local_fetch_1 << " < N);" << std::endl;


    stream << std::endl;
    stream << "//Outer loop" << std::endl;
    stream << "while(K >=" << p_.kL << ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();


    auto fetch_to_lds = [&](bool last_iteration)
    {
        stream << LocalBarrier(backend) << ";" << std::endl;
        stream << LocalPtr(backend) << " " << sdtype << "* ldsA = lA + idT.y*" << llda << " + idT.x;" << std::endl;
        stream << LocalPtr(backend) << " " << sdtype << "* ldsB = lB + idT.y*" << lldb << " + idT.x;" << std::endl;

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
                stream << VSTORE(VLOAD("0" ,"&Ai[" + mm +"][" + kk + "*lda]"), "0", "ldsA + " + to_string(k*llda+m)) << ";" << std::endl;
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
                    stream << VSTORE(VLOAD("0", "&Ai[" + mm + "][" + kk + ASTRIDE1 + "]"), "0", "ldsA + " + to_string(m*llda+k)) << ";" << std::endl;
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
                stream << VSTORE(VLOAD("0" ,"&Bi[" + nn +"][" + kk + "*ldb]"), "0", "ldsB + " + to_string(k*lldb+n)) << ";" << std::endl;
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
                  stream << VSTORE(VLOAD("0", "&Bi[" + nn + "][" + kk + BSTRIDE1 + "]"), "0", "ldsB + " + to_string(n*lldb+k)) << ";" << std::endl;
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

        stream << LocalBarrier(backend) << ";" << std::endl;

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
        for(unsigned int mm=0; mm < p_.mS; ++mm)
        {
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
          stream << res_str << "=" << "fma(" << lhs_str << "," << rhs_str << "," << res_str << ");" << std::endl;
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
        stream << "int Ky = K - idT.y;" << std::endl;
    if(A_trans_=='T' || B_trans_=='N')
        stream << "int Kx = K - idT.x;" << std::endl;

    if(A_trans_=='N' || B_trans_=='T')
        for(unsigned int k = 0; k < p_.kL; k += p_.local_fetch_1)
            stream << "int condy" << k << " = " << k << " < Ky;" << std::endl;

    if(A_trans_=='T' || B_trans_=='N')
    {
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
    stream << _size_t << " offx = (ids.x + ids.z*" << p_.simd_width << ")" << ";" << std::endl;
    stream << _size_t << " offy = (ids.y + ids.w*" << p_.simd_width << ");" << std::endl;
    stream << "C += " << "offx" << CSTRIDE1 << " + offy*ldc" << (has_depth?" + gidz*ldc*N;":"") << ";" << std::endl;
    stream << "M -= offx;" << std::endl;
    stream << "N -= offy;" << std::endl;
    stream << "int ibm[" << p_.mS << "];" << std::endl;
    for(unsigned int m=0; m < p_.mS; ++m)
    {
        string Ci = to_string((m/p_.simd_width)*(p_.local_size_0*p_.simd_width) + m%p_.simd_width);
        stream << "ibm[" << m << "] = " << Ci << " < M;" << std::endl;
    }


    for(unsigned int n=0; n < p_.nS; ++n)
    {
        string Cj = to_string((n/p_.simd_width)*(p_.local_size_1*p_.simd_width) + n%p_.simd_width);
        stream << "if(" << Cj << " >= N) return;" << std::endl;
        for(unsigned int m=0; m < p_.mS; ++m)
            stream << "rC[" << m << "][" << n << "] *= alpha;" << std::endl;
        for(unsigned int m=0; m < p_.mS; ++m)
        {
            string Ci = to_string((m/p_.simd_width)*(p_.local_size_0*p_.simd_width) + m%p_.simd_width);
            stream << "if(ibm[" << m << "]) ";
            stream << "C[" << Ci << CSTRIDE1 << "] = rC[" << m << "][" << n << "] + select((" << sdtype << ")0, C[" << Ci << CSTRIDE1 << "], beta>0);" << std::endl;
        }
        if((n+1)%p_.simd_width==0)
            stream << "C += ldc*" << p_.local_size_1*p_.simd_width - p_.simd_width + 1 << ";" << std::endl;
        else
            stream << "C += ldc;" << std::endl;

    }

    stream.dec_tab();
    stream << "}" << std::endl;


    if(has_depth)
    {
      stream << KernelPrefix(backend) << " void " << reduce_name << "(" << _size_t << " M, " << _size_t << " N, " << _size_t << " D, "
                                 << Global(backend) << " " << sdtype << "* Z, "  << _size_t << " Zld,"
                                 << Global(backend) << " " << sdtype << "* C, "  << _size_t << " ldc," << _size_t << " Cstart1," << _size_t << " Cstart2," << _size_t << " Cstride1, "  << _size_t << " Cstride2, "
                                 << sdtype << " beta)"
                                 << std::endl;
      stream << "{" << std::endl;
      stream.inc_tab();

      stream << "C += Cstart1 + Cstart2*ldc;" << std::endl;
      stream << "ldc *= Cstride2;" << std::endl;
      stream << "for(unsigned int i = " << GlobalIdx0(backend) << " ;  i < M ;  i += " << GlobalSize0(backend) << ")" << std::endl;
      stream << "{" << std::endl;
      stream.inc_tab();
      stream << "for(unsigned int j = " << GlobalIdx1(backend) << " ;  j < N ;  j += " << GlobalSize1(backend) << ")" << std::endl;
      stream << "{" << std::endl;
      stream.inc_tab();
      stream << sdtype << " acc = 0;" << std::endl;
      stream << "for(unsigned int k = 0 ;  k < D ;  k++)" << std::endl;
      stream.inc_tab();
      stream << "acc += Z[i + j*Zld + k*Zld*N];" << std::endl;
      stream.dec_tab();
      stream << "C[i*Cstride1 + j*ldc] = acc + beta*C[i + j*ldc];" << std::endl;
      stream.dec_tab();
      stream << "}" << std::endl;
      stream.dec_tab();
      stream << "}" << std::endl;

      stream.dec_tab();
      stream << "}" << std::endl;
    }


//    if(p_.simd_width>1)
//        std::cout << stream.str() << std::endl;
    return stream.str();

#undef VLOAD
#undef VST0RE
  }

  void gemm::enqueue_block(driver::CommandQueue & /*queue*/, int_t M, int_t N, int_t K,
                     array const & A, array const & B, array const & C,
                     value_scalar const & alpha, value_scalar const & beta,
                     driver::Program & program, const char * suffix, execution_options_type const & options)
  {
    using tools::align;

      if(M==0 || N==0 || K==0)
        return;

    char gemm_name[32] = {"gemm"};
    char reduce_name[32] = {"reduce"};
    strcat(gemm_name, suffix);
    strcat(reduce_name, suffix);

    bind_all_unique binder;

    array const * out = &C;
    std::unique_ptr<array> tmp;
    if(p_.depth > 1){
      tmp.reset(new array(M, N, p_.depth, C.dtype(), C.context()));
      out = tmp.get();
    }

//    std::cout << C << std::endl;
    driver::Kernel gemm(program, gemm_name);
    driver::NDRange local(p_.local_size_0, p_.local_size_1);

    driver::NDRange global(align(align(M,p_.mS)/p_.mS, p_.local_size_0), align(align(N,p_.nS)/p_.nS, p_.local_size_1), p_.depth);

    unsigned int current_arg = 0;
    set_arguments_functor helper(binder, current_arg, gemm);
    gemm.setSizeArg(current_arg++, M);
    gemm.setSizeArg(current_arg++, N);
    gemm.setSizeArg(current_arg++, K);
    gemm.setArg(current_arg++, out->data());
    gemm.setSizeArg(current_arg++, out->ld()*out->stride()[1]);
    gemm.setSizeArg(current_arg++, out->start()[0] + out->start()[1]*out->ld());
    gemm.setSizeArg(current_arg++, out->stride()[0]);

    helper.set_arguments(alpha.dtype(), alpha.values());
    gemm.setArg(current_arg++, A.data());
    gemm.setSizeArg(current_arg++, A.ld()*A.stride()[1]);
    gemm.setSizeArg(current_arg++, (A.start()[0] + A.start()[1]*A.ld()));
    gemm.setSizeArg(current_arg++, A.stride()[0]);

    gemm.setArg(current_arg++, B.data());
    gemm.setSizeArg(current_arg++, B.ld()*B.stride()[1]);
    gemm.setSizeArg(current_arg++, B.start()[0] + B.start()[1]*B.ld());
    gemm.setSizeArg(current_arg++, B.stride()[0]);

    helper.set_arguments(beta.dtype(), beta.values());
    options.enqueue(program.context(), gemm, global, local);

    if(p_.depth > 1)
    {
      unsigned int current_arg = 0;
      driver::Kernel reduce(program, reduce_name);
      driver::NDRange local(p_.local_size_0, p_.local_size_1);
      driver::NDRange global(align(M, p_.local_size_0), align(N, p_.local_size_1));
      set_arguments_functor helper(binder, current_arg, reduce);
      reduce.setSizeArg(current_arg++, M);
      reduce.setSizeArg(current_arg++, N);
      reduce.setSizeArg(current_arg++, p_.depth);
      reduce.setArg(current_arg++, out->data());
      reduce.setSizeArg(current_arg++, out->ld());
      reduce.setArg(current_arg++, C.data());
      reduce.setSizeArg(current_arg++, C.ld());
      reduce.setSizeArg(current_arg++, C.start()[0]);
      reduce.setSizeArg(current_arg++, C.start()[1]);
      reduce.setSizeArg(current_arg++, C.stride()[0]);
      reduce.setSizeArg(current_arg++, C.stride()[1]);
      helper.set_arguments(beta.dtype(), beta.values());
      options.enqueue(program.context(), reduce, global, local);
    }

  }

  array gemm::create_slice(array & M, int_t s0_0, int_t s0_1, int_t s1_0, int_t s1_1, bool swap)
  {
    slice s0(s0_0, s0_1);
    slice s1(s1_0, s1_1);
    if (swap)
      std::swap(s0, s1);
    return array(M, s0, s1);
  }

  std::vector<int_t> gemm::infos(expressions_tuple const & expressions, symbolic::preset::gemm::args& arguments) const
  {
    isaac::array_expression & array_expression = (*expressions.data().front());
    array_expression::container_type & array = array_expression.tree();
    std::size_t root = array_expression.root();
    arguments = symbolic::preset::gemm::check(array, root);
    int_t M = arguments.C->array->shape()[0];
    int_t N = arguments.C->array->shape()[1];
    int_t K = (A_trans_=='T')?arguments.A->array->shape()[0]:arguments.A->array->shape()[1];
    return {M, N, K};
  }

  gemm::gemm(gemm_parameters const & parameters, bool check_bounds, char A_trans, char B_trans) : base_impl<gemm, gemm_parameters>(parameters, BIND_ALL_UNIQUE), A_trans_(A_trans), B_trans_(B_trans), check_bounds_(check_bounds)
  {
    if(A_trans_=='N' && B_trans_=='N') type_ = GEMM_NN_TYPE;
    else if(A_trans_=='T' && B_trans_=='N') type_ = GEMM_TN_TYPE;
    else if(A_trans_=='N' && B_trans_=='T') type_ = GEMM_NT_TYPE;
    else if(A_trans_=='T' && B_trans_=='T') type_ = GEMM_TT_TYPE;
    else throw;
  }

  std::vector<int_t> gemm::input_sizes(expressions_tuple const & expressions) const
  {
    symbolic::preset::gemm::args dummy;
    return infos(expressions, dummy);
  }

  void gemm::enqueue(driver::CommandQueue & queue, driver::Program & program, const char * suffix, base & fallback_base, controller<expressions_tuple> const & ctr)
  {
    using namespace tools;
//    std::cout << p_.simd_width << " " << p_.mL << " " << p_.kL << " " << p_.mS << " " << p_.depth << " " << p_.local_size_0 << std::endl;

    gemm & fallback = (gemm&)fallback_base;
    expressions_tuple const & expressions = ctr.x();


    symbolic::preset::gemm::args args;
    std::vector<int_t> MNK = infos(expressions, args);

    int_t M = MNK[0];
    int_t N = MNK[1];
    int_t K = MNK[2];

    //Skip if empty
    if(M==0 || N == 0 || K ==0)
      return;
    //Extract
    array * pA = args.A->array;
    array * pB = args.B->array;
    array * pC = args.C->array;

    //Check if requires fallback
    int_t ldstrideA = pA->stride()[0];
    int_t ldstrideB = pB->stride()[0];
    int_t ldstrideC = pC->stride()[0];

    numeric_type dtype = args.C->dtype;

    //Enqueue
    value_scalar beta(0, dtype);
    if(args.beta) beta = value_scalar(args.beta->vscalar, dtype);

    value_scalar alpha(1, dtype);
    if(args.alpha) alpha = value_scalar(args.alpha->vscalar, dtype);


    execution_options_type const & options = ctr.execution_options();

    if (ldstrideA> 1 || ldstrideB > 1 || ldstrideC > 1)
    {
      fallback.enqueue_block(queue, M, N, K, *pA, *pB, *pC, alpha, beta, program, "fallback", options);
    }
    else
    {
        enqueue_block(queue,  M, N, K, *pA, *pB, *pC, alpha, beta, program, suffix, options);
    }
  }

  //
  gemm_nn::gemm_nn(unsigned int simd
                           , int_t ls0, int_t KL, int_t ls1, int_t D
                           , int_t ms, int_t ks, int_t ns
                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                           , int_t lfetch0, int_t lfetch1, bool check_bound) :
    gemm(gemm_parameters(simd, ls0, KL, ls1, D, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1), check_bound, 'N', 'N')
  { }

  //
  gemm_tn::gemm_tn(unsigned int simd
                           , int_t ls0, int_t KL, int_t ls1, int_t D
                           , int_t ms, int_t ks, int_t ns
                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                           , int_t lfetch0, int_t lfetch1, bool check_bound) :
    gemm(gemm_parameters(simd, ls0, KL, ls1, D, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1), check_bound, 'T', 'N')
  { }

  //
  gemm_nt::gemm_nt(unsigned int simd
                           , int_t ls0, int_t KL, int_t ls1, int_t D
                           , int_t ms, int_t ks, int_t ns
                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                           , int_t lfetch0, int_t lfetch1, bool check_bound) :
    gemm(gemm_parameters(simd, ls0, KL, ls1, D, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1), check_bound, 'N', 'T')
  { }

  //
  gemm_tt::gemm_tt(unsigned int simd
                           , int_t ls0, int_t KL, int_t ls1, int_t D
                           , int_t ms, int_t ks, int_t ns
                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                           , int_t lfetch0, int_t lfetch1, bool check_bound) :
    gemm(gemm_parameters(simd, ls0, KL, ls1, D, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1), check_bound, 'T', 'T')
  { }

}
}

