#include "isaac/array.h"
#include "isaac/backend/templates/mproduct.h"
#include "isaac/backend/keywords.h"
#include "isaac/tools/make_vector.hpp"
#include "isaac/tools/to_string.hpp"
#include "isaac/tools/miscellaneous.hpp"
#include "isaac/model/model.h"
namespace isaac
{

mproduct_parameters::mproduct_parameters(unsigned int simd_width
                          , int_t local_size_0, int_t KL, int_t local_size_1, int_t D
                          , int_t ms, int_t ks, int_t ns
                          , fetching_policy_type A_fetching_policy, fetching_policy_type B_fetching_policy
                          , int_t local_fetch_0, int_t local_fetch_1): base::parameters_type(simd_width, local_size_0, local_size_1, 1),
  kL(KL), depth(D), mS(ms), kS(ks), nS(ns), A_fetching_policy(A_fetching_policy), B_fetching_policy(B_fetching_policy),
  local_fetch_0(local_fetch_0), local_fetch_1(local_fetch_1),
  mL(ms*local_size_0), nL(ns*local_size_1){}


  unsigned int mproduct::lmem_usage(expressions_tuple const & expressions) const
  {
    isaac::array_expression const & array_expression = (*expressions.data().front());
    numeric_type numeric_t = lhs_most(array_expression.tree(), array_expression.root()).lhs.dtype;

    unsigned int N = 0;
    if (p_.A_fetching_policy==FETCH_FROM_LOCAL)
      N += p_.kL * p_.mL;
    if (p_.B_fetching_policy==FETCH_FROM_LOCAL)
      N += p_.nL * p_.kL;
    return N*size_of(numeric_t);
  }

  unsigned int mproduct::registers_usage(expressions_tuple const & expressions) const
  {
    isaac::array_expression const & array_expression = (*expressions.data().front());
    numeric_type numeric_t = lhs_most(array_expression.tree(), array_expression.root()).lhs.dtype;

    unsigned int N = p_.mS * p_.nS + p_.mS * p_.kS + p_.kS * p_.nS;
    return N*size_of(numeric_t);
  }

  int mproduct::is_invalid_impl(driver::Device const &, expressions_tuple const &) const
  {
    if (p_.A_fetching_policy!=FETCH_FROM_LOCAL && p_.B_fetching_policy!=FETCH_FROM_LOCAL&& (p_.local_fetch_0!=0 || p_.local_fetch_1!=0))
      return TEMPLATE_GLOBAL_MEMORY_REQUIRES_ZERO_LOCAL_FETCH;

    if ((p_.mS % p_.simd_width) > 0 || (p_.nS % p_.simd_width) > 0)
      return TEMPLATE_MS_NS_MUST_BE_SIMD_WIDTH_MULTIPLE;

    if(p_.mL > 256 || p_.nL > 256)
       return 1;

    if ( p_.kS % p_.kL == 0)
      return TEMPLATE_KS_MUST_BE_SMALLER_THAN_KL;

    if (!(A_trans_=='N' && B_trans_=='T') && p_.simd_width>1)
      return TEMPLATE_SIMD_WIDTH_MUST_BE_ONE;

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

  std::string mproduct::generate_impl(const char * suffix, expressions_tuple const & expressions, driver::Device const & device, std::vector<mapping_type> const &) const
  {
    using std::string;
    using tools::to_string;

    driver::backend_type backend = device.backend();
    bool is_a_fallback = check_bounds_ && p_.kS==1 && p_.depth==1;

#define HANDLE_BOUNDS(in_bounds, to_load) (!check_bounds_?string(to_load):string( string(in_bounds) + "?" + string(to_load) + ":0"))
#define VLOAD(offset, ptr) vload(p_.simd_width, sdtype, offset, ptr, backend)
#define VSTORE(value, offset, ptr) vstore(p_.simd_width, sdtype, value, offset, ptr, backend)
#define ASTRIDE1 string(is_a_fallback?"*Astride1":"")
#define BSTRIDE1 string(is_a_fallback?"*Bstride1":"")
#define CSTRIDE1 string(is_a_fallback?"*Cstride1":"")



    //////////////////
    /// INIT
    /// //////////////
    kernel_generation_stream stream;
    array_expression const & st = (*expressions.data().front());
    numeric_type dtype = lhs_most(st.tree(), st.root()).lhs.dtype;
    std::string sdtype = numeric_type_to_string(dtype);
    std::string vdtype = append_width(sdtype, p_.simd_width);
    std::string _size_t = size_type(device);

    size_t lAld = p_.mL;
    size_t lBld = p_.nL;

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
                               << Global(backend) << " " << sdtype << "* C, "  << _size_t << " Cld," << _size_t << " Coff," << _size_t << " Cstride1, "
                               << sdtype << " alpha,"
                               << Global(backend) << " " << vdtype << "* A, "  << _size_t << " Ald," << _size_t << " Aoff," << _size_t << " Astride1,"
                               << Global(backend) << " " << vdtype << "* B, "  << _size_t << " Bld," << _size_t << " Boff," << _size_t << " Bstride1,"
                               << sdtype << " beta)"
                               << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();

    stream << "A += Aoff;" << std::endl;
    stream << "B += Boff;" << std::endl;
    stream << "C += Coff;" << std::endl;


    stream << sdtype << " rC[" << p_.mS << "][" << p_.nS << "] = {{(" << sdtype << ")0}};" << std::endl;
    stream << vdtype << " rA[" << p_.kS << "][" << p_.mS/p_.simd_width << "];" << std::endl;
    stream << vdtype << " rB[" << p_.kS << "][" << p_.nS/p_.simd_width << "];" << std::endl;

    ///Result Values
    if (p_.A_fetching_policy==FETCH_FROM_LOCAL)
    {
      stream << Local(backend) << " " << sdtype << " lA[" << p_.kL*lAld << "];" << std::endl;
      stream << "size_t lAstart = 0;" << std::endl;
    }
    if (p_.B_fetching_policy==FETCH_FROM_LOCAL)
    {
      stream << Local(backend) << " " << sdtype << " lB[" << p_.kL*lBld << "];" << std::endl;
      stream << "size_t lBstart = 0;" << std::endl;
    }
    stream << std::endl;

    stream << "size_t gidx = " << GroupIdx0(backend) << ";" << std::endl;
    stream << "size_t gidy = " << GroupIdx1(backend) << ";" << std::endl;
    stream << "size_t idx =  " << LocalIdx0(backend) << ";" << std::endl;
    stream << "size_t idy =  " << LocalIdx1(backend) << ";" << std::endl;

    if(p_.depth > 1){
      stream << "size_t gidz = " << GroupIdx2(backend) << ";" << std::endl;
      stream << "size_t chunk_size = K/" << p_.depth << ";" << std::endl;
      stream << "size_t offz = chunk_size*gidz;" << std::endl;
    }
    else{
      stream << "size_t gidz = 0;" << std::endl;
      stream << "size_t chunk_size = K;" << std::endl;
      stream << "size_t offz = 0;" << std::endl;
    }

    if (p_.A_fetching_policy==FETCH_FROM_LOCAL || p_.B_fetching_policy==FETCH_FROM_LOCAL)
    {
      stream << std::endl;
      stream << "size_t idt = " << p_.local_size_0 << "*idy + idx;" << std::endl;
      stream << "size_t idxT = idt % " << p_.local_fetch_0 << ";" << std::endl;
      stream << "size_t idyT = idt / " << p_.local_fetch_0 << ";" << std::endl;
    }
    stream << std::endl;

    if (check_bounds_)
    {
      //Bounds checking for M (in A, C)
      stream << "bool in_bounds_m[" << p_.mS << "] = {" ;
      for(unsigned int m = 0; m < p_.mS ; m++)
      {
        if(m > 0) stream << ",";
        switch(p_.A_fetching_policy)
        {
          case FETCH_FROM_GLOBAL_CONTIGUOUS:
            stream << "gidx*" << p_.mL << " + idx*" << p_.mS << " + " << m  << "< M";
            break;
          default:
            stream << "gidx*" << p_.mL << " + idx + " << m * p_.local_size_0 << " < M";
            break;
        }
      }
      stream << "};" << std::endl;

      //Bounds checking for N (in B, C)
      stream << "bool in_bounds_n[" << p_.nS << "] = {";
      for(unsigned int n = 0; n < p_.nS ; n++)
      {
        if(n > 0) stream << ",";
        switch (p_.B_fetching_policy)
        {
        case FETCH_FROM_GLOBAL_CONTIGUOUS:
          stream << "gidy*" << p_.nL << " + idy*" << p_.nS << " + " << n << " < N";
          break;
        default:
          stream << "gidy*" << p_.nL << " + idy + " << n * p_.local_size_1 << " < N";
          break;
        }
      }
      stream << "};" << std::endl;



      //Bounds checking for A if Local
      if (p_.A_fetching_policy==FETCH_FROM_LOCAL)
      {
        unsigned int fetch_size = (A_trans_=='N'?p_.local_fetch_0*p_.simd_width:p_.local_fetch_1);
        stream << "bool in_bounds_A[" << p_.mL/fetch_size << "];" << std::endl;
        stream << "for(unsigned int m = 0; m < " << p_.mL/fetch_size << "; m++)" << std::endl;
        stream.inc_tab();
        stream << "in_bounds_A[m] = (gidx*" << p_.mL << " + " << (A_trans_=='N'?"idxT":"idyT") << " + m*" << fetch_size << ") < M;" << std::endl;
        stream.dec_tab();
      }

      //Bounds checking for B if Local
      if (p_.B_fetching_policy==FETCH_FROM_LOCAL)
      {
        unsigned int fetch_size = (B_trans_=='T'?p_.local_fetch_0*p_.simd_width:p_.local_fetch_1);
        stream << "bool in_bounds_B[" << p_.nL/fetch_size << "];" << std::endl;
        stream << "for(unsigned int n = 0; n < " <<  p_.nL/fetch_size << "; n++)" << std::endl;
        stream.inc_tab();
        stream << "in_bounds_B[n] = (gidy*" << p_.nL << " + " << (B_trans_=='T'?"idxT":"idyT") << " + n*" << fetch_size << ") < N;" << std::endl;
        stream.dec_tab();

//        for(unsigned int n = 0 ; n < p_.nL/fetch_size ; n++)
//            stream << n>0?",":"" << "(gidy*" << p_.nL << " + " << (B_trans_=='T'?"idxT":"idyT") << " + " << n*fetch_size << ") < N";
      }
    }

    switch (p_.A_fetching_policy)
    {
    case FETCH_FROM_LOCAL:
      if (A_trans_=='N')
        stream << "A += (gidx*" << p_.mL/p_.simd_width << " + idxT) " << ASTRIDE1 << " + idyT*Ald + offz*Ald;" << std::endl;
      else
        stream << "A += idxT" << ASTRIDE1 << " + gidx*" << p_.mL/p_.simd_width << "*Ald + idyT*Ald + offz;" << std::endl;
      break;

    case FETCH_FROM_GLOBAL_CONTIGUOUS:
      if (A_trans_=='N')
        stream << "A += (gidx*" << p_.mL/p_.simd_width << "+ idx*" << p_.mS/p_.simd_width << "+ offz*Ald)" << ASTRIDE1 << ";" << std::endl;
      else
        stream << "A += (gidx*" << p_.mL/p_.simd_width << "+ idx*" << p_.mS/p_.simd_width << ")*Ald + offz;" << std::endl;
      break;

    case FETCH_FROM_GLOBAL_STRIDED:
      if (A_trans_=='N')
        stream << "A += (gidx*" << p_.mL/p_.simd_width << "+ idx + offz*Ald)" << ASTRIDE1 << ";" << std::endl;
      else
        stream << "A += (gidx*" << p_.mL/p_.simd_width << "+ idx)*Ald + offz" << std::endl;
      break;

    default: break;
    }

    switch (p_.B_fetching_policy)
    {
    case FETCH_FROM_LOCAL:
      if (B_trans_=='T')
        stream << "B += (gidy*" << p_.nL/p_.simd_width << " + idxT)" << BSTRIDE1 << " + idyT*Bld + offz*Bld;" << std::endl;
      else
        stream << "B += idxT" << BSTRIDE1 << " + gidy*" << p_.nL/p_.simd_width << "*Bld + idyT*Bld + offz;" << std::endl;
      break;

    case FETCH_FROM_GLOBAL_CONTIGUOUS:
      if (B_trans_=='T')
        stream << "B += (gidy*" << p_.nL/p_.simd_width << " + idy*" << p_.nS/p_.simd_width << " + offz*Bld)" << BSTRIDE1 << ";" << std::endl;
      else
        stream << "B += (gidy*" << p_.nL/p_.simd_width << " + idy*" << p_.nS/p_.simd_width << ")*Bld + offz;" << std::endl;
      break;

    case FETCH_FROM_GLOBAL_STRIDED:
      if (B_trans_=='T')
        stream << "B += (gidy*" << p_.nL/p_.simd_width << "+ idy + offz*Bld)" << BSTRIDE1 << ";" << std::endl;
      else
        stream << "B += (gidy*" << p_.nL/p_.simd_width << "+ idy)*Bld + offz;" << std::endl;
      break;

    default: break;
    }

    stream << std::endl;

    if (p_.A_fetching_policy==FETCH_FROM_LOCAL && A_trans_=='N')
      stream << LocalPtr(backend) << " " << sdtype << "* lAstore = lA + idyT*" << lAld << " + idxT*" << p_.simd_width << ";" << std::endl;
    else if (p_.A_fetching_policy==FETCH_FROM_LOCAL && A_trans_=='T')
      stream << LocalPtr(backend) << " " << sdtype << "* lAstore = lA + idxT*" << lAld << " + idyT;" << std::endl;
    if (p_.B_fetching_policy==FETCH_FROM_LOCAL && B_trans_=='T')
      stream << LocalPtr(backend) << " " << sdtype << "* lBstore = lB + idyT*" << lBld << " + idxT*" << p_.simd_width << ";" << std::endl;
    else if (p_.B_fetching_policy==FETCH_FROM_LOCAL && B_trans_=='N')
      stream << LocalPtr(backend) << " " << sdtype << "* lBstore = lB + idxT*" << lBld << " + idyT;" << std::endl;


    stream << "//Outer loop" << std::endl;
    stream << "for(size_t block_k=0; block_k < chunk_size ; block_k+=" << p_.kL << "){" << std::endl;
    stream.inc_tab();

    if (p_.A_fetching_policy==FETCH_FROM_LOCAL || p_.B_fetching_policy==FETCH_FROM_LOCAL)
    {
      stream << LocalBarrier(backend) << ";" << std::endl;

        stream << "//Fetch A to local memory" << std::endl;
        if (p_.A_fetching_policy==FETCH_FROM_LOCAL && A_trans_=='N')
          for(int_t k = 0; k < p_.kL; k += p_.local_fetch_1)
            for(int_t m = 0; m < p_.mL; m += p_.local_fetch_0*p_.simd_width)
            {
              string in_bounds = "in_bounds_A[" + to_string(m/(p_.local_fetch_0*p_.simd_width)) + "] && (idyT + block_k < K)";
              string to_load = "A[" + to_string(k) + "*Ald + " + to_string(m/p_.simd_width) + ASTRIDE1  + "]";
              stream << VSTORE(HANDLE_BOUNDS(in_bounds, to_load), "0", "lAstore + lAstart + " + to_string(k*lAld+m)) << ";" << std::endl;
            }
        else if (p_.A_fetching_policy==FETCH_FROM_LOCAL && A_trans_=='T')
          for(int_t k = 0; k < p_.mL; k += p_.local_fetch_1)
            for(int_t m = 0; m < p_.kL; m += p_.local_fetch_0*p_.simd_width)
            {
              string in_bounds = "in_bounds_A[" + to_string(k/p_.local_fetch_1) + "] && (idxT + block_k < K)";
              string to_load = "A[" + to_string(k) + "*Ald + " + to_string(m/p_.simd_width) + ASTRIDE1 + "]";
              stream << VSTORE(HANDLE_BOUNDS(in_bounds, to_load), "0", "lAstore + lAstart + " + to_string(m*lAld+k)) << ";" << std::endl;
            }

        stream << "//Fetch B to local memory" << std::endl;
        if (p_.B_fetching_policy==FETCH_FROM_LOCAL && B_trans_=='T')
          for(int_t k = 0; k < p_.kL; k += p_.local_fetch_1)
            for(int_t n = 0; n < p_.nL; n += p_.local_fetch_0*p_.simd_width)
            {
              string in_bounds = "in_bounds_B[" + to_string(n/(p_.local_fetch_0*p_.simd_width)) + "] && (idyT + block_k < K)";
              string to_load = "B[" + to_string(k) + "*Bld + " + to_string(n/p_.simd_width) + BSTRIDE1 + "]";
              stream << VSTORE(HANDLE_BOUNDS(in_bounds, to_load), "0", "lBstore + lBstart + " + to_string(k*lBld+n)) << ";" << std::endl;
            }
        else if (p_.B_fetching_policy==FETCH_FROM_LOCAL && B_trans_=='N')
          for(int_t k = 0; k < p_.nL; k += p_.local_fetch_1)
            for(int_t n = 0; n < p_.kL; n += p_.local_fetch_0*p_.simd_width)
            {
              string in_bounds = "in_bounds_B[" + to_string(k/p_.local_fetch_1) + "]  && (idxT + block_k < K)";
              string to_load = "B[" + to_string(k) + "*Bld + " + to_string(n/p_.simd_width) + BSTRIDE1 + "]";
              stream << VSTORE(HANDLE_BOUNDS(in_bounds, to_load), "0", "lBstore + lBstart + " + to_string(n*lBld+k)) << ";" << std::endl;
            }
        stream << LocalBarrier(backend) << ";" << std::endl;
      }
      if (p_.A_fetching_policy==FETCH_FROM_LOCAL)
        stream << LocalPtr(backend) << " " << sdtype << "* readA = lA + idx*" << p_.simd_width << ";" << std::endl;
      if(p_.B_fetching_policy==FETCH_FROM_LOCAL)
        stream << LocalPtr(backend) << " " << sdtype << "* readB = lB + idy*" << p_.simd_width << ";" << std::endl;


    stream << "//Inner loop" << std::endl;
    if (check_bounds_)
      stream << "for(unsigned int k = 0; k < " << p_.kL << " && (block_k + k < chunk_size); k+=" << p_.kS << "){" << std::endl;
    else
      stream << "for(unsigned int k = 0; k < " << p_.kL << "; k+=" << p_.kS << "){" << std::endl;
    stream.inc_tab();

    stream << "//Fetch A to registers" << std::endl;
    stream << "#pragma unroll" << std::endl;
    stream << "for(unsigned int kk = 0; kk < " << p_.kS << "; kk++)" << std::endl;
    stream << "#pragma unroll " << p_.mS/p_.simd_width << std::endl;
    stream << "for(unsigned int mm = 0; mm < " << p_.mS/p_.simd_width << "; mm++)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    if(p_.A_fetching_policy==FETCH_FROM_LOCAL)
      stream << "rA[kk][mm] = "  << VLOAD("0", "readA + k*" + to_string(lAld) + " + mm*" + to_string(p_.local_size_0*p_.simd_width) + "+ kk*" + to_string(lAld)) << ";" << std::endl;
    else if(p_.A_fetching_policy==FETCH_FROM_GLOBAL_CONTIGUOUS)
    {
      if (A_trans_=='N')
        stream << "rA[kk][mm] = " << HANDLE_BOUNDS("in_bounds_m[mm]", "A[kk*Ald + mm" + ASTRIDE1 + "]") << ";" << std::endl;
      else
        stream << "rA[kk][mm] = " << HANDLE_BOUNDS("in_bounds_m[mm]", "A[mm*Ald + kk" + ASTRIDE1 + "]") << ";" << std::endl;
    }
    else if(p_.A_fetching_policy==FETCH_FROM_GLOBAL_STRIDED)
    {
      if (A_trans_=='N')
        stream << "rA[kk][mm] = " << HANDLE_BOUNDS("in_bounds_m[mm]", "A[kk*Ald + mm*" + to_string(p_.local_size_0) + ASTRIDE1 + "]") << ";" << std::endl;
      else
        stream << "rA[kk][mm] = " << HANDLE_BOUNDS("in_bounds_m[mm]", "A[mm*Ald*" + to_string(p_.local_size_0) + " + kk" + ASTRIDE1 + "]") << ";" << std::endl;
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
    if(p_.B_fetching_policy==FETCH_FROM_LOCAL)
      stream << "rB[kk][nn] = " << VLOAD("0", "readB + k*" + to_string(lBld) + " + nn*" + to_string(p_.local_size_1*p_.simd_width)  + "+ kk*" + to_string(lBld)) << ";" << std::endl;
    else if(p_.B_fetching_policy==FETCH_FROM_GLOBAL_CONTIGUOUS)
    {
      if (B_trans_=='T')
        stream << "rB[kk][nn] = " << HANDLE_BOUNDS("in_bounds_n[nn]", "B[kk*Bld + nn" + BSTRIDE1 + "]") << ";" << std::endl;
      else
        stream << "rB[kk][nn] = " << HANDLE_BOUNDS("in_bounds_n[nn]", "B[nn*Bld + kk" + BSTRIDE1 + "]") << ";" << std::endl;
    }
    else if(p_.B_fetching_policy==FETCH_FROM_GLOBAL_STRIDED)
    {
      if (B_trans_=='T')
        stream << "rB[kk][nn] = " << HANDLE_BOUNDS("in_bounds_n[nn]", "B[kk*Bld + nn*" + to_string(p_.local_size_1) + BSTRIDE1 + "]") << ";" << std::endl;
      else
        stream << "rB[kk][nn] = " << HANDLE_BOUNDS("in_bounds_n[nn]", "B[nn*Bld*" + to_string(p_.local_size_1) + " + kk" + BSTRIDE1 + "]") << ";" << std::endl;
    }
    stream.dec_tab();
    stream << "}" << std::endl;


   stream << "//Increment A pointers" << std::endl;
   if(p_.A_fetching_policy!=FETCH_FROM_LOCAL)
   {
     if (A_trans_=='N')
       stream << "A += " << p_.kS << "*Ald;" << std::endl;
     else
       stream << "A += " << p_.kS << ASTRIDE1 << ";" << std::endl;
   }

   stream << "//Increment B pointers" << std::endl;
   if(p_.B_fetching_policy!=FETCH_FROM_LOCAL)
   {
     if (B_trans_=='T')
       stream << "B += " << p_.kS << "*Bld;" << std::endl;
     else
       stream << "B += " << p_.kS << BSTRIDE1 << ";" << std::endl;
   }

    stream << "//FMA computations" << std::endl;
    for(int_t kk=0 ; kk < p_.kS; ++kk)
    for(int_t nn=0; nn < p_.nS; ++nn)
    for(int_t mm=0; mm < p_.mS; ++mm)
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

    //Increment global pointer if local memory is used
    //Else, it's incremented directly when fetching
    if (p_.A_fetching_policy==FETCH_FROM_LOCAL)
    {
      if (A_trans_=='N')
        stream << "A += "  << p_.kL << "*Ald;" << std::endl;
      else
        stream << "A += "  << p_.kL << ASTRIDE1 << ";" << std::endl;
    }

    if (p_.B_fetching_policy==FETCH_FROM_LOCAL)
    {
      if (B_trans_=='T')
        stream << "B += " << p_.kL << "*Bld;" << std::endl;
      else
        stream << "B += " << p_.kL << BSTRIDE1 << ";" << std::endl;
    }

    stream.dec_tab();
    stream << "}" << std::endl;



    stream << "//Write back C" << std::endl;
    unsigned int ministartstride0 = p_.A_fetching_policy==FETCH_FROM_GLOBAL_CONTIGUOUS?p_.mS:p_.simd_width;
    unsigned int ministartstride1 = p_.B_fetching_policy==FETCH_FROM_GLOBAL_CONTIGUOUS?p_.nS:p_.simd_width;
    stream << "C += (gidx*" << p_.mL << " + idx*" << ministartstride0 << ")" << CSTRIDE1 << "  + (gidy*" << p_.nL << " + idy*" << ministartstride1 << ")*Cld + gidz*Cld*N;" << std::endl;
    for(int_t m=0; m < p_.mS; ++m)
    for(int_t n=0; n < p_.nS; ++n)
    {
      unsigned int ministride0 = p_.A_fetching_policy==FETCH_FROM_GLOBAL_CONTIGUOUS?1:p_.local_size_0;
      unsigned int ministride1 = p_.B_fetching_policy==FETCH_FROM_GLOBAL_CONTIGUOUS?1:p_.local_size_1;

      string Ci = to_string((m/p_.simd_width)*(ministride0*p_.simd_width) + m%p_.simd_width);
      string Cj = to_string((n/p_.simd_width)*(ministride1*p_.simd_width) + n%p_.simd_width);
      if (check_bounds_)
        stream << "if (in_bounds_m[" << m << "] && in_bounds_n[" << n << "]) " ;
      stream << "C[" << Ci << CSTRIDE1 << " + " << Cj << "*Cld] = rC[" << m << "][" << n << "]*alpha + ((beta==0)?0:beta*C[" << Ci << " + " << Cj << "*Cld]);" << std::endl;
    }
    stream.dec_tab();
    stream << "}" << std::endl;


    if(p_.depth > 1)
    {
      stream << KernelPrefix(backend) << " void " << reduce_name << "(" << _size_t << " M, " << _size_t << " N, " << _size_t << " D, "
                                 << Global(backend) << " " << sdtype << "* Z, "  << _size_t << " Zld,"
                                 << Global(backend) << " " << sdtype << "* C, "  << _size_t << " Cld," << _size_t << " Cstart1," << _size_t << " Cstart2," << _size_t << " Cstride1, "  << _size_t << " Cstride2, "
                                 << sdtype << " beta)"
                                 << std::endl;
      stream << "{" << std::endl;
      stream.inc_tab();

      stream << "C += Cstart1 + Cstart2*Cld;" << std::endl;
      stream << "Cld *= Cstride2;" << std::endl;
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
      stream << "C[i*Cstride1 + j*Cld] = acc + beta*C[i + j*Cld];" << std::endl;
      stream.dec_tab();
      stream << "}" << std::endl;
      stream.dec_tab();
      stream << "}" << std::endl;

      stream.dec_tab();
      stream << "}" << std::endl;
    }

//    std::cout << stream.str() << std::endl;
    return stream.str();

#undef HANDLE_BOUNDS
#undef VLOAD
#undef VST0RE
  }

  void mproduct::enqueue_block(driver::CommandQueue & queue, int_t M, int_t N, int_t K,
                     array const & A, array const & B, array const & C,
                     value_scalar const & alpha, value_scalar const & beta,
                     driver::Program & program, const char * suffix, execution_options_type const & options)
  {

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

    driver::Kernel gemm(program, gemm_name);
    driver::NDRange local(p_.local_size_0, p_.local_size_1);

    using tools::align;
    driver::NDRange global = (strcmp(suffix,"fallback")==0)?driver::NDRange(align(align(M,p_.mS)/p_.mS, p_.local_size_0), align(align(N,p_.nS)/p_.nS, p_.local_size_1), p_.depth):driver::NDRange(M/p_.mS, N/p_.nS, p_.depth);
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
    gemm.setSizeArg(current_arg++, A.ld()*A.stride()[1]/p_.simd_width);
    gemm.setSizeArg(current_arg++, A.start()[0] + A.start()[1]*A.ld()/p_.simd_width);
    gemm.setSizeArg(current_arg++, A.stride()[0]);

    gemm.setArg(current_arg++, B.data());
    gemm.setSizeArg(current_arg++, B.ld()*B.stride()[1]/p_.simd_width);
    gemm.setSizeArg(current_arg++, B.start()[0] + B.start()[1]*B.ld()/p_.simd_width);
    gemm.setSizeArg(current_arg++, B.stride()[0]);

    helper.set_arguments(beta.dtype(), beta.values());
    options.enqueue(program.context(), gemm, global, local);

    if(p_.depth > 1)
    {
      unsigned int current_arg = 0;
      driver::Kernel reduce(program, reduce_name);
      driver::NDRange local(p_.local_size_0, p_.local_size_1);
      driver::NDRange global = driver::NDRange(M, N);
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

  array mproduct::create_slice(array & M, int_t s0_0, int_t s0_1, int_t s1_0, int_t s1_1, bool swap)
  {
    slice s0(s0_0, s0_1);
    slice s1(s1_0, s1_1);
    if (swap)
      std::swap(s0, s1);
    return array(M, s0, s1);
  }

  std::vector<int_t> mproduct::infos(expressions_tuple const & expressions, lhs_rhs_element *& C, lhs_rhs_element *& A, lhs_rhs_element *& B, lhs_rhs_element *&, lhs_rhs_element *&)
  {
    isaac::array_expression & array_expression = (*expressions.data().front());
    array_expression::container_type & array = array_expression.tree();
    std::size_t root = array_expression.root();
    C = &array[root].lhs;
    A = &array[array[root].rhs.node_index].lhs;
    B = &array[array[root].rhs.node_index].rhs;
    int_t M = C->array->shape()[0];
    int_t N = C->array->shape()[1];
    int_t K = (A_trans_=='T')?A->array->shape()[0]:A->array->shape()[1];
    return {M, N, K};
  }

  mproduct::mproduct(mproduct_parameters const & parameters, bool check_bounds, char A_trans, char B_trans) : base_impl<mproduct, mproduct_parameters>(parameters, BIND_ALL_UNIQUE), A_trans_(A_trans), B_trans_(B_trans), check_bounds_(check_bounds)
  {
    if(A_trans_=='N' && B_trans_=='N') type_ = MATRIX_PRODUCT_NN_TYPE;
    else if(A_trans_=='T' && B_trans_=='N') type_ = MATRIX_PRODUCT_TN_TYPE;
    else if(A_trans_=='N' && B_trans_=='T') type_ = MATRIX_PRODUCT_NT_TYPE;
    else if(A_trans_=='T' && B_trans_=='T') type_ = MATRIX_PRODUCT_TT_TYPE;
    else throw;
  }

  std::vector<int_t> mproduct::input_sizes(expressions_tuple const & expressions)
  {
    lhs_rhs_element *d0, *d1, *d2, *d3, *d4;
    return infos(expressions, d0, d1, d2, d3, d4);
  }

  void mproduct::enqueue(driver::CommandQueue & queue, driver::Program & program, const char * suffix, base & fallback_base, controller<expressions_tuple> const & ctr)
  {
    using namespace tools;

    mproduct & fallback = (mproduct&)fallback_base;
    expressions_tuple const & expressions = ctr.x();


    lhs_rhs_element  * eC, *eA, *eB, *ealpha, *ebeta;
    std::vector<int_t> MNK = infos(expressions, eC, eA, eB, ealpha, ebeta);

    int_t M = MNK[0];
    int_t N = MNK[1];
    int_t K = MNK[2];

    //Skip if empty
    if(M==0 || N == 0 || K ==0)
      return;
    //Extract
    array * pA = eA->array;
    array * pB = eB->array;
    array * pC = eC->array;

    //Check if requires fallback
    int_t ldstrideA = pA->stride()[0];
    int_t ldstrideB = pB->stride()[0];
    int_t ldstrideC = pC->stride()[0];
    int_t ldstartA = pA->start()[0];
    int_t ldstartB = pB->start()[0];

    numeric_type dtype = eC->dtype;

    //Enqueue
    bool swap_A = (A_trans_=='T');
    bool swap_B = (B_trans_=='T');


    value_scalar beta(0, dtype);
    value_scalar alpha(1, dtype);
    value_scalar _1(1, dtype);


    execution_options_type const & options = ctr.execution_options();

    if (M < p_.mL || N < p_.nL || K < p_.kL || ldstrideA> 1 || ldstrideB > 1 || ldstrideC > 1
        || (p_.simd_width>1 && (ldstartA % p_.simd_width > 0 || ldstartB % p_.simd_width > 0 || pA->ld()%p_.simd_width > 0 || pB->ld()%p_.simd_width > 0)))
    {
      fallback.enqueue_block(queue, M, N, K, create_slice(*pA, 0, M, 0, K, swap_A), create_slice(*pB, 0, K, 0, N,  swap_B),
                             create_slice(*pC, 0, M, 0, N, false), alpha, beta, program, "fallback", options);
      return;
    }

    int_t lM = M / p_.mL * p_.mL;
    int_t lN = N / p_.nL * p_.nL;
    int_t lK = K / (p_.kL*p_.depth) * p_.kL*p_.depth;

    enqueue_block(queue,  lM, lN, lK, create_slice(*pA, 0, lM, 0, lK, swap_A), create_slice(*pB, 0, lK, 0, lN, swap_B), create_slice(*pC, 0, lM, 0, lN, false), alpha, beta, program, suffix, options);
    fallback.enqueue_block(queue,  lM, lN, K - lK, create_slice(*pA, 0, lM, lK, K, swap_A), create_slice(*pB, lK, K, 0, lN, swap_B), create_slice(*pC, 0, lM, 0, lN, false), alpha, _1, program, "fallback", options);

    fallback.enqueue_block(queue,  lM, N - lN, lK, create_slice(*pA, 0, lM, 0, lK, swap_A), create_slice(*pB, 0, lK, lN, N, swap_B), create_slice(*pC, 0, lM, lN, N, false), alpha, beta, program, "fallback", options);
    fallback.enqueue_block(queue,  lM, N - lN, K - lK, create_slice(*pA, 0, lM, lK, K, swap_A), create_slice(*pB, lK, K, lN, N, swap_B), create_slice(*pC, 0, lM, lN, N, false), alpha, _1, program, "fallback", options);

    fallback.enqueue_block(queue,  M - lM, lN, lK, create_slice(*pA, lM, M, 0, lK, swap_A), create_slice(*pB, 0, lK, 0, lN, swap_B), create_slice(*pC, lM, M, 0, lN, false), alpha, beta, program, "fallback", options);
    fallback.enqueue_block(queue,  M - lM, lN, K - lK, create_slice(*pA, lM, M, lK, K, swap_A), create_slice(*pB, lK, K, 0, lN, swap_B), create_slice(*pC, lM, M, 0, lN, false), alpha, _1, program, "fallback", options);

    fallback.enqueue_block(queue,  M - lM, N - lN, lK, create_slice(*pA, lM, M, 0, lK, swap_A), create_slice(*pB, 0, lK, lN, N, swap_B), create_slice(*pC, lM, M, lN, N, false), alpha, beta, program, "fallback", options);
    fallback.enqueue_block(queue,  M - lM, N - lN, K - lK, create_slice(*pA, lM, M, lK, K, swap_A), create_slice(*pB, lK, K, lN, N, swap_B), create_slice(*pC, lM, M, lN, N, false), alpha, _1, program, "fallback", options);
  }

  //
  mproduct_nn::mproduct_nn(unsigned int simd
                           , int_t ls0, int_t KL, int_t ls1, int_t D
                           , int_t ms, int_t ks, int_t ns
                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                           , int_t lfetch0, int_t lfetch1, bool check_bound) :
    mproduct(mproduct_parameters(simd, ls0, KL, ls1, D, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1), check_bound, 'N', 'N')
  { }

  //
  mproduct_tn::mproduct_tn(unsigned int simd
                           , int_t ls0, int_t KL, int_t ls1, int_t D
                           , int_t ms, int_t ks, int_t ns
                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                           , int_t lfetch0, int_t lfetch1, bool check_bound) :
    mproduct(mproduct_parameters(simd, ls0, KL, ls1, D, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1), check_bound, 'T', 'N')
  { }

  //
  mproduct_nt::mproduct_nt(unsigned int simd
                           , int_t ls0, int_t KL, int_t ls1, int_t D
                           , int_t ms, int_t ks, int_t ns
                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                           , int_t lfetch0, int_t lfetch1, bool check_bound) :
    mproduct(mproduct_parameters(simd, ls0, KL, ls1, D, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1), check_bound, 'N', 'T')
  { }

  //
  mproduct_tt::mproduct_tt(unsigned int simd
                           , int_t ls0, int_t KL, int_t ls1, int_t D
                           , int_t ms, int_t ks, int_t ns
                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                           , int_t lfetch0, int_t lfetch1, bool check_bound) :
    mproduct(mproduct_parameters(simd, ls0, KL, ls1, D, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1), check_bound, 'T', 'T')
  { }
}


