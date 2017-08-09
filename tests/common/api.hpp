#ifndef TEST_COMMON_HPP_
#define TEST_COMMON_HPP_

#include <cassert>
#include <vector>
#include <algorithm>
#include <functional>
#include "isaac/array.h"

namespace sc = isaac;

typedef isaac::int_t int_t;

template<typename T> struct numeric_trait;
template<> struct numeric_trait<float> { static constexpr float epsilon = 1e-4; static const char* name; };
template<> struct numeric_trait<double> { static constexpr double epsilon = 1e-8; static const char* name; };
const char * numeric_trait<float>::name = "float";
const char * numeric_trait<double>::name = "double";

template<class T> struct BLAS;
template<> struct BLAS<float> { template<class FT, class DT> static FT F(FT SAXPY, DT ) { return SAXPY; } };
template<> struct BLAS<double> { template<class FT, class DT> static DT F(FT , DT DAXPY) { return DAXPY; } };

enum interface_t
{
    clBLAS,
    cuBLAS,
    CPP
};

cl_mem cl(sc::array_base const & x) { return x.data().handle().cl(); }
CUdeviceptr cu(sc::array_base const & x) { return x.data().handle().cu(); }
int off(sc::array_base const & x) { return x.start(); }
int inc(sc::array_base const & x) { return x.stride()[0]; }
int ld(sc::array_base const & x) { return x.stride()[1]; }

template<class Iterator>
std::string join(Iterator begin, Iterator end, std::string delimiter){
  std::string result;
  while(begin!=end){
    result += *begin;
    if(++begin!=end) result += delimiter;
  }
  return result;
}

/*------ Simple Vector ---------*/
template<class T>
class simple_vector_base
{
public:
    typedef T value_type;

    simple_vector_base(int_t start, int_t end, int_t stride, std::vector<T> & data) : N_((end-start)/stride), start_(start),
                                                                                     stride_(stride), data_(data)
    { }
    int_t size() const { return N_; }
    int_t start() const { return start_; }
    int_t stride() const { return stride_; }
    std::vector<T> & data() { return data_; }
    T & operator[](int_t i) { return data_[start_ + i*stride_]; }
    T operator[](int_t i) const { return data_[start_ + i*stride_]; }
private:
    int_t N_;
    int_t start_;
    int_t stride_;
    std::vector<T> & data_;
};

template<class T>
class simple_vector : public simple_vector_base<T>
{
public:
    simple_vector(int_t N) :  simple_vector_base<T>(0, N, 1, data_), data_(N){}
private:
    std::vector<T> data_;
};

template<class T>
class simple_vector_s : public simple_vector_base<T>
{
public:
    simple_vector_s(simple_vector_base<T> & x, int_t start, int_t end, int_t stride) :
        simple_vector_base<T>(start, end, stride, x.data()){ }
};

/*------ Simple Matrix ---------*/
template<class T>
class simple_matrix_base
{
public:
    simple_matrix_base(int_t start1, int_t end1, int_t stride1,
                       int_t start2, int_t end2, int_t stride2,
                       int_t ld, std::vector<T> & data) : M_((end1 - start1)/stride1), start1_(start1), stride1_(stride1),
                                                                                         N_((end2-start2)/stride2), start2_(start2), stride2_(stride2),
                                                                                         ld_(ld), data_(data)
    {}

    int_t size1() const { return M_; }
    int_t start1() const { return start1_; }
    int_t stride1() const { return stride1_; }

    int_t size2() const { return N_; }
    int_t start2() const { return start2_; }
    int_t stride2() const { return stride2_; }

    std::vector<T> & data() { return data_; }

    int_t ld() const { return ld_; }

    T operator()(int_t i, int_t j) const
    {
      int_t ii = start1_ + i*stride1_;
      int_t jj = start2_ + j*stride2_;
      return data_[ii + jj*ld_];
    }

    T& operator()(int_t i, int_t j)
    {
      int_t ii = start1_ + i*stride1_;
      int_t jj = start2_ + j*stride2_;
      return data_[ii + jj*ld_];
    }
#ifdef _MSC_VER
//In Windows, as class simple_matrix's copy constructor needs these parameters,
//protected is used here to replace the private.
protected:
#else
private:
#endif
    int_t M_;
    int_t start1_;
    int_t stride1_;

    int_t N_;
    int_t start2_;
    int_t stride2_;

    int_t ld_;

    std::vector<T> & data_;
};


template<class T>
class simple_matrix : public simple_matrix_base<T>
{
public:
    simple_matrix(int_t M, int_t N) :  simple_matrix_base<T>(0, M, 1, 0, N, 1, M, data_), data_(M*N){}
    simple_matrix(int_t M, int_t N, std::vector<T> data) : simple_matrix_base<T>(0, M, 1, 0, N, 1, M, data_), data_(data){}
#ifdef _MSC_VER
//In Windows, the copy constructor has to be defined manually.
    simple_matrix(simple_matrix& c) : simple_matrix_base<T>(0, c.M_, 1, 0, c.N_, 1, c.M_, data_) { data_ = c.data(); }
#endif
private:
    std::vector<T> data_;
};

template<class T>
class simple_matrix_s : public simple_matrix_base<T>
{
public:
    simple_matrix_s(simple_matrix_base<T> & A,
                        int_t start1, int_t end1, int_t stride1,
                        int_t start2, int_t end2, int_t stride2) :
        simple_matrix_base<T>(start1, end1, stride1, start2, end2, stride2, A.ld(), A.data()){ }
};

template<class T>
std::ostream& operator<<(std::ostream& os, simple_matrix_base<T> const & x)
{
  for(size_t i = 0 ; i < (size_t)x.size1() ; i++){
    for(size_t j = 0; j < (size_t)x.size2() ; j++)
      os << ((j>0)?",":"") << x(i,j);
    os << std::endl;
  }
  return os;
}

/*------ Initializer ---------*/

template<class T>
std::vector<T> random(size_t N)
{
  std::vector<T> res(N);
  for (size_t i = 0; i < res.size(); ++i)
    res[i] = (T)rand()/RAND_MAX;
  return res;
}

template<typename T>
void init_rand(simple_vector_base<T> & x)
{
  for (int_t i = 0; i < x.size(); ++i)
    x[i] = (T)rand()/RAND_MAX;
}

template<typename T>
void init_rand(simple_matrix_base<T> & A)
{
  for (int_t i = 0; i < A.size1(); ++i)
    for(int_t j = 0 ; j < A.size2() ; ++j)
      A(i,j) = (T)rand()/RAND_MAX;
}

template<typename T>
simple_matrix<T> simple_trans(simple_matrix_base<T> const & A)
{
  simple_matrix<T> res(A.size2(), A.size1());
  for(int_t i = 0 ; i < A.size1(); ++i)
    for(int_t j = 0; j < A.size2(); ++j)
      res(j, i) = A(i, j);
  return res;
}

/*------ Compare -----------*/
template<typename T>
bool diff(T a, T b, T epsilon, typename std::enable_if<std::is_arithmetic<T>::value>::type* = 0)
{
  T delta = std::abs(a - b);
  if(std::max(a, b)!=0)
    delta/=std::abs(std::max(a, b));
  return delta > epsilon;
}

template<class VecType1, class VecType2>
bool diff(VecType1 const & x, VecType2 const & y, typename VecType1::value_type epsilon)
{
  assert((size_t)x.size()==(size_t)y.size());
  typedef typename VecType1::value_type T;
  T max = 0;
  for(int_t i = 0 ; i < (int_t)x.size() ; ++i)
  {
    if(diff(x[i], y[i], epsilon)){
      return true;
    }
  }
  return false;
}

template<class VecType>
bool diff(VecType const & x, isaac::array const & iy, typename VecType::value_type epsilon)
{
  VecType y(x.size());
  isaac::copy(iy, y);
  return diff(x, y, epsilon);
}

template<class VecType>
bool diff(isaac::array const & x, VecType const & y, typename VecType::value_type epsilon)
{ return diff(y, x, epsilon); }

#define INIT_VECTOR(N, SUBN, START, STRIDE, CPREFIX, PREFIX, CTX) \
    simple_vector<T> CPREFIX(N);\
    simple_vector_s<T> CPREFIX ## _s(CPREFIX, START, START + STRIDE*SUBN, STRIDE);\
    init_rand(CPREFIX);\
    isaac::array PREFIX(CPREFIX.data(), CTX);\
    isaac::view PREFIX ## _s = PREFIX[{START, START + STRIDE*SUBN, STRIDE}];

#define INIT_MATRIX(M, SUBM, START1, STRIDE1, N, SUBN, START2, STRIDE2, CPREFIX, PREFIX, CTX) \
    simple_matrix<T> CPREFIX(M, N);\
    simple_matrix_s<T> CPREFIX ## _s(CPREFIX, START1, START1 + STRIDE1*SUBM, STRIDE1,\
                                                                 START2, START2 + STRIDE2*SUBN, STRIDE2);\
    init_rand(CPREFIX);\
    isaac::array PREFIX(M, N, CPREFIX.data(), CTX);\
    isaac::view PREFIX ## _s(PREFIX({START1, START1 + STRIDE1*SUBM, STRIDE1},\
                                                   {START2, START2 + STRIDE2*SUBN, STRIDE2}));\
    simple_matrix<T> CPREFIX ## T = simple_trans(CPREFIX);\
    isaac::array PREFIX ## T(N, M, CPREFIX ## T.data(), CTX);\
    isaac::view PREFIX ## T_s(PREFIX ## T( {START2, START2 + STRIDE2*SUBN, STRIDE2},\
                                                    {START1, START1 + STRIDE1*SUBM, STRIDE1}));\

template<typename test_fun_t>
int run_test(test_fun_t const & testf, test_fun_t const & testd)
{
    int nfail = 0;
    int npass = 0;
    std::list<isaac::driver::Context const *> data;
    sc::driver::backend::contexts::get(data);
    for(isaac::driver::Context const * context : data)
    {
      sc::driver::Device device = sc::driver::backend::queues::get(*context,0).device();
      if(device.type() != sc::driver::Device::Type::GPU)
          continue;
      std::cout << "Device: " << device.name() << " on " << device.platform().name() << " " << device.platform().version() << std::endl;
      std::cout << "---" << std::endl;
      testf(*context, nfail, npass);
      if(device.fp64_support())
        testd(*context, nfail, npass);
      std::cout << "---" << std::endl;
    }
    if(nfail>0)
      return EXIT_FAILURE;
    return EXIT_SUCCESS;
}

#define ADD_TEST_1D_EW(NAME, CPU_LOOP, GPU_EXPR) \
  {\
    simple_vector<T> buffer(N);\
    std::cout << NAME << "..." << std::flush;\
    for(int_t i = 0 ; i < N ; ++i)\
      CPU_LOOP;\
    GPU_EXPR;\
    isaac::copy(y, buffer.data());\
    if(diff(cy, buffer, numeric_trait<T>::epsilon)){\
      nfail++;\
      std::cout << " [FAIL] " << std::endl;\
    }\
    else{\
      npass++;\
      std::cout << std::endl;\
    }\
  }

#define ADD_TEST_1D_RD(NAME, CPU_REDUCTION, INIT, ASSIGNMENT, GPU_REDUCTION) \
  {\
    T tmp = 0;\
    std::cout << NAME << "..." << std::flush;\
    cs = INIT;\
    for(int_t i = 0 ; i < N ; ++i)\
      CPU_REDUCTION;\
    cs= ASSIGNMENT ;\
    GPU_REDUCTION;\
    tmp = ds;\
    if(std::isnan((T)tmp) || (std::abs(cs - tmp)/std::max(cs, tmp)) > numeric_trait<T>::epsilon){\
      nfail++;\
      std::cout << " [FAIL] " << std::endl;\
    }\
    else{\
      npass++;\
      std::cout << std::endl;\
    }\
  }

  #define ADD_TEST_2D_EW(NAME, CPU_LOOP, GPU_EXPR) \
    {\
      std::cout << NAME << "..." << std::flush;\
      for(int_t i = 0 ; i < M ; ++i)\
        for(int_t j = 0 ; j < N ; ++j)\
            CPU_LOOP;\
      GPU_EXPR;\
      isaac::copy(C, buffer.data());\
      std::vector<T> cCbuffer(M*N);\
      for(int i = 0 ; i < M ; ++i)\
        for(int j = 0 ; j < N ; ++j)\
          cCbuffer[i + j*M] = cC(i,j);\
      if(diff(cCbuffer, buffer, numeric_trait<T>::epsilon)) {\
        nfail++;\
        std::cout << " [Failure!]" << std::endl;\
      }\
      else{\
        npass++;\
        std::cout << std::endl;\
      }\
    }

  #define ADD_TEST_2D_RD(NAME, SIZE1, SIZE2, NEUTRAL, REDUCTION, ASSIGNMENT, GPU_REDUCTION, RES, BUF, CRES)\
    {\
      std::cout << NAME << "..." << std::flush;\
      for(int i = 0 ; i < SIZE1 ; ++i)\
      {\
        yi = NEUTRAL;\
        xi = NEUTRAL;\
        for(int j = 0 ; j < SIZE2 ; ++j)\
          REDUCTION;\
        ASSIGNMENT;\
      }\
      GPU_REDUCTION;\
      sc::copy(RES, BUF.data());\
      if(diff(CRES, BUF, numeric_trait<T>::epsilon)){\
        nfail++;\
        std::cout << " [FAIL] " << std::endl;\
      }\
      else{\
        npass++;\
        std::cout << std::endl;\
      }\
    }

#define ADD_TEST_MATMUL(NAME, GPU_OP)\
  {\
    std::cout << NAME << "..." << std::flush;\
    GPU_OP;\
    sc::copy(C, buffer);\
    if(diff(buffer, cCbuffer, numeric_trait<T>::epsilon))\
    {\
      nfail++;\
      std::cout << " [Failure!]" << std::endl;\
    }\
    else{\
      npass++;\
      std::cout << std::endl;\
    }\
  }

#endif

