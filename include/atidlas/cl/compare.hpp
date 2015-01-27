#ifndef ATIDLAS_CL_COMPARE_HPP
#define ATIDLAS_CL_COMPARE_HPP

namespace atidlas
{

namespace cl_ext
{

struct compare{
public:
  template<class T>
  bool operator()(T const & x, T const & y){ return x() < y(); }
};

}
}

#endif
