#ifndef ATIDLAS_MODEL_TOOLS_HPP
#define ATIDLAS_MODEL_TOOLS_HPP

#include <vector>
#include "rapidjson/document.h"

namespace atidlas
{

  namespace tools
  {
    template<class T>
    std::vector<T> to_int_array(rapidjson::Value const & a)
    {
      size_t N = a.Size();
      std::vector<T> res(N);
      for(size_t i = 0 ; i < N ; ++i) res[i] = a[i].GetInt();
      return res;
    }

    template<class T>
    std::vector<T> to_float_array(rapidjson::Value const & a)
    {
      size_t N = a.Size();
      std::vector<T> res(N);
      for(size_t i = 0 ; i < N ; ++i) res[i] = a[i].GetDouble();
      return res;
    }
  }

}

#endif
