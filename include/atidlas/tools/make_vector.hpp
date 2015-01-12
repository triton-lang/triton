#ifndef ATIDLAS_TOOLS_MAKE_VECTOR_HPP
#define ATIDLAS_TOOLD_MAKE_VECTOR_HPP

#include <vector>

namespace atidlas
{

namespace tools
{

template <typename T>
class make_vector
{
public:
  typedef make_vector<T> my_type;
  my_type& operator<< (const T& val) {
    data_.push_back(val);
    return *this;
  }
  operator std::vector<T>() const {
    return data_;
  }
private:
  std::vector<T> data_;
};

}

}

#endif
