#ifndef ISAAC_TOOLS_MAKE_VECTOR_HPP
#define ISAAC_TOOLS_MAKE_VECTOR_HPP

#include <vector>

namespace isaac
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
