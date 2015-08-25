#include "isaac/kernels/keywords.h"

namespace isaac
{

keyword::keyword(driver::backend_type backend, std::string const & opencl, std::string const & cuda) : backend_(backend), opencl_(opencl), cuda_(cuda)
{

}

std::string const & keyword::get() const
{
  switch(backend_)
  {
    case driver::OPENCL:
      return opencl_;
    case driver::CUDA:
      return cuda_;
    default: throw;
  }
}

std::ostream &  operator<<(std::ostream & ss, keyword const & kw)
{
  return ss << kw.get();
}


}
