#include "isaac/common/simple_half.h"
#include "isaac/common/numeric_type.h"
#include "isaac/value_scalar.h"
namespace isaac{


half::operator int() const{return (int)(us);}

half half::operator+(half &h)
{
  return h;
}
half half::operator-(half &h)
{
  return h;
}
half half::operator*(half &h)
{
  return h;
}
half half::operator/(half &h)
{
  return h;
}
half half::operator+=(half &h)
{
  return h;
}

half half::operator>(half &h) { return h; }
half half::operator>=(half &h) { return h; }
half half::operator<(half &h) { return h; }
half half::operator<=(half &h) { return h; }
half half::operator==(half &h) { return h; }
half half::operator!=(half &h) { return h; }

half& half::operator=(const char x) { us = x; return *this; }
half& half::operator=(const unsigned char x) { us = x; return *this; }
half& half::operator=(const int x) { us = x; return *this; }
half& half::operator=(const unsigned int x) { us = x; return *this; }
half& half::operator=(const short x) { us = x; return *this; }
half& half::operator=(const unsigned short x) { us = x; return *this; }
half& half::operator=(const float x) { us = x; return *this; }
half& half::operator=(const double x) { us = x; return *this; }
half& half::operator=(const long x) { us = x; return *this; }
half& half::operator=(const unsigned long x) { us = x; return *this; }
half& half::operator=(const long long x) { us = x; return *this; }
half& half::operator=(const unsigned long long x) { us = x; return *this; }

std::ostream & half::operator<<(std::ostream & os)
{
  return os<<this->us<<std::endl;
}

half::half(char &x) { us = x; }
half::half(unsigned char &x) { us = x; }
half::half(int &x) { us = x; }
half::half(unsigned int &x) { us = x; }
half::half(short &x) { us = x; }
half::half(unsigned short &x) { us = x; }
half::half(float &x) { us = x; }
half::half(double &x) { us = x; }
half::half(long &x) { us = x; }
half::half(unsigned long &x) { us = x; }
half::half() { }

std::ostream & operator<<(std::ostream & os, half h)
{
  return os<<h.us<<std::endl;
}

#define INSTANTIATEVS  \
  std::cout << "WARNING: This function has no practical significance, it will not give correct output." << std::endl;\
  values_holder vh;\
  vh.float16 = h;\
  return value_scalar(vh, HALF_TYPE);

#define INSTANTIATEHALFOP(TYPE)\
  value_scalar operator+(TYPE, half h)\
  {\
    INSTANTIATEVS\
  }\
  value_scalar operator-(TYPE, half h)\
  {\
    INSTANTIATEVS\
  }\
  value_scalar operator*(TYPE, half h)\
  {\
    INSTANTIATEVS\
  }\
  value_scalar operator/(TYPE, half h)\
  {\
    INSTANTIATEVS\
  }\
   \
  value_scalar operator+(half h, TYPE)\
  {\
    INSTANTIATEVS\
  }\
  value_scalar operator-(half h, TYPE)\
  {\
    INSTANTIATEVS\
  }\
  value_scalar operator*(half h, TYPE)\
  {\
    INSTANTIATEVS\
  }\
  value_scalar operator/(half h, TYPE)\
  {\
    INSTANTIATEVS\
  }

INSTANTIATEHALFOP(char)
INSTANTIATEHALFOP(unsigned char)
INSTANTIATEHALFOP(short)
INSTANTIATEHALFOP(unsigned short)
INSTANTIATEHALFOP(int)
INSTANTIATEHALFOP(unsigned int)
INSTANTIATEHALFOP(long)
INSTANTIATEHALFOP(unsigned long)
INSTANTIATEHALFOP(long long)
INSTANTIATEHALFOP(unsigned long long)
INSTANTIATEHALFOP(float)
INSTANTIATEHALFOP(double)

}

