#ifndef MYHALF_H_
#define MYHALF_H_

#include <iostream>

namespace isaac{
class  value_scalar;
struct    values_holder;

class half{
public:
  unsigned short us;
  operator int() const;

  half operator+(half &);
  half operator-(half &);
  half operator*(half &);
  half operator/(half &);
  half operator+=(half &);
  half operator>(half &);
  half operator>=(half &);
  half operator<(half &);
  half operator<=(half &);
  half operator==(half &);
  half operator!=(half &);

  half & operator=(const char);
  half & operator=(const unsigned char);
  half & operator=(const int);
  half & operator=(const unsigned int);
  half & operator=(const short);
  half & operator=(const unsigned short);
  half & operator=(const float);
  half & operator=(const double);
  half & operator=(const long);
  half & operator=(const unsigned long);
  half & operator=(const long long);
  half & operator=(const unsigned long long);

  std::ostream & operator<<(std::ostream &);
  half(char &);
  half(unsigned char &);
  half(int &);
  half(unsigned int &);
  half(short &);
  half(unsigned short &);
  half(float &);
  half(double &);
  half(long &);
  half(unsigned long &);
  half();

};

std::ostream & operator<<(std::ostream & os,half h);

#define DECLAREHALFOP(TYPE)\
  value_scalar operator+(TYPE, half);\
  value_scalar operator-(TYPE, half);\
  value_scalar operator*(TYPE, half);\
  value_scalar operator/(TYPE, half);\
  value_scalar operator+(half, TYPE);\
  value_scalar operator-(half, TYPE);\
  value_scalar operator*(half, TYPE);\
  value_scalar operator/(half, TYPE);

DECLAREHALFOP(char)
DECLAREHALFOP(unsigned char)
DECLAREHALFOP(short)
DECLAREHALFOP(unsigned short)
DECLAREHALFOP(int)
DECLAREHALFOP(unsigned int)
DECLAREHALFOP(long)
DECLAREHALFOP(unsigned long)
DECLAREHALFOP(long long)
DECLAREHALFOP(unsigned long long)
DECLAREHALFOP(float)
DECLAREHALFOP(double)

}
#endif
