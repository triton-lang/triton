
#include"isaac/common/simple_half.h"
#include"isaac/common/numeric_type.h"
#include"isaac/value_scalar.h"
namespace isaac{


half::operator int() const{return (int)(us);}
// half::  operator float()const {return (float)us;}
// half:: operator double() const {return (double)us;}
//   operator int() const {return 0;}
//  half:: operator unsigned int() const {return (unsigned int)us;}
// half:: operator short() const {return (short)us;}
// half::  operator unsigned short() const {return us;}
//  half:: operator char() const{return (char)us;}
//   operator int() const{}
//   operator int() const{}

half half::operator+(half &h1)
{
  return h1;
}
half half::operator-(half &h1 )
{
  return h1;
}
half half::operator*(half &h1)
{
  return h1;
}
half half::operator/(half &h1)
{
  return h1;
}
half half::operator+=(half &h1)
{
  return h1;
}

half half::operator>(half &h1) { return h1; }
half half::operator>=(half &h1) { return h1; }
half half::operator<(half &h1) { return h1; }
half half::operator<=(half &h1) { return h1; }
half half::operator==(half &h1) { return h1; }
half half::operator!=(half &h1) { return h1; }

//half::half(char &x):us(x){}
//half::half(int &x):us(x){}
//half::half(unsigned short &x):us(x){}
//half::half( short &x):us(x){}
//half::half(unsigned int &x):us(x){}
//half::half(float &x):us(x){}
//half::half(double &x):us(x){}

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

// half::half ( char x){us=x;}
// half::half (unsigned char x){us=x;}
// half::half ( int x){us=x;}
// half::half (unsigned int x){us=x;}
// half::half( short x){us=x;}
// half::half ( unsigned short x){us=x;}
// half::half ( float x){us=x;}
// half::half ( double x){us=x;}
// half::half( long x){us=x;}
// half::half ( unsigned long x){us=x;}
std::ostream & operator<<(std::ostream & os, half h)
{
  return os<<h.us<<std::endl;
}

value_scalar operator+(char h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(char h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(char h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(char h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}

value_scalar operator+(half h2, char h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(half h2, char h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(half h2, char h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(half h2, char h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}

value_scalar operator+(unsigned char h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(unsigned char h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(unsigned char h1, half h2)
{
   values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(unsigned char h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator+(half h2, unsigned char h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(half h2, unsigned char h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(half h2, unsigned char h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(half h2, unsigned char h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}

value_scalar operator+(short h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(short h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(short h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(short h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator+(half h2, short h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(half h2, short h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(half h2, short h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(half h2, short h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}

value_scalar operator+(unsigned short h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(unsigned short h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(unsigned short h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(unsigned short h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator+(half h2, unsigned short h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(half h2, unsigned short h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(half h2, unsigned short h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(half h2, unsigned short h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}

value_scalar operator+(int h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(int h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(int h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(int h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator+(half h2,int h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(half h2, int h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(half h2, int h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(half h2, int h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}

value_scalar operator+(unsigned int h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(unsigned int h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(unsigned int h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(unsigned int h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator+(half h2, unsigned int h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(half h2, unsigned int h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(half h2, unsigned int h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(half h2, unsigned int h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}

value_scalar operator+(long h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(long h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(long h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(long h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator+(half h2, long h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(half h2, long h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(half h2, long h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(half h2, long h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}

value_scalar operator+(unsigned long h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(unsigned long h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(unsigned long h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(unsigned long h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator+(half h2, unsigned long h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(half h2, unsigned long h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(half h2, unsigned long h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(half h2, unsigned long h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}

value_scalar operator+(long long h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(long long h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(long long h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(long long h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator+(half h2, long long h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(half h2, long long h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(half h2, long long h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(half h2, long long h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}

value_scalar operator+(unsigned long long h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(unsigned long long h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(unsigned long long h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(unsigned long long h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator+(half h2, unsigned long long h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(half h2, unsigned long long h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(half h2, unsigned long long h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(half h2, unsigned long long h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}

value_scalar operator+(float h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(float h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(float h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(float h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator+(half h2, float h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(half h2, float h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(half h2, float h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(half h2, float h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}

value_scalar operator+(double h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(double h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(double h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(double h1, half h2)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator+(half h2, double h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator-(half h2, double h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator*(half h2, double h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}
value_scalar operator/(half h2, double h1)
{
  values_holder vh;
  vh.float16 = h2;
  return value_scalar(vh, HALF_TYPE);
}

}

