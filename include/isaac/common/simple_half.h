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
//   operator float()const ;
//   operator double() const ;
//   operator unsigned int() const ;
//   operator short() const ;
//   operator unsigned short() const ;
//   operator char() const;

  half operator+(half &h1);
  half operator-(half &h1);
  half operator*(half &h1);
  half operator/(half &h1);
  half operator+=(half &h1);
  half operator>(half &h1);
  half operator>=(half &h1);
  half operator<(half &h1);
  half operator<=(half &h1);
  half operator==(half &h1);
  half operator!=(half &h1);

  half & operator=(const char x);
  half & operator=(const unsigned char x);
  half & operator=(const int x);
  half & operator=(const unsigned int x);
  half & operator=(const short x);
  half & operator=(const unsigned short x);
  half & operator=(const float x);
  half & operator=(const double x);
  half & operator=(const long x);
  half & operator=(const unsigned long x);
  half & operator=(const long long x);
  half & operator=(const unsigned long long x);
  //half & operator=(const int8_t x);
  //half & operator=(const uint8_t x);

  std::ostream & operator<<(std::ostream & os);
  half(char &x);
  half(unsigned char &x);
  half(int &x);
  half(unsigned int &x);
  half(short &x);
  half(unsigned short &x);
  half(float &x);
  half(double &x);
  half(long &x);
  half(unsigned long &x);
  half();

// half ( char x);
// half (unsigned char x);
// half ( int x);
// half (unsigned int x);
// half( short x);
// half ( unsigned short x);
// half ( float x);
// half ( double x);
// half( long x);
// half ( unsigned long x);

};

std::ostream & operator<<(std::ostream & os,half h);
value_scalar operator+(char h1, half h2);
value_scalar operator-(char h1, half h2);
value_scalar operator*(char h1, half h2);
value_scalar operator/(char h1, half h2);
value_scalar operator+(half h2, char h1);
value_scalar operator-(half h2, char h1);
value_scalar operator*(half h2, char h1);
value_scalar operator/(half h2, char h1);
value_scalar pow(half h2, char h1);
value_scalar pow(char h1, half h2);

value_scalar operator+(unsigned char h1, half h2);
value_scalar operator-(unsigned char h1, half h2);
value_scalar operator*(unsigned char h1, half h2);
value_scalar operator/(unsigned char h1, half h2);
value_scalar operator+(half h2, unsigned char h1);
value_scalar operator-(half h2, unsigned char h1);
value_scalar operator*(half h2, unsigned char h1);
value_scalar operator/(half h2, unsigned char h1);

value_scalar operator+(short h1, half h2);
value_scalar operator-(short h1, half h2);
value_scalar operator*(short h1, half h2);
value_scalar operator/(short h1, half h2);
value_scalar operator+(half h2, short h1);
value_scalar operator-(half h2, short h1);
value_scalar operator*(half h2, short h1);
value_scalar operator/(half h2, short h1);

value_scalar operator+(unsigned short h1, half h2);
value_scalar operator-(unsigned short h1, half h2);
value_scalar operator*(unsigned short h1, half h2);
value_scalar operator/(unsigned short h1, half h2);
value_scalar operator+(half h2, unsigned short h1);
value_scalar operator-(half h2, unsigned short h1);
value_scalar operator*(half h2, unsigned short h1);
value_scalar operator/(half h2, unsigned short h1);

value_scalar operator+(int h1, half h2);
value_scalar operator-(int h1, half h2);
value_scalar operator*(int h1, half h2);
value_scalar operator/(int h1, half h2);
value_scalar operator+(half h2, int h1);
value_scalar operator-(half h2, int h1);
value_scalar operator*(half h2, int h1);
value_scalar operator/(half h2, int h1);

value_scalar operator+(unsigned int h1, half h2);
value_scalar operator-(unsigned int h1, half h2);
value_scalar operator*(unsigned int h1, half h2);
value_scalar operator/(unsigned int h1, half h2);
value_scalar operator+(half h2, unsigned int h1);
value_scalar operator-(half h2, unsigned int h1);
value_scalar operator*(half h2, unsigned int h1);
value_scalar operator/(half h2, unsigned int h1);

value_scalar operator+(long h1, half h2);
value_scalar operator-(long h1, half h2);
value_scalar operator*(long h1, half h2);
value_scalar operator/(long h1, half h2);
value_scalar operator+(half h2, long h1);
value_scalar operator-(half h2, long h1);
value_scalar operator*(half h2, long h1);
value_scalar operator/(half h2, long h1);

value_scalar operator+(unsigned long h1, half h2);
value_scalar operator-(unsigned long h1, half h2);
value_scalar operator*(unsigned long h1, half h2);
value_scalar operator/(unsigned long h1, half h2);
value_scalar operator+(half h2, unsigned long h1);
value_scalar operator-(half h2, unsigned long h1);
value_scalar operator*(half h2, unsigned long h1);
value_scalar operator/(half h2, unsigned long h1);

value_scalar operator+(long long h1, half h2);
value_scalar operator-(long long h1, half h2);
value_scalar operator*(long long h1, half h2);
value_scalar operator/(long long h1, half h2);
value_scalar operator+(half h2, long long h1);
value_scalar operator-(half h2, long long h1);
value_scalar operator*(half h2, long long h1);
value_scalar operator/(half h2, long long h1);

value_scalar operator+(unsigned long long h1, half h2);
value_scalar operator-(unsigned long long h1, half h2);
value_scalar operator*(unsigned long long h1, half h2);
value_scalar operator/(unsigned long long h1, half h2);
value_scalar operator+(half h2, unsigned long long h1);
value_scalar operator-(half h2, unsigned long long h1);
value_scalar operator*(half h2, unsigned long long h1);
value_scalar operator/(half h2, unsigned long long h1);

value_scalar operator+(float h1, half h2);
value_scalar operator-(float h1, half h2);
value_scalar operator*(float h1, half h2);
value_scalar operator/(float h1, half h2);
value_scalar operator+(half h2, float h1);
value_scalar operator-(half h2, float h1);
value_scalar operator*(half h2, float h1);
value_scalar operator/(half h2, float h1);

value_scalar operator+(double h1, half h2);
value_scalar operator-(double h1, half h2);
value_scalar operator*(double h1, half h2);
value_scalar operator/(double h1, half h2);
value_scalar operator+(half h2, double h1);
value_scalar operator-(half h2, double h1);
value_scalar operator*(half h2, double h1);
value_scalar operator/(half h2, double h1);

// value_scalar operator/(unsigned char h1, half h2);
// value_scalar operator+=(short h1, half h2);
// value_scalar operator+=(unsigned short h1, half h2);
// value_scalar operator+(int h1, half h2);
// value_scalar operator*(unsigned int h1, half h2);
// value_scalar operator+=(long h1, half h2);
// value_scalar operator+=(unsigned long h1, half h2);
// value_scalar operator+=(long long h1, half h2);
// value_scalar operator+=(double h1, half h2);
// value_scalar operator-( half h2,char h1);
// value_scalar operator/( half h2,unsigned char h1);
// value_scalar operator+=( half h2,short h1);
// value_scalar operator+=( half h2,unsigned short h1);
// value_scalar operator+( half h2,int h1);
// value_scalar operator*( half h2,unsigned int h1);
// value_scalar  operator+=( half h2,long h1);
// value_scalar operator+=( half h2,unsigned long h1);
// value_scalar operator+=( half h2,long long h1);

}
#endif
