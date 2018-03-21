#include <cmath>
#include <vector>
#include <iostream>

double max_rounding_error(int32_t){ return 0; }
double max_rounding_error(double x){ return std::pow(2, int(std::log2(x)) - 52); }
double max_rounding_error(float x){ return std::pow(2, int(std::log2(x)) - 23); }
double max_rounding_error(half_float::half x){ return std::pow(2, int(std::log2(x)) - 10); }

template<class T>
bool is_correct(std::vector<T> const & iO, std::vector<T> const & rO, double eps){

  if(iO.size() != rO.size()){
    std::cout << "inputs don't have the same size" << std::endl;
    return false;
  }
  for(size_t i = 0 ; i < iO.size(); ++i){
    T io = iO[i], ro = rO[i];
    T denom = std::max(std::fabs(io), std::fabs(ro));
    denom = (denom==0)?1:denom;
    if(std::fabs(io - ro)/denom > eps || std::isnan(io) || (std::isinf(io) ^ std::isinf(ro))){
      std::cout << "idx " << i << ": " <<  io << " != " << ro << std::endl;
      return false;
    }
  }
  return true;
}

// Pack increment
template<class T> struct pack_increment
{ enum{ VALUE = 1}; };

template<> struct pack_increment<int>
{ enum{ VALUE = 4}; };


// Clamp
template <class T> T clamp(T x, T lo, T hi)
{
  return std::max<T>(lo, std::min<T>(x, hi));
}


// Pack
template<class T, class U>
T pack(U* tmp, U scale);

template<>
double pack<double, double>(double* tmp, double scale)
{ return tmp[0]*scale; }

template<>
float pack<float, float>(float* tmp, float scale)
{ return tmp[0]*scale; }

template<>
int pack<int, float>(float* tmp, float scale)
{
  int res = 0;
  for(int i = 0; i < 4; i++){
    int8_t clamped = std::round(clamp(tmp[i]*scale, (float)-128, (float)127));
    res |= (clamped & 0xFF) << (8*i);
  }
  return res;
}

// Unpacked type
template<class T> struct unpack_type { typedef T Type; };
template<> struct unpack_type<int> { typedef float Type; };

// Unpack
float* unpack(float* ptr, float value, float scale)
{
  *ptr = value/scale;
  return ptr;
}

float* unpack(float* ptr, int value, float scale)
{
  for(int i = 0; i < 4; i++){
    int shifted = (value >> (8*i) & 0xff);
    ptr[i] = ((float)(*(int8_t*)(&shifted)))/scale;
  }
  return ptr;
}

// Dot
template<class T>
inline T dot(T x, T y, T z)
{
  return std::fma(x, y, z);
}

inline int dot(int x, int y, int z){
  int res = 0;
  for(int i = 0; i < 4; i++){
    int32_t a = ((x >> (8*i)) & 0x000000FF);
    int32_t b = ((y >> (8*i)) & 0x000000FF);
    res +=  (*(int8_t*)(&a)) * (*(int8_t*)(&b));
  }
  return res + z;
}

// Accumulation
template<class T>
T max(T x, T y)
{ return std::max(x, y); }

template<class T>
T plus(T x, T y)
{ return x + y; }

// Index computation
inline int32_t idx(int32_t x, int32_t y, int32_t z, int32_t w, int32_t u,
                   int32_t /*s0*/, int32_t s1, int32_t s2, int32_t s3, int32_t s4)
{ return u + w*s4 + z*s4*s3 + y*s4*s3*s2 + x*s4*s3*s2*s1; }


template<char> inline int matidx(int i, int j, int ld);
template<> inline int matidx<'T'>(int i, int j, int ld){ return j + i*ld; }
template<> inline int matidx<'N'>(int i, int j, int ld){ return i + j*ld; }
