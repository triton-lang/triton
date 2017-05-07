#include <cmath>
#include <vector>
#include <iostream>

double max_rounding_error(double x){ return std::pow(2, int(std::log2(x)) - 52); }
double max_rounding_error(float x){ return std::pow(2, int(std::log2(x)) - 23); }
double max_rounding_error(half_float::half x){ return std::pow(2, int(std::log2(x)) - 10); }

template<class T>
bool is_correct(std::vector<T> const & iO, std::vector<T> const & rO, double eps){
  for(size_t i = 0 ; i < iO.size(); ++i)
    if(std::abs((iO[i] - rO[i])/rO[i]) > eps || std::isnan(iO[i])){
      std::cout << "idx " << i << ": " <<  iO[i] << " != " << rO[i] << std::endl;
      return false;
    }
  return true;
}
