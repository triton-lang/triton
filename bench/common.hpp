#ifndef ISAAC_BENCH_COMMON_HPP_
#define ISAAC_BENCH_COMMON_HPP_

#include <chrono>
#include <algorithm>

template<std::size_t> struct int_{};

template <class Tuple, size_t Pos>
std::ostream& print_tuple(std::ostream& out, const Tuple& t, int_<Pos> )
{
    out << std::get< std::tuple_size<Tuple>::value-Pos >(t) << ',';
    return print_tuple(out, t, int_<Pos-1>());
}

template <class Tuple>
std::ostream& print_tuple(std::ostream& out, const Tuple& t, int_<1> )
{
    return out << std::get<std::tuple_size<Tuple>::value-1>(t);
}

template <class... Args>
std::ostream& operator<<(std::ostream& out, const std::tuple<Args...>& t)
{
    print_tuple(out, t, int_<sizeof...(Args)>());
    return out;
}

int ceil(int N, int pad)
{
    return (N%pad==0)?N:(N+pad-1)/pad*pad;
}

std::vector<int> create_log_range(int min, int max, int N, int pad)
{
  std::vector<int> res(N);
  for(int i = 0 ; i < N ; ++i)
  {
    res[i] = static_cast<int>(std::exp(std::log(min) + (float)(std::log(max) - std::log(min))*i/N));
    res[i] = ceil(res[i], pad);
  }
  return res;
}

std::vector<int> create_full_range(int min, int max, int pad)
{
    std::vector<int> N;
    for(int i = ceil(min, pad) ; i < ceil(max, pad) ; i+=pad)
        N.push_back(i);
    return N;
}


template<class T>
T median(std::vector<T> x)
{
  size_t size = x.size();
  std::sort(x.begin(), x.end());
  if (size  % 2 == 0)
      return (x[size / 2 - 1] + x[size / 2]) / 2;
  else
      return x[size / 2];
}

template<class T>
T min(std::vector<T> x)
{
  return *std::min_element(x.begin(), x.end());
}

template<class T>
T mean(std::vector<T> x)
{
  T res = 0;
  int N = x.size();
  for(int i = 0 ; i < N ; ++i)
    res += x[i];
  return res/N;
}

class Timer
{
    typedef std::chrono::high_resolution_clock high_resolution_clock;
    typedef std::chrono::nanoseconds nanoseconds;

public:
    explicit Timer(bool run = false)
    {
        if (run)
            start();
    }

    void start()
    {
        _start = high_resolution_clock::now();
    }

    nanoseconds get() const
    {
        return std::chrono::duration_cast<nanoseconds>(high_resolution_clock::now() - _start);
    }

private:
    high_resolution_clock::time_point _start;
};



#endif
