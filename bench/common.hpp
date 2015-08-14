#ifndef ISAAC_BENCH_COMMON_HPP_
#define ISAAC_BENCH_COMMON_HPP_

#include <chrono>

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
