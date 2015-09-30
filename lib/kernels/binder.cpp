#include "isaac/kernels/binder.h"

namespace isaac
{

symbolic_binder::~symbolic_binder()
{
}

symbolic_binder::symbolic_binder() : current_arg_(0)
{
}

unsigned int symbolic_binder::get()
{
    return current_arg_++;
}

//Sequential
bind_sequential::bind_sequential()
{
}

bool bind_sequential::bind(driver::Buffer const & ph, bool)
{
    return memory.insert(std::make_pair(ph, current_arg_)).second;
}

unsigned int bind_sequential::get(driver::Buffer const & ph, bool is_assigned)
{
    return bind(ph, is_assigned)?current_arg_++:memory.at(ph);
}

//Independent
bind_independent::bind_independent()
{
}

bool bind_independent::bind(driver::Buffer const & ph, bool is_assigned)
{
    return is_assigned?true:memory.insert(std::make_pair(ph, current_arg_)).second;
}

unsigned int bind_independent::get(driver::Buffer const & ph, bool is_assigned)
{
    return bind(ph, is_assigned)?current_arg_++:memory.at(ph);
}

}
