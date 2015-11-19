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

bool bind_sequential::bind(array_base const * a, bool)
{
    return memory.insert(std::make_pair(a, current_arg_)).second;
}

unsigned int bind_sequential::get(array_base const * a, bool is_assigned)
{
    return bind(a, is_assigned)?current_arg_++:memory.at(a);
}

//Independent
bind_independent::bind_independent()
{
}

bool bind_independent::bind(array_base const * a, bool is_assigned)
{
    return is_assigned?true:memory.insert(std::make_pair(a, current_arg_)).second;
}

unsigned int bind_independent::get(array_base const * a, bool is_assigned)
{
    return bind(a, is_assigned)?current_arg_++:memory.at(a);
}

}
