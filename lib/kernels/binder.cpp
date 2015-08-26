#include "isaac/kernels/binder.h"

namespace isaac
{

symbolic_binder::~symbolic_binder(){ }

symbolic_binder::symbolic_binder() : current_arg_(0)
{}

unsigned int symbolic_binder::get()
{ return current_arg_++; }

bind_to_handle::bind_to_handle()
{ }

//
bool bind_to_handle::bind(driver::Buffer const & ph)
{ return memory.insert(std::make_pair(ph, current_arg_)).second; }

unsigned int bind_to_handle::get(driver::Buffer const & ph)
{ return bind(ph)?current_arg_++:memory.at(ph); }

//
bind_all_unique::bind_all_unique()
{ }

bool bind_all_unique::bind(driver::Buffer const &)
{return true;}

unsigned int bind_all_unique::get(driver::Buffer const &)
{ return current_arg_++;}

}
