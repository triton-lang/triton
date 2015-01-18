#include "atidlas/backend/binder.h"

namespace atidlas
{

symbolic_binder::~symbolic_binder(){ }

bind_to_handle::bind_to_handle() : current_arg_(0)
{ }

//
bool bind_to_handle::bind(cl_mem ph)
{ return (ph==NULL)?true:memory.insert(std::make_pair((void*)ph, current_arg_)).second; }

unsigned int bind_to_handle::get(cl_mem ph)
{ return bind(ph)?current_arg_++:memory.at(ph); }

//
bind_all_unique::bind_all_unique() : current_arg_(0)
{ }

bool bind_all_unique::bind(cl_mem)
{return true;}

unsigned int bind_all_unique::get(cl_mem)
{ return current_arg_++;}

}
