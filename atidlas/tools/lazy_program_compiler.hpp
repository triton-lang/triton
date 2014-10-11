#ifndef ATIDLAS_LAZY_PROGRAM_COMPILER_HPP
#define ATIDLAS_LAZY_PROGRAM_COMPILER_HPP

#include <map>
#include "viennacl/ocl/context.hpp"

namespace atidlas
{

  class lazy_program_compiler
  {
  public:

    lazy_program_compiler(viennacl::ocl::context * ctx, std::string const & name, std::string const & src, bool force_recompilation) : ctx_(ctx), program_(NULL), name_(name), src_(src), force_recompilation_(force_recompilation){ }
    lazy_program_compiler(viennacl::ocl::context * ctx, std::string const & name, bool force_recompilation) : ctx_(ctx), program_(NULL), name_(name), force_recompilation_(force_recompilation){ }

    void add(std::string const & src) {  src_+=src; }

    std::string const & src() const { return src_; }

    viennacl::ocl::program & program()
    {
      if(program_==NULL)
      {
        if (force_recompilation_ && ctx_->has_program(name_))
          ctx_->delete_program(name_);
        if (!ctx_->has_program(name_))
          ctx_->add_program(src_, name_);
        program_ = &ctx_->get_program(name_);
      }
      return *program_;
    }

  private:
    viennacl::ocl::context * ctx_;
    viennacl::ocl::program * program_;
    std::string name_;
    std::string src_;
    bool force_recompilation_;
  };

}
#endif
