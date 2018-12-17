#include "ast.h"

namespace tdl{

namespace ast{

void translation_unit::codegen(module *mod)
{ decls_->codegen(mod); }


}

}
