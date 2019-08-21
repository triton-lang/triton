#include "triton/lang/wgtcc/t_code_gen.h"
#include "triton/lang/wgtcc/evaluator.h"
#include "triton/lang/wgtcc/parser.h"
#include "triton/lang/wgtcc/token.h"

void Generator::Gen() {
  VisitTranslationUnit(parser_->Unit());
}
