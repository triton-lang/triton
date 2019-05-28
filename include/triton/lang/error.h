#ifndef TRITON_INCLUDE_LANG_ERROR_H
#define TRITON_INCLUDE_LANG_ERROR_H

#include "parser.hpp"


namespace triton{
namespace lang{


void update_location(const char *t);
void print_error(const char *error);
char return_impl(char t, const char * yytext);
yytokentype return_impl(yytokentype t, const char * yytext);
void return_void(const char * yytext);

}
}

#endif
