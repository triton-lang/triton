#include "triton/ast/error.h"


namespace triton{

namespace ast{

static int current_line = 0;
static int current_column = 0;

// begin token
void update_location(const char *text) {
  for (int i = 0; text[i] != '\0'; i++){
    if (text[i] == '\n'){
      current_column = 0;
      current_line++;
    }
    else if (text[i] == '\t')
      current_column += 8 - (current_column % 8);
    else
      current_column++;
  }
}

void print_error(const char *cerror) {
  std::string error(cerror);
  auto it = error.find("syntax error,");
  error.replace(it, 13, "");
  std::cerr << "error at line " << current_line << " (column " << current_column << "): " << error << std::endl;
  throw std::runtime_error("compilation failed");
}

char return_impl(char t, const char * yytext) {
  update_location(yytext);
  return t;
}

yytokentype return_impl(yytokentype t, const char * yytext){
  update_location(yytext);
  return t;
}

void return_void(const char * yytext){
  update_location(yytext);
}

}

}
