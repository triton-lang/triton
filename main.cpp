#include <cstring>
#include <cstdio>

typedef struct yy_buffer_state * YY_BUFFER_STATE;
extern int yyparse();
extern YY_BUFFER_STATE yy_scan_string(const char * str);
extern void yy_delete_buffer(YY_BUFFER_STATE buffer);

const char src[] =
"\
void test(int32 id){\
  fp32 c[16, 16] = {0};\
  int32 i = 0;\
  i += 1;\
}\
";

int main() {
   YY_BUFFER_STATE buffer = yy_scan_string(src);
   yyparse();
   yy_delete_buffer(buffer);
   return 0;
}
