#include <cstring>
#include <cstdio>

typedef struct yy_buffer_state * YY_BUFFER_STATE;
extern int yyparse();
extern YY_BUFFER_STATE yy_scan_string(const char * str);
extern void yy_delete_buffer(YY_BUFFER_STATE buffer);

int main() {
   char string[] = "void test(int);";
   YY_BUFFER_STATE buffer = yy_scan_string(string);
   yy_delete_buffer(buffer);
   return 0;
}
