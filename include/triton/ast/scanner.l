D			[0-9]
L			[a-zA-Z_]
H			[a-fA-F0-9]
E			[Ee][+-]?{D}+
FS			(f|F|l|L)
IS			(u|U|l|L)*

%{
#include <stdio.h>
#include "parser.hpp"
%}

%%
"__constant__"     { return(CONSTANT_SPACE); }
"const"            { return(CONST); }
"tunable"          { return(TUNABLE); }
"kernel"           { return(KERNEL); }
"restrict"         { return(RESTRICT); }
"read_only"         { return(READONLY); }
"write_only"        { return(WRITEONLY); }
"@"                { return(AT); }
"newaxis"		       { return(NEWAXIS); }
"if"			         { return(IF); }
"else"			       { return(ELSE); }
"for"			         { return(FOR); }
"void"			       { return(VOID); }
"uint1"            { return(UINT1); }
"uint8"			       { return(UINT8); }
"uint16"		       { return(UINT16); }
"uint32"		       { return(UINT32); }
"uint64"		       { return(UINT64); }
"int1"             { return(INT1); }
"int8"			       { return(INT8); }
"int16"			       { return(INT16); }
"int32"			       { return(INT32); }
"int64"			       { return(INT64); }
"fp32"			       { return(FP32); }
"fp64"			       { return(FP64); }
"..."              { return(ELLIPSIS); }
"get_global_range" { return GET_GLOBAL_RANGE; }
"dot"              { return DOT;}
"continue"         { return(CONTINUE); }
"alloc_const"      { return(ALLOC_CONST); }
{L}({L}|{D})*		{ return(IDENTIFIER); }

0[xX]{H}+{IS}?		{ return(CONSTANT); }
0{D}+{IS}?		{ return(CONSTANT); }
{D}+{IS}?		{ return(CONSTANT); }
L?'(\\.|[^\\'])+'	{ return(CONSTANT); }

{D}+{E}{FS}?		{ return(CONSTANT); }
{D}*"."{D}+({E})?{FS}?	{ return(CONSTANT); }
{D}+"."{D}*({E})?{FS}?	{ return(CONSTANT); }

L?\"(\\.|[^\\"])*\"	{ return(STRING_LITERAL); }

">>="			{ return(RIGHT_ASSIGN); }
"<<="			{ return(LEFT_ASSIGN); }
"+="			{ return(ADD_ASSIGN); }
"-="			{ return(SUB_ASSIGN); }
"*="			{ return(MUL_ASSIGN); }
"/="			{ return(DIV_ASSIGN); }
"%="			{ return(MOD_ASSIGN); }
"&="			{ return(AND_ASSIGN); }
"^="			{ return(XOR_ASSIGN); }
"|="			{ return(OR_ASSIGN); }
">>"			{ return(RIGHT_OP); }
"<<"			{ return(LEFT_OP); }
"++"			{ return(INC_OP); }
"--"			{ return(DEC_OP); }
"->"			{ return(PTR_OP); }
"&&"			{ return(AND_OP); }
"||"			{ return(OR_OP); }
"<="			{ return(LE_OP); }
">="			{ return(GE_OP); }
"=="			{ return(EQ_OP); }
"!="			{ return(NE_OP); }
";"			{ return(';'); }
("{"|"<%")		{ return('{'); }
("}"|"%>")		{ return('}'); }
","			{ return(','); }
":"			{ return(':'); }
"="			{ return('='); }
"("			{ return('('); }
")"			{ return(')'); }
("["|"<:")		{ return('['); }
("]"|":>")		{ return(']'); }
"."			{ return('.'); }
"&"			{ return('&'); }
"!"			{ return('!'); }
"~"			{ return('~'); }
"-"			{ return('-'); }
"+"			{ return('+'); }
"*"			{ return('*'); }
"/"			{ return('/'); }
"%"			{ return('%'); }
"<"			{ return('<'); }
">"			{ return('>'); }
"^"			{ return('^'); }
"|"			{ return('|'); }
"?"			{ return('?'); }

[ \t\v\n\f]		{ }
.       { /* ignore bad characters */ }

%%

int yywrap()
{ return(1); }

void yyerror (const char *s)  /* Called by yyparse on error */
{
  printf ("Error: %s\n", s);
}
