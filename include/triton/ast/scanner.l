D			[0-9]
L			[a-zA-Z_]
H			[a-fA-F0-9]
E			[Ee][+-]?{D}+
FS			(f|F|l|L)
IS			(u|U|l|L)*

%{
#include <stdio.h>
#include "parser.hpp"
#include "../include/triton/ast/ast.h"
using triton::ast::return_impl;
using triton::ast::return_void;
%}

%%
"__constant__"          { return return_impl(CONSTANT_SPACE, yytext); }
"const"                 { return return_impl(CONST, yytext); }
"tunable"               { return return_impl(TUNABLE, yytext); }
"kernel"                { return return_impl(KERNEL, yytext); }
"restrict"              { return return_impl(RESTRICT, yytext); }
"read_only"             { return return_impl(READONLY, yytext); }
"write_only"            { return return_impl(WRITEONLY, yytext); }
"@"                     { return return_impl(AT, yytext); }
"newaxis"		        { return return_impl(NEWAXIS, yytext); }
"if"			        { return return_impl(IF, yytext); }
"else"			        { return return_impl(ELSE, yytext); }
"for"			        { return return_impl(FOR, yytext); }
"while"			        { return return_impl(WHILE, yytext); }
"void"			        { return return_impl(VOID, yytext); }
"uint1"                 { return return_impl(UINT1, yytext); }
"uint8"			        { return return_impl(UINT8, yytext); }
"uint16"		        { return return_impl(UINT16, yytext); }
"uint32"		        { return return_impl(UINT32, yytext); }
"uint64"		        { return return_impl(UINT64, yytext); }
"int1"                  { return return_impl(INT1, yytext); }
"int8"			        { return return_impl(INT8, yytext); }
"int16"			        { return return_impl(INT16, yytext); }
"int32"			        { return return_impl(INT32, yytext); }
"int64"			        { return return_impl(INT64, yytext); }
"fp32"			        { return return_impl(FP32, yytext); }
"fp64"			        { return return_impl(FP64, yytext); }
"..."                   { return return_impl(ELLIPSIS, yytext); }
"get_global_range"      { return return_impl(GET_GLOBAL_RANGE, yytext); }
"get_range_id"          { return return_impl(GET_RANGE_ID, yytext); }
"__atomic_cas"          { return return_impl(ATOMIC_CAS, yytext); }
"__atomic_exchg"    { return return_impl(ATOMIC_EXCHG, yytext); }
"dot"                   { return return_impl(DOT, yytext); }
"max"                   { return return_impl(MAX, yytext); }
"min"                   { return return_impl(MIN, yytext); }
"select"                { return return_impl(SELECT, yytext); }
"trans"                 { return return_impl(TRANS, yytext); }
"continue"              { return return_impl(CONTINUE, yytext); }
"alloc_const"           { return return_impl(ALLOC_CONST, yytext); }
{L}({L}|{D})*           { return return_impl(IDENTIFIER, yytext); }
0[xX]{H}+{IS}?	        { return return_impl(CONSTANT, yytext); }
0{D}+{IS}?              { return return_impl(CONSTANT, yytext); }
{D}+{IS}?		        { return return_impl(CONSTANT, yytext); }
L?'(\\.|[^\\'])+'      	{ return return_impl(CONSTANT, yytext); }
{D}+{E}{FS}?		    { return return_impl(CONSTANT, yytext); }
L?\"(\\.|[^\\"])*\"	    { return return_impl(STRING_LITERAL, yytext); }
">>="                   { return return_impl(RIGHT_ASSIGN, yytext); }
"<<="			        { return return_impl(LEFT_ASSIGN, yytext); }
"+="			        { return return_impl(ADD_ASSIGN, yytext); }
"-="			        { return return_impl(SUB_ASSIGN, yytext); }
"*="			        { return return_impl(MUL_ASSIGN, yytext); }
"/="			        { return return_impl(DIV_ASSIGN, yytext); }
"%="			        { return return_impl(MOD_ASSIGN, yytext); }
"&="			        { return return_impl(AND_ASSIGN, yytext); }
"^="			        { return return_impl(XOR_ASSIGN, yytext); }
"|="			        { return return_impl(OR_ASSIGN, yytext); }
">>"			        { return return_impl(RIGHT_OP, yytext); }
"<<"			        { return return_impl(LEFT_OP, yytext); }
"++"			        { return return_impl(INC_OP, yytext); }
"--"			        { return return_impl(DEC_OP, yytext); }
"->"			        { return return_impl(PTR_OP, yytext); }
"&&"			        { return return_impl(AND_OP, yytext); }
"||"			        { return return_impl(OR_OP, yytext); }
"<="			        { return return_impl(LE_OP, yytext); }
">="			        { return return_impl(GE_OP, yytext); }
"=="			        { return return_impl(EQ_OP, yytext); }
"!="			        { return return_impl(NE_OP, yytext); }
";"			            { return return_impl(';', yytext); }
("{"|"<%")	           	{ return return_impl('{', yytext); }
("}"|"%>")	           	{ return return_impl('}', yytext); }
","			            { return return_impl(',', yytext); }
":"			            { return return_impl(':', yytext); }
"="			            { return return_impl('=', yytext); }
"("			            { return return_impl('(', yytext); }
")"			            { return return_impl(')', yytext); }
("["|"<:")		        { return return_impl('[', yytext); }
("]"|":>")		        { return return_impl(']', yytext); }
"."			            { return return_impl('.', yytext); }
"&"			            { return return_impl('&', yytext); }
"!"			            { return return_impl('!', yytext); }
"~"			            { return return_impl('~', yytext); }
"-"			            { return return_impl('-', yytext); }
"+"			            { return return_impl('+', yytext); }
"*"			            { return return_impl('*', yytext); }
"/"			            { return return_impl('/', yytext); }
"%"			            { return return_impl('%', yytext); }
"<"			            { return return_impl('<', yytext); }
">"			            { return return_impl('>', yytext); }
"^"			            { return return_impl('^', yytext); }
"|"			            { return return_impl('|', yytext); }
"?"			            { return return_impl('?', yytext); }
[ \t\v\n\f]             { return_void(yytext);}
.       				{ /* ignore bad characters */ }

%%

int yywrap()
{ return(1); }
