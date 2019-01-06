%{
namespace tdl{
namespace ast{
class node;
}
}
using namespace tdl::ast;
#define YYSTYPE node*
#include "../include/ast/ast.h"

extern char* yytext;
void yyerror(const char *s);
int yylex(void);

translation_unit *ast_root;

/* wrap token in AST node */
struct token: public node{
  token(ASSIGN_OP_T value): assign_op(value){ }
  token(BIN_OP_T value): bin_op(value){ }
  token(UNARY_OP_T value): unary_op(value){ }
  token(TYPE_T value): type(value){ }

  union {
    ASSIGN_OP_T assign_op;
    BIN_OP_T bin_op;
    UNARY_OP_T unary_op;
    TYPE_T type;
  };
};

/* shortcut to append in list */
template<class T>
node* append_ptr_list(node *result, node *in){
  return static_cast<list<T*>*>(result)->append((T*)in);
}

/* shortcut to access token value */
ASSIGN_OP_T get_assign_op(node *op) { return ((token*)op)->assign_op; }
UNARY_OP_T get_unary_op(node *op) { return ((token*)op)->unary_op; }
TYPE_T get_type_spec(node *op) { return ((token*)op)->type; }

%}
 
%token IDENTIFIER CONSTANT STRING_LITERAL
%token PTR_OP INC_OP DEC_OP LEFT_OP RIGHT_OP LE_OP GE_OP EQ_OP NE_OP
%token AND_OP OR_OP MUL_ASSIGN DIV_ASSIGN MOD_ASSIGN ADD_ASSIGN
%token SUB_ASSIGN LEFT_ASSIGN RIGHT_ASSIGN AND_ASSIGN
%token XOR_ASSIGN OR_ASSIGN TYPE_NAME
%token VOID UINT8 UINT16 UINT32 UINT64 INT8 INT16 INT32 INT64 FP32 FP64
%token IF ELSE FOR
%token NEWAXIS

%start translation_unit
%%


/* -------------------------- */
/*         Types              */
/* -------------------------- */

type_specifier
  : VOID { $$ = new token(VOID_T); }
  | UINT8 { $$ = new token(UINT8_T); }
  | UINT16 { $$ = new token(UINT16_T); }
  | UINT32 { $$ = new token(UINT32_T); }
  | UINT64 { $$ = new token(UINT64_T); }
  | INT8 { $$ = new token(INT8_T); }
  | INT16 { $$ = new token(INT16_T); }
  | INT32 { $$ = new token(INT32_T); }
  | INT64 { $$ = new token(INT64_T); }
  | FP32 { $$ = new token(FLOAT32_T); }
  | FP64 { $$ = new token(FLOAT64_T); }
	;

pointer
  : '*' { $$ = new pointer(nullptr); }
  | '*' pointer { $$ = new pointer($1); }
	
abstract_declarator
	: pointer { $$ = $1; }
  | pointer direct_abstract_declarator { $$ = ((declarator*)$2)->set_ptr($1); }
	| direct_abstract_declarator { $$ = $1; }
	;

direct_abstract_declarator
    : '[' constant_list ']' { $$ = new tile(nullptr, $1); }

constant : 
	CONSTANT { $$ = new constant(atoi(yytext)); }
	;
	
constant_list
	: constant  { $$ = new list<constant*>((constant*)$1); }
  | constant_list ',' constant { $$ = append_ptr_list<constant>($1, $3); }
	;

type_name
  : declaration_specifiers { $$ = new type_name($1, nullptr); }
  | declaration_specifiers abstract_declarator { $$ = new type_name($1, $2); }
	;

/* -------------------------- */
/*         Expressions        */
/* -------------------------- */

identifier
	: IDENTIFIER { $$ = new identifier(yytext); }
	;
	
primary_expression
  : identifier  { $$ = new named_expression($1); }
	| constant { $$ = $1; }
	| STRING_LITERAL { $$ = new string_literal(yytext); }
  | '(' expression ')' { $$ = $1; }
	;

range
  : ':' { $$ = new range(tdl::ast::ALL); }
  | NEWAXIS { $$ = new range(tdl::ast::NEWAXIS); }

range_list
  : range { $$ = new list<range*>((range*)$1); }
  | range_list ',' range { $$ = append_ptr_list<range>($1, $2); }

postfix_expression
  : primary_expression { $$ = $1;}
  | identifier '[' range_list ']' { $$ = new indexing_expression($1, $2);}
  ;

unary_expression
  : postfix_expression { $$ = $1; }
  | INC_OP unary_expression { $$ = new unary_operator(INC, $2); }
  | DEC_OP unary_expression { $$ = new unary_operator(DEC, $2); }
  | unary_operator cast_expression { $$ = new unary_operator(get_unary_op($1), $2); }
	;

unary_operator
  : '&' { $$ = new token(ADDR); }
  | '*' { $$ = new token(DEREF); }
  | '+' { $$ = new token(PLUS); }
  | '-' { $$ = new token(MINUS); }
  | '~' { $$ = new token(COMPL); }
  | '!' { $$ = new token(NOT); }
	;

cast_expression
	: unary_expression { $$ = $1; }
  | '(' type_name ')' cast_expression { $$ = new cast_operator($1, $2); }
	;

multiplicative_expression
	: cast_expression { $$ = $1; }
  | multiplicative_expression '*' cast_expression { $$ = new binary_operator(MUL, $1, $3); }
  | multiplicative_expression '/' cast_expression { $$ = new binary_operator(DIV, $1, $3); }
  | multiplicative_expression '%' cast_expression { $$ = new binary_operator(MOD, $1, $3); }
	;

additive_expression
	: multiplicative_expression { $$ = $1; }
  | additive_expression '+' multiplicative_expression { $$ = new binary_operator(ADD, $1, $3); }
  | additive_expression '-' multiplicative_expression { $$ = new binary_operator(SUB, $1, $3); }
	;

shift_expression
	: additive_expression { $$ = $1; }
  | shift_expression LEFT_OP additive_expression { $$ = new binary_operator(LEFT_SHIFT, $1, $3); }
  | shift_expression RIGHT_OP additive_expression { $$ = new binary_operator(RIGHT_SHIFT, $1, $3); }
	;

/* Comparison */
relational_expression
	: shift_expression { $$ = $1; }
  | relational_expression '<' shift_expression { $$ = new binary_operator(LT, $1, $3); }
  | relational_expression '>' shift_expression { $$ = new binary_operator(GT, $1, $3); }
  | relational_expression LE_OP shift_expression { $$ = new binary_operator(LE, $1, $3); }
  | relational_expression GE_OP shift_expression { $$ = new binary_operator(GE, $1, $3); }
	;

equality_expression
	: relational_expression { $$ = $1; }
  | equality_expression EQ_OP relational_expression { $$ = new binary_operator(EQ, $1, $3); }
  | equality_expression NE_OP relational_expression { $$ = new binary_operator(NE, $1, $3); }
	;

/* Binary */
and_expression
	: equality_expression { $$ = $1; }
  | and_expression '&' equality_expression { $$ = new binary_operator(AND, $1, $3); }
	;

exclusive_or_expression
	: and_expression { $$ = $1; }
  | exclusive_or_expression '^' and_expression { $$ = new binary_operator(XOR, $1, $3); }
	;

inclusive_or_expression
	: exclusive_or_expression { $$ = $1; }
  | inclusive_or_expression '|' exclusive_or_expression { $$ = new binary_operator(OR, $1, $3); }
	;

/* Logical */
logical_and_expression
	: inclusive_or_expression { $$ = $1; }
  | logical_and_expression AND_OP inclusive_or_expression { $$ = new binary_operator(LAND, $1, $3); }
	;

logical_or_expression
	: logical_and_expression { $$ = $1; }
  | logical_or_expression OR_OP logical_and_expression { $$ = new binary_operator(LOR, $1, $3); }
	;

/* Conditional */
conditional_expression
	: logical_or_expression { $$ = $1; }
	| logical_or_expression '?' conditional_expression ':' conditional_expression { $$ = new conditional_expression($1, $2, $3); }
	;

/* Assignment */
assignment_operator
  : '=' { $$ = new token(ASSIGN); }
  | MUL_ASSIGN { $$ = new token(INPLACE_MUL); }
  | DIV_ASSIGN { $$ = new token(INPLACE_DIV); }
  | MOD_ASSIGN { $$ = new token(INPLACE_MOD); }
  | ADD_ASSIGN { $$ = new token(INPLACE_ADD); }
  | SUB_ASSIGN { $$ = new token(INPLACE_SUB); }
  | LEFT_ASSIGN { $$ = new token(INPLACE_LSHIFT); }
  | RIGHT_ASSIGN { $$ = new token(INPLACE_RSHIFT); }
  | AND_ASSIGN { $$ = new token(INPLACE_AND); }
  | XOR_ASSIGN { $$ = new token(INPLACE_XOR); }
  | OR_ASSIGN { $$ = new token(INPLACE_OR); }
	;

assignment_expression
	: conditional_expression { $$ = $1; }
  | unary_expression assignment_operator assignment_expression { $$ = new assignment_expression($1, get_assign_op($2), $3); }
	;

/* Expression */
expression
	: assignment_expression { $$ = $1; }
	;

/* Initialization */
initialization_expression
  : assignment_expression { $$ = $1; }
  | '{' constant '}' { $$ = $2; }
  ;


/* -------------------------- */
/*         Statements         */
/* -------------------------- */

statement
	: compound_statement { $$ = $1; }
	| expression_statement { $$ = $1; }
	| selection_statement { $$ = $1; }
	| iteration_statement { $$ = $1; }
	;

compound_statement
  : '{' '}' { $$ = new compound_statement(nullptr, nullptr); }
  | '{' statement_list '}' { $$ = new compound_statement(nullptr, $2); }
  | '{' declaration_list '}' { $$ = new compound_statement($2, nullptr); }
  | '{' declaration_list statement_list '}' { $$ = new compound_statement($2, $3);}
	;


declaration_list
  : declaration { $$ = new list<declaration*>((declaration*)$1); }
  | declaration_list declaration { $$ = append_ptr_list<declaration>($1, $2); }

statement_list
  : statement { $$ = new list<statement*>((statement*)$1); }
  | statement_list statement { $$ = append_ptr_list<statement>($1, $2); }
	;
	
expression_statement
	: ';' { $$ = new no_op(); }
  | expression ';' { $$ = new expression_statement($1); }
	;

selection_statement
  : IF '(' expression ')' statement { $$ = new selection_statement($3, $5); }
  | IF '(' expression ')' statement ELSE statement { $$ = new selection_statement($3, $5, $7); }
	;

iteration_statement
  : FOR '(' expression_statement expression_statement expression ')' statement { $$ = new iteration_statement($3, $4, $5, $7); }
	;


/* -------------------------- */
/*         Declarator         */
/* -------------------------- */


direct_declarator
  : identifier { $$ = $1; }
  | identifier '[' constant_list ']' { $$ = new tile($1, $3); }
  | identifier '(' parameter_list ')' { $$ = new function($1, $3); }
  | identifier '(' ')' { $$ = new function($1, nullptr); }
  ;


parameter_list
	: parameter_declaration { $$ = new list<parameter*>((parameter*)$1); }
  | parameter_list ',' parameter_declaration { $$ = append_ptr_list<parameter>($1, $3); }
	;

parameter_declaration
  : declaration_specifiers declarator { $$ = new parameter($1, $2); }
  | declaration_specifiers abstract_declarator { $$ = new parameter($1, $2); }
	;


declaration_specifiers
  : type_specifier { $$ = new declaration_specifier(get_type_spec($1)); }
	;

init_declarator_list
  : init_declarator { $$ = new list<initializer*>((initializer*)$1); }
  | init_declarator_list ',' init_declarator { $$ = append_ptr_list<initializer>($1, $3); }
	;

declaration
	: declaration_specifiers ';' { $$ = new declaration($1, nullptr); }
	| declaration_specifiers init_declarator_list ';' { $$ = new declaration($1, $2); }
	;
	
declarator
  : pointer direct_declarator { $$ = ((declarator*)$2)->set_ptr($1); }
	| direct_declarator { $$ = $1; }
	;


init_declarator
  : declarator { $$ = new initializer($1, nullptr); }
  | declarator '=' initialization_expression { $$ = new initializer($1, $3); }
	;

/* -------------------------- */
/*      Translation Unit 	  */
/* -------------------------- */

translation_unit
  : external_declaration { ast_root = new translation_unit($1); $$ = ast_root; }
	| translation_unit external_declaration { $$ = ((translation_unit*)($1))->add($2); }
	;
	
external_declaration
  : function_definition { $$ = $1; }
  | declaration { $$ = $1; }
  ;
	
function_definition
  : declaration_specifiers declarator compound_statement { $$ = new function_definition($1, $2, $3); }
	;

