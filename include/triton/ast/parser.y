%define parse.error verbose

%{
namespace triton{
namespace ast{
class node;
}
}
using namespace triton::ast;
#define YYSTYPE node*
#include "../include/triton/ast/ast.h"

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
  token(STORAGE_SPEC_T value): storage_spec(value){ }

  union {
    ASSIGN_OP_T assign_op;
    BIN_OP_T bin_op;
    UNARY_OP_T unary_op;
    TYPE_T type;
    STORAGE_SPEC_T storage_spec;
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
STORAGE_SPEC_T get_storage_spec(node *op) { return ((token*)op)->storage_spec;}
%}

%token IDENTIFIER CONSTANT STRING_LITERAL
%token TUNABLE KERNEL RESTRICT READONLY WRITEONLY CONST CONSTANT_SPACE
%token PTR_OP INC_OP DEC_OP LEFT_OP RIGHT_OP LE_OP GE_OP EQ_OP NE_OP
%token AND_OP OR_OP MUL_ASSIGN DIV_ASSIGN MOD_ASSIGN ADD_ASSIGN
%token SUB_ASSIGN LEFT_ASSIGN RIGHT_ASSIGN AND_ASSIGN
%token XOR_ASSIGN OR_ASSIGN TYPE_NAME
%token VOID UINT1 UINT8 UINT16 UINT32 UINT64 INT1 INT8 INT16 INT32 INT64 FP32 FP64
%token IF ELSE FOR CONTINUE
%token NEWAXIS ELLIPSIS AT
%token GET_GLOBAL_RANGE DOT ALLOC_CONST

%start translation_unit
%%


/* -------------------------- */
/*         Types              */
/* -------------------------- */

type_specifier
  : VOID { $$ = new token(VOID_T); }
  | UINT1 { $$ = new token(UINT1_T); }
  | UINT8 { $$ = new token(UINT8_T); }
  | UINT16 { $$ = new token(UINT16_T); }
  | UINT32 { $$ = new token(UINT32_T); }
  | UINT64 { $$ = new token(UINT64_T); }
  | INT1 { $$ = new token(INT1_T);}
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
    : '[' primary_expression_list ']' { $$ = new tile(nullptr, $1); }

constant:
    CONSTANT { $$ = new constant(atoi(yytext)); }
	;

constant_list:
      constant { $$ = new list<constant*>((constant*)$1); }
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

builtin
  : GET_GLOBAL_RANGE '[' primary_expression ']' '(' constant ')' { $$ = new get_global_range($3, $6); }
  | DOT '(' expression ',' expression ',' expression ')' { $$ = new matmul_expression($3, $5, $7); }
  | ALLOC_CONST type_specifier '[' constant ']' { $$ = new alloc_const(new typed_declaration_specifier(get_type_spec($2)), $4); }

primary_expression
  : identifier         { $$ = new named_expression($1); }
  | constant           { $$ = $1; }
  | primary_expression ELLIPSIS primary_expression { $$ = new constant_range($1, $3); }
  | builtin            { $$ = $1; }
  | STRING_LITERAL     { $$ = new string_literal(yytext); }
  | '(' expression ')' { $$ = $2; }
	;

primary_expression_list
  : primary_expression { $$ = new list<expression*>((expression*)$1); }
  | primary_expression_list ',' primary_expression { $$ = append_ptr_list<expression>($1, $3); }
  ;

slice
  : ':' { $$ = new slice(triton::ast::ALL); }
  | NEWAXIS { $$ = new slice(triton::ast::NEWAXIS); }

slice_list
  : slice { $$ = new list<slice*>((slice*)$1); }
  | slice_list ',' slice { $$ = append_ptr_list<slice>($1, $3); }

postfix_expression
  : primary_expression { $$ = $1;}
  | identifier '[' slice_list ']' { $$ = new indexing_expression($1, $3);}
  ;

unary_expression
  : postfix_expression             { $$ = $1; }
  | INC_OP unary_expression        { $$ = new unary_operator(INC, $2); }
  | DEC_OP unary_expression        { $$ = new unary_operator(DEC, $2); }
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
  | '(' type_name ')' cast_expression { $$ = new cast_operator($2, $4); }
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
  | logical_or_expression '?' conditional_expression ':' conditional_expression { $$ = new conditional_expression($1, $3, $5); }
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
  | '{' constant_list '}' { $$ = $2; }
  ;


/* -------------------------- */
/*         Statements         */
/* -------------------------- */

statement
	: compound_statement { $$ = $1; }
	| expression_statement { $$ = $1; }
	| selection_statement { $$ = $1; }
	| iteration_statement { $$ = $1; }
  | jump_statement { $$ = $1; }
	;

compound_statement
    : '{' '}' { $$ = new compound_statement(nullptr); }
    | '{' block_item_list '}' { $$ = new compound_statement($2); }

block_item_list
  : block_item { $$ = new list<block_item*>((block_item*)$1); }
  | block_item_list block_item { $$ = append_ptr_list<block_item>($1, $2); }

block_item
  : declaration { $$ = $1; }
  | statement { $$ = $1; }

expression_statement
	: ';' { $$ = new no_op(); }
  | expression ';' { $$ = new expression_statement($1); }
  | AT primary_expression expression ';' { $$ = new expression_statement($3, $2); }
  ;

selection_statement
  : IF '(' expression ')' statement { $$ = new selection_statement($3, $5); }
  | IF '(' expression ')' statement ELSE statement { $$ = new selection_statement($3, $5, $7); }
	;

iteration_statement
  : FOR '(' expression_statement expression_statement expression ')' statement { $$ = new iteration_statement($3, $4, $5, $7); }
  | FOR '(' declaration expression_statement ')' statement { $$ = new iteration_statement($3, $4, nullptr, $6); }
  | FOR '(' declaration expression_statement expression ')' statement { $$ = new iteration_statement($3, $4, $5, $7); }
	;

jump_statement
  : CONTINUE ';'          { $$ = new continue_statement(); }
;

/* -------------------------- */
/*         Declarator         */
/* -------------------------- */


direct_declarator
  : identifier { $$ = $1; }
  | identifier '[' primary_expression_list ']' { $$ = new tile($1, $3); }
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
  : type_specifier { $$ = new typed_declaration_specifier(get_type_spec($1)); }
  | storage_class_specifier declaration_specifiers { $$ = new storage_declaration_specifier(get_storage_spec($1), $2); }
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

storage_class_specifier
  : CONST     { $$ = new token(CONST_T); }
  | TUNABLE   { $$ = new token(TUNABLE_T); }
  | KERNEL    { $$ = new token(KERNEL_T); }
  | RESTRICT  { $$ = new token(RESTRICT_T); }
  | READONLY  { $$ = new token(READONLY_T); }
  | WRITEONLY { $$ = new token(WRITEONLY_T); }
  | CONSTANT_SPACE  { $$ = new token(CONSTANT_SPACE_T); }
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

%%
void yyerror (const char *s){
  print_error(s);
}
