#pragma once

#ifndef _WGTCC_ERROR_H_
#define _WGTCC_ERROR_H_


struct SourceLocation;
class Token;
class Expr;


[[noreturn]] void Error(const char* format, ...);
[[noreturn]] void Error(const SourceLocation& loc, const char* format, ...);
[[noreturn]] void Error(const Token* tok, const char* format, ...);
[[noreturn]] void Error(const Expr* expr, const char* format, ...);

#endif
