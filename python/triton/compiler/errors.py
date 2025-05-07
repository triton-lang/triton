import ast
from typing import Optional
from ..errors import TritonError


class CompilationError(TritonError):
    """Base class for all errors raised during compilation"""
    source_line_count_max_in_message = 12

    def _format_message(self) -> str:
        node = self.node
        if self.src is None:
            source_excerpt = " <source unavailable>"
        else:
            if hasattr(node, 'lineno'):
                source_excerpt = self.src.split('\n')[:node.lineno][-self.source_line_count_max_in_message:]
                if source_excerpt:
                    source_excerpt.append(' ' * node.col_offset + '^')
                    source_excerpt = '\n'.join(source_excerpt)
                else:
                    source_excerpt = " <source empty>"
            else:
                source_excerpt = self.src

        message = "at {}:{}:\n{}".format(node.lineno, node.col_offset, source_excerpt) if hasattr(
            node, 'lineno') else source_excerpt
        if self.error_message:
            message += '\n' + self.error_message
        return message

    def __init__(self, src: Optional[str], node: ast.AST, error_message: Optional[str] = None):
        self.src = src
        self.node = node
        self.error_message = error_message
        self.message = self._format_message()

    def __str__(self):
        return self.message

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return type(self), (self.src, self.node, self.error_message)


class CompileTimeAssertionFailure(CompilationError):
    """Specific exception for failed tests in `static_assert` invocations"""
    pass


class UnsupportedLanguageConstruct(CompilationError):
    pass

class MLIRCompilationError(TritonError):
    def __init__(self, stage_name: Optional[str], message: Optional[str] = None):
        self.stage_name = stage_name
        self.message = f"\n" \
            f"{self.format_line_delim('[ERROR][Triton][BEG]')}" \
            f"[{self.stage_name}] encounters error:\n" \
            f"{self.filter_message(message)}" \
            f"{self.format_line_delim('[ERROR][Triton][END]')}"
    def __str__(self):
        return self.message
    def filter_message(self, message):
        # Content starting from "Stack dump without symbol names" means nothing to the users
        return message.split("Stack dump without symbol names")[0]
    def format_line_delim(self, keyword):
        return f"///------------------{keyword}------------------\n"
