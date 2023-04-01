import ast
from typing import Optional, Union


class CompilationError(Exception):
    source_line_count_max_in_message = 12

    def _format_message(self) -> str:
        node = self.node
        if self.src is None:
            source_excerpt = " <source unavailable>"
        else:
            source_excerpt = self.src.split('\n')[:node.lineno][-self.source_line_count_max_in_message:]
            if source_excerpt:
                source_excerpt.append(' ' * node.col_offset + '^')
                source_excerpt = '\n'.join(source_excerpt)
            else:
                source_excerpt = " <source empty>"

        message = "at {}:{}:{}".format(node.lineno, node.col_offset, source_excerpt)
        if self.error_message:
            message += '\n' + self.error_message
        return message

    def __init__(self, src: Optional[str], node: ast.AST, error_message: Union[str, None]):
        self.src = src
        self.node = node
        self.error_message = error_message
        self.message = self._format_message()

    def set_source_code(self, src: Optional[str]):
        self.src = src
        self.message = self._format_message()

    def __str__(self):
        return self.message

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self.message)

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return type(self), (self.src, self.node, self.error_message)


class CompileTimeAssertionFailure(CompilationError):
    """Specific exception for failed tests in `static_assert` invocations"""
    pass


class UnsupportedLanguageConstruct(CompilationError):
    pass
