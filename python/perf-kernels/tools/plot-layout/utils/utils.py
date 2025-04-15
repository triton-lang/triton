import argparse
import subprocess


def run_bash_command(commandstring):
    proc = subprocess.run(commandstring, shell=True, check=False, executable='/bin/bash', stdout=subprocess.PIPE)
    return proc.returncode


class OneLineFormatter(argparse.HelpFormatter):

    def _format_action_invocation(self, action):
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar

        parts = []

        if action.nargs == 0:
            parts.extend(action.option_strings)
        else:
            default_metavar = self._get_default_metavar_for_optional(action)
            args_string = self._format_args(action, default_metavar)
            parts.extend([f'{opt} {args_string}' for opt in action.option_strings])

        return ', '.join(parts)

    def _get_help_string(self, action):
        help_text = action.help or ''
        if '%(default)' not in help_text and action.default is not argparse.SUPPRESS:
            if action.default is not None:
                help_text += f' (default: {action.default})'
        return help_text
