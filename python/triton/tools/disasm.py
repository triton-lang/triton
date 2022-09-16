# MIT License

# Copyright (c) 2020 Da Yan @ HKUST

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
import subprocess

FLINE_RE = re.compile(r'\s*/\*\w{4}\*/\s*([^;]*;)\s*/\* 0x(\w{16}) \*/\s*')
SLINE_RE = re.compile(r'\s*/\* 0x(\w{16}) \*/\s*')
FNAME_RE = re.compile(r'\s*Function : (\w+)\s*')
BRA_RE = re.compile(r'(.*BRA(?:\.U)? )(0x\w+);')


def parseCtrl(sline):
    enc = int(SLINE_RE.match(sline).group(1), 16)
    stall = (enc >> 41) & 0xf
    yld = (enc >> 45) & 0x1
    wrtdb = (enc >> 46) & 0x7
    readb = (enc >> 49) & 0x7
    watdb = (enc >> 52) & 0x3f

    yld_str = 'Y' if yld == 0 else '-'
    wrtdb_str = '-' if wrtdb == 7 else str(wrtdb)
    readb_str = '-' if readb == 7 else str(readb)
    watdb_str = '--' if watdb == 0 else f'{watdb:02d}'
    return f'{watdb_str}:{readb_str}:{wrtdb_str}:{yld_str}:{stall:x}'


def processSassLines(fline, sline, labels):
    asm = FLINE_RE.match(fline).group(1)
    # Remove tailing space
    if asm.endswith(" ;"):
        asm = asm[:-2] + ";"
    ctrl = parseCtrl(sline)
    # BRA target address
    if BRA_RE.match(asm) is not None:
        target = int(BRA_RE.match(asm).group(2), 16)
        if target in labels:
            pass
        else:
            labels[target] = len(labels)
    return (f'{ctrl}', f'{asm}')


def extract(file_path, fun):
    if fun is None:
        sass_str = subprocess.check_output(["cuobjdump", "-sass", file_path])
    else:
        sass_str = subprocess.check_output(["cuobjdump", "-fun", fun, "-sass", file_path])
    sass_lines = sass_str.splitlines()
    line_idx = 0
    while line_idx < len(sass_lines):
        line = sass_lines[line_idx].decode()
        # format:
        # function : <function_name>
        # .headerflags: ...
        # /*0000*/ asmstr /*0x...*/
        #                 /*0x...*/

        # Looking for new function header (function: <name>)
        while FNAME_RE.match(line) is None:
            line_idx += 1
            if line_idx < len(sass_lines):
                line = sass_lines[line_idx].decode()
            else:
                return

        fname = FNAME_RE.match(line).group(1)
        ret = ''
        ret += f'Function:{fname}\n'
        line_idx += 2  # bypass .headerflags
        line = sass_lines[line_idx].decode()
        # Remapping address to label
        labels = {}  # address -> label_idx
        # store sass asm in buffer and them print them (for labels)
        # (ctrl, asm)
        asm_buffer = []
        while FLINE_RE.match(line) is not None:
            # First line (Offset ASM Encoding)
            fline = sass_lines[line_idx].decode()
            line_idx += 1
            # Second line (Encoding)
            sline = sass_lines[line_idx].decode()
            line_idx += 1
            asm_buffer.append(processSassLines(fline, sline, labels))
            # peek the next line
            line = sass_lines[line_idx].decode()
        # Print sass
        # label naming convention: LBB#i
        for idx, (ctrl, asm) in enumerate(asm_buffer):
            # Print label if this is BRA target
            offset = idx * 16
            if offset in labels:
                label_name = f'LBB{labels[offset]}'
                ret += f'{label_name}:\n'
            ret += ctrl + '\t'
            # if this is BRA, remap offset to label
            if BRA_RE.match(asm):
                target = int(BRA_RE.match(asm).group(2), 16)
                target_name = f'LBB{labels[target]}'
                asm = BRA_RE.sub(rf'\1{target_name};', asm)
            ret += asm + '\n'
        ret += '\n'
        return ret
