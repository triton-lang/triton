import os
import argparse
import re
from collections import OrderedDict
import graphviz


class Options:

    def __init__(self, input_file, output_file, verbose, format):
        if not os.path.exists(input_file):
            raise RuntimeError('input file is not provided')

        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            raise RuntimeError('output directory does not exist')

        self.input_file = input_file
        self.output_file = output_file
        self.verbose = verbose
        self.format = format
        self.output_dir = output_dir


class Block:

    def __init__(self, label, code):
        self.label = label
        self.code = code
        self.edges = []


class Kernel:

    def __init__(self, kernel_name, blocks):
        self.name = kernel_name
        self.blocks = blocks
        self.cfg = None


begin_label = 'Begin'
end_label = 'End'


def find_kernel(text):
    func_name_expr = r'^([^\s^\.]\w.+):'
    func_name = None
    start = None
    for index, line in enumerate(text):
        match = re.search(func_name_expr, line)
        if match is not None:
            func_name = match[1]
            start = index
            break
    if start is None:
        return None, None, None

    end = None
    for index, line in enumerate(text):
        if re.search(r's_endpgm', line) is not None:
            end = index
            break

    if end is None:
        return None, None, None

    return func_name, text[start:end + 1], end


def find_label(kernel):
    label = None
    index = None
    for index, line in enumerate(kernel):
        match = re.search(r'^\.(\w+):', line)
        if match is not None:
            label = match[1]
            break
    return label, index


def get_block_list(kernel):
    label, index = find_label(kernel)

    blocks = OrderedDict()
    if (index > 1):
        blocks[begin_label] = Block(begin_label, kernel[:index - 1])

    while label is not None:
        kernel = kernel[index + 1:]
        next_label, next_index = find_label(kernel)
        if next_label is None:
            code = kernel[index:]
        else:
            code = kernel[:next_index]
        blocks[label] = Block(label, code)

        label = next_label
        index = next_index

    blocks[end_label] = Block(end_label, [])

    return blocks


def find_terminators(code):
    terminator_labels = []
    for line in code:
        branch = re.search(r'(c)?branch.*\s+\.?(.*)', line)
        if branch is not None:
            is_condional = True if len(branch.groups()) == 2 else False
            label_idx = 2 if is_condional else 1
            terminator_labels.append(branch[label_idx])
            if not is_condional:
                return terminator_labels, True
        end = re.search(r's_endpgm', line)
        if end is not None:
            terminator_labels.append(end_label)
            return terminator_labels, True

    return terminator_labels, False


def add_edges(kernel):
    keys = list(kernel.blocks.keys())
    for index, curr_label in enumerate(keys):
        if curr_label == end_label:
            continue

        code = kernel.blocks[curr_label].code
        terminators, is_last_unconditional = find_terminators(code[:-1])

        if is_last_unconditional:
            # unconditional jump in the middle of the block
            break

        # handle the last terminator in the current BB
        last_terminator, is_unconditional = find_terminators([code[-1]])

        is_conditional = not is_unconditional
        next_block_label = keys[index + 1]
        is_next_covered = next_block_label in terminators

        if last_terminator:
            terminators.extend(last_terminator)
            if is_conditional and not is_next_covered:
                next_block_label = keys[index + 1]
                terminators.append(next_block_label)
        else:
            if not is_next_covered:
                next_block_label = keys[index + 1]
                terminators.append(next_block_label)

        assert (len(terminators))
        kernel.blocks[curr_label].edges = terminators


def generate_cfg(kernel, options):
    graph = graphviz.Digraph(f'{kernel.name}')
    for curr_label in kernel.blocks:
        block = kernel.blocks[curr_label]
        asm = [line.strip() for line in block.code]
        if options.verbose:
            label_text = repr('\n'.join([f'{curr_label}', *asm]))
        else:
            label_text = curr_label
        graph.node(curr_label, shape='rect', labeljust='l', margin='0.01', label=label_text)

    for curr_label in kernel.blocks:
        block = kernel.blocks[curr_label]
        for edge in block.edges:
            graph.edge(curr_label, edge)

    return graph


def main(options):
    asm = []
    with open(options.input_file, 'r') as file:
        context = file.readlines()
        for line in context:
            asm.append(line[:-1])

    kernels = []
    last_end_index = 0
    while last_end_index is not None:
        func_name, kernel_asm, last_end_index = find_kernel(asm)
        if kernel_asm is None:
            break

        blocks = get_block_list(kernel_asm)
        kernel = Kernel(func_name, blocks)
        add_edges(kernel)

        cfg = generate_cfg(kernel, options)
        kernel.cfg = cfg
        kernels.append(kernel)
        asm = asm[last_end_index + 1:]

        for index, kernel in enumerate(kernels):
            output_file_name = f'{options.output_file}.kernel-{index}'
            if options.format == 'dot':
                with open(f'{output_file_name}.dot', 'w') as file:
                    file.write(str(kernel.cfg))
                    file.write('\n')
            else:
                kernel.cfg.render(
                    filename=f'{output_file_name}',
                    format=options.format,
                ).replace('\\', '/')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Generates Control Flow Graph (CFG) from amdgcn assembly file", )
    parser.add_argument("-i", "--input", type=str, default=None, help="input file")
    parser.add_argument("-o", "--output", type=str, default=None, help="output file prefix")
    parser.add_argument("-v", "--verbose", action='store_true', help='verbose output')
    parser.add_argument("-f", "--format", choices=['dot', 'svg', 'pdf'], default="dot", help="output format type")
    args = parser.parse_args()

    options = Options(args.input, args.output, args.verbose, args.format)

    main(options)
