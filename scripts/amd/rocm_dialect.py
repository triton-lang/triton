import tree_sitter
from shutil import copytree
from os import path, walk

# Load the C++ language parser
LANGUAGE = tree_sitter.Language('tree-sitter-cpp/build', 'cpp')
PARSER = tree_sitter.Parser()
PARSER.set_language(LANGUAGE)

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()

    # Parse the code
    tree = PARSER.parse(bytes(code, 'utf8'))

    # Iterate through the syntax tree and find macro definitions
    cursor = tree.walk()
    modifications = []
    while True:
        if cursor.node.type == 'preproc_def':
            start_byte = cursor.node.start_byte
            end_byte = cursor.node.end_byte
            old_macro = code[start_byte:end_byte]
            new_macro = old_macro.replace('OLD_MACRO', 'NEW_MACRO')  # Adjust as needed
            modifications.append((start_byte, end_byte, new_macro))
        if not cursor.goto_next_sibling():
            if not cursor.goto_parent():
                break  # Break when we've visited the entire tree

    # Apply modifications in reverse order (to not mess up the offsets)
    for start_byte, end_byte, new_macro in reversed(modifications):
        code = code[:start_byte] + new_macro + code[end_byte:]

    # Write the modified code back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(code)

def main():
    src_dir = '/path/to/source/directory'
    dest_dir = '/path/to/destination/directory'

    # Copy the source directory to the destination directory
    copytree(src_dir, dest_dir)

    # Process each file in the destination directory
    for root, _, files in walk(dest_dir):
        for file in files:
            if file.endswith('.cpp') or file.endswith('.h'):
                process_file(path.join(root, file))

if __name__ == '__main__':
    main()
