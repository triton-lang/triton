import os
import csv

def is_instruction(line):
    # rm whitespace
    line = line.strip()
    if not line:
        return False
    
    # starts with a letter, excluding:
    # comments (//), directives (.), labels ($)
    if line.startswith('@'):
        return True
    
    if not line[0].isalpha():
        return False
        
    if line.endswith(':'):
        return False
        
    return True

def main():
    root_dir = "/root/wkspace/triton/ptx_dump_all"
    output_file = "instructions.csv"
    
    unique_instructions = {}

    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} does not exist.")
        return

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".ptx"):
                filepath = os.path.join(dirpath, filename)
                # Use relative path for the filename column
                rel_filename = os.path.relpath(filepath, root_dir)
                
                try:
                    with open(filepath, 'r') as f:
                        for line_idx, line in enumerate(f, 1):
                            if is_instruction(line):
                                clean_line = line.strip()
                                # Extract instruction opcode (first word)
                                if clean_line.split()[0].startswith('@'):
                                    # e.g., @%p7 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r14], [%rd3, {%r21, %r125}], [%r19];
                                    # remove first token start with '@', end with space
                                    clean_line = ' '.join(clean_line.split()[1:])
                                opcode = clean_line.split()[0]
                                if opcode not in unique_instructions:
                                    unique_instructions[opcode] = {
                                        'instruction': clean_line,
                                        'file': rel_filename,
                                        'line': line_idx
                                    }
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

    instructions_list = list(unique_instructions.values())

    instructions_list.sort(key=lambda x: x['instruction'].split()[0])

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['instruction', 'file', 'line']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)

            writer.writeheader()
            for instr in instructions_list:
                writer.writerow(instr)

        print(f"Extracted {len(instructions_list)} unique instructions to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    main()
