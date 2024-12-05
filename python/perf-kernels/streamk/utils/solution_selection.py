import numpy as np
import yaml
from sklearn.neighbors import KDTree
from collections import defaultdict


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


# Path to the YAML file
yaml_file_path = './utils/tuned.yaml'

# Load YAML content from the file
data = read_yaml_file(yaml_file_path)

# Define the keys to extract
keys_to_extract = [
    'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'GROUP_SIZE_M', 'num_warps', 'num_stages', 'waves_per_eu',
    'matrix_instr_nonkdim', 'kpack'
]

# Dictionary to keep track of duplicates based on keys_to_extract
duplicate_entries = defaultdict(list)

# Extract values and assign new solution index
for entry in data:
    extracted_values = tuple(entry[key] for key in keys_to_extract)
    m_n_k = (entry['M'], entry['N'], entry['K'])
    duplicate_entries[extracted_values].append(m_n_k)

# Create a new list with unique key combinations and their solution index
unique_keys_list = list(duplicate_entries.keys())
unique_solution_index = {keys: idx for idx, keys in enumerate(unique_keys_list)}

# Print the maximum solution index
max_solution_index = len(unique_solution_index) - 1
print(f"Maximum solution index: {max_solution_index}")

# Create sizes list with (M, N, K) and the new solution index
sizes = []
for entry in data:
    extracted_values = tuple(entry[key] for key in keys_to_extract)
    solution_index = unique_solution_index[extracted_values]
    sizes_entry = {'M': entry['M'], 'N': entry['N'], 'K': entry['K'], 'SolutionIndex': solution_index}
    sizes.append(sizes_entry)

# Create solution_params list with unique key combinations
solution_params = [dict(zip(keys_to_extract, keys)) for keys in unique_keys_list]

searchpoints = np.array([(tsol['M'], tsol['N'], tsol['K']) for tsol in sizes])
tunedtree = KDTree(searchpoints)
tunedarr = np.array([tsol['SolutionIndex'] for tsol in sizes])
