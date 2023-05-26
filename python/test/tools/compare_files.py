import argparse
import glob
import os

import yaml


def list_files_with_extension(path, extension):
    files = glob.glob(os.path.join(path, f'*.{extension}'))
    return files


def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        content = yaml.safe_load(file)
    return content


def compare_files(file1, file2):
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        content1 = f1.read()
        content2 = f2.read()

    return content1 == content2


def get_json_file(path):
    # get json file in directory
    json_files = list_files_with_extension(path, "json")
    if len(json_files) == 0:
        return None
    # filter out json files with grp in their name
    json_files = [f for f in json_files if "grp" not in f]
    assert len(json_files) == 1, f"Found {len(json_files)} files in {path} with extension json!"
    json_file = os.path.join(path, json_files[0])
    return json_file


def get_name_to_hashes_dict(path):
    name_to_hashes = {}
    for hash in os.listdir(path):
        full_path = os.path.join(path, hash)
        assert os.path.isdir(full_path), f"Path {full_path} is not a directory!"
        json_file = get_json_file(full_path)
        if json_file is None:
            continue
        # load json file
        with open(json_file, 'r') as file:
            content = yaml.safe_load(file)
            # get name
            name = content["name"]
            name_to_hashes.setdefault(name, []).append(hash)
    return name_to_hashes


def compare_matching_files(name, extension, name_to_hashes1, name_to_hashes2, args):
    hashes1 = name_to_hashes1[name]
    hashes2 = name_to_hashes2[name]
    for hash1 in hashes1:
        json1 = get_json_file(os.path.join(args.path1, hash1))
        assert json1 is not None
        for hash2 in hashes2:
            json2 = get_json_file(os.path.join(args.path2, hash2))
            assert json2 is not None
            if not compare_files(json1, json2):
                continue
            ext_file1 = list_files_with_extension(os.path.join(args.path1, hash1), extension)[0]
            ext_file2 = list_files_with_extension(os.path.join(args.path2, hash2), extension)[0]
            if not compare_files(ext_file1, ext_file2):
                return (ext_file1, ext_file2)
    return ()


def main(args):
    assert args.path1 != args.path2, "Cannot compare files in the same directory!"
    name_to_hashes1 = get_name_to_hashes_dict(args.path1)
    name_to_hashes2 = get_name_to_hashes_dict(args.path2)

    yaml_file_path = args.kernels
    assert os.path.exists(yaml_file_path), f"Path {yaml_file_path} does not exist!"
    name_and_extension = load_yaml_file(yaml_file_path)["name_and_extension"]

    mismatches = {}
    for d in name_and_extension:
        name = d["name"]
        extension = d["extension"]
        result = compare_matching_files(name, extension, name_to_hashes1, name_to_hashes2, args)
        if len(result) > 0:
            mismatches[name] = result
    if len(mismatches) > 0:
        print(f"Found {len(mismatches)} mismatches:")
        for name, mismatch in mismatches.items():
            print(f"{name}: {mismatch}")
        assert False, "Found mismatches!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path1",
        type=str,
        default=None,
        required=True,
        help=("Path to first cache directory"),
    )
    parser.add_argument(
        "--path2",
        type=str,
        default=None,
        required=True,
        help=("Path to second cache directory"),
    )
    parser.add_argument(
        "--kernels",
        type=str,
        default=None,
        required=True,
        help=("Path to kernels yaml file"),
    )
    args = parser.parse_args()
    main(args)
