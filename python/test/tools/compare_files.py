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

    assert content1 == content2, f"Files {file1} and {file2} are not identical."


def main(args):
    yaml_file_path = args.hash
    assert os.path.exists(yaml_file_path), f"Path {yaml_file_path} does not exist!"
    hashes_and_extensions = load_yaml_file(yaml_file_path)["hashes_and_extensions"]

    assert args.path1 != args.path2, "Cannot compare files in the same directory!"

    for d in hashes_and_extensions:
        hash = d["hash"]
        extension = d["extension"]
        full_path1 = os.path.join(args.path1, hash)
        full_path2 = os.path.join(args.path2, hash)
        assert os.path.exists(full_path1), f"Path {full_path1} does not exist!"
        assert os.path.exists(full_path2), f"Path {full_path2} does not exist!"
        files1 = list_files_with_extension(full_path1, extension)
        files2 = list_files_with_extension(full_path2, extension)
        assert len(files1) == 1, f"Found {len(files1)} files in {full_path1} with extension {extension}!"
        assert len(files2) == 1, f"Found {len(files2)} files in {full_path2} with extension {extension}!"
        file1 = os.path.join(full_path1, files1[0])
        file2 = os.path.join(full_path2, files2[0])
        compare_files(file1, file2)


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
        "--hash",
        type=str,
        default=None,
        required=True,
        help=("Path to hash file"),
    )
    args = parser.parse_args()
    main(args)
