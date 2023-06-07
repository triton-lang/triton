import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import yaml


def listFilesWithExtension(path: str, extension: str) -> List[str]:
    files = glob.glob(os.path.join(path, f'*.{extension}'))
    return files


def getFileWithExtension(path: str, ext: str) -> Optional[str]:
    # get all files in directory with extension
    files = listFilesWithExtension(path, ext)
    if len(files) == 0:
        return None
    # filter out files with grp in their name
    files = [f for f in files if "__grp__" not in f]
    assert len(files) == 1, f"Found {len(files)} files in {path} with extension {ext}!"
    return files[0]


def loadYamlFile(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, 'r') as file:
        content = yaml.safe_load(file)
    return content


def compareFiles(file1: str, file2: str) -> bool:
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        content1 = f1.read()
        content2 = f2.read()

    return content1 == content2


def getNameToHashesDict(path: str) -> Dict[str, List[str]]:
    name_to_hashes = {}
    for hash in os.listdir(path):
        full_path = os.path.join(path, hash)
        assert os.path.isdir(full_path), f"Path {full_path} is not a directory!"
        json_file = getFileWithExtension(full_path, "json")
        if json_file is None:
            continue
        # load json file
        with open(json_file, 'r') as file:
            content = yaml.safe_load(file)
            # get name
            name = content["name"]
            name_to_hashes.setdefault(name, []).append(hash)
    return name_to_hashes


def getFileVec(path: str) -> List[Tuple[str, str]]:
    vec = []
    for ext in ["json", "ttir", "ttgir"]:
        file = getFileWithExtension(path, ext)
        if file is not None:
            vec.append((ext, file))
    return vec


def do_files_match(path1: str, path2: str) -> bool:
    filesVec1 = getFileVec(path1)
    filesVec2 = getFileVec(path2)
    # both hashes must at least have 1 json file and 1 ttir or ttgir file
    # and the number of files must match
    if len(filesVec1) <= 1 or len(filesVec2) <= 1 or len(filesVec1) != len(filesVec2):
        return False
    # if either path does not have a json file, return false
    if filesVec1[0][0] != "json" or filesVec2[0][0] != "json":
        return False

    for (ext1, file1), (ext2, file2) in zip(filesVec1, filesVec2):
        if ext1 != ext2:
            return False
        if not compareFiles(file1, file2):
            return False
        else:
            # once we actually compared a ttir or ttgir file, we can break
            if ext1 in ("ttir", "ttgir"):
                break
    return True


def compare_matching_files(name: str, extension: str, name_to_hashes1: Dict[str, List[str]], name_to_hashes2: Dict[str, List[str]], args) -> Tuple[str, str]:
    hashes1 = name_to_hashes1[name]
    hashes2 = name_to_hashes2[name]
    numComparisons = 0
    for hash1 in hashes1:
        path1 = os.path.join(args.path1, hash1)
        for hash2 in hashes2:
            path2 = os.path.join(args.path2, hash2)
            # check whether both paths have:
            # 1. json files that match
            # 2. ttir files that match (if they exist), otherwise ttgir files that match (if they exist)
            # if any of these contraints is not met, then we can skip this pair of hashes since they are not a match
            if not do_files_match(path1, path2):
                continue
            numComparisons += 1
            ext_file1 = listFilesWithExtension(path1, extension)[0]
            ext_file2 = listFilesWithExtension(path2, extension)[0]
            if not compareFiles(ext_file1, ext_file2):
                return (ext_file1, ext_file2)
    assert numComparisons > 0, f"Did not find any matching files for {name}"
    return ()


def main(args) -> None:
    assert args.path1 != args.path2, "Cannot compare files in the same directory!"
    # Get kernel name to hashes dict, these hashes would have the same kernel name
    nameToHashes1 = getNameToHashesDict(args.path1)
    nameToHashes2 = getNameToHashesDict(args.path2)

    yamlFilePath = args.kernels
    assert os.path.exists(yamlFilePath), f"Path {yamlFilePath} does not exist!"
    nameAndExtension = loadYamlFile(yamlFilePath)["name_and_extension"]

    mismatches = {}
    # iterate over the kernels that need to be checked
    for d in nameAndExtension:
        name = d["name"]  # kernel name
        extension = d["extension"]  # extension of the file to be compared (e.g. ptx)
        # Compare all hashes on path 1 with all hashes on path 2
        # result is either the mismatching (file1, file2) with "extension" or empty tuple if no mismatch
        result = compare_matching_files(name, extension, nameToHashes1, nameToHashes2, args)
        # If mismatch exists, add it to mismatches dict
        if len(result) > 0:
            mismatches[name] = result
    # If mismatches dict is not empty, print the mismatches and assert false
    if len(mismatches) > 0:
        print(f"Found {len(mismatches)} mismatches:")
        for name, mismatch in mismatches.items():
            print(f"{name}: {mismatch}")
        assert False, "Found mismatches!"
    else:
        print("No mismatches found!")


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
