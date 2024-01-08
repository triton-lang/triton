import argparse
import difflib
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import yaml


class ComparisonResult:

    def __init__(self, name: str, numComparisons: int, diffs: List[str] = None, errors: List[str] = None):
        self.name = name
        self.numComparisons = numComparisons
        self.diffs = [] if diffs is None else diffs
        self.errors = [] if errors is None else errors

    def isSuccess(self) -> bool:
        return len(self.diffs) == 0 and len(self.errors) == 0

    def __str__(self) -> str:
        return f"name={self.name}, numComparisons={self.numComparisons}, success={self.isSuccess()}"


def listFilesWithExtension(path: str, extension: str) -> List[str]:
    """
        Returns a list of files in the given path with the given extension
        The files are returned with their full path
    """
    files = glob.glob(os.path.join(path, f'*.{extension}'))
    return files


def getFileWithExtension(path: str, ext: str) -> Optional[str]:
    """
        Returns a single file in the given path with the given extension
    """
    # get all files in directory with extension
    files = listFilesWithExtension(path, ext)
    if len(files) == 0:
        return None
    # filter out files with grp in their name
    files = [f for f in files if "__grp__" not in f]
    if len(files) != 1:
        print(f"Found {len(files)} files in {path} with extension {ext}!")
        sys.exit(2)
    return files[0]


def loadYamlFile(filePath: str) -> List[Dict[str, str]]:
    """
        Loads a yaml file and returns its content as a list of dictionaries
    """
    with open(filePath, 'r') as file:
        content = yaml.safe_load(file)
    return content


def compareFiles(file1: str, file2: str) -> bool:
    """
        Compares two files and returns True if they are the same, False otherwise
    """
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        content1 = f1.read()
        content2 = f2.read()

    return content1 == content2


def diffFiles(file1, file2):
    with open(file1, 'r') as f1:
        file1_lines = f1.readlines()
    with open(file2, 'r') as f2:
        file2_lines = f2.readlines()

    diff = list(difflib.unified_diff(file1_lines, file2_lines, file1, file2))
    return diff


def getFileVec(path: str) -> List[Tuple[str, str]]:
    """
        Returns a list of tuples (extension, file) for the given path (note: the path includes the hash)
        The returned list must have extensions (json, ttir, ttgir)
        in this particular order, unless a file with a certain extension does not exist
    """
    vec = []
    for ext in ["json", "ttir", "ttgir"]:
        file = getFileWithExtension(path, ext)
        if file is not None:
            vec.append((ext, file))
    return vec


def getNameToHashesDict(path: str) -> Dict[str, List[str]]:
    """
        Returns a dictionary that maps kernel names to a list of hashes that have the same kernel name
        in the given path
        Note: the hashes must have a json file and either a ttir or ttgir file, otherwise they are ignored
    """
    nameToHashes = {}
    for hash in os.listdir(path):
        fullPath = os.path.join(path, hash)
        if not os.path.isdir(fullPath):
            print(f"Path {fullPath} is not a directory!")
            sys.exit(2)
        fileVec = getFileVec(fullPath)
        if len(fileVec) < 2 or fileVec[0][0] != "json":
            continue
        jsonFile = fileVec[0][1]
        # load json file
        with open(jsonFile, 'r') as file:
            content = yaml.safe_load(file)
            # get name
            name = content["name"]
            nameToHashes.setdefault(name, []).append(hash)
    return nameToHashes


def doFilesMatch(path1: str, path2: str) -> bool:
    """
        Returns True if the files in the given paths match, False otherwise
        The files are considered to match if:
        1. The number of files in both paths match
        2. The json files match
        3. Both paths have a ttir that match, if a ttir does not exist, the ttgir file must exist and match
    """
    filesVec1 = getFileVec(path1)
    filesVec2 = getFileVec(path2)
    # The number of files must match
    if len(filesVec1) != len(filesVec2):
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


def compareMatchingFiles(name: str, nameToHashes1: Dict[str, List[str]], nameToHashes2: Dict[str, List[str]],
                         args) -> ComparisonResult:
    """
        Compare files with the given name in all hashes in both paths
        Return the first mismatching files as a tuple (file1, file2), otherwise, return an empty tuple
    """
    hashes1 = nameToHashes1.get(name, [])
    hashes2 = nameToHashes2.get(name, [])
    diffs = []
    errors = []
    numComparisons = 0
    for hash1 in hashes1:
        path1 = os.path.join(args.path1, hash1)
        for hash2 in hashes2:
            path2 = os.path.join(args.path2, hash2)
            # check whether both paths have:
            # 1. json files that match
            # 2. ttir files that match (if they exist), otherwise ttgir files that match (if they exist)
            # if any of these constraints are not met, then we can skip this pair of hashes since they are not a match
            if not doFilesMatch(path1, path2):
                continue
            numComparisons += 1
            extFile1 = listFilesWithExtension(path1, "ptx")[0]
            extFile2 = listFilesWithExtension(path2, "ptx")[0]
            diff = diffFiles(extFile1, extFile2)
            if len(diff) > 0:
                diffs.append(diffFiles(extFile2, extFile1))
    if numComparisons == 0:
        errors.append(f"Did not find any matching files for {name}")
    return ComparisonResult(name=name, numComparisons=numComparisons, diffs=diffs, errors=errors)


def dumpResults(results: List[ComparisonResult], fileName: str):
    """
        Dumps the results to the given file
    """
    with open(fileName, 'w') as file:
        for result in results:
            file.write(str(result) + "\n")
            file.write("Diffs:\n")
            for diff in result.diffs:
                for line in diff:
                    file.write(line)
            file.write("Errors:\n")
            for error in result.errors:
                file.write(error)
            file.write("\n\n")


def main(args) -> bool:
    """
        Iterates over all kernels in the given yaml file and compares them
        in the given paths
    """
    if args.path1 == args.path2:
        print("Cannot compare files in the same directory!")
        sys.exit(2)
    # Get kernel name to hashes dict, these hashes would have the same kernel name
    nameToHashes1 = getNameToHashesDict(args.path1)
    nameToHashes2 = getNameToHashesDict(args.path2)

    # Get all kernels that need to be checked
    kernelNames = set(nameToHashes1.keys()).union(set(nameToHashes2.keys()))

    results = []
    # iterate over the kernels that need to be checked
    for name in kernelNames:
        # Compare all hashes on path 1 with all hashes on path 2
        # result is either the mismatching (file1, file2) with "extension" or empty tuple if no mismatch
        result = compareMatchingFiles(name, nameToHashes1, nameToHashes2, args)
        print(result)
        # Otherwise, add it to the mismatches
        results.append(result)

    # Dump results
    dumpResults(results, "kernels_reference_check.txt")

    success = all(result.isSuccess() for result in results)

    if not success:
        print("Failed!")
        sys.exit(1)

    print("Passed!")
    sys.exit(0)


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
    args = parser.parse_args()
    main(args)
