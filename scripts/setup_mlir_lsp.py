import os
import shutil
import subprocess
import traceback

from pathlib import Path


def print_good(s):
    print("\033[92m" + s + "\033[0m")


def print_bad(s):
    print("\033[91m" + s + "\033[0m")


def chdir_to_base_dir():
    base_dir = (subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())
    os.chdir(base_dir)
    print("Working dir: ", os.getcwd())


# Should be in sync with the one in ./setup.py
def get_triton_cache_path():
    user_home = os.getenv("TRITON_HOME")
    if not user_home:
        user_home = (os.getenv("HOME") or os.getenv("USERPROFILE") or os.getenv("HOMEPATH") or None)
    if not user_home:
        raise RuntimeError("Could not find user home directory")
    return os.path.join(user_home, ".triton")


# creat or overwrite a symlink
def update_symlink(link_path, source_path):
    source_path = Path(source_path)
    link_path = Path(link_path)

    if link_path.is_symlink():
        print("Removing existing symlink at: ", link_path)
        link_path.unlink()
    elif link_path.exists():
        print("Removing existing file at: ", link_path)
        link_path.unlink()

    print(f"Creating symlink: {link_path} -> {source_path}")
    link_path.symlink_to(source_path.absolute())


# let vscode analyze MLIR files (e.g. .mlir files)
# will also work for .ttir or .ttgir if you let VSCode parse them as MLIR
def setup_lsp_for_mlir():
    print("\n** setup_lsp_for_mlir **")
    lsp_bin_path = list(filter(
        lambda x: os.path.isfile(x),
        Path.cwd().glob("build/**/triton-lsp"),
    ))
    assert (
        len(lsp_bin_path) == 1), "Expected exactly one tablegen compile commands file, but found " + str(lsp_bin_path)
    # by default, VSCode extension will look for the mlir lsp binary ./mlir-lsp-server
    update_symlink("./mlir-lsp-server", lsp_bin_path[0])
    print_good(f"Setup successful! Reload VSCode window to see IDE analysed mlir files.")


# let vscode analyze tablegen files (.td files)
def setup_lsp_for_tablegen():
    print("\n** setup_lsp_for_tablegen **")
    # part 1: find the lsp binary from llvm
    llvm_dir_path = os.path.join(get_triton_cache_path(), "llvm")
    # points to the llvm used to build triton, either prebuilt or customized build
    symlinks = list(filter(lambda x: x.is_symlink(), os.scandir(llvm_dir_path)))
    assert (len(symlinks) == 1), f"Expected exactly one symlink in llvm dir, but found {symlinks} at {llvm_dir_path}"
    lsp_bin_path = os.path.join(os.path.realpath(symlinks[0].path), "bin", "tblgen-lsp-server")
    assert os.path.exists(lsp_bin_path), ("Expected to find tblgen-lsp-server bin at " + lsp_bin_path)
    # need to copy instead of symlink because vscode extension cannot seem to read outside
    # of the workspace folder
    bin_name = os.path.basename(lsp_bin_path)
    print(f"Copying {lsp_bin_path} to ./{bin_name} ")
    if os.path.exists(bin_name):
        os.unlink(bin_name)
    shutil.copy(lsp_bin_path, bin_name)

    # part 2: create a symlink to the tablegen_compile_commands.yml file
    td_compile_commands_path = list(
        filter(
            lambda x: not os.path.islink(x),
            Path.cwd().glob("build/**/tablegen_compile_commands.yml"),
        ))
    assert (len(td_compile_commands_path) == 1
            ), "Expected exactly one tablegen compile commands file, but found " + str(td_compile_commands_path)
    # put it under ./build so that MLIR vscode extension can find it
    update_symlink("./build/tablegen_compile_commands.yml", td_compile_commands_path[0])
    print_good(f"Setup successful! Reload VSCode window to see IDE analysed tablegen files.")


def main():
    print_good(
        "Starting VSCode setup. Make sure you've built triton successfully and installed the official MLIR extension.")
    chdir_to_base_dir()
    try:
        setup_lsp_for_mlir()
    except Exception as e:
        print_bad(f"Failed to setup lsp for mlir: {e}")
        traceback.print_exception(e)
    try:
        setup_lsp_for_tablegen()
    except Exception as e:
        print_bad(f"Failed to setup lsp for tablegen: {e}")
        traceback.print_exception(e)


if __name__ == "__main__":
    main()
