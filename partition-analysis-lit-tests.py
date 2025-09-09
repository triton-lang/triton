import subprocess
import os
from termcolor import colored
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', default=None)
    parser.add_argument('--test', default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--update', action='store_true')
    args = parser.parse_args()

    root_dir = (
        os.path.dirname(os.path.realpath(__file__)) + "/partition-analysis-lit-tests"
    )
    bin_dir = (
        os.path.dirname(os.path.realpath(__file__))
        + "/build/cmake.linux-x86_64-cpython-3.12/bin"
    )

    kinds = ["nvws", "main"]
    tests = os.listdir(os.path.join(root_dir, "input"))
    for kind in kinds:
        if args.kind is not None and args.kind != kind:
            continue
        for test in sorted(tests):
            if args.test is not None and args.test != test:
                continue

            print(kind, test, end="")

            run_dir = os.path.join(root_dir, "run-" + kind, test)
            os.makedirs(run_dir, exist_ok=True)

            output = subprocess.check_output(
                [
                    bin_dir + "/triton-opt",
                    "--tritongpu-hoist-tmem-alloc",
                    "--tritongpu-partition-analysis",
                    "-allow-unregistered-dialect",
                    os.path.join(root_dir, "input", test),
                ],
                env={
                    "PARTITION_ANALYSIS_ENABLE_DUMP_DOT": "1",
                    "PARTITION_ANALYSIS_NVWS_SERIALIZATION": (
                        "1" if kind == "nvws" else "0"
                    ),
                    "TRITON_OVERRIDE_ARCH": "sm100",
                },
                cwd=run_dir,
            ).decode()

            if args.update:
                with open(os.path.join(root_dir, f"output-{kind}", test), "w") as file:
                    file.write(output)

            with open(os.path.join(root_dir, f"output-{kind}", test), "r") as file:
                expected = file.read()
            if output != expected:
                print(colored(" FAIL", color="red"))
                if args.verbose:
                    print(output)
                    print()
                    with open("lit-test-result.mlir", "w") as file:
                        file.write(output)
                    subprocess.check_call(
                        [
                            "diff",
                            os.path.join(root_dir, f"output-{kind}", test),
                            "lit-test-result.mlir",
                        ]
                    )
            else:
                print(colored(" PASS", color="green"))


if __name__ == "__main__":
    main()
