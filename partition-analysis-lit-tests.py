import subprocess
import os
from termcolor import colored

root_dir = os.path.dirname(os.path.realpath(__file__)) + "/partition-analysis-lit-tests"
bin_dir = (
    os.path.dirname(os.path.realpath(__file__))
    + "/build/cmake.linux-x86_64-cpython-3.12/bin"
)

update = False

kinds = ["nvws", "main"]
tests = os.listdir(os.path.join(root_dir, "input"))
for kind in kinds:
    for test in sorted(tests):

        print(test, kind, end="")
        output = subprocess.check_output(
            [
                bin_dir + "/triton-opt",
                "--tritongpu-hoist-tmem-alloc",
                "--tritongpu-partition-analysis",
                "-allow-unregistered-dialect",
                os.path.join(root_dir, "input", test),
            ],
            env={
                "PARTITION_ANALYSIS_NVWS_SERIALIZATION": "1" if kind == "nvws" else "0",
                "TRITON_OVERRIDE_ARCH": "sm100",
            },
        ).decode()


        if update:
            with open(os.path.join(root_dir, f"output-{kind}", test), "w") as file:
                file.write(output)

        with open(os.path.join(root_dir, f"output-{kind}", test), "r") as file:
            expected = file.read()
        if output != expected:
            # print(output)
            # print()
            with open("lit-test-result.mlir", "w") as file:
                file.write(output)
            subprocess.check_call(
                [
                    "diff",
                    os.path.join(root_dir, f"output-{kind}", test),
                    "lit-test-result.mlir",
                ]
            )
            assert False
        print(colored(" PASS", color="green"))
