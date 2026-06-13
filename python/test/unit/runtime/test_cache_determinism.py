import concurrent.futures
import os
import random
import subprocess
import sys

import triton
import triton.language as tl


def make_child(value):
    shared = tl.constexpr(value)

    @triton.jit
    def child():
        return shared

    return child


identity_child_a = make_child(1)
identity_child_b = make_child(2)


@triton.jit
def identity_parent():
    identity_child_a()
    identity_child_b()


stress_children = [make_child(value) for value in range(16)]
(
    stress_child_00,
    stress_child_01,
    stress_child_02,
    stress_child_03,
    stress_child_04,
    stress_child_05,
    stress_child_06,
    stress_child_07,
    stress_child_08,
    stress_child_09,
    stress_child_10,
    stress_child_11,
    stress_child_12,
    stress_child_13,
    stress_child_14,
    stress_child_15,
) = stress_children


@triton.jit
def stress_parent():
    stress_child_00()
    stress_child_01()
    stress_child_02()
    stress_child_03()
    stress_child_04()
    stress_child_05()
    stress_child_06()
    stress_child_07()
    stress_child_08()
    stress_child_09()
    stress_child_10()
    stress_child_11()
    stress_child_12()
    stress_child_13()
    stress_child_14()
    stress_child_15()


def _subprocess_key(mode, seed, order="forward"):
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["TRITON_CACHE_KEY_TEST_ORDER"] = order
    env["TRITON_CACHE_KEY_TEST_SEED"] = str(seed)
    result = subprocess.run(
        [sys.executable, __file__, mode],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def test_cache_key_deterministic_across_process_global_identities():
    cases = [(seed, order) for seed in range(4) for order in ("forward", "reverse")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as executor:
        keys = list(executor.map(lambda case: _subprocess_key("identity", *case), cases))

    assert len(set(keys)) == 1


def test_cache_key_deterministic_under_concurrent_dependency_hashing():
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        keys = list(executor.map(lambda seed: _subprocess_key("stress", seed), range(16)))

    assert len(set(keys)) == 1


def _run_identity_case():
    children = [identity_child_a, identity_child_b]
    if os.environ["TRITON_CACHE_KEY_TEST_ORDER"] == "reverse":
        children.reverse()
    for child in children:
        child.cache_key
    print(identity_parent.cache_key)


def _run_stress_case():
    functions = [stress_parent, *stress_children]
    random.Random(int(os.environ["TRITON_CACHE_KEY_TEST_SEED"])).shuffle(functions)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda function: function.cache_key, functions))
    print(stress_parent.cache_key)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "identity":
        _run_identity_case()
    elif mode == "stress":
        _run_stress_case()
    else:
        raise ValueError(f"unknown mode: {mode}")
