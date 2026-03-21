from collections import deque
from dataclasses import dataclass

import pytest

from triton.tools.triton_to_gluon_translator.ordered_set import ordered_set
from triton.tools.triton_to_gluon_translator.stable_toposort import stable_toposort


@dataclass(frozen=True)
class Case:
    name: str
    graph: dict[int, ordered_set[int]]
    expected: list[int]


CASES: list[Case] = []


def add_case(name: str, mapping: list[tuple[int, list[int]]], expected_order: list[int]):
    graph = {node: ordered_set(children) for node, children in mapping}
    CASES.append(Case(name, graph, expected_order))


add_case("empty", [], [])
add_case("single_node", [(0, [])], [0])
add_case("single_self_loop", [(0, [0])], [0])
add_case("single_edge", [(0, [1]), (1, [])], [0, 1])
add_case("two_disconnected", [(0, []), (1, [])], [0, 1])
add_case("child_only_node", [(0, [1]), (1, [])], [0, 1])

for length in range(3, 11):
    mapping = [(idx, [idx + 1]) for idx in range(length - 1)]
    mapping.append((length - 1, []))
    add_case(f"chain_{length}", mapping, list(range(length)))

for size in range(2, 7):
    leaves = list(range(1, size + 1))
    mapping = [(0, leaves)] + [(leaf, []) for leaf in leaves]
    add_case(f"star_out_{size}", mapping, [0] + leaves)

for size in range(2, 7):
    leaves = list(range(1, size + 1))
    mapping = [(leaf, [0]) for leaf in leaves] + [(0, [])]
    add_case(f"star_in_{size}", mapping, leaves + [0])

add_case("diamond", [(0, [1, 2]), (1, [3]), (2, [3]), (3, [])], [0, 1, 2, 3])
add_case("wide_branch", [(0, [1, 2, 3]), (1, [4]), (2, [4]), (3, [4]), (4, [])], [0, 1, 2, 3, 4])
add_case(
    "deep_branch",
    [(0, [1]), (1, [2, 3]), (2, [4]), (3, [4]), (4, [5]), (5, [])],
    [0, 1, 2, 3, 4, 5],
)

add_case("two_cycle", [(1, [2]), (2, [1])], [1, 2])
add_case("three_cycle", [(0, [1]), (1, [2]), (2, [0])], [0, 1, 2])
add_case(
    "four_cycle_with_tail",
    [(0, [1]), (1, [2]), (2, [3]), (3, [0, 4]), (4, [])],
    [0, 1, 2, 3, 4],
)
add_case(
    "cycle_with_entry",
    [(5, [0]), (0, [1]), (1, [2]), (2, [0, 3]), (3, [])],
    [5, 0, 1, 2, 3],
)
add_case(
    "cycle_with_exit",
    [(0, [1]), (1, [2]), (2, [0, 3]), (3, [4]), (4, [])],
    [0, 1, 2, 3, 4],
)
add_case(
    "two_disjoint_cycles",
    [(0, [1]), (1, [0]), (2, [3]), (3, [2])],
    [0, 1, 2, 3],
)
add_case(
    "cycle_bridge",
    [(0, [1]), (1, [0, 2]), (2, [3]), (3, [4]), (4, [])],
    [0, 1, 2, 3, 4],
)
add_case(
    "self_loop_with_outgoing",
    [(0, [0, 1]), (1, [2]), (2, [])],
    [0, 1, 2],
)
add_case(
    "multi_roots",
    [(2, [3]), (0, [2]), (1, [3]), (3, [])],
    [0, 1, 2, 3],
)
add_case(
    "sink_only",
    [(0, [1, 2]), (1, []), (2, [])],
    [0, 1, 2],
)
add_case(
    "source_only",
    [(0, []), (1, [2]), (2, [3]), (3, [])],
    [0, 1, 2, 3],
)
add_case(
    "nested_cycles",
    [(0, [1]), (1, [2, 4]), (2, [0, 3]), (3, [2]), (4, [5]), (5, [])],
    [0, 1, 2, 3, 4, 5],
)
add_case(
    "cycle_and_chain",
    [(0, [1]), (1, [0]), (2, [3]), (3, [4]), (4, [])],
    [0, 1, 2, 3, 4],
)
add_case(
    "shared_child",
    [(0, [2]), (1, [2]), (2, [3]), (3, [])],
    [0, 1, 2, 3],
)
add_case(
    "shared_parent",
    [(0, [1, 2]), (1, []), (2, [])],
    [0, 1, 2],
)
add_case(
    "long_cycle_with_branch",
    [(0, [1]), (1, [2]), (2, [3, 5]), (3, [0]), (4, [0]), (5, [6]), (6, [])],
    [4, 0, 1, 2, 3, 5, 6],
)
add_case(
    "incoming_from_multiple",
    [(0, [3]), (1, [3]), (2, [3]), (3, [4]), (4, [])],
    [0, 1, 2, 3, 4],
)
add_case(
    "cycle_with_self_loop",
    [(0, [0, 1]), (1, [2]), (2, [0])],
    [0, 1, 2],
)
add_case(
    "reverse_chain",
    [(3, []), (2, [3]), (1, [2]), (0, [1])],
    [0, 1, 2, 3],
)
add_case(
    "disconnected_dag",
    [(0, [1]), (1, []), (2, [3]), (3, [])],
    [0, 2, 1, 3],
)
add_case(
    "disconnected_cycle_and_node",
    [(4, []), (0, [1]), (1, [2]), (2, [0])],
    [4, 0, 1, 2],
)
add_case(
    "cycle_with_multiple_exits",
    [(0, [1, 3]), (1, [2]), (2, [0]), (3, [4]), (4, [])],
    [0, 1, 2, 3, 4],
)
add_case(
    "parallel_paths",
    [(0, [2]), (1, [2]), (2, [3]), (3, [4]), (4, [])],
    [0, 1, 2, 3, 4],
)
add_case(
    "long_branching",
    [(0, [1, 2]), (1, [3, 4]), (2, [4]), (3, [5]), (4, [5]), (5, [])],
    [0, 1, 2, 3, 4, 5],
)
add_case(
    "looping_tail",
    [(0, [1]), (1, [2]), (2, [1, 3]), (3, [4]), (4, [])],
    [0, 1, 2, 3, 4],
)
add_case(
    "multi_component_dag",
    [(0, [1]), (1, []), (2, [3]), (3, []), (4, [5]), (5, [])],
    [0, 2, 4, 1, 3, 5],
)
add_case(
    "multiple_zero_indegree",
    [(2, [4]), (0, [3]), (1, [3]), (3, [5]), (4, [5]), (5, [])],
    [2, 0, 1, 4, 3, 5],
)
add_case(
    "interleaved_cycle",
    [(0, [2]), (1, [0]), (2, [1])],
    [0, 1, 2],
)
add_case(
    "self_loops_multiple",
    [(0, [0]), (1, [1]), (2, [2])],
    [0, 1, 2],
)
add_case(
    "cycle_with_disconnected_tail",
    [(0, [1]), (1, [0]), (2, []), (3, [4]), (4, [])],
    [0, 1, 2, 3, 4],
)
add_case(
    "cycle_with_child_only_nodes",
    [(0, [1, 2]), (1, [0]), (2, []), (3, [4]), (4, [])],
    [0, 1, 3, 2, 4],
)
add_case(
    "long_linear_with_skip",
    [(0, [1, 2]), (1, [3]), (2, [3]), (3, [4]), (4, [5]), (5, [])],
    [0, 1, 2, 3, 4, 5],
)
add_case(
    "incoming_chain_to_cycle",
    [(0, [1]), (1, [2]), (2, [3]), (3, [1])],
    [0, 1, 2, 3],
)
add_case(
    "outgoing_chain_from_cycle",
    [(0, [1]), (1, [0, 2]), (2, [3]), (3, [])],
    [0, 1, 2, 3],
)
add_case(
    "disconnected_self_loops",
    [(0, [0]), (2, [2]), (1, [])],
    [0, 2, 1],
)
add_case(
    "complex_cycle_mix",
    [(0, [1, 2]), (1, [0, 3]), (2, [4]), (3, [1, 5]), (4, [2]), (5, [])],
    [0, 1, 3, 2, 4, 5],
)
add_case(
    "isolated_node",
    [(0, []), (2, [3]), (3, [])],
    [0, 2, 3],
)
add_case(
    "large_cycle",
    [(0, [1]), (1, [2]), (2, [3]), (3, [4]), (4, [0])],
    [0, 1, 2, 3, 4],
)
add_case(
    "cycle_with_reentry",
    [(0, [1]), (1, [2, 3]), (2, [0]), (3, [4]), (4, [2])],
    [0, 1, 2, 3, 4],
)
add_case(
    "multi_branch_cycle",
    [(0, [1, 2]), (1, [0]), (2, [3]), (3, [4, 5]), (4, []), (5, [])],
    [0, 1, 2, 3, 4, 5],
)
add_case(
    "redundant_edges",
    [(0, [1, 1, 1]), (1, [2, 2]), (2, [])],
    [0, 1, 2],
)
add_case(
    "multiple_component_cycles",
    [(0, [1]), (1, [0]), (2, [3]), (3, [4]), (4, [2])],
    [0, 1, 2, 3, 4],
)
add_case(
    "child_only_multiple",
    [(0, [1, 2, 3]), (1, []), (2, []), (3, [])],
    [0, 1, 2, 3],
)
add_case(
    "complex_dag",
    [
        (0, [1, 2, 3]),
        (1, [4, 5]),
        (2, [5, 6]),
        (3, [6]),
        (4, [7]),
        (5, [7]),
        (6, [7]),
        (7, []),
    ],
    [0, 1, 2, 3, 4, 5, 6, 7],
)
add_case(
    "mixed_self_and_cycle",
    [(0, [0, 1]), (1, [2]), (2, [3]), (3, [1])],
    [0, 1, 2, 3],
)


def normalize_graph(graph: dict[int, ordered_set[int]]) -> dict[int, ordered_set[int]]:
    nodes = ordered_set(graph)
    for children in graph.values():
        nodes.update(children)
    return {node: ordered_set(graph.get(node, ordered_set())) for node in nodes}


def reachable(graph: dict[int, ordered_set[int]], src: int, dst: int) -> bool:
    if src == dst:
        return True
    seen = {src}
    queue = deque([src])
    while queue:
        node = queue.popleft()
        for nbr in graph.get(node, ()):  # type: ignore[arg-type]
            if nbr == dst:
                return True
            if nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)
    return False


def span_order(nodes: list[int]) -> dict[int, int]:
    return {node: idx for idx, node in enumerate(nodes)}


def check_case(graph: dict[int, ordered_set[int]], expected_order: list[int]) -> None:
    graph_copy = {node: ordered_set(children) for node, children in graph.items()}
    original = normalize_graph(graph_copy)
    result = stable_toposort(graph_copy)
    assert list(result) == expected_order
    assert ordered_set(result) == ordered_set(original)

    order = span_order(list(result))
    for src, children in original.items():
        for dst in children:
            if not reachable(original, dst, src):
                assert order[src] < order[dst]

    for node in original:
        assert node in result


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_stable_toposort_cases(case: Case) -> None:
    check_case(case.graph, case.expected)


def test_input_not_mutated() -> None:
    graph = {0: {1}, 1: {2}, 2: ordered_set()}
    snapshot = {k: ordered_set(v) for k, v in graph.items()}
    stable_toposort(graph)
    assert graph == snapshot
