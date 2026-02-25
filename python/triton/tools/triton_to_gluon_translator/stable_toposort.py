from __future__ import annotations

from collections import deque
from collections.abc import Mapping

from triton.tools.triton_to_gluon_translator.ordered_set import ordered_set


def stable_toposort(edges: Mapping[int, ordered_set[int]]) -> list[int]:  # noqa: C901
    nodes: list[int] = list(edges.keys())
    order = {node: idx for idx, node in enumerate(nodes)}
    index: dict[int, int] = {}
    low: dict[int, int] = {}
    stack: list[int] = []
    on_stack: ordered_set[int] = ordered_set()
    comp_of: dict[int, int] = {}
    comps: list[tuple[int, ...]] = []

    def dfs(node: int) -> None:
        idx = len(index)
        index[node] = low[node] = idx
        stack.append(node)
        on_stack.add(node)
        for nxt in edges[node]:
            if nxt not in index:
                dfs(nxt)
                low[node] = min(low[node], low[nxt])
            elif nxt in on_stack:
                low[node] = min(low[node], index[nxt])
        if low[node] == index[node]:
            comp: list[int] = []
            while True:
                cur = stack.pop()
                on_stack.remove(cur)
                comp.append(cur)
                comp_of[cur] = len(comps)
                if cur == node:
                    break
            comps.append(tuple(sorted(comp, key=order.__getitem__)))

    for node in nodes:
        if node not in index:
            dfs(node)

    priority = [min(order[n] for n in comp) for comp in comps]
    indegree = [0] * len(comps)
    outgoing: list[ordered_set[int]] = [ordered_set() for _ in comps]
    for node in nodes:
        src = comp_of[node]
        for nbr in edges[node]:
            dst = comp_of[nbr]
            if src != dst and dst not in outgoing[src]:
                outgoing[src].add(dst)
                indegree[dst] += 1

    queue = deque(
        sorted((cid for cid, deg in enumerate(indegree) if deg == 0), key=priority.__getitem__)
    )
    result: list[int] = []
    while queue:
        cid = queue.popleft()
        result.extend(comps[cid])
        for nxt in sorted(outgoing[cid], key=priority.__getitem__):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    assert len(result) == len(nodes), "component graph contains a cycle"
    return result
