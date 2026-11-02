"""Compressed radix tree for KV cache prefix reuse.

Each node owns the token segment on the edge from its parent, so a run of N
tokens stored in one edge costs a single node instead of N.  The root is a
sentinel and carries no edge.

Semantics pinned down together with the test suite
1. insert([]) is a no-op, re-inserting a stored sequence changes nothing
   structurally but does refresh its LRU stamps
2. a global clock ticks once per insert / match_prefix call; every node that
   the call walks through (or creates) is stamped with the new value, so the
   relative order of calls is recoverable from the stamps.  A partial hit in
   match_prefix splits the edge at the hit boundary and stamps only the matched
   prefix: the untouched tail keeps its old stamp, so eviction can drop the tail
   while keeping the prefix that was actually reused
3. splitting an edge: the new middle node inherits the old node's stamp
   (existing data must not look freshly used), the leaf created for the tail of
   the incoming sequence gets the current stamp
4. only leaves may be evicted; eviction removes whole leaves until the released
   token count reaches the budget, therefore it may overshoot, and it returns
   the number of tokens actually released
5. a parent left without children becomes a leaf candidate for later eviction
   rounds, but it is not swept inside the same call once the budget is met
6. evict(0) releases nothing and returns 0; a negative budget raises ValueError
"""
from collections.abc import Sequence, Iterable
import heapq
from typing import Any


class Node:
    """One edge (``key``) plus the subtree hanging below it."""

    def __init__(self, key: list, parent: "Node | None" = None, last_access_time: int = 0) -> None:
        self.key: list = key
        self.children: dict[int, Node] = {}
        self.parent: "Node | None" = parent
        self.last_access_time: int = last_access_time

    def __lt__(self, other):
        return self.last_access_time < other.last_access_time


def longest_prefix(prefix: list[int], target: list[int], start: int = 0) -> int:
    n = min(len(prefix) - start, len(target))
    for i in range(n):
        if prefix[start + i] != target[i]:
            return i
    return n


class RadixTree:
    def __init__(self) -> None:
        self.root = Node([])
        self._clock = 0
        self.cached_token_length : int = 0

    def insert(self, tokens: list[int]) -> None:
        """Store ``tokens``, sharing whatever prefix the tree already holds."""
        self._clock += 1
        curr = self.root
        i = 0
        n = len(tokens)
        while i < n:
            token = tokens[i]
            child = curr.children.get(token)
            if child is None:
                newNode = Node(tokens[i:], curr, self._clock)
                curr.children[token] = newNode
                self.cached_token_length += n - i
                return
            common_prefix_length = longest_prefix(tokens, child.key, i)
            if common_prefix_length == len(child.key):
                child.last_access_time = self._clock
                curr = child
                i += common_prefix_length
                continue
            old_half = self._split_node(child, common_prefix_length, curr, token)
            if i + common_prefix_length < n:
                tail = tokens[i + common_prefix_length:]
                old_half.children[tail[0]] = Node(tail, old_half, self._clock)
                self.cached_token_length += len(tail)
            return

    def _split_node(self, child: Node, common_prefix_length: int, curr: Node | Any, token: int) -> Node:
        old_half = Node(child.key[:common_prefix_length], curr, child.last_access_time)
        curr.children[token] = old_half
        child.key = child.key[common_prefix_length:]
        child.parent = old_half
        old_half.children[child.key[0]] = child
        return old_half

    def match_prefix(self, tokens: list[int]) -> int:
        """Return the length of the longest prefix of ``tokens`` present in the tree.

        A partial hit splits the edge at the hit boundary and stamps only the
        matched prefix, so the untouched tail keeps its old stamp (the same split
        SGLang's ``_match_prefix_helper`` performs, but with a precise LRU stamp
        instead of refreshing the whole child).
        """
        self._clock += 1
        curr = self.root
        n = len(tokens)
        i = 0
        while i < n:
            token = tokens[i]
            child = curr.children.get(token)
            if child is None:
                return i
            common_prefix_length = longest_prefix(tokens, child.key, i)
            if common_prefix_length == len(child.key):
                child.last_access_time = self._clock
                curr = child
                i += common_prefix_length
                continue
            matched = self._split_node(child, common_prefix_length, curr, token)
            matched.last_access_time = self._clock
            return i + common_prefix_length
        return i




    def evict(self, num_tokens: int) -> int:
        """Evict least recently used leaves until at least ``num_tokens`` are freed."""
        evicted_tokens = 0
        if num_tokens < 0:
            raise ValueError(f"cannot evict negative number of tokens: {num_tokens}")
        leaves = list(self.iter_leaves(self.root))
        heapq.heapify(leaves)
        while evicted_tokens < num_tokens and leaves:
            candidate = heapq.heappop(leaves)
            freed = len(candidate.key)
            parent = candidate.parent
            del parent.children[candidate.key[0]]
            candidate.parent = None
            evicted_tokens += freed
            self.cached_token_length -= freed
            if not parent.children and parent.parent is not None:
                heapq.heappush(leaves, parent)
        return evicted_tokens

    @property
    def total_tokens(self) -> int:
        """Sum of the lengths of every edge in the tree."""
        return self.cached_token_length

    def iter_leaves(self, node: Node) -> Iterable[Node]:
        stack = [node]
        while stack:
            curr = stack.pop()
            if curr.children:
                stack.extend(curr.children.values())
            elif curr.parent is not None: # ensure not root
                yield curr
