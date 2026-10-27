"""Radix prefix cache tests: compressed structure, split semantics, LRU eviction.

Tests in this directory are co-designed with DeepSeek V4.1 Flash.

Structural contract these tests rely on
1. ``RadixTree()`` takes no required argument and exposes ``root`` and ``total_tokens``
2. ``total_tokens`` equals the sum of ``len(node.key)`` over every non-root node
3. ``node.key`` is the token segment on the edge parent -> node; ``root.key`` is empty
4. ``node.children`` maps first-token -> child; ``node.parent`` points back

Behaviour pinned down by this suite
1. insert([]) is a no-op; re-inserting a stored sequence changes nothing structurally
2. the global clock ticks once per insert / match_prefix call and stamps every node
   that call touches
3. splitting an edge makes the middle node inherit the old stamp while the new leaf
   gets the current one
4. only leaves are evicted, whole leaves at a time, so the release may overshoot the
   budget; evict returns the tokens actually released
5. a parent whose children are gone becomes a leaf candidate for later rounds, but is
   not swept in the same call once the budget is met
6. evict(0) is a no-op returning 0, a negative budget raises ValueError

Acceptance criteria (from README)
1. total_tokens equals the sum of all edge lengths
2. sibling edges start with pairwise different tokens
3. insert then match_prefix of the same sequence always returns the full length
4. evict only removes leaves, and every remaining root-to-node path is still a prefix
   of some previously inserted sequence
"""
import random

import pytest

from radix_prefix_cache.radix_tree import RadixTree


def iter_nodes(tree: RadixTree):
    """Pre-order walk over every node, root included."""
    stack = [tree.root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children.values())


def path_of(node) -> tuple:
    """Token sequence from the root down to ``node``."""
    segments = []
    while node.parent is not None:
        segments.append(tuple(node.key))
        node = node.parent
    out = ()
    for segment in reversed(segments):
        out += segment
    return out


def leaf_paths(tree: RadixTree) -> list[tuple]:
    """Sorted root-to-leaf token sequences; the compressed shape in one glance."""
    return sorted(path_of(node) for node in iter_nodes(tree)
                  if node.parent is not None and not node.children)


def assert_invariants(tree: RadixTree, known=()) -> None:
    """Structure, counters and prefix provenance; never mutates LRU state."""
    known = [tuple(seq) for seq in known]
    root = tree.root
    assert root.parent is None, "root must not have a parent"
    assert len(root.key) == 0, "root carries no edge"

    edge_total = 0
    for node in iter_nodes(tree):
        if node.parent is not None:
            assert len(node.key) > 0, f"non-root node at {path_of(node)} has an empty edge"
            edge_total += len(node.key)
            assert node.parent.children.get(node.key[0]) is node, (
                f"parent of {path_of(node)} does not list it under token {node.key[0]}")
        firsts = [child.key[0] for child in node.children.values()]
        assert len(firsts) == len(set(firsts)), (
            f"duplicate first token among children of {path_of(node)}: {firsts}")
        for first, child in node.children.items():
            assert child.key[0] == first, (
                f"children key {first} != first token {child.key[0]} of its edge")
            assert child.parent is node, "child does not point back to its parent"

    assert tree.total_tokens == edge_total, (
        f"total_tokens {tree.total_tokens} != sum of edge lengths {edge_total}")

    if known:
        for node in iter_nodes(tree):
            path = path_of(node)
            assert any(seq[:len(path)] == path for seq in known), (
                f"path {path} is not a prefix of any inserted sequence")


def assert_every_leaf_matches_itself(tree: RadixTree) -> None:
    """Cross-check match_prefix against the tree shape; refreshes LRU stamps."""
    for leaf in [n for n in iter_nodes(tree) if n.parent is not None and not n.children]:
        path = path_of(leaf)
        assert tree.match_prefix(list(path)) == len(path), (
            f"leaf {path} does not match itself fully")


class TestInsertStructure:
    def test_single_sequence_becomes_one_compressed_edge(self):
        """3 tokens stored as 1 node, not 3."""
        tree = RadixTree()
        tree.insert([1, 2, 3])
        assert leaf_paths(tree) == [(1, 2, 3)]
        assert tree.total_tokens == 3
        assert len(tree.root.children) == 1
        only_child = next(iter(tree.root.children.values()))
        assert tuple(only_child.key) == (1, 2, 3)
        assert_invariants(tree, [[1, 2, 3]])

    def test_empty_sequence_is_noop(self):
        tree = RadixTree()
        tree.insert([])
        assert tree.total_tokens == 0
        assert tree.root.children == {}
        assert leaf_paths(tree) == []

    def test_duplicate_insert_does_not_grow_the_tree(self):
        tree = RadixTree()
        tree.insert([1, 2, 3])
        tree.insert([1, 2, 3])
        assert tree.total_tokens == 3
        assert leaf_paths(tree) == [(1, 2, 3)]
        assert_invariants(tree, [[1, 2, 3]])

    def test_disjoint_sequences_are_siblings_of_root(self):
        tree = RadixTree()
        for seq in ([1, 2, 3], [4, 5, 6], [7, 8, 9]):
            tree.insert(seq)
        assert sorted(tree.root.children) == [1, 4, 7]
        assert tree.total_tokens == 9
        assert leaf_paths(tree) == [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        assert_invariants(tree, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_shared_prefix_is_stored_once(self):
        """[1,2,3] and [1,2,4] cost 4 tokens, not 6."""
        tree = RadixTree()
        tree.insert([1, 2, 3])
        tree.insert([1, 2, 4])
        assert tree.total_tokens == 4
        assert leaf_paths(tree) == [(1, 2, 3), (1, 2, 4)]
        assert_invariants(tree, [[1, 2, 3], [1, 2, 4]])

    def test_appending_extends_the_existing_edge(self):
        """Storing a shorter sequence first must not block the longer one."""
        tree = RadixTree()
        tree.insert([1, 2])
        tree.insert([1, 2, 3, 4])
        assert tree.total_tokens == 4
        assert leaf_paths(tree) == [(1, 2, 3, 4)]
        assert_invariants(tree, [[1, 2], [1, 2, 3, 4]])

    def test_sequence_that_is_a_prefix_of_stored_one_splits(self):
        """[1,2] inserted after [1,2,3] still forces a split at offset 2."""
        tree = RadixTree()
        tree.insert([1, 2, 3])
        tree.insert([1, 2])
        assert tree.total_tokens == 3
        assert leaf_paths(tree) == [(1, 2, 3)]
        only_child = next(iter(tree.root.children.values()))
        assert tuple(only_child.key) == (1, 2)
        assert tree.match_prefix([1, 2]) == 2
        assert_invariants(tree, [[1, 2, 3], [1, 2]])

    def test_tokens_may_be_any_hashable(self):
        """The tree must not assume integer token ids."""
        tree = RadixTree()
        tree.insert(["system", "prompt", "hi"])
        tree.insert(["system", "prompt", "bye"])
        assert tree.total_tokens == 4
        assert tree.match_prefix(["system", "prompt", "?"]) == 2
        assert_invariants(tree, [["system", "prompt", "hi"], ["system", "prompt", "bye"]])


class TestMatchPrefix:
    def test_match_after_insert_returns_full_length(self):
        """Acceptance 3, checked for every shape of insertion order."""
        tree = RadixTree()
        for seq in ([1, 2, 3], [1, 2, 4], [5, 6], [1, 2, 3, 7, 8]):
            tree.insert(seq)
            assert tree.match_prefix(seq) == len(seq)

    def test_match_stops_at_the_divergence(self):
        tree = RadixTree()
        tree.insert([1, 2, 3])
        tree.insert([1, 2, 4])
        assert tree.match_prefix([1, 2, 9]) == 2
        assert tree.match_prefix([1, 2]) == 2
        assert tree.match_prefix([1]) == 1

    def test_match_stops_inside_a_compressed_edge(self):
        """The comparison has to happen mid-edge, not only at node boundaries."""
        tree = RadixTree()
        tree.insert([1, 2, 3, 4])
        assert tree.match_prefix([1, 2, 9]) == 2
        assert tree.match_prefix([1, 2, 3, 9]) == 3

    def test_match_miss_returns_zero(self):
        tree = RadixTree()
        tree.insert([1, 2, 3])
        assert tree.match_prefix([9, 9]) == 0
        assert tree.match_prefix([]) == 0

    def test_match_does_not_change_the_structure(self):
        tree = RadixTree()
        tree.insert([1, 2, 3])
        before = leaf_paths(tree)
        tree.match_prefix([1, 2, 3])
        assert tree.total_tokens == 3
        assert leaf_paths(tree) == before


class TestSplit:
    def test_split_creates_mid_node_with_common_prefix(self):
        tree = RadixTree()
        tree.insert([1, 2, 3, 4])
        tree.insert([1, 2, 3, 9])
        top = next(iter(tree.root.children.values()))
        assert tuple(top.key) == (1, 2, 3)
        assert sorted(tuple(c.key) for c in top.children.values()) == [(4,), (9,)]
        assert tree.total_tokens == 5
        assert_invariants(tree, [[1, 2, 3, 4], [1, 2, 3, 9]])

    def test_split_preserves_total_tokens(self):
        tree = RadixTree()
        tree.insert([1, 2, 3, 4, 5])
        before = tree.total_tokens
        tree.insert([1, 2, 3, 9])
        assert before == 5
        assert tree.total_tokens == 6
        assert len(tree.root.children) == 1

    def test_split_at_the_first_token_of_a_long_edge(self):
        tree = RadixTree()
        tree.insert([1, 2, 3])
        tree.insert([1, 9])
        top = next(iter(tree.root.children.values()))
        assert tuple(top.key) == (1,)
        assert sorted(tuple(c.key) for c in top.children.values()) == [(2, 3), (9,)]
        assert tree.total_tokens == 4
        assert_invariants(tree, [[1, 2, 3], [1, 9]])

    def test_three_way_split_keeps_sibling_tokens_distinct(self):
        tree = RadixTree()
        for tail in (3, 4, 5):
            tree.insert([1, 2, tail])
        top = next(iter(tree.root.children.values()))
        assert tuple(top.key) == (1, 2)
        assert sorted(top.children) == [3, 4, 5]
        assert tree.total_tokens == 5
        assert_invariants(tree, [[1, 2, 3], [1, 2, 4], [1, 2, 5]])

    def test_repeated_splits_keep_the_tree_consistent(self):
        tree = RadixTree()
        known = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6], [1, 7], [8]]
        for seq in known:
            tree.insert(seq)
        assert tree.total_tokens == 8
        assert leaf_paths(tree) == [(1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 6), (1, 7), (8,)]
        assert_invariants(tree, known)

    def test_insert_after_split_matches_both_branches(self):
        tree = RadixTree()
        tree.insert([1, 2, 3])
        tree.insert([1, 2, 4])
        assert tree.match_prefix([1, 2, 3]) == 3
        assert tree.match_prefix([1, 2, 4]) == 3
        assert tree.match_prefix([1, 2, 5]) == 2


class TestEvict:
    def test_evict_releases_a_whole_leaf_even_if_budget_is_smaller(self):
        tree = RadixTree()
        tree.insert([1, 2, 3])
        assert tree.evict(1) == 3
        assert tree.total_tokens == 0
        assert leaf_paths(tree) == []

    def test_evict_removes_the_least_recently_used_leaf(self):
        tree = RadixTree()
        tree.insert([1, 2, 3])
        tree.insert([4, 5, 6])
        assert tree.evict(3) == 3
        assert leaf_paths(tree) == [(4, 5, 6)]
        assert tree.match_prefix([1, 2, 3]) == 0
        assert tree.match_prefix([4, 5, 6]) == 3

    def test_evict_stops_once_the_budget_is_met(self):
        """Budget 4 against 3-token leaves releases 6, i.e. two leaves."""
        tree = RadixTree()
        for base in (1, 4, 7):
            tree.insert([base, base + 1, base + 2])
        assert tree.evict(4) == 6
        assert tree.total_tokens == 3
        assert leaf_paths(tree) == [(7, 8, 9)]

    def test_evict_more_than_the_tree_holds_releases_everything(self):
        tree = RadixTree()
        tree.insert([1, 2, 3])
        tree.insert([4, 5])
        assert tree.evict(100) == 5
        assert tree.total_tokens == 0
        assert tree.root.children == {}

    def test_evict_zero_releases_nothing(self):
        tree = RadixTree()
        tree.insert([1, 2, 3])
        assert tree.evict(0) == 0
        assert tree.total_tokens == 3
        assert leaf_paths(tree) == [(1, 2, 3)]

    def test_evict_rejects_negative_budget(self):
        tree = RadixTree()
        with pytest.raises(ValueError):
            tree.evict(-1)

    def test_evict_on_empty_tree_returns_zero(self):
        assert RadixTree().evict(5) == 0

    def test_match_prefix_refreshes_lru_order(self):
        tree = RadixTree()
        tree.insert([1, 2, 3])
        tree.insert([4, 5, 6])
        tree.match_prefix([1, 2, 3])
        assert tree.evict(3) == 3
        assert leaf_paths(tree) == [(1, 2, 3)]

    def test_reinsert_refreshes_lru_order(self):
        tree = RadixTree()
        tree.insert([1, 2, 3])
        tree.insert([4, 5, 6])
        tree.insert([1, 2, 3])
        tree.evict(3)
        assert leaf_paths(tree) == [(1, 2, 3)]

    def test_parent_becomes_a_leaf_candidate_only_in_a_later_round(self):
        tree = RadixTree()
        tree.insert([1, 2, 3])
        tree.insert([1, 2, 4])
        assert tree.total_tokens == 4
        assert tree.evict(1) == 1
        assert tree.total_tokens == 3
        assert tree.evict(1) == 1
        assert tree.total_tokens == 2
        assert leaf_paths(tree) == [(1, 2)]
        assert tree.evict(2) == 2
        assert tree.total_tokens == 0

    def test_evict_keeps_remaining_paths_as_inserted_prefixes(self):
        """Acceptance 4 with a branching tree."""
        tree = RadixTree()
        known = [[1, 2, 3], [1, 2, 4], [1, 5], [6, 7]]
        for seq in known:
            tree.insert(seq)
        tree.evict(2)
        assert_invariants(tree, known)

    def test_insert_after_full_eviction_works(self):
        tree = RadixTree()
        tree.insert([1, 2, 3])
        tree.evict(100)
        tree.insert([1, 2, 3])
        assert tree.match_prefix([1, 2, 3]) == 3
        assert tree.total_tokens == 3
        assert_invariants(tree, [[1, 2, 3]])


class TestRandomOps:
    def test_random_mix_keeps_invariants(self):
        rng = random.Random(20261025)
        tree = RadixTree()
        known: list[list[int]] = []

        for _ in range(400):
            op = rng.choices(["insert", "match", "evict"], weights=[5, 3, 2])[0]
            seq = [rng.randrange(4) for _ in range(rng.randrange(0, 7))]

            if op == "insert":
                if seq and seq not in known:
                    known.append(seq)
                tree.insert(seq)
                assert tree.match_prefix(seq) == len(seq), (
                    f"insert {seq} did not become fully matchable")
            elif op == "match":
                matched = tree.match_prefix(seq)
                assert 0 <= matched <= len(seq)
                assert matched <= tree.total_tokens
            else:
                budget = rng.randrange(0, 6)
                before = tree.total_tokens
                released = tree.evict(budget)
                assert released == before - tree.total_tokens, (
                    "evict must report exactly the tokens it removed")
                if budget == 0:
                    assert released == 0
                elif before > 0:
                    assert released > 0

            assert_invariants(tree, known)

        assert_every_leaf_matches_itself(tree)
