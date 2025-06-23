import pytest

from bosquet.trees import cache


@pytest.fixture(autouse=True)
def reset_tree_cache():
    # Remove or reset all cache attributes before each test
    for attr in ["_f_index_cache", "_f_color_cache", "_f_both_cache", "_f_invalid_cache"]:
        if hasattr(Tree, attr):
            delattr(Tree, attr)


# Minimal Tree class for testing
class Tree:
    def __init__(self, a, b):
        self.index = (a, b)
        self.call_count = 0

    @cache("index")
    def f_index(self):
        self.call_count += 1
        return self.index[0]

    @cache("color")
    def f_color(self):
        self.call_count += 1
        return self.index[1]

    @cache("both")
    def f_both(self):
        self.call_count += 1
        return self.index

    @cache("invalid")
    def f_invalid(self):
        self.call_count += 1
        return 42


class TestCache:
    def test_cache_basic_and_caching(self):
        t1 = Tree(1, 2)

        assert t1.f_index() == 1
        assert t1.call_count == 1  # First call should compute
        assert t1.f_index() == 1
        assert t1.call_count == 1  # Second call should hit cache

        assert t1.f_color() == 2
        assert t1.call_count == 2  # First call should compute
        assert t1.f_color() == 2
        assert t1.call_count == 2  # Second call should hit cache

        assert t1.f_both() == (1, 2)
        assert t1.call_count == 3  # First call should compute
        assert t1.f_both() == (1, 2)
        assert t1.call_count == 3  # Second call should hit cache

        assert hasattr(Tree, "_f_index_cache")
        assert isinstance(Tree._f_index_cache, dict)
        assert Tree._f_index_cache == {1: 1}

        assert hasattr(Tree, "_f_color_cache")
        assert isinstance(Tree._f_color_cache, dict)
        assert Tree._f_color_cache == {2: 2}

        assert hasattr(Tree, "_f_both_cache")
        assert isinstance(Tree._f_both_cache, dict)
        assert Tree._f_both_cache == {(1, 2): (1, 2)}

    def test_cache_shared_across_instances(self):
        t1 = Tree(1, 2)
        t2 = Tree(1, 3)  # Same index[0] as t1 for f_index

        assert t1.f_index() == 1
        assert t1.call_count == 1

        assert t2.f_index() == 1  # Should hit cache, not increment t2.call_count
        assert t2.call_count == 0
        assert t1.call_count == 1  # Still 1

    def test_cache_multiple_keys(self):
        t1 = Tree(1, 2)
        t2 = Tree(3, 2)
        t3 = Tree(1, 4)

        t1.f_index()
        t2.f_index()
        t3.f_index()

        assert Tree._f_index_cache == {1: 1, 3: 3}

    def test_cache_invalid(self):
        t = Tree(1, 2)
        with pytest.raises(ValueError):
            t.f_invalid()

        assert t.call_count == 0
        assert not hasattr(Tree, "_f_invalid_cache")
