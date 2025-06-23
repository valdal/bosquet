from collections import Counter

import pytest

from bosquet.trees import Tree


@pytest.fixture(autouse=True)
def reset_tree():
    Tree.set_num_colors(1)


class TestTree:
    def test_initialization(self):
        t1 = Tree(0, 0)
        assert t1.index == (0, 0)

        t2 = Tree(10)
        assert t2.index == (10, 0)

        with pytest.raises(ValueError):
            Tree(1, 1)

        with pytest.raises(ValueError):
            Tree(-1)

        with pytest.raises(ValueError):
            Tree(1, -1)

        Tree.set_num_colors(2)
        t3 = Tree(1, 1)
        assert t3.index == (1, 1)

    def test_generation(self):
        # Test generating trees with a single color
        assert Tree._order_max == 0
        trees_up_to_3 = [(0, 0), (1, 0), (1, 1), (2, 1), (1, 2)]
        colors_up_to_3 = [[(0, 0)], [(0, 0)], [(0, 0)], [(0, 0)], [(0, 0)]]

        Tree.generate_trees(3)
        assert Tree._order_max == 3
        assert Tree._index_first == [0, 1, 2, 3, 5]
        assert Tree._comp == trees_up_to_3
        assert Tree._comp_colors == colors_up_to_3

        Tree.generate_index(10)
        assert Tree._order_max == Tree(10).order

        Tree.generate_trees(10)
        assert all(len(comp) == 1 for comp in Tree._comp_colors)

        with pytest.raises(ValueError):
            Tree.generate_trees(-1)

        with pytest.raises(ValueError):
            Tree.generate_index(-1)

        # Test generating trees with multiple colors
        Tree.set_num_colors(2)
        Tree.generate_trees(2)
        assert Tree._order_max == 2
        assert Tree._index_first == [0, 1, 2, 3]
        assert Tree._comp == [(0, 0), (1, 0), (1, 1)]
        assert Tree._comp_colors == [[(0, 0)], [(0, 0), (1, 0)], [(0, 0), (1, 0), (0, 1), (1, 1)]]

        Tree.set_num_colors(5)
        Tree.generate_trees(1)
        assert len(Tree._comp_colors[1]) == 5

    def test_generation_linear(self):
        # Test generating linear trees with a single color
        Tree.set_num_colors(1, [True])
        Tree.generate_trees(3)
        assert Tree._order_max == 3
        assert Tree._index_first == [0, 1, 2, 3, 4]
        assert Tree._comp == [(0, 0), (1, 0), (1, 1), (1, 2)]

        # Test generating linear trees with multiple colors
        Tree.set_num_colors(2, [False, True])
        Tree.generate_trees(3)
        assert Tree._order_max == 3
        assert Tree._index_first == [0, 1, 2, 3, 5]

    def test_clear(self):
        # Test clearing with no previous generation
        Tree.clear()

        # Test clearing the tree cache
        t = Tree(10)
        _ = t.order
        assert Tree._order_max > 0
        assert hasattr(Tree, "_order_cache")

        Tree.clear()
        assert Tree._order_max == 0
        assert Tree._comp == [(0, 0)]
        assert Tree._comp_colors == [[(0, 0)]]
        assert Tree._index_first == [0, 1]

    def test_num_colors(self):
        # Test the default number of colors
        assert Tree.num_colors() == 1

        # Test setting the number of colors
        Tree.set_num_colors(3)
        assert Tree.num_colors() == 3

        with pytest.raises(ValueError):
            Tree.set_num_colors(0)

    def test_is_linear(self):
        # Test the default linearity
        assert Tree.is_linear(0) is False

        # Test setting linearity for a single color
        Tree.set_num_colors(1, [True])
        assert Tree.is_linear(0) is True

        # Accessing linearity for a non-existent color should raise an error
        with pytest.raises(ValueError):
            Tree.is_linear(1)

        # Setting wrong number of elements for linearity should raise an error
        with pytest.raises(ValueError):
            Tree.set_num_colors(1, [True, False])

        # Test setting multiple colors and checking linearity
        Tree.set_num_colors(2)
        assert Tree.is_linear(0) is False
        assert Tree.is_linear(1) is False

    def test_cmp(self):
        # Test comparing trees
        t1 = Tree(1)
        t2 = Tree(10)

        assert t1 < t2
        assert not (t2 < t1)

        # Test equality
        assert t1 == Tree(1, 0)
        assert hash(t1) == hash(Tree(1, 0))
        assert t2 != Tree(5, 0)
        assert hash(t2) != hash(Tree(5, 0))

        class NotTree:
            pass

        assert Tree(1) != NotTree()
        with pytest.raises(TypeError):
            assert Tree(1) < NotTree()

        # Test comparison with different colors
        Tree.set_num_colors(2)
        assert Tree(1, 0) < Tree(1, 1)

    def test_repr(self):
        # Test string representation of trees
        assert repr(Tree(0)) == "Tree(0)"
        assert repr(Tree(1, 0)) == "Tree(1)"
        assert repr(Tree(10)) == "Tree(10)"

        # Test with multiple colors
        Tree.set_num_colors(3)
        assert repr(Tree(0, 0)) == "Tree(0, 0)"
        assert repr(Tree(1, 2)) == "Tree(1, 2)"
        assert repr(Tree(3, 1)) == "Tree(3, 1)"

    def test_indices(self):
        # Test getting indices of trees
        Tree.set_num_colors(1)
        assert Tree.indices(0) == range(0, 1)
        assert Tree.indices(1) == range(1, 2)
        assert Tree.indices(3) == range(3, 5)

        assert Tree.colorings(3) == range(0, 1)

        # Test with multiple colors
        Tree.set_num_colors(3)
        assert Tree.indices(0) == range(0, 1)
        assert Tree.indices(1) == range(1, 2)
        assert Tree.indices(3) == range(3, 5)

        assert Tree.colorings(1) == range(0, 3)
        assert Tree.colorings(3) == range(0, 27)

    def test_decompose(self):
        t = Tree(10)
        assert t.left == Tree(6)
        assert t.right == Tree(1)
        assert t.decompose == (Tree(6), Tree(1))

        Tree.set_num_colors(3)
        t = Tree(10, 5)
        assert t.left == Tree(6, 1)
        assert t.right == Tree(1, 2)
        assert t.decompose == (Tree(6, 1), Tree(1, 2))

    def test_circ(self):
        # Test circ product for all trees up to order 5
        for i in range(19):
            t = Tree(i)
            assert Tree.circ(t.left, t.right) == t

        # Test circ product with non-standard left, right decomposition
        assert Tree.circ(Tree(4), Tree(1)) == Tree.circ(Tree(2), Tree(2))

    def test_from_children(self):
        # Test from children with Counter input
        children = Counter({Tree(1): 2, Tree(2): 1})
        t = Tree.from_children(children)
        assert t == Tree(10)

        # Test from children with a list of trees
        children_list = [Tree(1), Tree(1), Tree(2)]
        t = Tree.from_children(children_list)
        assert t == Tree(10)

        # Test from children with multiple colors
        Tree.set_num_colors(3)
        children_list = [Tree(1, 2), Tree(3, 5)]
        t = Tree.from_children(children_list)
        assert t == Tree(11, 47)

        # Test from children with an empty Counter
        t = Tree.from_children(Counter(), root=2)
        assert t == Tree(1, 2)

    def test_merge_root(self):
        t1 = Tree(4)
        t2 = Tree(2)

        assert Tree.merge_root(t1, t2) == Tree(6)

        Tree.set_num_colors(3)
        t1 = Tree(4, 1)
        t2 = Tree(2, 3)
        assert Tree.merge_root(t1, t2) == Tree(6, 1)

    def test_root(self):
        assert Tree(10).root == 0
        assert Tree(0).root == -1

        Tree.set_num_colors(5)
        for i in range(5):
            assert Tree(1, i).root == i

    def test_order(self):
        # Test order of trees with a single color
        assert Tree(0).order == 0
        assert Tree(1).order == 1
        assert Tree(2).order == 2
        assert Tree(10).order == 5

        # Test order of trees with multiple colors
        Tree.set_num_colors(3)
        assert Tree(0, 0).order == 0
        assert Tree(1, 0).order == 1
        assert Tree(10, 1).order == 5

    def test_gamma(self):
        expected_gamma = [1, 1, 2, 3, 6, 4, 8, 12, 24, 5, 10, 15, 30, 20, 20, 40, 60, 120]
        for i in range(len(expected_gamma)):
            assert Tree(i).gamma == expected_gamma[i]

        Tree.set_num_colors(3)
        assert Tree(10, 5).gamma == Tree(10).gamma

    def test_sigma(self):
        expected_sigma = [1, 1, 1, 2, 1, 6, 1, 2, 1, 24, 2, 2, 1, 2, 6, 1, 2, 1]
        for i in range(len(expected_sigma)):
            assert Tree(i).sigma == expected_sigma[i]

        Tree.set_num_colors(3)
        assert Tree(10, 5).sigma == Tree(10).sigma

    def test_height(self):
        expected_height = [0, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 2, 3, 3, 4]
        for i in range(len(expected_height)):
            assert Tree(i).height == expected_height[i]

        Tree.set_num_colors(3)
        assert Tree(10, 5).height == Tree(10).height

    def test_width(self):
        expected_width = [0, 1, 1, 2, 1, 3, 2, 2, 1, 4, 3, 3, 2, 2, 3, 2, 2, 1]
        for i in range(len(expected_width)):
            assert Tree(i).width == expected_width[i]

        Tree.set_num_colors(3)
        assert Tree(10, 5).width == Tree(10).width

    def test_children(self):
        # Test children of trees with a single color
        t = Tree(10)
        assert t.children == Counter({Tree(1): 2, Tree(2): 1})

        expected_children = [
            Counter(),
            Counter(),
            Counter({Tree(1): 1}),
            Counter({Tree(1): 2}),
            Counter({Tree(2): 1}),
            Counter({Tree(1): 3}),
            Counter({Tree(2): 1, Tree(1): 1}),
            Counter({Tree(3): 1}),
            Counter({Tree(4): 1}),
        ]
        for i in range(9):
            assert Tree(i).children == expected_children[i]

        # Test children of trees with multiple colors
        Tree.set_num_colors(3)
        assert Tree(10, 5).children == Counter({Tree(2, 0): 1, Tree(1, 1): 1, Tree(1, 2): 1})
