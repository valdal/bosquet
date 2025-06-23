from collections import Counter
from functools import cache as fun_cache
from functools import reduce, total_ordering
from itertools import product
from math import ceil, factorial
from typing import Callable, ClassVar, TypeVar, cast

TreeIndex = tuple[int, int]
"""Type for tree index represented as a tuple of two integers: (uncolored tree index, color index)."""

# Caching decorator for tree methods
R = TypeVar("R")


def cache(key_part: str) -> Callable[[Callable[["Tree"], R]], Callable[["Tree"], R]]:
    def decorator(func: Callable[["Tree"], R]) -> Callable[["Tree"], R]:
        cache_attr = f"_{func.__name__}_cache"

        def wrapper(self: "Tree") -> R:
            key: int | tuple[int, int]
            if key_part == "index":
                key = self.index[0]
            elif key_part == "color":
                key = self.index[1]
            elif key_part == "both":
                key = self.index
            else:
                raise ValueError(f"Invalid key part: {key_part}, must be 'index', 'color' or 'both'")

            cache = getattr(self.__class__, cache_attr, None)
            if cache is None:
                cache = {}
                setattr(self.__class__, cache_attr, cache)

            if key in cache:
                return cast(R, cache[key])

            result = func(self)
            cache[key] = result
            return result

        return wrapper

    return decorator


@total_ordering
class Tree:
    """A class representing a rooted tree and its operations.

    A tree is represented by its index in the sequence of all trees.
    """

    _num_colors: ClassVar[int] = 1
    _is_linear: ClassVar[list[bool]] = [False]

    _order_max: ClassVar[int] = 0
    _comp: ClassVar[list[tuple[int, int]]] = [(0, 0)]
    _comp_colors: ClassVar[list[list[tuple[int, int]]]] = [[(0, 0)]]
    _index_first: ClassVar[list] = [0, 1]

    @classmethod
    def clear(cls) -> None:
        """Clears all generated trees and caches values for this class."""

        cls._order_max = 0
        cls._comp = [(0, 0)]
        cls._comp_colors = [[(0, 0)]]
        cls._index_first = [0, 1]

        # Clear all cached values
        for attr_name in dir(cls):
            if attr_name.endswith("_cache") and attr_name.startswith("_"):
                delattr(cls, attr_name)

    @classmethod
    def set_num_colors(cls, n: int, is_linear: list[bool] | None = None) -> None:
        """Set the number of color for the trees

        Args:
            n (int): Number of colors
        """
        if n < 1:
            raise ValueError("Number of colors must be at least 1")
        if is_linear is None:
            is_linear = [False] * n
        if len(is_linear) != n:
            raise ValueError(f"len(is_linear) must be {n}, got {len(is_linear)}")

        cls._num_colors = n
        cls._is_linear = is_linear

        cls.clear()

    @classmethod
    def num_colors(cls) -> int:
        return cls._num_colors

    @classmethod
    def is_linear(cls, root: int) -> bool:
        if root < 0 or root >= cls._num_colors:
            raise ValueError(f"Root must be in range [0, {cls._num_colors - 1}], got {root}")
        return cls._is_linear[root]

    def __init__(self, index: int, coloring: int = 0) -> None:
        """Initialize a Tree with the given index

        Args:
            index int: The index of the uncolored tree.
            coloring (int, optional): The index of the coloring of the tree. Defaults to 0.
        """
        if index < 0:
            raise ValueError("Tree index must be non-negative")
        if coloring < 0:
            raise ValueError("Color index must be non-negative")

        # Generate trees until we have the index
        self.generate_index(index)

        if coloring >= len(self._comp_colors[index]):
            raise ValueError(f"Color index must be less than {len(self._comp_colors[index])}, got {coloring}")

        self.index = (index, coloring)

    def __eq__(self, other: object) -> bool:
        """Return True if the trees have the same index."""
        # Ensure both are from same dynamic class (same number of colors)
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.index == other.index

    def __hash__(self) -> int:
        """Return the hash of the tree's index."""
        return hash((type(self), self.index))

    def __lt__(self, other: object) -> bool:
        """Return True if the first trees have smaller index than other."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.index < other.index

    def __repr__(self) -> str:
        """Return a string representation of the tree."""
        if self.num_colors() == 1:
            return f"Tree({self.index[0]})"
        else:
            return f"Tree({self.index[0]}, {self.index[1]})"

    @classmethod
    def generate_trees(cls, new_order: int) -> None:
        """Generate all trees up to the specified order.

        Args:
            new_order (int): The maximum order of trees to generate

        Examples:
            >>> Tree.generate_trees(5)
        """
        if new_order < 0:
            raise ValueError("Order must be non-negative")

        for order in range(cls._order_max + 1, new_order + 1):
            for j in range(1, ceil(order / 2)):
                idxLeft = cls.indices(order - j)
                idxRight = cls.indices(j)
                for left, right in product(idxLeft, idxRight):
                    if not cls.is_linear(Tree(left).root) and Tree(left).right >= Tree(right):
                        cls._comp.append((left, right))
                        cls._comp_colors.append([])
                        for lc, rc in product(cls.colorings(left), cls.colorings(right)):
                            cls._comp_colors[-1].append((lc, rc))

            for r in cls.indices(order - 1):
                cls._comp.append((1, r))
                cls._comp_colors.append([])
                for rc in cls.colorings(r):
                    for root in range(cls._num_colors):
                        cls._comp_colors[-1].append((root, rc))

            cls._index_first.append(len(cls._comp))
            cls._order_max = order

    @classmethod
    def generate_index(cls, index: int) -> None:
        if index < 0:
            raise ValueError("Index must be non-negative")

        order = cls._order_max + 1
        while index >= len(cls._comp):
            cls.generate_trees(order)
            order += 1

    @classmethod
    def indices(cls, order: int) -> range:
        """Return the indices for all trees of specified order.

        Args:
            order: The order of trees to get indices for

        Returns:
            A range of indices for trees of the specified order

        Examples:
            >>> [Tree(i) for i in Tree.indices(4)]
            [Tree(5), Tree(6), Tree(7), Tree(8)]
        """
        cls.generate_trees(order)
        return range(cls._index_first[order], cls._index_first[order + 1])

    @classmethod
    def colorings(cls, index: int) -> range:
        cls.generate_index(index)
        return range(len(cls._comp_colors[index]))

    @property
    def left(self) -> "Tree":
        """Return the left tree in the circ product.

        Returns:
            A Tree instance representing the left tree

        Examples:
            >>> t = Tree(4)
            >>> t.left.index
            1
        """
        idx = self._comp[self.index[0]][0]
        col = self._comp_colors[self.index[0]][self.index[1]][0]
        return Tree(idx, col)

    @property
    def right(self) -> "Tree":
        """Return the right tree in the circ product.

        Returns:
            A Tree instance representing the right tree

        Examples:
            >>> t = Tree(4)
            >>> t.right.index
            2
        """
        idx = self._comp[self.index[0]][1]
        col = self._comp_colors[self.index[0]][self.index[1]][1]
        return Tree(idx, col)

    @property
    def decompose(self) -> tuple["Tree", "Tree"]:
        """Return a tuple containing the left and right trees.

        Returns:
            A tuple (left_tree, right_tree)

        Examples:
            >>> t = Tree(4)
            >>> left, right = t.decompose()
            >>> l.index, r.index
            (1, 2)
        """
        idx = self._comp[self.index[0]]
        col = self._comp_colors[self.index[0]][self.index[1]]
        return Tree(idx[0], col[0]), Tree(idx[1], col[1])

    @classmethod
    @fun_cache
    def circ(cls, left: "Tree", right: "Tree") -> "Tree":
        sum_order = left.order + right.order
        cls.generate_trees(sum_order)
        try:
            # Get the index (without coloring)
            idx = cls._comp.index(
                (left.index[0], right.index[0]), cls._index_first[sum_order], cls._index_first[sum_order + 1]
            )
            # Find the color index
            col = cls._comp_colors[idx].index((left.index[1], right.index[1]))
            return Tree(idx, col)
        except ValueError:
            return cls.from_children(left.children + Counter([right]), left.root, check_order=False)

    @classmethod
    def from_children(cls, children: Counter["Tree"] | list["Tree"], root: int = 0, check_order: bool = True) -> "Tree":
        if not isinstance(children, Counter):
            children = Counter(children)

        sorted_children = sorted(children.elements(), reverse=True)

        if check_order:
            order = sum(t.order * k for t, k in children.items())
            cls.generate_trees(order)
        return reduce(cls.circ, sorted_children, Tree(1, root))

    @classmethod
    def merge_root(cls, left: "Tree", right: "Tree", root: int = 0) -> "Tree":
        return cls.from_children(left.children + right.children, root)

    @property
    @cache(key_part="both")
    def root(self) -> int:
        """Return the root of the tree"""
        if self.index[0] == 0:
            return -1
        if self.index[0] == 1:
            return self.index[1]
        return self.left.root

    @property
    @cache(key_part="index")
    def order(self) -> int:
        """ """
        if self.index[0] == 0:
            return 0
        if self.index[0] == 1:
            return 1
        left, right = self.decompose
        return left.order + right.order

    @property
    @cache(key_part="index")
    def gamma(self) -> int:
        """Return the density of the tree.

        Returns:
            The density of the tree

        Examples:
            >>> t = Tree(5)
            >>> t.gamma
            4
        """
        if self.index[0] <= 1:
            return 1
        g = self.order
        for tree, count in self.children.items():
            g *= tree.gamma**count

        return g

    @property
    @cache(key_part="index")
    def sigma(self) -> int:
        """Return the symmetry of the tree.

        Returns:
            The symmetry of the tree

        Examples:
            >>> t = Tree(5)
            >>> t.sigma
            6
        """
        s = 1
        for c in self.children.items():
            s *= factorial(c[1]) * c[0].sigma ** c[1]
        return s

    @property
    def height(self) -> int:
        """Return the height of the tree.

        The height is defined as:
        - height(t_0) = height(t_1) = 0
        - height(t) = max(height(t') for t' in children) + 1

        Returns:
            The height of the tree

        Examples:
            >>> t = Tree(4)
            >>> t.height
            2
        """
        if self.index[0] <= 1:
            return 0
        return max(c.height for c, _ in self.children.items()) + 1

    @property
    def width(self) -> int:
        """Return the width of the tree.

        The width is defined as:
        - width(t_0) = 0
        - width(t_1) = 1
        - width(t) = sum(k * width(t') for t',k in children.items())

        Returns:
            The width of the tree

        Examples:
            >>> t = Tree(4)
            >>> t.width
            1
        """
        if self.index[0] <= 1:
            return self.index[0]
        return sum(k * c.width for c, k in self.children.items())

    @property
    @cache(key_part="both")
    def children(self) -> Counter["Tree"]:
        """Return the trees obtained after removing the root.

        The children are returned as a Counter mapping each subtree
        to its multiplicity (how many times it appears).

        Returns:
            A Counter mapping Tree instances to their multiplicities

        Examples:
            >>> t = Tree(4)
            >>> t.children  # Tree 4 has two copies of Tree 1 and one of Tree 2
            Counter({Tree(2): 1})

            >>> t = Tree(10)
            >>> for child, mult in t.children.items():
            ...     print(f"Tree {child.index} appears {mult} times")
            Tree 2 appears 1 times
            Tree 1 appears 2 times

            >>> # Sum multiplicities of all children
            >>> sum(t.children.values())
            3
        """
        if self.index[0] <= 1:
            return Counter()

        left, right = self.decompose
        return left.children + Counter([right])
