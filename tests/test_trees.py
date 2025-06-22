from bosquet.trees import Tree


def test_tree_generation():
    trees_up_to_3 = [
        [((0, 0), (0, 0))],
        [((1, 0), (0, 0))],
        [((1, 0), (1, 0))],
        [((2, 0), (1, 0))],
        [((1, 0), (2, 0))]
]
    
    Tree.generate_trees(3)
    assert Tree._comp == trees_up_to_3