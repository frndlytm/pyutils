"""
@author: frndlytm
@name: utils.ml.features.analysis
@description:

    Utility functions for common feature analysis tasks, like
    counting activations, and getting sizes of categorical
    features.

    These utility functions should keep preparing sets for
    visualizations in mind.
"""
from itertools import combinations
from copy import deepcopy

def count_activations(data, features=None, axis=0):
    """count_activations tests for how sparsely populated the features of a
    DataFrame are. For example, a row full of zero-valued features might
    not be a useful record for making predictions. Grouping activations
    might also lend insight into the relationship between categorical
    features.
    """
    if not features: features = list(data.columns)

    # Get the featured data, transpose if necessary.
    activations = deepcopy(data[features])
    if axis: activations = activations.T

    # Sum the True features across the rows.
    to_activation = lambda df: df.fillna(0).astype(bool).sum(axis=1)
    activations = to_activation(activations)
    return activations


def get_sizes_by_categories(data, categories):
    """get_sizes_by_categories takes some data and some categories to analyze
    their relative sizes.
    """
    # Build the combinations of categories to analyze
    combos = []
    for n in range(len(categories)):
        combos += list(combinations(categories, n + 1))

    # Show sizes.
    for combo in combos:
        yield data.groupby(combo).size()