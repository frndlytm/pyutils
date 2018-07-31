"""
@author: frndlytm
@name: utils.ml.features
@description:

    Utility functions for common feature generation tasks, like
    categorical features and interactivity features.
"""

from itertools import combinations
from copy import deepcopy

from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


"""
Categroical Features
"""
def encode_categories(data, keys=None, categories=None, prefix='tkn', copy=True):
    """Personally, I don't like the sklearn implementation of a OneHotEncoder
    since it's finnicky and returns a SparseDataFrame. In a Pipeline, I'm sure
    this works great, but for exploration and feature generation, not so much.
    I find pandas.get_dummies() significantly more intuitive.
    """
    if copy: keys, categories = deepcopy(keys), deepcopy(categories)

    if not keys:
        raise ValueError(f'"keys" cannot be {keys}. Please provide and valid Index.')

    # Make sure categories is populated. By default, encode the whole DataFrame.
    if not categories: categories = list(data.drop(keys))

    # For each category, prepare the list of output column names using a prefix.
    for feature in categories:
        keys = keys + [f'{prefix}_' + f for f in list(data[feature].drop_duplicates())]

    # Return a one-hot encoded DataFrame, unique on keys.
    results = (
        pd.get_dummies(
            data, prefix=prefix, columns=categories
        )[keys].drop_duplicates()
    )
    return results


def unstack_categories(data, keys=None, categories=None, features=None, sep='_'):
    """unstack_categories takes some data and trades rows for columns for the
    given categories. For example, given the following data:

        x | category | numeric1 | numeric2
        1 | 1        | 50       | 100
        1 | 2        | 10       | 20
        2 | 1        | 20       | 40
        3 | 1        | 20       | 40
        3 | 2        | 50       | 100

    Unstack transforms the DataFrame as follows

        x | numeric1,1 | numeric1,2 | numeric2,1 | numeric2,2
        1 | 50         | 10         | 100        | 20
        2 | 20         | NaN        | 40         | NaN
        3 | 20         | 50         | 40         | 100

    Unstacking features is useful when there is a category which causes duplication
    in the level you're trying to learn at.

    For example, there might be categories on a product, and categories of services
    the products are attached to that define a price you're trying to predict.

    Since the service and the product are independent, you can choose to learn about
    the service, the product, or the (service,product) depending on the application;
    however, learning about them independently requires rows to be uniquely defined
    by one or the other.

    The function takes the parameters:
        - data:         a pandas.DataFrame
        - keys:         a list of look-up columns that are required.
        - categories:   a list of categories to unstack.
        - features:     a list of features to unstack by category.
        - sep:          a separator when joining the feature name and category
                            values into a list of new column headers.
    """
    if not keys:
        raise ValueError(f'"keys" cannot be {keys}. Please provide and valid Index.')

    if not features:
        features = list(data.drop(categories + keys))

    # Build a result set by unstacking the data.
    results = data.set_index(keys + categories, verify_integrity=True)[features]
    results = results.unstack(categories, fill_value=0.0)
    results.reset_index(inplace=True)

    # Clean up the column names by removing the axis=1 MultiIndex.
    def to_safe_str(tup, sep):
        out = f'{sep}'.join([str(x) for x in tup])
        out = out[:-1] if out[-1] == sep else out
        return out

    labels = list(map(lambda tup: to_safe_str(tup, sep), list(results.columns)))
    results.set_axis(labels=labels, axis='columns', inplace=True)

    return results


def add_interactions(data, degree=2):
    """add_interactions was in a presentation given by April Chen here:

        https://www.youtube.com/watch?v=V0u6bxQOUJ8

    To add feature interactions to your DataFrame, get combinations of
    column names and fit_transform PolynomialFeatures transformer on the
    data frame.

    Layer back in the column names and do some clean-up.

    In her implementation, she used feature pairs, but to make this a little
    more dynamic, I added the degree parameter to manage the number of interacting
    features.
    """
    # Get feature names
    combos = list(combinations(list(data.columns), degree))
    colnames = list(data.columns) + ['_'.join(x) for x in combos]

    # Find interactions
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    data = poly.fit_transform(data)
    data = pd.DataFrame(data)
    data.set_axis(colnames, axis=1)

    # Remove interaction terms with all 0 values
    noints = [i for i, x in enumerate(list((data == 0).all())) if x]
    data = data.drop(noints, axis=1)

    return data



"""
Feature Analysis
"""
def count_activations(data, features=None, axis=0):
    """count_activations tests for how sparsely populated the features of a
    DataFrame are. For example, a row full of zero-valued features might
    not be a useful record for making predictions. Grouping activations
    might also lend insight into the relationship between categorical
    features.
    """
    if not features: features = list(data.columns)

    # Get the featured data, transpose if necessary.
    activations = data[features]
    if axis: activations = activations.T

    # Sum the True features across the rows.
    to_activation = lambda df: df.fillna(0).astype(bool).sum(axis=0)
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
        yield data.grouby(combo).size()
