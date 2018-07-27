"""
@author: frndlytm
@name: utils.ml.features
@description:

    Utility functions for common feature generation tasks, like
    categorical features and interactivity features.
"""
import pandas as pd
from copy import deepcopy


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
    for category in categories:
        results = results.unstack(category)
    results.reset_index(inplace=True)

    # Clean up the column names by removing the axis=1 MultiIndex.
    def to_safe_str(tup, sep):
        out = f'{sep}'.join([str(x) for x in tup])
        out = out[:-1] if out[-1] == sep else out
        return out

    labels = list(map(lambda tup: to_safe_str(tup, sep), list(results)))
    results.set_axis(labels=labels, axis='columns', inplace=True)

    return results

