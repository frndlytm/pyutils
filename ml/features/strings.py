"""
@author: frndlytm
@name: utils.ml.features.strings
@description:

    Utility functions for common feature generation tasks, like
    categorical features and interactivity features.
"""
import string
from copy import deepcopy

def try_cast(s, sub, t=int):
    """given a string, try casting elementwise to int.
    If True, return sub"""
    for i in s:
        try:
            i = t(i)
            yield sub
        except:
            yield i


def pad_string(s, n_chars, sub, side):
    """pad_string takes a string and makes sure it is
    at least n_chars. If not, it pads the string to the
    right with the substitution character.
    """
    l = len(s)
    if l < n_chars:
        to_pad = (n_chars - l) * sub
        if side == 'right':
            s = s + to_pad
        elif side == 'left':
            s = to_pad + s
        else:
            raise ValueError("'side' must by one of: ['right', 'left'].")
    return s


def clean_charset(s, charset):
    """clean_punctuation removes punctuation from a
    string.
    """
    s = filter(lambda x: x not in charset, s)
    s = ''.join(list(s))
    return s


def string_to_categorical(data, sub='X', n_chars=6, clean=True, pad=True, side='right'):
    """chars_to_categorical turns a Series of text into
    a categorical column by selecting the first n_chars
    from the string after some given substitutions.
    """
    # copy the data so it doesn't transform the data in
    # memory, overwriting your DataFrame
    result = deepcopy(data)
    result.fillna(value='X', inplace=True)

    # clean the string and pad it.
    if clean: result = result.apply(lambda s: clean_charset(s, string.punctuation+' '))
    if pad: result = result.apply(lambda s: pad_string(s, n_chars, sub, side))

    # convert numerics to chars; return the uppercase string.
    result = result.apply(lambda s: ''.join(list(try_cast(s, sub))))
    result = result.apply(lambda s: s[:n_chars].upper())
    return result