"""
@author: frndlytm
@name: utils.sql
@description:

    Python 3 functions for cleaning and parsing SQL queries.

"""
import re

def remove_expression(text, regex):
    """remove_expression takes a body of text
    and a regex and removes all occurences
    of the expression from the text
    """
    pattern = re.compile(regex)
    query, _ = pattern.subn('', text)
    return query


def remove_comments(query):
    """remove_comments strips the block comments
    and the line_comments from SQL stored
    procedures.
    """
    expressions = [r'(((/\*)+?[\w\W]+?(\*/)+))', r'(--.*)', r'\[\]']
    for regex in expressions:
        query = remove_expression(query, regex)
    return query

def clean_query(query, comments=True):
    """clean_query is a general utility that
    performs all clean-up operations on the
    query.
    """
    if comments:
        query = remove_comments(query)
    return query