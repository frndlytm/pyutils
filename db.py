"""
@author: frndlytm
@name: utils.db
@description:

    Utility functions for working with databases in Python 3.
    In order for this project to work as intended, it will need
    to be added to the Python PATH in the OS.

"""
from urllib.parse import quote_plus
from utils.sql import clean_query

def connect(driver, port, server, database, flavor='mssql'):
    """Build the SQLAlchemy pyodbc connection string from
    configuration parameters for a given flavor (default to
    'mssql').
    """
    def params(driver, port, server, database):
        """Using quote_plus, build the odbc_connect param
        for the connection string.
        """
        return quote_plus(
            'DRIVER={};PORT={};SERVER={};DATABASE={};Trusted_Connection=yes;'
            .format(driver, port, server, database)
        )

    # Prepare params, and format the string.
    p = params(driver, port, server, database)
    return '{}+pyodbc:///?odbc_connect={}'.format(flavor, p)



def read_query_from_file(file, cleanup=False):
    """Take a well-formatted .sql file and read the query,
    returned as output. If cleanup is requested, run some
    clean-up utilities and return.
    """
    with open(file, 'r') as f:
        query = f.read()

    if cleanup:
        query = clean_query(query)

    return query

