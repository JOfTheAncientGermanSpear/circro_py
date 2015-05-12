import tempfile, shutil, os

def inputs_to_list(*args):
    """
    >>> inputs_to_list('abc')
    ['abc']
    >>> inputs_to_list('abc', 'bde')
    ['abc', 'bde']
    """
    return list(args)

#http://preshing.com/20110920/the-python-with-statement-by-example/
class temp_dir:
    """
    >>> import pandas as pd
    >>> left = pd.Series(['l1', 'l2']) 
    >>> right = pd.Series(['r1', 'r2']) 
    >>> labels_df = pd.concat([left, right], axis = 1)
    >>> with temp_dir(labels_df) as fs:
    ...    with open(fs[0]) as f:
    ...        print(f.read())
    0,1
    l1,r1
    l2,r2
    <BLANKLINE>
    """
    def __init__(self, *dfs):
        self.d = tempfile.mkdtemp()
        self.create_tmp = lambda df: create_temporary_file(df, self.d)
        self._dfs = inputs_to_list(*dfs)

    def __enter__(self):
        self._paths = map(self.create_tmp, self._dfs)
        return self._paths

    def __exit__(self, type, value, traceback):
        for p in self._paths:
            f = open(p)
            if not f.closed:
                f.close()
        shutil.rmtree(self.d)


#http://stackoverflow.com/questions/6587516/how-to-concisely-create-a-temporary-file-that-is-a-copy-of-another-file-in-pytho
def create_temporary_file(df, temp_d):
    (_, f_path) = tempfile.mkstemp(dir = temp_d, text=True)
    df.to_csv(f_path, index=False)
    return f_path


if __name__ == "__main__":
    import doctest
    doctest.testmod()
