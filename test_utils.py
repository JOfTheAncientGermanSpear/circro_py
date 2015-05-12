import tempfile, shutil, os

def inputs_to_list(*args):
    """
    >>> inputs_to_list('abc')
    ['abc']
    >>> inputs_to_list('abc', 'bde')
    ['abc', 'bde']
    """
    return list(args)


def _inputs_to_dict(**kwords):
    return kwords


#http://preshing.com/20110920/the-python-with-statement-by-example/
class temp_dir:
    """
    >>> import pandas as pd
    >>> left = pd.Series(['l1', 'l2']) 
    >>> right = pd.Series(['r1', 'r2']) 
    >>> labels_df = pd.concat([left, right], axis = 1)
    >>> import os
    >>> with temp_dir(labels = labels_df) as d:
    ...    f_path = os.path.join(d, 'labels.csv')
    ...    with open(f_path) as f:
    ...        print(f.read())
    0,1
    l1,r1
    l2,r2
    <BLANKLINE>
    """
    def __init__(self, labels = None, sizes = None, colors = None, edge_matrix = None):
        f_names = {'edge_matrix': 'edge_matrix.csv',
                'labels': 'labels.csv', 'sizes': 'sizes.csv', 'colors': 'colors.csv'}
        inputs_dict = _inputs_to_dict(labels = labels, sizes = sizes, colors = colors, edge_matrix = edge_matrix)
        self.d = tempfile.mkdtemp()
        self.create_tmp = lambda t: create_temporary_file(inputs_dict[t], self.d, f_names[t], t != 'edge_matrix')
        self._ts = {t for t in f_names if inputs_dict[t] is not None}

    def __enter__(self):
        self._paths = map(self.create_tmp, self._ts)
        return self.d

    def __exit__(self, type, value, traceback):
        for p in self._paths:
            f = open(p)
            if not f.closed:
                f.close()
        shutil.rmtree(self.d)


#http://stackoverflow.com/questions/6587516/how-to-concisely-create-a-temporary-file-that-is-a-copy-of-another-file-in-pytho
def create_temporary_file(df, temp_d, f_name, header):
    f_path = os.path.join(temp_d, f_name)
    df.to_csv(f_path, index=False, header=header)
    return f_path


if __name__ == "__main__":
    import doctest
    doctest.testmod()
