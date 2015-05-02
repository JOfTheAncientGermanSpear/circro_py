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
    >>> with temp_dir('data/labels.csv') as fs:
    ...    print("hello")
    hello
    >>> import os
    >>> base = lambda f: os.path.basename(f)
    >>> with temp_dir('data/labels.csv', 'data/sizes.csv') as fs:
    ...    print(base(fs[0]) + ', ' + base(fs[1]))
    labels.csv, sizes.csv
    """
    def __init__(self, *paths):
        self.d = tempfile.mkdtemp()
        self.create_tmp = lambda p: create_temporary_copy(p, self.d)
        self._paths = inputs_to_list(*paths)

    def __enter__(self):
        self.cps = map(self.create_tmp, self._paths)
        return self.cps

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self.d)


#http://stackoverflow.com/questions/6587516/how-to-concisely-create-a-temporary-file-that-is-a-copy-of-another-file-in-pytho
def create_temporary_copy(src_f, temp_d):
    f_name = os.path.basename(src_f)
    temp_path = os.path.join(temp_d, f_name)

    shutil.copy2(src_f, temp_path)
    return temp_path


if __name__ == "__main__":
    import doctest
    doctest.testmod()
