from pandas import read_csv

def namedInputsToDict(**inputs):
    return inputs

class InputError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

def make_circro(labels = None, sizes = None, colors = None, edge_matrix = None):
    inputs = namedInputsToDict(labels = labels, sizes = sizes, 
            colors = colors, edge_matrix = edge_matrix)

    if not len(inputs):
        raise InputError("at least one of the method inputs must be set")

    return {i: (read_csv(v, header=None) if isinstance(v, str) else v)
        for i,v in inputs.items() if v}
