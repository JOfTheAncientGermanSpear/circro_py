from pandas import read_csv, concat, Series, DataFrame
import numpy as np

def inputsToDict(**kwords):
    return kwords

class InputError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

def read_node_file(filename):
    df = read_csv(filename, header=None)
    return concat([df[0], df[1]], keys=['left', 'right']).to_frame()


def make_circro(labels = None, sizes = None, colors = None, edge_matrix = None):
    data_filenames = inputsToDict(labels = labels, sizes = sizes, 
            colors = colors, edge_matrix = edge_matrix)

    file_keys = ['labels', 'sizes', 'colors', 'edge_matrix']

    res = {}

    if not any(f in data_filenames for f in file_keys):
        raise InputError("at least one of {} inputs must be set".format(file_keys))

    node_dfs = {k: read_node_file(f) 
            for k,f in data_filenames.items() if f and 'edge' not in k}

    for k, df in node_dfs.items():
        df.columns = [k[:-1]] 


    res['nodes'] = concat(list(node_dfs.values()), axis = 1)

    if edge_matrix:
        res['edges'] = read_csv(edge_matrix, header=None)

    if labels:
        labels = res['nodes']['label']

        if 'edges' in res:
            res['edges'].columns = labels.values
            res['edges'].index = labels.values

    return res
