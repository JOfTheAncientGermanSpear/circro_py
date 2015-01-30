from pandas import read_csv, concat, Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt

def _inputs_to_dict(**kwords):
    return kwords

class InputError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

def _read_node_file(filename):
    df = read_csv(filename, header=None)
    return concat([df[0], df[1]], keys=['left', 'right']).to_frame()


def make_circro(labels = None, sizes = None, colors = None, edge_matrix = None):
    data_filenames = _inputs_to_dict(labels = labels, sizes = sizes, 
            colors = colors, edge_matrix = edge_matrix)

    file_keys = ['labels', 'sizes', 'colors', 'edge_matrix']

    res = {}

    if not any(f in data_filenames for f in file_keys):
        raise InputError("at least one of {} inputs must be set".format(file_keys))

    node_dfs = {k: _read_node_file(f) 
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

def _get_draw_info(node, side, index, rad_per_node):
    res = {}
    res['inner_r'] = 1.0
    res['outer_r'] = res['inner_r'] + node['size'] if 'size' in node else .5
    res['width'] = rad_per_node if side is 'right' else -1 * rad_per_node
    res['theta'] = index * res['width']
    return res

def plot_circro(my_circ, inner_radius = 1):
    ax = plt.subplot(111, polar = True)
    nodes = my_circ['nodes']
    rad_per_node = 2 * np.pi/len(nodes)

    for (side, index), n in nodes.iterrows():
        n = _get_draw_info(n, side, index, rad_per_node)
        bar = ax.bar(n['theta'], n['outer_r'], width=n['width'], bottom = n['inner_r'])
        if side is 'left':
            bar[0].set_facecolor([1, 0, 0])

    plt.show()
