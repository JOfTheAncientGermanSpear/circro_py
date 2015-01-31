from __future__ import division
from pandas import read_csv, concat, Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt

class InputError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

def _inputs_to_dict(**kwords):
    return kwords

def _read_node_file(filename):
    df = read_csv(filename, header=None)
    return concat([df[0], df[1]], keys=['left', 'right']).to_frame()


def _create_nodes_df(filename_dict):
    node_file_keys = ['labels', 'sizes', 'colors']
    node_dfs = {k: _read_node_file(f) for k,f in filename_dict.items() 
            if f and k in node_file_keys}
    for k, df in node_dfs.items():
        df.columns = [k[:-1]] 
    return concat(list(node_dfs.values()), axis = 1)


def _create_edges_df(edge_file, labels, nodes):
    edges = read_csv(edge_file, header=None)
    if labels:
        labels = nodes['label']
        if 'edges' in res:
            edges.columns = labels.values
            edges.index = labels.values
    return edges


def make_circro(labels = None, sizes = None, colors = None, edge_matrix = None,
        inner_r=1.0, start_radian=0.0, edge_threshold=.5):
    inputs = _inputs_to_dict(labels = labels, sizes = sizes, 
            colors = colors, edge_matrix = edge_matrix)

    file_keys = ['labels', 'sizes', 'colors', 'edge_matrix']


    if not any(f in inputs for f in file_keys):
        raise InputError("at least one of {} inputs must be set".format(file_keys))

    res = {}
    res['nodes'] = _create_nodes_df(inputs) 
    if edge_matrix:
        res['edges'] = _create_edges_df(edge_matrix, labels, res['nodes'])

    res['inner_r'] = inner_r;
    res['start_radian'] = start_radian
    res['edge_threshold'] = edge_threshold

    num_nodes = len(res['nodes']) if 'nodes' in res else len(res['edges'])
    rad_per_node = 2 * np.pi/num_nodes

    res['nodes']['width'] = rad_per_node
    res['nodes']['width'].right = rad_per_node * -1
    res['nodes']['theta'] = res['nodes']['width'] * res['nodes'].index.labels[1]
    res['nodes']['label_loc'] = res['nodes']['theta'] * 180/np.pi
    res['nodes']['label_loc'].right = 360 + res['nodes']['label_loc'].right
    deg_per_node = 180/np.pi * rad_per_node
    res['nodes']['label_loc'].left = res['nodes']['label_loc'].left + deg_per_node/2
    res['nodes']['label_loc'].right = res['nodes']['label_loc'].right - deg_per_node/2

    return res


def plot_circro(my_circ):
    nodes = my_circ['nodes']
    start_radian = my_circ['start_radian']
    inner_r = my_circ['inner_r']

    ax = plt.subplot(111, polar = True)
    plt.thetagrids(nodes['label_loc'], nodes['label'])
    plt.grid(False, axis='y', which='both') #turn off radial lines
    plt.grid(False, axis='x', which='minor') #turn off radial lines
    ax.set_yticklabels([]) #turn off radial labels

    for (side, index), n in nodes.iterrows():
        bar = ax.bar(n['theta'] + start_radian, n['size'], width=n['width'], bottom = inner_r)

    plt.show()
