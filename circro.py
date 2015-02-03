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


def _create_edges_df(edge_file, nodes):
    edges = read_csv(edge_file, header=None)
    if 'label' in nodes:
        labels = nodes['label']
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
        res['edges'] = _create_edges_df(edge_matrix, res['nodes'])

    res['inner_r'] = inner_r
    res['start_radian'] = start_radian
    res['edge_threshold'] = edge_threshold

    return res

def _plot_info(circ):
    num_nodes = len(circ['nodes']) if 'nodes' in circ else len(circ['edges'])
    rad_per_node = 2 * np.pi/num_nodes

    info = {}
    nodes = DataFrame()

    #get the index & main data fram circ
    nodes['label'] = circ['nodes']['label']
    nodes['size'] = circ['nodes']['size']

    nodes['width'] = rad_per_node
    nodes['width'].right = rad_per_node * -1

    start_radian = circ['start_radian']
    nodes['theta'] = nodes['width'] * nodes.index.labels[1]

    nodes['label_loc'] = nodes['theta'] * 180/np.pi
    nodes['label_loc'].right = 360 + nodes['label_loc'].right

    deg_per_node = np.rad2deg(rad_per_node)
    nodes['label_loc'].left = nodes['label_loc'].left + deg_per_node/2
    nodes['label_loc'].right = nodes['label_loc'].right - deg_per_node/2

    start_deg = np.rad2deg(start_radian)
    nodes['label_loc'] = nodes['label_loc'] + start_deg
    nodes['theta'] = nodes['theta'] + start_radian

    nodes['label_loc'] = np.mod(nodes['label_loc'], 360.0)
    nodes['theta'] = np.mod(nodes['theta'], 2*np.pi)

    info['nodes'] = nodes

    return info

def plot_circro(my_circ):
    info = _plot_info(my_circ)
    nodes = info['nodes']

    inner_r = my_circ['inner_r']

    ax = plt.subplot(111, polar = True)
    plt.thetagrids(nodes['label_loc'], nodes['label'])

    plt.grid(False, axis='y', which='both') #turn off radial lines
    plt.grid(False, axis='x', which='minor') #turn off radial lines
    ax.set_yticklabels([]) #turn off radial labels

    for (side, index), n in nodes.iterrows():
        bar = ax.bar(n['theta'], n['size'], width=n['width'], bottom = inner_r)

    plt.show()
