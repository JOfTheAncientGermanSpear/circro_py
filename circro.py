from __future__ import division

import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

class InputError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

def _inputs_to_dict(**kwords):
    """
    >>> _inputs_to_dict(k1="v1", k2="v2")
    {'k2': 'v2', 'k1': 'v1'}
    """
    return kwords

def _scale_matrix(mat):
    """
    >>> import numpy as np
    >>> x = np.array([[1, 2], [3, 5]])
    >>> _scale_matrix(x)
    array([[ 0.  ,  0.25],
           [ 0.5 ,  1.  ]])
    """
    mx = mat.max()
    mn = mat.min()
    rg = mx - mn
    return (mat - mn)/rg


def _lazy_df(fn, df):
    """
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame(np.array([[1, 2], [3, 5]]))
    >>> fn = lambda i: [i - 1, i, i + 1]
    >>> lz_df = _lazy_df(fn, df)
    >>> lz_df.iloc[0][0]()
    [0, 1, 2]
    >>> lz_df.iloc[0][1]()
    [1, 2, 3]
    >>> lz_df.iloc[1][0]()
    [2, 3, 4]
    >>> lz_df.iloc[1][1]()
    [4, 5, 6]
    """
    lazy_fn = lambda i: lambda: fn(i)
    return df.applymap(lazy_fn)


def _read_node_file(filename, node_type):
    """
    >>> x = 'test_data/labels.csv'
    >>> (df, cols) = _read_node_file(x, 'labels')
    >>> df
            labels
    left  0    BA1
          1    BA2
          2    BA3
    right 0    BA4
          1    BA5
          2    BA6
    >>> cols
    ['Regions L', 'Regions R']
    """
    data = pd.read_csv(filename)
    (left, right) = data.columns
    df = pd.concat([data[left], data[right]], keys=['left', 'right']).to_frame()
    df.columns = [node_type]
    return (df, [left, right])

def _create_nodes_df(filename_dict):
    """
    >>> x = {'labels': 'test_data/labels.csv', 'sizes': 'test_data/sizes.csv'}
    >>> (df, cols) = _create_nodes_df(x)
    >>> df
            labels  sizes
    left  0    BA1    1.0
          1    BA2    2.0
          2    BA3    3.0
    right 0    BA4    1.5
          1    BA5    2.5
          2    BA6    3.5
    >>> cols
    {'labels': ['Regions L', 'Regions R'], 'sizes': ['Sizes L', 'Sizes R']}
    """
    node_file_keys = ['labels', 'sizes', 'colors']
    dfs_cols = [_read_node_file(f, k) for k,f in filename_dict.items() 
            if f and k in node_file_keys]
    dfs = [df_col[0] for df_col in dfs_cols]
    cols = {df_col[0].columns[0]: df_col[1] for df_col in dfs_cols}
    df = pd.concat(dfs, axis = 1)
    return (df, cols)


def _create_edges_df(edge_file, nodes):
    """
    >>> import pandas as pd
    >>> left = pd.Series(['BA1', 'BA2', 'BA3'])
    >>> right = pd.Series(['BA4', 'BA5', 'BA6'])
    >>> nodes = pd.concat([left, right], keys = ['left', 'right']).to_frame()
    >>> nodes.columns = ['labels']
    >>> edge_file = 'test_data/edge_matrix.csv'
    >>> df = _create_edges_df(edge_file, nodes)
    >>> df
         BA1  BA2  BA3  BA4  BA5  BA6
    BA1  0.0  1.2  1.3  1.4  1.5  1.6
    BA2  1.2  0.0  2.3  0.0  0.0  0.0
    BA3  1.3  2.3  0.0  0.0  0.0  3.6
    BA4  1.4  0.0  0.0  0.0  4.5  0.0
    BA5  1.5  0.0  0.0  4.5  0.0  0.0
    BA6  1.6  0.0  3.6  0.0  0.0  0.0
    """
    edges = pd.read_csv(edge_file, header=None)
    if 'labels' in nodes:
        labels = nodes['labels']
        edges.columns = labels.values
        edges.index = labels.values
    return edges

def _raise_input_error(inputs):
    raise InputError("at least one of {} inputs must be set".format(inputs))


def make_circro(labels = None, sizes = None, colors = None, edge_matrix = None,
        inner_r=1.0, start_radian=0.0, edge_threshold=.5, node_cm = 'jet', edge_cm = 'jet'):
    inputs = _inputs_to_dict(labels = labels, sizes = sizes, 
            colors = colors, edge_matrix = edge_matrix)

    file_keys = ['labels', 'sizes', 'colors', 'edge_matrix']


    if not any(inputs[f] for f in file_keys):
        _raise_input_error(file_keys)

    res = {}
    res['nodes'], res['_node_columns'] = _create_nodes_df(inputs) 
    if edge_matrix:
        res['edges'] = _create_edges_df(edge_matrix, res['nodes'])

    res['inner_r'] = inner_r
    res['start_radian'] = start_radian
    res['edge_threshold'] = edge_threshold

    res['node_cm'] = node_cm
    res['edge_cm'] = edge_cm

    return res

def make_circro_from_dir(src_dir, inner_r = 1.0, start_radian = 0.0, edge_threshold = .5, node_cm = 'jet', edge_cm = 'jet'):
    """
    >>> src_dir = 'data' #data has files for labels, colors, sizes, edge_matrix 
    >>> my_circ_dir = make_circro_from_dir(src_dir)
    >>> import os
    >>> prep = lambda(l): os.path.join(src_dir, l + '.csv')
    >>> my_circ = make_circro(labels = prep('labels'), sizes = prep('sizes'), colors = prep('colors'), edge_matrix = prep('edge_matrix'))
    >>> from pandas.util.testing import assert_frame_equal
    >>> assert_frame_equal(my_circ['nodes'], my_circ_dir['nodes'])
    >>> assert_frame_equal(my_circ['edges'], my_circ_dir['edges'])
    """
    file_keys = {'labels', 'colors', 'sizes', 'edge_matrix'}

    def add_file_input(acc, i):
        f = os.path.join(src_dir, i + '.csv')
        acc[i] = f if os.path.isfile(f) else None
        return acc
    
    inputs = reduce(add_file_input, file_keys, dict())

    if all(inputs[i] == None for i in file_keys):
        _raise_input_error(file_keys)

    inputs.update(_inputs_to_dict(inner_r = inner_r, start_radian = start_radian,
        edge_threshold = edge_threshold, node_cm = node_cm, edge_cm = edge_cm))

    return make_circro(**inputs)


def _calculate_radial_arc(start_radian, end_radian, radius): 
    [start_radian, end_radian] = np.sort([start_radian, end_radian])

    theta_gap_orig = end_radian - start_radian

    theta_gap =  theta_gap_orig if theta_gap_orig < np.pi else 2*np.pi - theta_gap_orig
    
    theta_mid = np.pi/2
    theta_left = theta_mid - theta_gap/2
    theta_right = theta_mid + theta_gap/2
    thetas = [theta_left, theta_mid, theta_right]

    xs = np.cos(thetas)

    h_top = np.sin(theta_left)
    dip_coeff = np.cos(theta_gap/2)
    hs = [h_top, h_top * dip_coeff, h_top]

    h_fn = interpolate.interp1d(xs, hs, kind = 'quadratic')
    xs = np.linspace(start = xs[0], stop = xs[2], num = 20)
    hs = h_fn(xs)
    rs = np.linalg.norm([hs, xs], axis = 0)
    thetas = np.arctan2(hs, xs)
    thetas = thetas - np.min(thetas)
    
    if theta_gap_orig > np.pi:
        thetas = 2*np.pi - thetas

    thetas = thetas + start_radian
    
    return (rs * radius, thetas)

def _plot_info(circ):
    num_nodes = len(circ['nodes']) if 'nodes' in circ else len(circ['edges'])
    rad_per_node = 2 * np.pi/num_nodes

    info = {}
    nodes = pd.DataFrame()

    #get the index & main data fram circ
    nodes['label'] = circ['nodes']['labels'] if 'labels' in circ['nodes'] else circ['nodes'].index.labels[1]
    nodes['size'] = circ['nodes']['sizes'] if 'sizes' in circ['nodes'] else 1

    nodes['width'] = rad_per_node
    nodes['width'].right = rad_per_node * -1

    node_cm = getattr(cm, circ['node_cm'])

    info['node_colors'] = node_cm(circ['nodes']['colors'] if 'colors' in circ['nodes'] else 1.0)

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

    if 'edges' in circ:
        scaled_edges = _scale_matrix(circ['edges'])
        edge_cm = getattr(cm, circ['edge_cm'])
        info['edge_colors'] = _lazy_df(edge_cm, scaled_edges) 

    return info

def plot_circro(my_circ, draw = True):
    info = _plot_info(my_circ)
    nodes = info['nodes']

    inner_r = my_circ['inner_r']

    ax = plt.subplot(111, polar = True)
    plt.thetagrids(nodes['label_loc'], nodes['label'])

    plt.grid(False, axis='y', which='both') #turn off radial lines
    plt.grid(False, axis='x', which='minor') #turn off radial lines
    ax.set_yticklabels([]) #turn off radial labels

    ax.bar(nodes['theta'], nodes['size'], nodes['width'], bottom = inner_r, color = info['node_colors'])


    if 'edges' in my_circ:
        edges = my_circ['edges'].T

        for start_label in edges:
            end_edges = edges[start_label][:start_label][:-1] #label slices are inclusive
            start_node = nodes[nodes['label'] == start_label]
            start_theta = np.deg2rad(start_node['label_loc'][0])
            for (end_label, weight) in end_edges.iteritems():
                if (weight > my_circ['edge_threshold']):
                    end_node = nodes[nodes['label'] == end_label]
                    end_theta = np.deg2rad(end_node['label_loc'][0])
                    (radii, thetas) = _calculate_radial_arc(start_theta, end_theta, inner_r)
                    clr = info['edge_colors'][start_label][end_label]()
                    ax.plot(thetas, radii, color = clr)

    if draw:
        plt.show()

def plot_circros(my_circs):
    for c in my_circs:
        plot_circro(c, draw = False)
    plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
