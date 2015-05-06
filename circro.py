from __future__ import division

import os

import matplotlib as mpl
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
    Return a dictionary from named arguments

    Examples
    --------
    >>> _inputs_to_dict(k1="v1", k2="v2")
    {'k2': 'v2', 'k1': 'v1'}
    """
    return kwords


def _scale_matrix(mat, new_min=0, new_max=1, selectors=None):
    """
    Return a scaled version of a matrix

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1, 2], [3, 5]])
    >>> _scale_matrix(x)
    array([[ 0.  ,  0.25],
           [ 0.5 ,  1.  ]])
    >>> _scale_matrix(x, 2, 4)
    array([[ 2. ,  2.5],
           [ 3. ,  4. ]])
    >>> x = np.array([[0, 1], [3, 5]])
    >>> _scale_matrix(x, 4, 8, x!=0)
    array([[ 0.,  4.],
           [ 6.,  8.]])
    """
    if selectors is None:
        selectors = np.ones(mat.shape) == 1
    mx = mat[selectors].max()
    mn = mat[selectors].min()
    rg = mx - mn
    new_rg = new_max - new_min
    ratio = new_rg/rg
    new_mat = mat.astype(float)
    new_mat[selectors] = (mat[selectors] - mn) * ratio + new_min
    return new_mat


def _lazy_df(fn, df):
    """
    maps each element of a Pandas dataframe into a function
    this allows for executing code in a lazy manner - only when necessary
    it also facilitates changing the dimension of each element in the dataframe
    
    Examples
    --------
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
    lazy_gen = lambda i: lambda: fn(i)
    return df.applymap(lazy_gen)


def _read_node_file(filename, node_type):
    """
    Reads a CSV File into a Pandas Dataframe
    The CSV file is expected to have 2 columns and a header row
    Left column for left nodes
    Right column for right nodes

    Examples
    --------
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
    return df, [left, right]


def _create_nodes_df(filename_dict):
    """
    Converts a dictionary of node_type, node_csv_file to a dataframe
    Each column of the resultant dataframe corresponds to a different
    node type

    Examples
    --------
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
    >>> del x['labels']
    >>> (df, cols) = _create_nodes_df(x)
    >>> df
             sizes
    left  0    1.0
          1    2.0
          2    3.0
    right 0    1.5
          1    2.5
          2    3.5
    >>> cols
    {'sizes': ['Sizes L', 'Sizes R']}
    """
    node_file_keys = ['labels', 'sizes', 'colors']
    dfs_cols = [_read_node_file(f, k) for k, f in filename_dict.items()
                if f and k in node_file_keys]
    dfs = [df_col[0] for df_col in dfs_cols]
    cols = {df_col[0].columns[0]: df_col[1] for df_col in dfs_cols}
    df = pd.concat(dfs, axis=1)
    return df, cols


def _create_edges_df(edge_file, left_len, right_len):
    """
    Reads a CSV file with a header row & data with dimensions
    number of nodes by number of nodes

    Which rows/cols are specified as right or left set by inputs
    left_len and right_len

    Examples
    --------
    >>> import pandas as pd
    >>> left = pd.Series(['BA1', 'BA2', 'BA3'])
    >>> right = pd.Series(['BA4', 'BA5', 'BA6'])
    >>> nodes = pd.concat([left, right], keys = ['left', 'right']).to_frame()
    >>> edge_file = 'test_data/edge_matrix.csv'
    >>> df = _create_edges_df(edge_file, len(nodes.loc['left']), len(nodes.loc['right']))
    >>> df
             left            right          
                0    1    2      0    1    2
    left  0   0.0  1.2  1.3    1.4  1.5  1.6
          1   1.2  0.0  2.3    0.0  0.0  0.0
          2   1.3  2.3  0.0    0.0  0.0  3.6
    right 0   1.4  0.0  0.0    0.0  4.5  0.0
          1   1.5  0.0  0.0    4.5  0.0  0.0
          2   1.6  0.0  3.6    0.0  0.0  0.0
    """
    outer_index = ['left']*left_len + ['right']*right_len
    inner_index = range(left_len) + range(right_len)
    index = pd.MultiIndex.from_arrays([outer_index, inner_index])

    edges = pd.read_csv(edge_file, header=None)
    edges.columns = index
    edges.index = index
    return edges


def _raise_input_error(inputs):
    raise InputError("at least one of {} inputs must be set".format(inputs))


def make_circro(labels=None, sizes=None, colors=None, edge_matrix=None,
                inner_r=1.0, start_radian=0.0, edge_threshold=.5, node_cm='jet', edge_cm='jet',
                draw_labels=True, draw_nodes_colorbar=None, edge_render_thickness=None):
    """
    Generates a circular diagram data structure that contains data to be rendered with
    plot_circro. See plot_circro

    Examples
    --------
    >>> from test_utils import temp_dir
    >>> with temp_dir('test_data/sizes.csv') as fs:
    ...    my_circ = make_circro(sizes = fs[0])
    >>> sorted(my_circ.keys())
    ['_node_columns', 'draw_labels', 'draw_nodes_colorbar', 'edge_cm', 'edge_render_thickness', 'edge_threshold', 'inner_r', 'node_cm', 'nodes', 'start_radian']
    >>> my_circ['nodes']
             sizes
    left  0    1.0
          1    2.0
          2    3.0
    right 0    1.5
          1    2.5
          2    3.5
    """
    inputs = _inputs_to_dict(labels=labels, sizes=sizes,
                             colors=colors, edge_matrix=edge_matrix)

    file_keys = ['labels', 'sizes', 'colors', 'edge_matrix']

    if not any(inputs[f] for f in file_keys):
        _raise_input_error(file_keys)

    res = dict()
    res['nodes'], res['_node_columns'] = _create_nodes_df(inputs) 
    if edge_matrix:
        res['edges'] = _create_edges_df(edge_matrix,
                                        len(res['nodes'].loc['left']),
                                        len(res['nodes'].loc['right']))

    res['inner_r'] = inner_r
    res['start_radian'] = start_radian
    res['edge_threshold'] = edge_threshold

    res['node_cm'] = node_cm
    res['edge_cm'] = edge_cm

    res['draw_labels'] = draw_labels
    res['draw_nodes_colorbar'] = draw_nodes_colorbar \
        if draw_nodes_colorbar is not None else 'colors' in res['nodes']

    res['edge_render_thickness'] = edge_render_thickness

    return res


def make_circro_from_dir(src_dir, inner_r=1.0, start_radian=0.0, edge_threshold=.5,
                         node_cm='jet', edge_cm='jet', draw_labels=True,
                         draw_nodes_colorbar=True, edge_render_thickness=None):
    """
    Wrapper for make_circro
    src_dir must contain at least one of the following files:
        labels.csv, colors.csv, sizes.csv, edge_matrix.csv

    Examples
    --------
    >>> src_dir = 'data' #data has files for labels, colors, sizes, edge_matrix 
    >>> my_circ_dir = make_circro_from_dir(src_dir)
    >>> import os
    >>> prep = lambda(l): os.path.join(src_dir, l + '.csv')
    >>> my_circ = make_circro(labels = prep('labels'),
    ...     sizes = prep('sizes'), colors = prep('colors'),
    ...     edge_matrix = prep('edge_matrix'))
    >>> from pandas.util.testing import assert_frame_equal
    >>> assert_frame_equal(my_circ['nodes'], my_circ_dir['nodes'])
    >>> assert_frame_equal(my_circ['edges'], my_circ_dir['edges'])
    >>> from test_utils import temp_dir
    >>> my_circ = make_circro(sizes = prep('sizes'))
    >>> with temp_dir('data/sizes.csv') as fs:
    ...    my_circ_dir = make_circro_from_dir(os.path.dirname(fs[0]))
    >>> assert_frame_equal(my_circ['nodes'], my_circ_dir['nodes'])
    """
    file_keys = {'labels', 'colors', 'sizes', 'edge_matrix'}

    def add_file_input(acc, i):
        f = os.path.join(src_dir, i + '.csv')
        acc[i] = f if os.path.isfile(f) else None
        return acc
    
    inputs = reduce(add_file_input, file_keys, dict())

    if all([inputs[i] is None for i in file_keys]):
        _raise_input_error(file_keys)

    inputs.update(_inputs_to_dict(inner_r=inner_r, start_radian=start_radian,
                                  edge_threshold=edge_threshold, node_cm=node_cm, edge_cm=edge_cm,
                                  draw_labels=draw_labels, draw_nodes_colorbar=draw_nodes_colorbar,
                                  edge_render_thickness=edge_render_thickness
                                  ))

    return make_circro(**inputs)


def _calculate_radial_arc(start_radian, end_radian, radius): 
    """
    Calculates the radii and thetas corresponding to an arc between start_radian and end_radian

    Examples
    --------
    >>> import numpy as np
    >>> (rs, ts) = _calculate_radial_arc(0, np.pi, 1)
    >>> assert rs.max() == 1
    >>> assert rs.min() >= 0
    >>> assert np.logical_or(ts == 0, ts == np.pi).all()
    >>> (rs, ts) = _calculate_radial_arc(0, np.pi, 2)
    >>> assert rs.max() == 2
    >>> (rs, ts) = _calculate_radial_arc(np.pi * .5, np.pi * 1.5, 1)
    >>> assert ts.min() == np.pi * .5
    >>> assert ts.max() == np.pi * 1.5
    >>> assert np.logical_or(ts[rs == 2] == np.pi * .5, ts[rs == 2] == np.pi * 1.5).all()
    """
    [start_radian, end_radian] = np.sort([start_radian, end_radian])

    theta_gap_orig = end_radian - start_radian

    theta_gap = theta_gap_orig if theta_gap_orig < np.pi else 2*np.pi - theta_gap_orig
    
    theta_mid = np.pi/2
    theta_left = theta_mid - theta_gap/2
    theta_right = theta_mid + theta_gap/2
    thetas = [theta_left, theta_mid, theta_right]

    xs = np.cos(thetas)

    h_top = np.sin(theta_left)
    dip_coeff = np.cos(theta_gap/2)
    hs = [h_top, h_top * dip_coeff, h_top]
    h_fn = interpolate.interp1d(xs, hs, kind='quadratic')

    xs = np.linspace(start=xs[0], stop=xs[2], num=20)
    hs = h_fn(xs)
    rs = np.linalg.norm([hs, xs], axis=0)
    thetas = np.arctan2(hs, xs)
    thetas = thetas - np.min(thetas)
    
    if theta_gap_orig > np.pi:
        thetas = 2*np.pi - thetas

    thetas = thetas + start_radian
    
    return rs * radius, thetas


def _plot_info(circ):
    """
    Calculates render inputs from a circro structure
    see plot_circro, make_circro

    Examples
    --------
    >>> from test_utils import temp_dir
    >>> with temp_dir('test_data/sizes.csv') as fs:
    ...    my_circ = make_circro(sizes = fs[0])
    >>> info = _plot_info(my_circ)
    """
    num_nodes = len(circ['nodes']) if 'nodes' in circ else len(circ['edges'])
    rad_per_node = 2 * np.pi/num_nodes

    info = {}
    nodes = pd.DataFrame()

    nodes['label'] = circ['nodes']['labels'] \
        if 'labels' in circ['nodes'] else circ['nodes'].index.labels[1]
    nodes.index = circ['nodes'].index

    nodes['size'] = circ['nodes']['sizes'] if 'sizes' in circ['nodes'] else 1

    nodes['width'] = rad_per_node
    nodes['width'].right = rad_per_node * -1

    node_cm = getattr(cm, circ['node_cm'])

    info['node_colors'] = node_cm(circ['nodes']['colors'] if 'colors' in circ['nodes'] else 1.0)
    if circ['draw_nodes_colorbar']:
        info['node_colors_norm'] = \
            mpl.colors.Normalize(vmin=circ['nodes']['colors'].min().min(),
                                 vmax=circ['nodes']['colors'].max().max())

    start_radian = circ['start_radian']
    nodes['theta'] = nodes['width'] * nodes.index.labels[1]

    nodes['label_loc'] = nodes['theta'] * 180/np.pi
    nodes['label_loc'].right += 360

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
        mask = circ['edges'] > circ['edge_threshold']
        scaled_edges = _scale_matrix(circ['edges'], selectors=mask)
        edge_cm = getattr(cm, circ['edge_cm'])
        info['edge_colors'] = _lazy_df(edge_cm, scaled_edges)
        info['edge_colors_norm'] = mpl.colors.Normalize(
            vmin=circ['edges'][mask].min().min(),
            vmax=circ['edges'][mask].max().max()
        )

    return info


def plot_circro(my_circ, draw=True):
    """
    Renders diagram associated with a circro structure

    Examples
    --------
    my_circ = circro.make_circro_from_dir('test_data')
    plot_circro(my_circ)
    """
    info = _plot_info(my_circ)
    nodes = info['nodes']

    inner_r = my_circ['inner_r']

    ax = plt.subplot(111, projection='polar')

    if my_circ['draw_labels']:
        plt.thetagrids(nodes['label_loc'], nodes['label'])

#   turn off radial lines
    plt.grid(False, axis='y', which='both')
    plt.grid(False, axis='x', which='both')
    ax.set_yticklabels([])

    ax.bar(nodes['theta'], nodes['size'], nodes['width'], bottom=inner_r, color=info['node_colors'])
    if 'node_colors_norm' in info:
        norm = info['node_colors_norm']
        ax_color, params = mpl.colorbar.make_axes(ax, location='left')
        mpl.colorbar.ColorbarBase(ax_color, norm=norm,
                                  cmap=getattr(cm, my_circ['node_cm']), **params)

    if 'edges' in my_circ:
        edges = my_circ['edges'].T

        index_to_theta = lambda i: np.deg2rad(nodes.loc[i]['label_loc'])

        if my_circ['edge_render_thickness']:
            new_min, new_max = my_circ['edge_render_thickness']
            edge_thicknesses = _scale_matrix(edges, new_min, new_max,
                                             edges > my_circ['edge_threshold'])
        else:
            edge_thicknesses = edges

        for start_index in edges:
            end_edges = edges[start_index][:start_index][:-1]
            start_theta = index_to_theta(start_index)
            for end_index in end_edges.index[end_edges > my_circ['edge_threshold']]:
                end_theta = index_to_theta(end_index)
                if start_theta == end_theta:
                    continue
                (radii, thetas) = _calculate_radial_arc(start_theta, end_theta, inner_r)
                clr = info['edge_colors'][start_index][end_index]()
                ax.plot(thetas, radii, color=clr, ls='-', lw=edge_thicknesses[start_index][end_index])

        norm = info['edge_colors_norm']
        ax_color, params = mpl.colorbar.make_axes(ax, location='right')
        mpl.colorbar.ColorbarBase(ax_color, norm=norm,
                                  cmap=getattr(cm, my_circ['edge_cm']), **params)

    if draw:
        plt.show()


def plot_circros(my_circs):
    """
    Renders diagram for a list of circros

    Examples
    --------
    my_circ = make_circro_from_dir('test_dir', draw_labels=False)
    my_circ2 = make_circro(labels='test_dir/labels.csv', inner_r = 2)
    plot_circros([my_circ, my_circ2])
    """
    for c in my_circs:
        plot_circro(c, draw=False)
    plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
