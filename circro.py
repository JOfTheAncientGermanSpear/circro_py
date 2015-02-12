from __future__ import division
from pandas import read_csv, concat, Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

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


    if not any(inputs[f] for f in file_keys):
        raise InputError("at least one of {} inputs must be set".format(file_keys))

    res = {}
    res['nodes'] = _create_nodes_df(inputs) 
    if edge_matrix:
        res['edges'] = _create_edges_df(edge_matrix, res['nodes'])

    res['inner_r'] = inner_r
    res['start_radian'] = start_radian
    res['edge_threshold'] = edge_threshold

    return res

def _calculate_radial_arc(start_radian, end_radian, radius):
	def calc_gap():
		return np.abs(end_radian - start_radian)

	theta_gap = calc_gap()
	if theta_gap > np.pi:
		theta_gap = theta_gap - np.pi
	
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

	return (rs * radius, thetas + start_radian)

def _plot_info(circ):
    num_nodes = len(circ['nodes']) if 'nodes' in circ else len(circ['edges'])
    rad_per_node = 2 * np.pi/num_nodes

    info = {}
    nodes = DataFrame()

    #get the index & main data fram circ
    nodes['label'] = circ['nodes']['label'] if 'label' in circ['nodes'] else circ['nodes'].index.labels[1]
    nodes['size'] = circ['nodes']['size'] if 'size' in circ['nodes'] else 1

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

    ax.bar(nodes['theta'], nodes['size'], nodes['width'], bottom = inner_r)


    if 'edges' in my_circ:
        edges = my_circ['edges'].T

        connections = edges.copy()
        connections[connections < my_circ['edge_threshold']] = 0

        for start_label in edges:
            end_edges = edges[start_label][:start_label][:-1] #label slices are inclusive
            start_node = nodes[nodes['label'] == start_label]
            start_theta = np.deg2rad(start_node['label_loc'][0])
            for (end_label, weight) in end_edges.iteritems():
                if (weight > my_circ['edge_threshold']):
                    end_node = nodes[nodes['label'] == end_label]
                    end_theta = np.deg2rad(end_node['label_loc'][0])
                    (start_theta, end_theta) = np.sort([start_theta, end_theta])
                    (radii, thetas) = _calculate_radial_arc(start_theta, end_theta, inner_r)
                    ax.plot(thetas, radii)

    plt.show()
