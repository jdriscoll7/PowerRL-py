import copy

import igraph as ig
import numpy as np
import plotly
import gravis as gv
from juliacall import DictValue
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

from rl_power.power.powermodels_interface import recursive_dict_cast, ConfigurationManager


def _dict_to_multiline_string(input_dict):

    return "{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in input_dict.items()) + "}\n"


def _pm_get_edge_and_id_list(network: dict):

    bus_ids_and_edges = (list(network["branch"].keys()), [(int(b["f_bus"])-1, int(b["t_bus"])-1)
                                                          for k, b in network["branch"].items() if b["br_status"] == 1])
    n_bus = len(network["bus"])

    load_ids_and_edges = (list(network["load"].keys()), [(int(load["load_bus"])-1, int(load["index"])+n_bus-1) for k, load in network["load"].items()])
    n_load = len(load_ids_and_edges[0])

    gen_ids_and_edges = (list(network["gen"].keys()), [(int(g["gen_bus"])-1, int(g["index"])+n_bus+n_load-1) for k, g in network["gen"].items()])

    return bus_ids_and_edges, load_ids_and_edges, gen_ids_and_edges


def pm_to_igraph(network: dict, solution: dict) -> ig.Graph:

    network = recursive_dict_cast(network)
    n_bus = len(network["bus"])

    (branch_ids, edges), (load_ids, load_edges), (gen_ids, gen_edges) = _pm_get_edge_and_id_list(network)

    graph = ig.Graph(edges + load_edges + gen_edges)

    # Bus, loads, generators.
    vertex_labels = [f"b{i}" for i in range(n_bus // 2)]
    vertex_labels += [f"bb{i}" for i in range(n_bus // 2)]
    vertex_labels += [f"ld{i}" for i in range(len(load_ids))]
    vertex_labels += [f"g{i}" for i in range(len(gen_ids))]
    graph.vs["label"] = vertex_labels

    vertex_sizes = [20 for i in range(n_bus)]
    vertex_sizes += [20 for i in range(len(load_ids))]
    vertex_sizes += [20 for i in range(len(gen_ids))]
    graph.vs["size"] = vertex_sizes

    vertex_shapes = ["circle" for i in range(n_bus)]
    vertex_shapes += ["square" for i in range(len(load_ids))]
    vertex_shapes += ["diamond" for i in range(len(gen_ids))]
    graph.vs["shape"] = vertex_shapes

    branch_solutions_dicts = [{} if b not in dict(solution["solution"]["branch"]) else dict(solution["solution"]["branch"][b])
                              for b in branch_ids]

    branch_param_dicts = [{} if b not in dict(solution["solution"]["branch"]) else dict(network["branch"][b])
                          for b in branch_ids]

    branch_solution_data = [_dict_to_multiline_string(x) + _dict_to_multiline_string(branch_param_dicts[i])
                            for i, x in enumerate(branch_solutions_dicts)]

    node_solution_data = [_dict_to_multiline_string(dict(solution["solution"]["bus"][str(b + 1)])) + _dict_to_multiline_string(dict(network["bus"][str(b + 1)]))
                          for b in range(len(graph.vs[:n_bus]))]

    graph.es['hover'] = branch_solution_data
    graph.vs['hover'] = node_solution_data

    cmap1 = LinearSegmentedColormap.from_list("vertex_cmap", ["red", "blue"])
    cmap2 = LinearSegmentedColormap.from_list("edge_cmap", ["red", "blue"])
    cmap3 = LinearSegmentedColormap.from_list("load_cmap", ["red", "blue"])
    cmap4 = LinearSegmentedColormap.from_list("gen_cmap", ["red", "blue"])

    bus_vm_list = ig.rescale([solution["solution"]["bus"][str(b + 1)]["vm"] for b in range(len(graph.vs[:n_bus]))], clamp=True)
    branch_pt_list = ig.rescale([0 if str(b) not in solution["solution"]["branch"] else solution["solution"]["branch"][str(b)]["pt"]
                                 for b in branch_ids], clamp=True)
    gen_powers = [np.linalg.norm([solution["solution"]["gen"][str(g)]["pg"], solution["solution"]["gen"][str(g)]["qg"]], ord=2)
                  for g in gen_ids]
    gen_power_list = ig.rescale(gen_powers, clamp=True)

    vertex_colors = [cmap1(v) for v in bus_vm_list]
    vertex_colors += ["green" for load in load_ids]
    vertex_colors += [cmap4(g) for g in gen_power_list]
    graph.vs['color'] = vertex_colors

    edge_colors = [cmap2(p) for p in branch_pt_list]
    edge_colors += ["black" for load in load_ids]
    edge_colors += ["black" for load in gen_ids]

    graph.es['color'] = edge_colors

    return graph


def draw_pm_igraph(graph: ig.Graph, layout: ig.layout, target_ax=None) -> None:
    ig.plot(graph, backend="matplotlib", layout=layout, vertex_label_size=6, vertex_label_color="white", target=target_ax)

    # cmap1 = LinearSegmentedColormap.from_list("vertex_cmap", ["red", "blue"])
    # cmap2 = LinearSegmentedColormap.from_list("edge_cmap", ["red", "blue"])

    # norm1 = ScalarMappable(norm=Normalize(0, 1), cmap=cmap1)
    # norm2 = ScalarMappable(norm=Normalize(0, 1), cmap=cmap2)
    # plt.colorbar(norm1, orientation="vertical", label='vm')
    # plt.colorbar(norm2, orientation="vertical", label='pt')

    # plt.show()

def draw_pm_configuration(nm: ConfigurationManager, target_ax=None, layout=None):

    # unconfigured_graph = pm_to_igraph(nm.network, nm.solution)
    configured_graph = pm_to_igraph(nm.configured_network, nm.config_solution)

    draw_pm_igraph(graph=configured_graph, layout=layout, target_ax=target_ax)

    return layout


class PMSolutionRenderer:
    def __init__(self):
        self.layout = None
        self.fig, self.axes = plt.subplots()

    def update_frame(self, nm: ConfigurationManager):

        nm_copy = copy.deepcopy(nm)

        if self.layout is None:
            unconfigured_graph = pm_to_igraph(nm_copy.network, nm_copy.solution)
            self.layout = unconfigured_graph.layout(layout="fr")

        self.axes.cla()

        draw_pm_configuration(nm=nm_copy, layout=self.layout, target_ax=self.axes)
        cost = nm_copy.config_solution["objective"]
        self.axes.set_title(f"Cost: {cost}")


if __name__ == "__main__":

    from rl_power.power.powermodels_interface import Configuration, load_test_case, ConfigurationManager

    import pickle
    from rl_power.power.powermodels_interface import solve_opf

    with open('network.pickle', 'rb') as f:
        x = pickle.load(f)

    solve_opf(x)

    network = load_test_case(path="ieee_data/pglib_opf_case30_ieee.m")
    config_manager = ConfigurationManager(network)
    # config_manager.apply_network_configuration(123)
    config_manager.solve_branch_configuration(binary_configuration=0b011, branch_id="3")
    config_manager.solve_branch_configuration(binary_configuration=0b011, branch_id="3")
    config_manager.solve_branch_configuration(binary_configuration=0b011, branch_id="3")
    renderer = PMSolutionRenderer()
    renderer.update_frame(config_manager)
    plt.show()

    config_manager.solve_configuration()


    # fig = gv.d3(graph, node_hover_tooltip=True, edge_hover_tooltip=True)
    # fig.display()
