from dataclasses import asdict
import networkx as nx


def get_adjacent_branches(network: dict, bus_ids: list[str]) -> (list[str], list[dict]):
    branch_ids = [k for k, v in network["branch"].items() if (str(v["f_bus"]) in bus_ids or str(v["t_bus"]) in bus_ids)]
    branch_dict_data = [network["branch"][k] for k in branch_ids]
    branch_ids.sort()

    return branch_ids, branch_dict_data


def powermodel_dict_to_graph(network: dict):
    # nodes = [int(k) for k in network["bus"].keys()]
    edges = [(b["f_bus"], b["t_bus"]) for b in network["branch"].values()]

    graph = nx.Graph(edges)

    return graph
