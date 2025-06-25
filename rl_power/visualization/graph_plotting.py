import igraph as ig
import matplotlib.pyplot as plt


def action_to_edge(a, bus_1, bus_2, n_bus=3):
    edge = None

    if a == 0:
        edge = (bus_1, bus_2)
    elif a == 1:
        edge = (bus_1, bus_2 + n_bus)
    elif a == 2:
        edge = (bus_1 + n_bus, bus_2)
    elif a == 3:
        edge = (bus_1 + n_bus, bus_2 + n_bus)
    elif a == 4:
        edge = None

    return edge


if __name__ == "__main__":

    fig, axs = plt.subplots(5, 5)

    graph = ig.Graph()
    graph.add_vertices(6)
    layout = graph.layout(layout="grid")
    graph.vs["label"] = [i for i in range(6)]

    for i in range(5):
        for j in range(5):
            graph.delete_edges(graph.es)
            e_1 = action_to_edge(i, 0, 1)
            e_2 = action_to_edge(j, 1, 2)

            if e_1 is not None:
                graph.add_edge(*e_1)
            if e_2 is not None:
                graph.add_edge(*e_2)

            ig.plot(graph, backend="matplotlib", layout=layout, vertex_label_size=8, vertex_label_color="black", target=axs[i, j])
            axs[i, j].spines["top"].set_visible(True)
            axs[i, j].spines["bottom"].set_visible(True)
            axs[i, j].spines["left"].set_visible(True)
            axs[i, j].spines["right"].set_visible(True)

    # fig.tight_layout()
    plt.show()
