import igraph as ig
from igraph import Graph
import matplotlib.pyplot as plt


if __name__ == "__main__":
    g = Graph.Tree(n=8, children=4)
    ig.plot(g, backend="matplotlib", vertex_label_size=8, vertex_label_color="black")
    plt.show()

    print()