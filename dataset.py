from utils import *
from torch_geometric.data import Dataset, InMemoryDataset
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Dataset

class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root: str, num_graphs: int = 100, verbose=0, **kwargs):
        self.num_graphs = num_graphs
        self.graph_kwargs = kwargs
        self.verbose = verbose
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        self.graphs = []
        for i in range(self.num_graphs):
            local_kwargs = {k: v for k, v in self.graph_kwargs.items() if k != "seed"}
            local_kwargs["seed"] = self.graph_kwargs["seed"] + i
            graph = make_graph(**local_kwargs)
            
            self.graphs.append(graph)
            if self.verbose > 0:
                print(f"Generated graph {i + 1}")

        data_list = self.graphs 
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def len(self):
        return self.num_graphs

def plot_results_initial(anchors, X, weighted_means, intersections, M):
    n_anchors = anchors.shape[0]
    plt.scatter(anchors[:, 0], anchors[:, 1], label="anchors", c="r", marker='*') # anchors nodes
    plt.scatter(X[n_anchors:, 0], X[n_anchors:, 1], label="true", c="g", marker='P') # target nodes
    plt.scatter(weighted_means[n_anchors:, 0], weighted_means[n_anchors:, 1], label="preds", c="y", marker="X") # preds
    plt.plot([X[n_anchors:, 0], weighted_means[n_anchors:, 0]], [X[n_anchors:, 1], weighted_means[n_anchors:, 1]], "k--")
    for i, xt in enumerate(X):
        if i < n_anchors:
            plt.annotate(f"A_{i}", xt)
        else:
            """ bbox = intersections[i]
            xmin, xmax, ymin, ymax = bbox
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin]) """
            plt.scatter(M[i, :, 0], M[i, :, 1], marker=".", s=10)
            plt.annotate(f"t_{i}", xt)
            plt.annotate(f"p_{i}", weighted_means[i - n_anchors])
    plt.legend()
    plt.title(f"Initial, Predictions with initial weights")
    plt.show()

def make_graph(
    seed: int = None,
    num_nodes: int = 100,
    d_dim: int = 2,
    num_particles: int = 100,
    num_anchors: int = 7,
    meters: int = 100,
    radius: int = 22,
    noise: float = 1.0,
    priors: bool = False,
    hop: str = 'two',
):
    """
    Create a PyTorch Geometric `Data` object representing a graph with:
    - Nodes: Positions of target and anchor nodes.
    - Edges: Connectivity defined by n-hop neighbors.
    - Edge attributes: Distances (1-hop or 2-hop approximations).
    - Optional node features (particles) if priors=True.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    num_nodes : int
        Total number of nodes (anchors + targets).
    d_dim : int
        Dimension of the node coordinates (e.g., 2 for 2D).
    num_particles : int
        Number of particles if using particle-based node features.
    num_anchors : int
        Number of anchor nodes.
    meters : int
        Size of the area in meters (for node generation).
    radius : int
        Communication radius for edge formation.
    noise : float
        Noise level for simulated distances.
    priors : bool
        If True, node features are sampled particles; if False, no node features.
    hop : str
        Neighborhood type: 'one' for 1-hop edges only, 'two' for also including
        2-hop distances.

    Returns
    -------
    Data
        A torch_geometric.data.Data instance containing:
        - x: Node features (if priors=True) or None.
        - edge_index: The graph connectivity.
        - edge_attr: Edge attributes (distance-based features).
        - y: Ground-truth node positions.
    """

    np.random.seed(seed)
    if num_anchors == 0:
        priors = False

    # Generate ground-truth node positions
    X_true, area = generate_targets(seed, (num_nodes, d_dim), meters, num_anchors, False)

    # Generate anchor positions and place them in the beginning of X_true
    if num_anchors is not None and num_anchors > 0:
        anchors = generate_anchors(area, num_anchors, border_offset=np.sqrt(meters)*1)
        num_anchors = anchors.shape[0]
        # Place anchors at the start of X_true
        X_true[:num_anchors] = anchors
    else:
        num_anchors = 0
        # Instead of None, use an empty array so subsequent code can still handle it
        anchors = np.zeros((0, d_dim), dtype=float)

    # Obtain distances and connectivity matrices
    full_D, D, B, RSS = get_distance_matrix(X_true, num_anchors, radius, noise)
    # `network` is an adjacency-like matrix with either 1-hop or 2-hop distances
    network = get_graphs(D)[hop]

    # Particle sampling logic (if priors are used)
    intersection_bbox, bbox = create_bbox(D, anchors, area, radius)
    particles, prior_beliefs = sample_particles(
        intersections=intersection_bbox,
        anchors=anchors,
        n_particles=num_particles,
        priors=priors,
        mm=meters
    )

    init_particles = particles.copy()
    messages = np.ones((num_nodes, num_nodes, num_particles))
    weights = prior_beliefs / np.sum(prior_beliefs, axis=1, keepdims=True)

    # Convert sampled particles to a Torch tensor and make them learnable
    particles = torch.tensor(particles, requires_grad=True, dtype=torch.float)
    particles = nn.Parameter(particles)

    # Mask for anchors: anchors are fixed (non-learnable)
    fixed_mask = torch.zeros(particles.shape[0], dtype=torch.bool)
    fixed_mask[:num_anchors] = True
    particles.data[fixed_mask] = particles.data[fixed_mask].detach()

    # Flatten particles if priors are used. Shape: (num_nodes, num_particles * d_dim)
    #x = particles.view(particles.shape[0], -1) if priors else None
    #x = torch.mean(particles, dim=1) if priors else None
    x = torch.mean(particles, dim=1)
    #print(x.shape)

    np_x = x.detach().numpy()
    plot_results_initial(anchors, X_true, np_x, intersection_bbox, init_particles)
    print(f"Init RMSE (mean particle estimates): {RMSE(X_true, np_x)}")

    # Construct edge_index from the adjacency matrix `network`
    row, col = np.nonzero(network)
    edge_index = torch.tensor(np.stack([row, col], axis=0), dtype=torch.long)

    # Edge attributes: e.g., we can store both the hop-based distance and the RSS-based B
    edge_attr = torch.tensor(
        np.stack([network[row, col], B[row, col]], axis=1),
        dtype=torch.float
    )


    # y: ground-truth node positions
    y = torch.tensor(X_true, dtype=torch.float)
    # remove if necessary
    anchor_mask = torch.zeros(num_nodes, dtype=torch.bool)
    anchor_mask[:num_anchors] = True
    
    # Create PyTorch Geometric Data object
    graph_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )

    # If we didn't provide x and we want to ensure correct node count, set num_nodes explicitly
    graph_data.num_nodes = num_nodes
    graph_data.num_anchors = num_anchors
    graph_data.radius = radius

    # remove if necesssary
    graph_data.anchor_mask = anchor_mask

    # for iterative NBP
    graph_data.particles = init_particles
    graph_data.messages = messages
    graph_data.weights = weights
    graph_data.B = B
    graph_data.D = network

    return graph_data

if __name__ == "__main__":
    graph = make_graph(
        seed=21,
        num_nodes=100,
        d_dim=2,
        num_particles=50,
        num_anchors=7,
        meters=100,
        radius=22,
        noise=1,
        priors=False,
        hop="two"
    )
    print(graph.num_edge_features)

    """ x = np.arange(10).reshape(2,5)
    print(x)
    print(x.T) """

