import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import euclidean_distances
import networkx as nx
from torch_geometric.data import Data
from scipy.stats import norm, gaussian_kde

def generate_targets(seed: int = None,
                     shape: tuple[int, int] = (50, 2),
                     deployment_area: int = 100, # write method for custom deployment area
                     n_anchors:  int = 6,
                     show = False
                     ):

    np.random.seed(seed)
    area = np.array([-deployment_area/2, deployment_area/2, -deployment_area/2, deployment_area/2])
    deployment_bbox = area.reshape(-1, 2)
    X = np.empty(shape)
    n, d = shape

    for j in range(d):
        X[:, j] = np.random.uniform(deployment_bbox[j, 0], deployment_bbox[j, 1], size=n)

    if show:
        plt.scatter(X[:n_anchors, 0], X[:n_anchors, 1], c='r', marker='*')
        plt.scatter(X[n_anchors:, 0], X[n_anchors:, 1], c='y', marker='+')
        plt.show()
    
    return X, area

def get_distance_matrix(X_true: np.ndarray, n_anchors: int, communication_radius: float, noise: float = 0.2, alpha: float = 3, d0: float = 1) -> np.ndarray:
    D = euclidean_distances(X_true)
    full_D = D.copy()
    anchors = D[:n_anchors, :n_anchors].copy()
    D[D > communication_radius] = 0
    B = D > 0
    P_i = -np.linspace(10, 20, D.shape[0])


    def distance_2_RSS(P_i, D, alpha, d0):
        with np.errstate(divide='ignore'):
            s = 10 * alpha * np.log10((D) / d0)
        return P_i[:, np.newaxis] - s
    
    if noise is None:
        rss_noiseless = distance_2_RSS(P_i, D, alpha, d0)
        return full_D, D, B, rss_noiseless
    
    def RSS_2_distance(P_i, RSS, alpha, d0, sigma: float = 0.2, add_noise: bool = True):
        if add_noise:
            noise_matrix = np.random.lognormal(mean=0, sigma=sigma, size=RSS.shape)
            noise_matrix = (noise_matrix + noise_matrix.T) / 2  # Ensure symmetry
            RSS += noise_matrix
            
        with np.errstate(divide='ignore', invalid='ignore'):
            d = d0 * 10 ** ((P_i[:, np.newaxis] - RSS) / (10 * alpha))
        np.fill_diagonal(d, 0)
        return d, RSS

    RSS = distance_2_RSS(P_i, D, alpha, d0)
    DD, rss_noisy = RSS_2_distance(P_i, RSS, alpha, d0, sigma=noise, add_noise=True)

    DD[:n_anchors, :n_anchors] = D[:n_anchors, :n_anchors]
    DD = np.abs(DD) * B

    return full_D, DD, B, rss_noisy

def generate_anchors(deployment_area: np.ndarray, anchor_count: int, border_offset: float) -> np.ndarray:
    x_min, x_max, y_min, y_max = deployment_area
    width = x_max - x_min
    height = y_max - y_min

    # Adjust deployment area to include border offset
    x_min += border_offset
    x_max -= border_offset
    y_min += border_offset
    y_max -= border_offset
    width -= 2 * border_offset
    height -= 2 * border_offset

    # Calculate number of points along each axis, adjusted for equal spacing
    points_per_axis = int(np.ceil(np.sqrt(anchor_count)))
    
    # Ensure we have enough points
    if points_per_axis ** 2 < anchor_count:
        points_per_axis += 1

    # Generate anchors using a hexagonal grid pattern
    anchors = []
    x_step = width / (points_per_axis - 1)
    y_step = height / (points_per_axis - 1)
    
    for j in range(points_per_axis):
        for i in range(points_per_axis):
            x = x_min + i * x_step
            y = y_min + j * y_step
            # Offset every other row for hexagonal pattern
            if j % 2 == 0:
                x += x_step / 2
            if len(anchors) < anchor_count:
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    anchors.append([x, y])
    
    anchors = np.array(anchors)
    
    #plt.scatter(anchors[:, 0], anchors[:, 1])
    #plt.show()
    #print(anchors)
    return anchors

def get_graphs(D: np.ndarray) -> dict:
    graphs = {}
    graphs["full"] = D
    one_hop = D.copy()
    graphs["one"] = one_hop

    G = nx.from_numpy_array(one_hop)
    two_hop = one_hop.copy()
    for j, paths in nx.all_pairs_shortest_path(G, 2):
        for q, _ in paths.items():
            two_hop[j, q] = nx.shortest_path_length(G, j, q, weight='weight')
    
    graphs["two"] = two_hop
    return graphs

def create_bbox(D: np.ndarray, anchors: np.ndarray, limits: np.ndarray, radius: int):
    """
    Creates bounding boxes for each node relative to anchors, then computes
    their intersection. If num_anchors=0, fills with the entire 'limits'.
    """
    n_samples = D.shape[0]

    # If anchors is empty, we treat it as 0 anchors
    n_anchors = anchors.shape[0] if anchors is not None else 0
    d = limits.shape[0] // 2  # Because 'limits' is shaped (d_dim*2,)

    # bboxes: (n_samples, n_anchors, 2*d)
    bboxes = np.zeros((n_samples, n_anchors, 2*d))
    intersection_bboxes = np.zeros((n_samples, 2*d))

    # If there are no anchors, just fill everything with the entire 'limits'.
    if n_anchors == 0:
        for i in range(n_samples):
            for k in range(d):
                intersection_bboxes[i, 2*k]   = limits[2*k]     # min in dimension k
                intersection_bboxes[i, 2*k+1] = limits[2*k + 1] # max in dimension k
        return intersection_bboxes, bboxes

    # Otherwise, proceed with the anchor-based bounding boxes
    gap = 0  # You had a variable gap=0 in your code
    for i in range(n_samples):
        # If the node is itself an anchor, we just do a simple bounding box
        if i < n_anchors:
            for k in range(d):
                intersection_bboxes[i, 2*k] = max(anchors[i, k] - radius, limits[2*k])
                intersection_bboxes[i, 2*k+1] = min(anchors[i, k] + radius, limits[2*k+1])
            continue

        # Otherwise, compute bounding boxes with respect to each anchor
        for j in range(n_anchors):
            if i == j or D[i, j] == 0:
                bboxes[i, j] = limits
                continue

            for k in range(d):
                bboxes[i, j, 2*k] = max(anchors[j, k] - D[i, j], limits[2*k])
                bboxes[i, j, 2*k+1] = min(anchors[j, k] + D[i, j], limits[2*k+1])

        # Intersection across all anchors
        for k in range(d):
            intersection_bboxes[i, 2*k]   = np.max(bboxes[i, :, 2*k],   axis=0)
            intersection_bboxes[i, 2*k+1] = np.min(bboxes[i, :, 2*k+1], axis=0)

            # Ensure min <= max
            if intersection_bboxes[i, 2*k] > intersection_bboxes[i, 2*k+1]:
                intersection_bboxes[i, 2*k], intersection_bboxes[i, 2*k+1] = (
                    intersection_bboxes[i, 2*k+1],
                    intersection_bboxes[i, 2*k]
                )

            # Expand intersection if it's below some gap threshold
            if intersection_bboxes[i, 2*k+1] - intersection_bboxes[i, 2*k] < gap:
                intersection_bboxes[i, 2*k]   -= gap
                intersection_bboxes[i, 2*k+1] += gap

    return intersection_bboxes, bboxes

def sample_particles(
    intersections: np.ndarray,
    anchors: np.ndarray,
    n_particles: int,
    priors: bool,
    mm: int
):
    """
    Generates particles from intersections of bounded boxes for each target sample.
    - If priors=False, we ignore the bounding boxes and just use [-mm/2, mm/2]
      or any default bounding region you prefer.
    - If anchors is empty (0 anchors), all nodes are effectively "targets."
    """
    assert intersections.ndim == 2 and intersections.shape[1] % 2 == 0, \
        "intersections must be a 2D array with an even number of columns"

    # If anchors is None or empty, treat it as shape (0, d)
    if anchors is None or anchors.shape[0] == 0:
        d = intersections.shape[1] // 2
        anchors = np.zeros((0, d), dtype=float)

    # If we still have anchors, ensure dimension consistency
    n_anchors = anchors.shape[0]
    d = anchors.shape[1]
    assert intersections.shape[1] == 2 * d, \
        "anchors.shape[1] * 2 must match intersections.shape[1]"

    # Basic checks
    assert isinstance(n_particles, int) and n_particles > 0, "n_particles must be a positive integer"

    n_samples = intersections.shape[0]

    # Initialize all particles: (n_samples, n_particles, d)
    all_particles = np.zeros((n_samples, n_particles, d))

    # Fill anchor rows with repeated anchor positions
    anchor_particles = np.repeat(anchors[:, np.newaxis, :], repeats=n_particles, axis=1)
    all_particles[:n_anchors] = anchor_particles
    prior_beliefs = np.ones((n_samples, n_particles))

    # For target nodes, sample from intersection bboxes
    for i in range(n_anchors, n_samples):
        if not priors:
            # If priors=False, use a default bounding box (e.g., [-mm/2, mm/2, -mm/2, mm/2])
            intersections[i] = np.array([-mm/2, mm/2] * d)

        bbox = intersections[i].reshape(-1, 2)
        for j in range(d):
            all_particles[i, :, j] = np.random.uniform(
                low=bbox[j, 0],
                high=bbox[j, 1],
                size=n_particles
            )
        prior_beliefs[i] = mono_potential_bbox(intersections[i])(all_particles[i])

    return all_particles, prior_beliefs

def relative_spread(particles_u: np.ndarray, particles_r: np.ndarray, d_ru: float):
    dist_ur = particles_u - particles_r
    angle_samples = np.arctan2(dist_ur[:, 1], dist_ur[:, 0])
    
    # Add 2π, 0, -2π for 2π-periodicity
    extended_samples = np.concatenate([angle_samples, angle_samples + 2*np.pi, angle_samples - 2*np.pi])
    kde = gaussian_kde(extended_samples)
    
    samples = kde.resample(particles_u.shape[0]).T
    samples = np.mod(samples + np.pi, 2*np.pi) - np.pi  # Ensure samples are in [-π, π]
    w_xy = kde(samples.T) #+ 1e-7  # Prevent division by zero
    
    particle_noise = np.random.normal(0, 1, size=particles_u.shape[0]) * 1
    cos_u = (d_ru + particle_noise).reshape(-1, 1) * np.cos(samples)
    sin_u = (d_ru + particle_noise).reshape(-1, 1) * np.sin(samples)
    d_xy = np.column_stack([cos_u, sin_u])
    
    return d_xy, w_xy

def random_spread(particles_r: np.ndarray, d_ru: float):
    particle_noise = np.random.normal(0, 1, size=particles_r.shape[0]) * 1
    thetas = np.random.uniform(0, 2*np.pi, size=particles_r.shape[0])
    cos_u = (d_ru + particle_noise) * np.cos(thetas)
    sin_u = (d_ru + particle_noise) * np.sin(thetas)
    d_xy = np.column_stack([cos_u, sin_u])
    w_xy = np.ones(shape=particles_r.shape[0])

    return d_xy, w_xy

def plot_data_graph(
    data: Data, 
    n_anchors: int, 
    radius: float = None, 
    alpha: float = 0.6, 
    name: str = None,
    ax=None
):
    """
    Plot the graph using node positions and edges from a PyTorch Geometric Data object.
    Parameters
    ----------
    data : torch_geometric.data.Data
        The graph data object containing x (optional), y (node coords), edge_index, and edge_attr.
    n_anchors : int
        The number of anchors (used for special plotting as stars).
    r : float
        The radius threshold for highlighting 1-hop edges if you want to replicate previous style.
    alpha : float
        Edge transparency.
    name : str
        Title of the plot.
    ax : matplotlib axes
        If provided, plot on this axis.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Extract node coordinates
    # data.y is expected to be of shape (num_nodes, d_dim)
    X_true = data.y.cpu().numpy()

    # Extract edges
    edge_index = data.edge_index.cpu().numpy()  # shape (2, num_edges)
    row, col = edge_index[0, :], edge_index[1, :]

    # Extract edge attributes. Assuming edge_attr[ :, 0] is the distance measure.
    edge_attr = data.edge_attr.cpu().numpy() if data.edge_attr is not None else None

    # Create a networkx graph from edge_index
    G = nx.Graph()
    G.add_nodes_from(range(X_true.shape[0]))
    for i in range(len(row)):
        # Add edges with attributes if available
        attr_dict = {}
        if edge_attr is not None:
            attr_dict['distance'] = edge_attr[i, 0]
            attr_dict['B'] = edge_attr[i, 1]
        G.add_edge(int(row[i]), int(col[i]), **attr_dict)

    # pos dict for networkx
    pos = {i: (X_true[i, 0], X_true[i, 1]) for i in range(X_true.shape[0])}

    # If we want to highlight immediate neighbors (like in the original code),
    # we can filter edges by distance <= r (if r is given).
    if radius is not None and edge_attr is not None:
        # Filter edges by the radius
        one_hop_edges = []
        for u, v, data_dict in G.edges(data=True):
            if 'distance' in data_dict and data_dict['distance'] <= radius:
                one_hop_edges.append((u,v))

        # Draw edges that are within radius r
        nx.draw_networkx_edges(G, pos, edgelist=one_hop_edges, 
                               edge_color='black', width=1.337, alpha=alpha, ax=ax)
    else:
        # If no filtering by radius, just draw all edges
        nx.draw_networkx_edges(G, pos, edge_color='black', width=1.337, alpha=alpha, ax=ax)

    # Plot anchors
    ax.scatter(X_true[:n_anchors, 0], X_true[:n_anchors, 1], 
               marker="*", c="r", label=r"$N_{a}$", s=150)
    for i in range(n_anchors):
        ax.annotate(rf"$A_{{{i}}}$", (X_true[i, 0], X_true[i, 1]), 
                    textcoords="offset points", xytext=(0, 10), 
                    ha='center', fontsize=12, color='r')

    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    if name is not None:
        ax.set_title(name, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_network(X_true: np.ndarray, B: np.ndarray,
                  n_anchors: int, r: float = None,
                    alpha: float = 0.6, subset=None,
                      D: np.ndarray = None, zoom: int = None,
                        ax = None, name:str = None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    
    G = nx.from_numpy_array(D)
    pos = {i: (X_true[i][0], X_true[i][1]) for i in range(len(X_true))}

    # Draw immediate neighbor edges (within radius r)
    one_hop = D.copy()
    one_hop *= B
    one_hop[one_hop > r] = 0
    nx.draw_networkx_edges(nx.from_numpy_array(one_hop), pos, edge_color='black', width=1.337, alpha=alpha, ax=ax)
    
    # Plot anchors
    ax.scatter(X_true[:n_anchors, 0], X_true[:n_anchors, 1], marker="*", c="r", label=r"$N_{a}$", s=150)
    for i in range(n_anchors):
        ax.annotate(rf"$A_{{{i}}}$", (X_true[i, 0], X_true[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='r')

    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    #plt.xlabel("X-coordinate", fontsize=14)
    #plt.ylabel("Y-coordinate", fontsize=14)
    #plt.title("1-Hop True Connectivity Network", fontsize=16)
    #plt.title("True Graph", fontsize=16)
    ax.set_title(name, fontsize=16)
    plt.show()

def plot_results(X, predicts, n_anchors):
    fig, ax = plt.subplots()
    print(X.shape)
    # Plot anchors, true positions, and predictions
    ax.scatter(X[:n_anchors, 0], X[:n_anchors, 1], label=rf"$N_{{a}}$", c="red", marker='*')
    ax.scatter(X[n_anchors:, 0], X[n_anchors:, 1], label=rf"$N_{{t}}$", c="green", marker='P')
    ax.scatter(predicts[n_anchors:, 0], predicts[n_anchors:, 1], label="Estimates", c="orange", marker="X")

    ax.plot([X[:, 0], predicts[:, 0]], [X[:, 1], predicts[:, 1]], "k--")

    # Annotate node numbers
    for j, p in enumerate(X):
        if j < n_anchors:
            ax.annotate(rf"$A_{{{j}}}$", p)
        else:
            ax.annotate(rf"$T_{{{j}}}$", p)

    # Set plot properties
    ax.set_title("Localization Results")
    #plt.ylim((30, 70))
    #plt.xlim((3, 88))
    ax.legend()
    """ n_view = n_anchors
    print(f" X_true[:{n_view}]: {X[:n_anchors + n_view]}")
    print(f" y_pred[:{n_view}]: {predicts[:n_anchors + n_view]}") """
    plt.show()

""" def graph_vis():
    graph = make_data(seed=21, priors=True)

    particles = graph['x'].detach().numpy().reshape(100, 50, 2)
    X_true = graph['y'].numpy()

    edge_index = graph.edge_index
    edge_attr = graph.edge_attr

    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    edge_attr_np = edge_attr.numpy()

    network_values = edge_attr_np[:, 0]
    connectivity_values = edge_attr_np[:, 1]

    num_nodes = graph.num_nodes
    network_reconstructed = np.zeros((num_nodes, num_nodes), dtype=network_values.dtype)
    connectivity_reconstructed = np.zeros((num_nodes, num_nodes), dtype=connectivity_values.dtype)
    network_reconstructed[row, col] = network_values
    connectivity_reconstructed[row, col] = connectivity_values


    plot_network(X_true, D=network_reconstructed, B=connectivity_reconstructed, r=22, n_anchors=7) """

def spring_loss(node_positions, edge_index, edge_attr, two_hop_index, lambda1=1.0, lambda2=1.0):
    """
    Computes the combined spring loss for a graph.

    Args:
        node_positions (torch.Tensor): Positions of nodes (n_nodes, d_dim).
        edge_index (torch.Tensor): Direct edges (1-hop neighbors).
        edge_attr (torch.Tensor): Edge attributes (optional, not used here).
        two_hop_index (torch.Tensor): 2-hop edges (n_edges_2hop, 2).
        lambda1 (float): Weight for 1-hop attractive force.
        lambda2 (float): Weight for 2-hop repulsive force.

    Returns:
        torch.Tensor: Combined spring loss.
    """
    # Attractive loss (1-hop neighbors)
    src, tgt = edge_index
    diff = node_positions[src] - node_positions[tgt]
    attractive_loss = torch.sum(diff.pow(2))  # Minimize squared distances

    # Repulsive loss (2-hop neighbors)
    src_2hop, tgt_2hop = two_hop_index
    diff_2hop = node_positions[src_2hop] - node_positions[tgt_2hop]
    distances_2hop = torch.norm(diff_2hop, dim=1)
    repulsive_loss = torch.sum(1.0 / (distances_2hop.pow(2) + 1e-6))  # Maximize distances

    # Combine losses
    total_loss = lambda1 * attractive_loss + lambda2 * repulsive_loss
    return total_loss

def ERR(y_true: np.ndarray, y_pred: np.ndarray)->float:
    return np.sqrt(np.sum((y_true - y_pred)**2, axis=1))

def MAE(y_true, y_pred):
    return np.mean(ERR(y_true, y_pred))

def RMSE(targets: np.ndarray, predicts: np.ndarray):
    error = ERR(targets, predicts)
    mse = np.mean(error**2)
    rmse = np.sqrt(mse)

    mean_error = np.mean(error)
    median_error = np.median(error)
    return rmse, median_error

def plot_generalization(train_losses: list, eval_losses: list):
    plt.plot(train_losses, label="Train loss")
    plt.plot(eval_losses, label="Validation loss")
    plt.yticks(np.arange(31))
    plt.xticks(np.arange(50))
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.show()


def mono_potential_bbox(bbox: np.ndarray):
    """
    Returns a function of prior probability for a given bounded box.

    Parameters
    --
    bbox: np.ndarray
        Bounded box for a node in 2D space.

    Returns
    --
    joint_pdf: function
        Returns a uniform pdf function for a given bounded box created for a node.
    """
    x_min, x_max, y_min, y_max = bbox
    bbox_area = (x_max - x_min) * (y_max - y_min)
    
    def joint_pdf(r: np.ndarray) -> np.ndarray:
        inside = (x_min <= r[:, 0]) & (r[:, 0] <= x_max) & (y_min <= r[:, 1]) & (r[:, 1] <= y_max)
        return inside / bbox_area

    return joint_pdf
"""
# Assume we have graph data and a GNN model
node_positions = model(graph_data)  # Output of the GNN (node embeddings)
two_hop_index = get_two_hop_edges(edge_index, num_nodes=node_positions.shape[0])

# Compute loss
loss = spring_loss(node_positions, edge_index, edge_attr, two_hop_index)

# Backpropagation
optimizer.zero_grad()
loss.backward()
optimizer.step()
"""