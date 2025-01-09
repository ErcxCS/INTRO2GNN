import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import euclidean_distances

import networkx as nx


g_count = 0
def data_handling_of_graphs():
    from torch_geometric.data import Data
    edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.t().contiguous())

    print(f"keys: {data.keys()}")
    print(f"data as dict: data['x']: {data['x']}")
    print(f"data as dict: data['edge_index']: {data['edge_index']}")

    for key, item in data:
        print(f'{key} found in data')
        print(f'item at key: {item}')

    print('edge_attr' in data)

    print(f"number of nodes: {data.num_nodes}")
    print(f"number of edges: {data.num_edges}")
    print(f"number of node features: {data.num_node_features}")


    print(f"data has isoleted nodes: {data.has_isolated_nodes()}")
    print(f"data has loop: {data.has_self_loops()}")

    print(f"is graph directed: {data.is_directed()}")
    print(f"{data}")
    device = torch.device('cuda')
    data = data.to(device)
    print(device)


def common_benchmark_datasets():
    from torch_geometric.datasets import TUDataset
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    print(type(dataset))
    print(f"edge attrs: {'edge_attr' in dataset}")

    print(f"dataset size: {len(dataset)}")
    print(f"number of classes: {dataset.num_classes}")
    print(f"number of node features: {dataset.num_node_features}")

    print(f"data: {dataset[0]}")
    print(f"type: {type(dataset[0])}")
    print(f"is undirected: {dataset[0].is_undirected()}")

    """ for key, item in dataset[0]:
        print(f"key: {key}, data['key']: {dataset[0][key]}")
        print(f"item: {item}") """
    
    for data in dataset[111]:
        print(f"type(data): {type(data)}")
        print(data)

def plot_graph():
    import networkx as nx
    import matplotlib.pyplot as plt

    # Convert edge_index to a NetworkX graph
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    graph = nx.Graph()
    graph.add_edges_from(edge_index.t().numpy())  # Add edges

    # Add isolated nodes (those not in edge_index)
    for node in range(5):  # Total nodes are 5
        if node not in graph:
            graph.add_node(node)

    # Draw the graph
    nx.draw(graph, with_labels=True, node_color='lightblue')
    plt.show()

def edge_to_node_embeddings():
    from torch_geometric.nn import NNConv

    # Example: Node features, edge index, and edge attributes
    node_features = torch.randn((5, 4))  # 5 nodes, 4 features each
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])  # Connectivity
    edge_attr = torch.randn((4, 3))  # 4 edges, 3 features each

    # Neural network to compute edge weights
    nn = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16))

    # NNConv layer
    conv = NNConv(in_channels=4, out_channels=16, nn=nn)
    updated_node_features = conv(node_features, edge_index, edge_attr)

def simple_GNN():
    import torch.nn.functional as F
    from torch_geometric.nn import NNConv
    from torch_geometric.data import Data

    torch.manual_seed(42)
    class GNN(torch.nn.Module):
        def __init__(self, d: int = 2):
            super().__init__()
            # First NNConv layer
            self.conv1 = NNConv(
                in_channels=4, # node features
                out_channels=16,
                nn=torch.nn.Sequential(
                    torch.nn.Linear(3, 64),  # 3 edge features (edge_attr) -> 64
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 64)  # Final output: 16 * 4 = 64
                )
            )
            # Second NNConv layer
            self.conv2 = NNConv(
                in_channels=16,
                out_channels=32,
                nn=torch.nn.Sequential(
                    torch.nn.Linear(3, 128),  # 3 (edge_attr) -> 128
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 512)  # Final output: 32 * 16 = 512
                )
            )
            self.fc = torch.nn.Linear(32, d)  # Final layer for prediction (e.g., regression)
        
        def forward(self, data: Data):
            # Extract features and graph structure from the Data object
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x = F.relu(self.conv2(x, edge_index, edge_attr))

            return self.fc(x)
        
    n_nodes = 5
    d_dimension = 2
    node_features = 4
    edge_features = 3

    node_features = torch.randn((n_nodes, node_features)) # 5 nodes, 4 features each
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long) # not fully connected
    edge_attr = torch.randn((4, edge_features)) # 4 edges, 3 features each
    y = torch.randn((n_nodes, d_dimension)) # true node values

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # Usage
    model = GNN(d=d_dimension)
    output = model(data)
    print(f"output: {output}")
    
    # report
    criterion = torch.nn.MSELoss()
    loss = criterion(output, data.y)
    print(loss)

def testing_on_benchmark_dataset():
    # Imports
    import torch.nn.functional as F
    from torch_geometric.nn import NNConv, global_mean_pool
    from torch_geometric.data import Batch, Data
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    import numpy as np
    from sklearn.metrics import classification_report
    from sklearn.metrics._regression import root_mean_squared_error


    def load_and_batch_dataset(seed: int=42, percent: float=0.66, batch_size: int=32):
        #dataset = TUDataset(root='/tmp/TUDataset', name='MUTAG')
        dataset = test()

        torch.manual_seed(seed)
        dataset.shuffle()
        train_size = int(len(dataset)*percent)
        print(f"size: {len(dataset)}")

        train_dataset = dataset[:train_size]
        test_dataset = dataset[train_size:]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"num classes: {dataset.num_classes}")
        
        return dataset, train_loader, test_loader
    
    def get_model(num_nodes: int,
            in_channels: int,
                hidden_channels: int,
                d_dim: int,
                num_edge_features: int):

        class GNN(torch.nn.Module):
            def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, num_classes: int, num_edge_features: int):
                super().__init__()

                self.node_embeddings = torch.nn.Parameter(torch.randn(num_nodes, 1))

                self.conv1 = NNConv(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    nn=torch.nn.Sequential(
                        torch.nn.Linear(num_edge_features, 64),
                        torch.nn.ReLU(),
                        torch.nn.Linear(64, hidden_channels * in_channels)
                    )
                )

                self.conv2 = NNConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    nn=torch.nn.Sequential(
                        torch.nn.Linear(num_edge_features, 128),
                        torch.nn.ReLU(),
                        torch.nn.Linear(128, hidden_channels * hidden_channels)
                    )
                )

                self.fc = torch.nn.Linear(hidden_channels, num_classes)

            def forward(self, data: Data):
                x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
                x = data.x if data.x is not None else self.node_embeddings[:data.num_nodes]
                x = F.relu(self.conv1(x, edge_index, edge_attr))
                x = F.relu(self.conv2(x, edge_index, edge_attr))
                logits = self.fc(x)
                return logits

        return GNN(num_nodes, in_channels, hidden_channels, d_dim, num_edge_features)


    
    def train_and_evaluate():
        def train_model(model, loader, optimizer, criterion, device):
            model.train()
            total_loss = 0
            for data in loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)
        
        def evaluate_model(model, loader, device):
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    out = model(data)
                    y_true.append(data.y.cpu().numpy())
                    y_pred.append(out.cpu().numpy())
                    #y_pred.append((out > 0).long().cpu().numpy())  # Threshold logits at 0
            return np.vstack(y_true), np.vstack(y_pred)
        
        dataset, train_loader, test_loader = load_and_batch_dataset(seed=42, percent=0.66, batch_size=32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(num_nodes=100,
                        in_channels=1,
                      hidden_channels=32,
                      d_dim=dataset.num_classes,
                      num_edge_features=dataset.num_edge_features,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        #criterion = torch.nn.CrossEntropyLoss()
        #criterion = torch.nn.BCEWithLogitsLoss()
        #criterion = spring_loss()
        criterion = torch.nn.MSELoss()

        epochs = 20
        for epoch in range(1, epochs + 1):
            loss = train_model(model, train_loader, optimizer, criterion, device)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        y_true, y_pred = evaluate_model(model, test_loader, device)
        print(f"Regression report: shapes: {y_true.shape}")
        idx = 18
        y_true_graphs = y_true.reshape(34, 100, 2)
        y_pred_graphs = y_pred.reshape(34, 100, 2)
        print(root_mean_squared_error(y_true, y_pred))
        plot_results(y_true_graphs[idx], y_pred_graphs[idx], 7)


    train_and_evaluate()
        
def make_data(seed: int=None,
                 n_node: int=100,
                 d_dim: int=2,
                 p_particles: int=50,
                 a_anchors: int=7,
                 meters: int=100,
                 communication_radius: int=22,
                 noise: float = 1,
                 priors: bool = True,
                 hop: str = 'two',
                 ):
    from torch_geometric.data import Data
    import torch.nn as nn

    np.random.seed(seed)

    X_true, deployment_area = generate_targets(seed, (n_node, d_dim), meters, a_anchors, False)
    generated_anchors = generate_anchors(deployment_area, a_anchors, border_offset=np.sqrt(meters)*1)
    a_anchors = len(generated_anchors)
    X_true[:a_anchors] = generated_anchors
    full_D, D, B, RSS = get_distance_matrix(X_true, a_anchors, communication_radius, noise)
    network = get_graphs(D)[hop] # D, or 2-hop D ['one', 'two', 'full']
    #plot_network(X_true, B, a_anchors, communication_radius, D=D)

    # particle sampling logic
    intersection_bbox, bbox = create_bbox(network, generated_anchors, deployment_area, communication_radius)
    particles = sample_particles(intersection_bbox, generated_anchors, p_particles, priors, meters)

    # learnable embeddings
    particles = torch.tensor(particles, requires_grad=True, dtype=torch.float)
    particles = nn.Parameter(particles)

    # decoupling anchors from learnable embeddings
    fixed_mask = torch.zeros(particles.shape[0], dtype=torch.bool)
    fixed_mask[:a_anchors] = True
    particles.data[fixed_mask] = particles.data[fixed_mask].detach()

    # casting particles from (n_nodes, p_particles, d_dim) -> (n_nodes, p_particles * d_dim)
    x = particles.view(particles.shape[0], -1)

    # Generating graph data
    row, col = np.nonzero(network)
    edge_index = torch.tensor(np.stack([row, col], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(
        np.stack([network[row, col], B[row, col]], axis=1),
        dtype=torch.float)
    y = torch.tensor(X_true, dtype=torch.float)

    if not priors:
        x = None

    graph_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )
    graph_data.num_nodes = n_node
    # for simplycity remove x

    """
    since each graph layer applys messsage passing, each next layer will receive (n-1)-hop embeddings
    therefore, passing 2-hop data may not be necessariy
    on the other hand, providing this data intrinsicly, decouples n-hop logic from layer amount.
    Additionally, loss function may be challenging to formulate in case it depends on layer amount.
    """

    return graph_data


def generate_graph_dataset():
    from torch_geometric.data import Dataset, Data
    import torch

    class CustomGraphDataset(Dataset):
        def __init__(self, root: str | None = None,
                     num_graphs: int = 100,
                     transform=None,
                     pre_transform=None,
                     **kwargs):
            self.num_graphs = num_graphs
            self.graph_kwargs = kwargs
            super().__init__(root, transform, pre_transform)

            self.graphs = []
            for i in range(self.num_graphs):
                local_kwargs = {k: v for k, v in kwargs.items() if k != "seed"}
                local_kwargs["seed"] = kwargs["seed"] + i
                self.graphs.append(make_data(**local_kwargs))

        def len(self):
            return self.num_graphs
        
        def get(self, idx):
            return self.graphs[idx]

    kwargs = {
        "seed" : 21,
        "num_nodes" : 100,
        "d_dim" : 2,
        "num_particles" : 50,
        "num_anchors" : 7,
        "meters" : 100,
        "radius" : 22,
        "noise" : 1,
        "priors" : False,
        "hop" : 'two',
    }

    return CustomGraphDataset(root="/tmp/CustomGraphDataset", **kwargs)
    

    
def main(choice: int=0):
    funcs = {
        '0': data_handling_of_graphs,
        '1': common_benchmark_datasets,
        '2': plot_graph,
        '3': edge_to_node_embeddings,
        '4': simple_GNN,
        '5': testing_on_benchmark_dataset,
        '6': generate_graph_dataset,
    }
    funcs[str(choice)]()

if __name__ == "__main__":
    
    main(5)

