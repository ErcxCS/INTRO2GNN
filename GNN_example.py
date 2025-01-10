
from utils import *
from dataset import CustomGraphDataset
from model import GNN
from optimized_NBP import NBP

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from torch_geometric.loader import DataLoader
from sklearn.metrics._regression import root_mean_squared_error
from loss import EuclideanLoss

def generate_graph_dataset(num_graphs: int = 100) -> CustomGraphDataset:
    kwargs = {
        "seed" : 21,
        "num_nodes" : 100,
        "d_dim" : 2,
        "num_particles" : 50,
        "num_anchors" : 7,
        "meters" : 100,
        "radius" : 22,
        "noise" : 1,
        "priors" : True,
        "hop" : 'two',
    }

    return CustomGraphDataset(root="./Graphs/NoPriorMeanParticles", num_graphs=num_graphs, **kwargs)

def load_and_batch_dataset(dataset: CustomGraphDataset, percent: float=0.66, batch_size: int=32, eval_size: int=20):
    torch.manual_seed(dataset.graph_kwargs["seed"])
    dataset = dataset.shuffle()
    train_size = int(len(dataset)*percent)
    
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    eval_dataset = test_dataset[:eval_size]
    remaining_test_dataset = test_dataset[eval_size:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(remaining_test_dataset, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    return dataset, train_loader, test_loader, eval_loader

def get_model(num_nodes: int, d_dim, in_channels: int, hidden_channels: int, num_edge_features: int) -> GNN:
    return GNN(num_nodes=num_nodes,
               d_dim=d_dim,
               in_channels=in_channels,
               hidden_channels=hidden_channels,
               num_edge_features=num_edge_features)

def train_model(model: GNN, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device):
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

def test_model(model: GNN, loader: DataLoader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            y_true.append(data.y.cpu().numpy())
            y_pred.append(out.cpu().numpy())
    return np.vstack(y_true), np.vstack(y_pred)

def evaluate_model(model: GNN, loader: DataLoader, criterion: nn.Module, device):
    one_epoch = []
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            one_epoch.append(loss.item())
            total_loss += loss.item()
    return total_loss / len(loader), one_epoch

def run_NBP(loader: DataLoader):
    eval_rmse = []
    graph_preds = []
    for batch in loader:
        graphs = batch.to_data_list()
        for graph in graphs:
            X_true = graph.y.cpu().numpy()

            nbp = NBP(X_true, graph.num_anchors, 10)
            l_rmse, estimates = nbp.iterative_NBP(graph)
            
            graph_preds.append(estimates)

            targets = X_true[graph.num_anchors:]
            print(f"iter/rmse: {l_rmse}")

            eval_rmse.append(l_rmse)

    return eval_rmse


def plot_compare_GNN_vs_NBP(gnn_epoch_losses: np.ndarray, nbp_epoch_losses: np.ndarray, train_losses: np.ndarray):
    mean_gnn_losses = np.mean(gnn_epoch_losses, axis=1)
    mean_nbp_losses = np.mean(nbp_epoch_losses, axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 10 + 1), mean_gnn_losses, label="Eval Mean GNN Loss", marker='o', linestyle='--')
    plt.plot(range(1, 10 + 1), train_losses, label="Train Mean GNN Loss", marker='*', linestyle='dotted')
    plt.plot(range(1, 10 + 1), mean_nbp_losses, label="Mean NBP RMSE", marker='x', linestyle='-')

    plt.xlabel("Iterations (Epochs)")
    plt.ylabel("Mean Loss / RMSE")
    plt.title("Mean GNN Loss vs. Mean NBP RMSE Across Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_best_mixed_GNN_vs_NBP(gnn_epoch_losses: np.ndarray, nbp_epoch_losses: np.ndarray):
    final_gnn_losses = gnn_epoch_losses[-1]
    final_nbp_losses = nbp_epoch_losses[-1]

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 20 + 1), final_gnn_losses, label="Final GNN losses", marker="o", linestyle="--")
    plt.plot(range(1, 20 + 1), final_nbp_losses, label="Final NBP losses", marker="x", linestyle="-")

    plt.xlabel("Graphs")
    plt.ylabel("final RMSE")
    plt.title("Final GNN rmse losses vs. Final NBP rmse losses")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """ graph = make_graph(priors=False, hop='two', seed=22, num_anchors=0)
    plot_data_graph(graph, graph.num_anchors, graph.radius, name="Generated Graph") """
    
    d_size = 1000
    percent = 0.78
    b_size = 32

    dataset = generate_graph_dataset(d_size)
    dataset, train_loader, test_loader, eval_loader = load_and_batch_dataset(dataset, percent=percent, batch_size=b_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")
    # in_channels: num_node_features
    model = get_model(num_nodes=100, d_dim=2, in_channels=1*2, hidden_channels=64, num_edge_features=2)
    model = model.to(device)
    
    from torchinfo import summary
    summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    #criterion = nn.MSELoss()
    criterion = EuclideanLoss()

    train_losses = []
    eval_losses = []
    gnn_epoch_losses = []
    epochs = 50
    for epoch in range(1, epochs + 1):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        eval_loss, one_epoch = evaluate_model(model, eval_loader, criterion, device)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}')

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        gnn_epoch_losses.append(one_epoch)

    plot_generalization(train_losses, eval_losses)
    gnn_epoch_losses = np.array(gnn_epoch_losses)
    #nbp_epoch_losses = np.array(run_NBP(eval_loader)).T

    """ for i in range(len(gnn_epoch_losses)):
        print(f"GNN: {gnn_epoch_losses[i]}")
        print(f"NBP: {nbp_epoch_losses[i]}")
        print("#"*15)

    plot_compare_GNN_vs_NBP(gnn_epoch_losses, nbp_epoch_losses, train_losses)
    plot_best_mixed_GNN_vs_NBP(gnn_epoch_losses, nbp_epoch_losses) """

    test_size = d_size - int(d_size * percent) - 20
    y_true, y_pred = test_model(model, test_loader, device)
    y_true_graphs = y_true.reshape(test_size, 100, 2)
    y_pred_graphs = y_pred.reshape(test_size, 100, 2)

    rmse_scores = np.zeros((test_size,))
    rmse_scores_2 = np.zeros((test_size,))
    for i, (y_true_g, y_pred_g) in enumerate(zip(y_true_graphs, y_pred_graphs)):
        rmse_scores[i] = root_mean_squared_error(y_true_g, y_pred_g)
        rmse_scores_2[i], _ = RMSE(y_true_g, y_pred_g)
    #print(f"Regression report: {rmse_scores}")
    #print(f"rmse scores: {rmse_scores_2}")
    print(f"average rmse_ score: {np.mean(rmse_scores)}")
    print(f"average rmse score: {np.mean(rmse_scores_2)}")

    plot_results(y_true_graphs[12], y_pred_graphs[12], 11)

if __name__ == "__main__":
    main()
    #run_NBP()