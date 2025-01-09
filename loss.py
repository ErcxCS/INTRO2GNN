import torch
import torch.nn as nn

class EuclideanLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EuclideanLoss, self).__init__()

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        squared_diff = torch.pow(diff, 2)
        squared_sum = torch.sum(squared_diff, dim=1)
        distance = torch.sqrt(squared_sum)

        #mse = torch.mean(torch.pow(distance, 2))
        #rmse = torch.sqrt(mse)
        #return rmse
    
        mse_out = torch.mean(distance)
        return mse_out
    
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