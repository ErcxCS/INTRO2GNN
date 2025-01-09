import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, BatchNorm
from torch_geometric.data import Data

class GNN(nn.Module):
    def __init__(
        self, 
        num_nodes: int, 
        d_dim: int, 
        in_channels: int, 
        hidden_channels: int, 
        num_edge_features: int
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_dim = d_dim
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_edge_features = num_edge_features
        
        # The 'nn' submodule for NNConv:
        edge_network_1 = nn.Sequential(
            nn.Linear(num_edge_features, 128),
            #BatchNorm(128),
            nn.ReLU(),
            #nn.Dropout(p=0.2),

            nn.Linear(128, 128),
            #BatchNorm(128),
            nn.ReLU(),
            #nn.Dropout(p=0.2),

            nn.Linear(128, 128),
            #BatchNorm(128),
            nn.ReLU(),
            #nn.Dropout(p=0.2),

            nn.Linear(128, hidden_channels * in_channels),
        )

        self.conv1 = NNConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            nn=edge_network_1,
        )
        
        edge_network_2 = nn.Sequential(
            nn.Linear(num_edge_features, 128),
            #BatchNorm(128),
            nn.ReLU(),
            #nn.Dropout(p=0.2),

            nn.Linear(128, 128),
            #BatchNorm(128),
            nn.ReLU(),
            #nn.Dropout(p=0.2),

            nn.Linear(128, 128),
            #BatchNorm(128),
            nn.ReLU(),
            #nn.Dropout(p=0.2),

            nn.Linear(128, hidden_channels * hidden_channels),
        )

        self.conv2 = NNConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            nn=edge_network_2
        )

        #self.drop1 = nn.Dropout(p=0.2)
        #self.drop2 = nn.Dropout(p=0.2)

        self.batch1 = BatchNorm(hidden_channels)
        self.batch2 = BatchNorm(hidden_channels)

        self.fc = nn.Linear(hidden_channels, d_dim)

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # If x is None, create or fetch a dummy or learnable embedding
        if x is None:
            x = torch.ones(data.num_nodes, self.in_channels, device=edge_index.device)

        # ------ Layer 1 -------------
        x = self.conv1(x, edge_index, edge_attr)
        x = self.batch1(x)
        x = F.relu(x)
        #x = self.drop1(x)

        # ------ Layer 2 -------------
        x = self.conv2(x, edge_index, edge_attr)
        x = self.batch2(x)
        x = F.relu(x)
        #x = self.drop2(x)

        # Final regression
        anchor_mask = data.anchor_mask.to(edge_index.device) if hasattr(data, 'anchor_mask') else None

        if anchor_mask is not None:
            logits = torch.zeros_like(data.y)
            anchor_repr = x[anchor_mask]
            non_anchor_repr = x[~anchor_mask]
            
            non_anchor_out = self.fc(non_anchor_repr)
            anchor_out = data.y[anchor_mask]
            logits[anchor_mask] = anchor_out
            logits[~anchor_mask] = non_anchor_out
        else:
            logits = self.fc(x)  # shape: [num_nodes, d_dim]
            
        return logits
