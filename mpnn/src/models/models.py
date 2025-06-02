
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch_geometric.utils import add_self_loops

class DMPNN(nn.Module):
    """
    D-MPNN model from "Analyzing Learned Molecular Representations for Property Prediction" (Yang et al. 2019)

    Args:
        d_v: Dimension of node features
        d_e: Dimension of edge features
        d_h: Dimension of hidden features (for edge, node, and graph hidden states)
        T: Number of message passing steps
        output_dim: Dimension of output features
        task: "classification" or "regression"
    """
    def __init__(self, d_v, d_e, d_h, T=5, output_dim=1, task="classification"):
        super().__init__()
        self.T = T
        self.d_v = d_v
        self.d_e = d_e
        self.d_h = d_h
        self.output_dim = output_dim
        self.task = task

        print("Initializing DMPNN model with dimensions (dv, de, dh, T):", d_v, d_e, d_h, T)

        self.W_i = nn.Linear(d_v + d_e, d_h, bias=False) # For edge hidden feature initialization
        self.W_m = nn.Linear(d_h, d_h, bias=False) # For message function
        self.W_a = nn.Linear(d_v + d_h, d_h, bias=True) # For conversion to node features after message passing
        self.batch_norm = nn.BatchNorm1d(d_h, eps=1e-5, momentum=0.1, affine=True)

        self.readout = nn.Sequential( # For readout to graph-level representation
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(d_h, output_dim)
        )

    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric Data object
            data.x: (num_nodes, d_v)
            data.edge_index: (2, num_edges)
            data.edge_attr: (num_edges, d_e)
            data.rev_map: (num_edges)
            data.batch: (num_nodes)
        """
        node_feats, edge_adj, edge_feats, rev, batch = data.x, data.edge_index, data.edge_attr, data.rev_map, data.batch
        device = node_feats.device

        # Initialize hidden states
        src, dst = edge_adj # each of src and dst is (num_edges)
        edge_input = torch.cat([node_feats[src], edge_feats], dim=1) # (num_edges, d_v + d_e)
        h0 = self.W_i(edge_input) # (num_edges, d_h)
        h = F.relu(h0)
        
        # Message passing phase
        for _ in range(self.T):
            # Get hidden states of all reverse edges
            #rev = self.get_reverse_map(src, dst) # Indices of reverse edges (e.g. rev of (1,0) is the index corresponding to (0,1))
            h_rev = h[rev] # (num_edges, d_h)

            # Get sums of all edges into each node
            sum_per_node = torch.zeros(node_feats.shape[0], self.d_h, device=device)
            sum_per_node.index_add_(dim=0, index=src, source=h_rev)
            #sum_per_node.scatter_add_(dim=0, index=src.unsqueeze(1).expand(-1, self.d_h), src=h_rev) # (num_nodes, d_h)

            # Get messages for all edges
            msg = sum_per_node[src] - h_rev # (num_edges, d_h)

            # Update hidden states
            h = F.relu(h0 + self.W_m(msg)) # (num_edges, d_h)

        
        # Convert to node features
        m_v = torch.zeros(node_feats.shape[0], self.d_h, device=device) # (num_nodes, d_h)
        #m_v.scatter_add_(dim=0, index=src.unsqueeze(-1).expand(-1, self.d_h), src=h) # (num_nodes, d_h)
        m_v.index_add_(dim=0, index=src, source=h)
        h_v = F.relu(self.W_a(torch.cat([node_feats, m_v], dim=1))) # (num_nodes, d_h)

        # Get graph-level representation
        h_graph = torch.zeros(batch.max() + 1, self.d_h, device=device) # (num_graphs, d_h)
        #h_graph.scatter_add_(dim=0, index=batch.unsqueeze(-1).expand(-1, self.d_h), src=h_v) # (num_graphs, d_h)
        h_graph.index_add_(dim=0, index=batch, source=h_v)

        # Readout
        h_graph = self.batch_norm(h_graph)

        out = self.readout(h_graph) # (num_graphs, output_dim)
        
        return out
            
    # def get_reverse_map(self, src, dst):
    #     """
    #     Returns a map of reverse edges.
    #     """
    #     index_map = {(s.item(), d.item()): i for i, (s, d) in enumerate(zip(src, dst))}
    #     rev = torch.tensor([index_map[(d.item(), s.item())] for s, d in zip(src, dst)])
    #     return rev



########################################################
# DEPRECATED MODELS
########################################################
class EdgeGCN(MessagePassing):
    """ Defines a layer of a GCN that makes use of both node and edge features in constructing the message."""
    def __init__(self, in_channels, out_channels, edge_dim):
        super(EdgeGCN, self).__init__(aggr='add')
        self.lin_nodes = nn.Linear(in_channels, out_channels)
        self.lin_edges = nn.Linear(edge_dim, out_channels) 
        self.lin = nn.Linear(in_channels *2, out_channels)
    
    def forward (self, x, edge_index, edge_attr):
        #x = self.lin_nodes(x) # Map initial node features to hidden representations
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        dist = edge_attr[:, 0].view(-1, 1)
        edge_features = self.lin_edges(edge_attr)
        combined_features = torch.cat([x_j, edge_features], dim=-1)
        return self.lin(combined_features)

class SimpleGraphEncoder(nn.Module):
    """ Defines a graph encoder model using simple GCN layers. DEPRECATED"""
    def __init__(self, d_v, d_e, d_h, output_dim):
        print("Initializing GraphEncoder model with dimensions")
        print("Node features:", d_v)
        print("Edge features:", d_e)
        print("Hidden features:", d_h)
        super(SimpleGraphEncoder, self).__init__()

        self.lin = nn.Linear(d_h, output_dim)
        self.conv1 = GCNConv(d_v, d_h)
        self.conv2 = GCNConv(d_h, d_h)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, p=0.15, training=self.training)
        x = self.conv2(x, edge_index)

        # Readout layer; combines information from each node in a given graph
        x = global_mean_pool(x, data.batch)

        x = self.lin(x)
        return x

class GraphEncoder(nn.Module):
    """ Defines a graph encoder model using GCN and EdgeGCN layers."""
    def __init__(self, d_v, d_e, d_h, output_dim):
        print("Initializing GraphEncoder model with dimensions (dv, de, dh):", d_v, d_e, d_h)
        super(GraphEncoder, self).__init__()

        self.gcn = GCNConv(d_v, d_h)
        self.conv1 = EdgeGCN(d_h, d_h, d_e)
        self.lin = nn.Linear(d_h, output_dim)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.gcn(x, edge_index)
        x = F.relu(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)

        # Readout layer: combines information from each node to give a graph-level representation
        x = global_mean_pool(x, data.batch)

        x = self.lin(x)

        return x
