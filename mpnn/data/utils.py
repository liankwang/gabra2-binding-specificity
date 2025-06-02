from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

def dataset_to_loaders(dataset, batch_size):
    """ Takes dataset and converts it to train, validation, and test Pytorch DataLoaders."""
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def add_reverse_edges(data: Data) -> Data:
    """
    Adds reverse edges and duplicates edge features for use in D-MPNN-style models.

    Args:
        data (Data): A PyG Data object with attributes:
                     - edge_index: Tensor of shape (2, num_edges)
                     - edge_attr: Tensor of shape (num_edges, d_e)
    
    Returns:
        Data: A new Data object with bidirectional edges and duplicated features.
    """
    # Original edges
    src, dst = data.edge_index
    edge_attr = data.edge_attr

    # Reverse edges
    rev_edge_index = torch.stack([dst, src], dim=0)
    full_edge_index = torch.cat([data.edge_index, rev_edge_index], dim=1)

    # Duplicate edge features for reverse edges
    full_edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

    # Return new Data object
    return Data(
        x=data.x,
        edge_index=full_edge_index,
        edge_attr=full_edge_attr,
        batch=data.batch
    )