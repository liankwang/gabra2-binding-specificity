# For model testing
from models.models import DMPNN
import torch
from torch_geometric.data import Data
from data.utils import add_reverse_edges

# For mol object testing
from data.GraphObjectGenerator.graph_builder.mol_to_graph import mol_to_graph_data_obj


def test_mol_object():
    test_smiles = "C([*:1])([*:2])N"
    graph = mol_to_graph_data_obj(test_smiles)
    print(graph)


def test_dmpnn():
    ##### EASY TEST CASE ########
    # 3 nodes with 2-dimensional features
    node_feats = torch.tensor([
        [1.0, 0.0],  # node 0
        [0.0, 1.0],  # node 1
        [1.0, 1.0],  # node 2
    ], dtype=torch.float)

    # 4 edges (directed), with 1-dimensional edge features
    edge_index = torch.tensor([
        [0, 2, 0, 1],  # src
        [1, 0, 2, 0],  # dst
    ], dtype=torch.long)

    edge_feats = torch.tensor([
        [0.1],  # 0→1
        [0.1],  # 1→0
        [0.2],  # 0→2
        [0.2],  # 2→0
    ], dtype=torch.float)

    # All nodes belong to same graph in the batch
    batch = torch.tensor([0, 0, 0], dtype=torch.long)




    ######### HARDER TEST CASE #########

    # Graph 1: 4 nodes, 5 directed edges
    node_feats_1 = torch.tensor([
        [1.0, 0.0],  # node 0
        [0.0, 1.0],  # node 1
        [1.0, 1.0],  # node 2
        [0.5, 0.5],  # node 3
    ], dtype=torch.float)

    edge_index_1 = torch.tensor([
        [0, 1, 2, 3, 0],
        [1, 2, 3, 0, 2],
    ], dtype=torch.long)

    edge_feats_1 = torch.tensor([
        [0.1],
        [0.2],
        [0.3],
        [0.4],
        [0.5],
    ], dtype=torch.float)

    batch_1 = torch.tensor([0, 0, 0, 0], dtype=torch.long)  # All nodes in graph 0


    # Graph 2: 3 nodes, 3 directed edges
    node_feats_2 = torch.tensor([
        [0.2, 0.8],  # node 0
        [0.9, 0.1],  # node 1
        [0.4, 0.4],  # node 2
    ], dtype=torch.float)

    edge_index_2 = torch.tensor([
        [0, 1, 1],
        [1, 2, 0],
    ], dtype=torch.long)

    edge_feats_2 = torch.tensor([
        [0.6],
        [0.7],
        [0.8],
    ], dtype=torch.float)

    batch_2 = torch.tensor([1, 1, 1], dtype=torch.long)  # All nodes in graph 1

    # Graph 3: equivalent to graph 2 but reverse
    node_feats_3 = torch.tensor([
        [0.2, 0.8],  # node 0
        [0.9, 0.1],  # node 1
        [0.4, 0.4],  # node 2
    ], dtype=torch.float)

    edge_index_3 = torch.tensor([
        [1, 2, 0],
        [0, 1, 1]
    ], dtype=torch.long)

    edge_feats_3 = torch.tensor([
        [0.8],
        [0.7],
        [0.6],
    ], dtype=torch.float)

    batch_3 = torch.tensor([2, 2, 2], dtype=torch.long) 

    # Combine graphs
    node_feats = torch.cat([node_feats_1, node_feats_2, node_feats_3], dim=0)
    edge_index = torch.cat([
        edge_index_1,
        edge_index_2 + node_feats_1.shape[0],  # shift node indices
        edge_index_3 + node_feats_1.shape[0] + node_feats_2.shape[0],
    ], dim=1)
    edge_feats = torch.cat([edge_feats_1, edge_feats_2, edge_feats_3], dim=0)
    batch = torch.cat([batch_1, batch_2, batch_3], dim=0)

    # Wrap in PyG Data object
    data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats, batch=batch)
    data = add_reverse_edges(data)

    print(data.edge_index)


    model = DMPNN(d_v=node_feats.shape[1], 
                d_e=edge_feats.shape[1], 
                d_h=3, T=2, output_dim=1, 
                task="classification")



    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
    print(out)


if __name__ == "__main__":
    # test_dmpnn()
    test_mol_object()