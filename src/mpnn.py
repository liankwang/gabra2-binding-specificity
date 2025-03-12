import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool, SAGEConv, GraphConv,GINEConv
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
from torch_geometric.utils import add_self_loops, to_undirected, is_undirected
from torch.optim.lr_scheduler import StepLR

from create_graph_dataset import MolGraphDataset


random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


class MPNNLayer(MessagePassing): # NOT IN USE
    """
    NOT CURRENTLY IN USE!!!!!!

    Message Passing Neural Network Layer.
    Args:
        d_v (int): Node feature dimensions.
        d_e (int): Edge feature dimensions.
        d_h (int): Hidden message dimensions.
    """
    def __init__(self, d_v, d_e, d_h):
        super(MPNNLayer, self).__init__(aggr='mean')
        print("Initializing MPNN layer with dimensions:", d_v, d_e, d_h)
        # self.node_transform = nn.Linear(d_v, d_h)
        # self.edge_transform = nn.Linear(d_e, d_h)

        self.W_i = nn.Linear(d_v + d_e, d_h) # for computing messages
        #self.W_i = nn.Linear(d_h * 2, d_h) # for computing messages
        self.W_o = nn.Linear(d_v + d_h, d_h) # for updating features
        #self.W_o = nn.Linear(d_h, d_h)
        self.W_h = nn.Linear(d_v + d_h, d_h) # for updating features

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the MPNN layer.
        Args:
            x: Node features (N x d_v).
            edge_index: Defines edges in the graph (2 x E).
            edge_attr: Edge features (E x d_e).
        """

        if edge_attr.shape[0] * 2 == edge_index.shape[1]:  # Check for undirected graph duplication
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        # x = self.node_transform(x)
        # edge_attr = self.edge_transform(edge_attr)

        # Propagate calls message, aggregate, then update functions, and returns updated edge feature matrix
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        """
        Message function to compute messages from neighbors.
        Args:
            x_j: Node features of neighbors (E x d_v).
            edge_attr: Edge features (E x d_e).
        """
        # Concatenate node features and edge features 
        #print("Message function. X_j has dimensions ", x_j.shape, "and edge_attr has dimensions", edge_attr.shape)
        m = torch.cat([x_j, edge_attr], dim=-1) # (E x (d_v + d_e))
        #print("Concatenated message shape:", m.shape)
        m = self.W_i(m) # (E x d_h)
        return m

    def update(self, aggr_out, x):
        """
        Update node features by combining aggregated messages with original node features
        Args:
            aggr_out: Aggregated messages (N x d_h)
            x: Original node features (N x d_v)
        """
        # Concatenate node features and aggregated messages
        out = torch.cat([x, aggr_out], dim=-1) # (N x (d_v + d_h))
        #out = x + aggr_out
        out = self.W_o(out) # (N x d_h)
        return out

class EdgeGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(EdgeGCN, self).__init__(aggr='add')
        self.lin_nodes = nn.Linear(in_channels, out_channels)
        self.lin_edges = nn.Linear(edge_dim, out_channels)
        self.lin = nn.Linear(in_channels *2, out_channels)
    
    def forward (self, x, edge_index, edge_attr):
        # Edge_index has bidirectional edges. Duplicate edge_attr for each direction
        num_uni_edges = edge_attr.shape[0]
        edge_attr = edge_attr.repeat(1, 2).reshape(num_uni_edges * 2, edge_attr.shape[1])
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr)

        #x = self.lin_nodes(x) # Map initial node features to hidden representations
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        dist = edge_attr[:, 0].view(-1, 1)
        edge_features = self.lin_edges(edge_attr)
        combined_features = torch.cat([x_j, edge_features], dim=-1)
        return self.lin(combined_features)

class SimpleGraphEncoder(nn.Module):
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

def dataset_to_loaders(dataset, batch_size):
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

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): Number of epochs to wait for improvement before stopping
            verbose (bool): Print status messages
            delta (float): Minimum change to qualify as an improvement
            path (str): Path to save the model checkpoint
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        """
        Call this function after each validation epoch to check if training should stop
        """
        score = -val_loss  # We want to minimize validation loss, so we negate it

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves the model when validation loss decreases'''
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model...')
        torch.save(model.state_dict(), self.path)
        self.best_model_wts = model.state_dict()

def train(model, train_loader, optimizer, scheduler, criterion, num_epochs, val_loader=None, with_val=False):
    model.train()
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=10, verbose=True, path='../output/checkpoint.pth')

    for epoch in range(num_epochs):
        total_train_loss = 0
        for batch in train_loader:
            # Compute weights for positive class
            if (batch.y == 1).float().sum() == 0 or (batch.y == 0).float().sum() == 0:
                pos_weight = torch.tensor([1.0])
            else:
                pos_weight = (batch.y == 0).float().sum() / (batch.y == 1).float().sum()

            # Train
            optimizer.zero_grad()
            out = model(batch)
            #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = criterion(out, batch.y.view(-1,1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        ave_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {ave_train_loss:.4f}')
        train_losses.append(ave_train_loss)
    
        if with_val:
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                for batch in val_loader:
                    out = model(batch)
                    loss = criterion(out, batch.y.view(-1,1))
                    total_val_loss += loss.item()
            ave_val_loss = total_val_loss / len(val_loader)
            val_losses.append(ave_val_loss)
            print(f'Validation Loss: {ave_val_loss:.4f}')
        
        # early_stopping(ave_val_loss, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        
        #scheduler.step()

    return model, train_losses, val_losses

def plot_roc_curve(model, loader):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            probs = torch.sigmoid(out).numpy().flatten()
            labels = batch.y.numpy()

            all_probs.extend(probs)
            all_labels.extend(labels)

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def evaluate(model, test_loader):
    model.eval()
    preds = []
    probs = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            prob = torch.sigmoid(out).numpy().flatten()
            probs.append(prob)
            #pred = (prob > 0.5).float()
            #preds.append(pred)
            true_labels.append(batch.y)
    probs = np.concatenate(probs)
    true_labels = np.concatenate(true_labels)

    # Compute ROC curve and AUROC
    fpr, tpr, roc_thresholds = roc_curve(true_labels, probs)
    roc_auc = auc(fpr, tpr)

    # Compute precision-recall curve
    precisions, recalls, pr_thresholds = precision_recall_curve(true_labels, probs)

    # Compute best threshold using Youden's J Statistic (TPR - FPR)
    best_threshold_roc = roc_thresholds[np.argmax(tpr - fpr)]
    preds = (probs > best_threshold_roc).astype(int)
    print(f'Best threshold (ROC): {best_threshold_roc}')

    # Compute F1 score
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_threshold_f1 = pr_thresholds[f1_scores.argmax()]
    print(f'Best threshold (F1): {best_threshold_f1}')

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, preds)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, preds)
    print(f'Confusion Matrix:\n{cm}')

    print(f'AUROC: {roc_auc * 100:.2f}%')

    # Calculate precision, recall, f1
    precision = precision_score(true_labels, preds)
    recall = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')

    # Plot ROC Curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Plot Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recalls, precisions, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.savefig('../output/mpnn_roc_curve.png')
    plt.show()

    # Plot Score Distribution Histogram
    plt.figure(figsize=(6, 5))
    plt.hist(probs, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(best_threshold_roc, color='red', linestyle='dashed', linewidth=2, label=f'Threshold: {best_threshold_roc:.4f}')
    plt.xlabel('Predicted Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    plt.savefig('../output/mpnn_score_dist.png')
    plt.show()


if __name__ == "__main__":
    # filepath = '../data/processed/iuphar_labeled2.csv'
    # df = pd.read_csv(filepath)
    # smiles_list = df['Smiles'].tolist()
    # labels = torch.tensor(df['Interaction'].tolist(), dtype=torch.long)

    # # Calculate weight for positive samples
    # pos_weight = (labels == 0).float().sum() / (labels == 1).float().sum()
    # print(f'Positive sample weight: {pos_weight}')

    # # Create dataset of mol objects
    # mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    # dataset = MolGraphDataset(mols, labels)

    dataset = torch.load('../data/processed/iuphar_labeled2.pt')
    print(f'Loaded dataset with {len(dataset)} samples.')

    # Calculate weight for positive samples
    labels = dataset.labels
    # Calculate weight for positive samples
    pos_weight = (labels == 0).float().sum() / (labels == 1).float().sum()
    print(f'Positive sample weight: {pos_weight}')

    # Training hyperparameters
    d_v = dataset[0].x.shape[1]
    d_e = dataset[0].edge_attr.shape[1]
    d_h = 32
    output_dim = 1
    learning_rate = 1e-3
    num_epochs = 150
    batch_size = 16

    # Split dataset and convert to PyG DataLoaders
    train_loader, val_loader, test_loader = dataset_to_loaders(dataset, batch_size)

    # Initialize model, optimizer, and loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = GraphEncoder(d_v, d_e, d_h, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.01)

    # Training loop
    model, train_losses, val_losses = train(model, 
                                            train_loader, 
                                            optimizer, 
                                            scheduler,
                                            criterion, 
                                            num_epochs, 
                                            val_loader, with_val=True)

    # Plotting the training and validation losses on the same plot
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig('../output/mpnn_training_curve.png')
    plt.show()

    # Evaluating the model
    start_time = time.time()
    evaluate(model, val_loader)
    print(f'Evaluation time: {time.time() - start_time:.2f} seconds')
