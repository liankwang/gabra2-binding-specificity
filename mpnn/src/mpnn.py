import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
from torch_geometric.utils import add_self_loops
#from torch.optim.lr_scheduler import StepLR

from create_graph_dataset import MolGraphDataset

"""
This script trains, evaluates, and makes predictions using a Graph Neural Network model on a molecular dataset.
The dataset is created using the MolGraphDataset class from create_graph_dataset.py, which converts RDKit molecules
to PyTorch Geometric Data objects.

Arguments:
--input_path: Path to the PyTorch dataset file created by create_graph_dataset.py (should end in .pt)
--output_path: Folder path to save the model, loss curve, and predictions (if applicable)
--save_path (optional): Path to save the trained model checkpoint
--checkpoint_path (optional): Path to load a trained model checkpoint for predictions (should end in .pt)
"""

# random_seed = 42
# torch.manual_seed(random_seed)
# np.random.seed(random_seed)


class EdgeGCN(MessagePassing):
    """ Defines a layer of a GCN that makes use of both node and edge features in constructing the message."""
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


def train(dataset, output_path, train_loader, val_loader, hparams):
    """ Trains a GraphEncoder model on the given dataset. Can substitute with SimpleGraphEncoder in model definition."""
    # Training hyperparams
    learning_rate = hparams['learning_rate']
    num_epochs = hparams['num_epochs']
    d_v = hparams['d_v']
    d_e = hparams['d_e']
    d_h = hparams['d_h']
    output_dim = hparams['output_dim']

    # Compute positive samples weight
    labels = dataset.labels
    pos_weight = (labels == 0).float().sum() / (labels == 1).float().sum()
    print(f'Positive sample weight: {pos_weight}')

    # Initialize model, optimizer, and loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = GraphEncoder(d_v, d_e, d_h, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = StepLR(optimizer, step_size=10, gamma=0.01)

    # Training loop
    model, train_losses, val_losses = training_loop(model, 
                                            train_loader, 
                                            optimizer,
                                            criterion, 
                                            num_epochs, 
                                            val_loader)

    # Plotting the training and validation losses on the same plot
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(f'{output_path}/loss_curve.png',dpi=300)
    plt.show()

    return model

def training_loop(model, train_loader, optimizer, criterion, num_epochs, val_loader=None):
    """ Trains the model on the training data for the specified number of epochs."""
    model.train()
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        total_train_loss = 0
        for batch in train_loader:
            # Train
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1,1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        ave_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {ave_train_loss:.4f}')
        train_losses.append(ave_train_loss)
    
        if val_loader is not None:
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

def extract_embeddings(model, data_loader):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in data_loader:
            x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
            x = model.gcn(x, edge_index)
            x = F.relu(x)
            x = model.conv1(x, edge_index, edge_attr)
            x = global_mean_pool(x, batch.batch)
            embeddings.append(x.numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings

def evaluate(model, test_loader, output_path):
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
    print(f'Best threshold (ROC): {round(best_threshold_roc,2)}')

    # Compute F1 score
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_threshold_f1 = pr_thresholds[f1_scores.argmax()]
    print(f'Best threshold (F1): {round(best_threshold_f1,2)}')

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
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()

    # Plot Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recalls, precisions, label='Precision-Recall curve')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()

    plt.savefig(f'{output_path}/roc_curve_test.png', dpi=300)
    plt.show()

    # Plot Score Distribution Histogram
    plt.figure(figsize=(6, 5))
    plt.hist(probs, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(best_threshold_roc, color='red', linestyle='dashed', linewidth=2, label=f'Threshold: {best_threshold_roc:.4f}')
    plt.xlabel('Predicted Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    plt.savefig(f'{output_path}/score_dist.png', dpi=300)
    plt.show()

def predict(model, test_loader, threshold=0.4):
    """ Makes predictions on the test set (with no true labels) using the trained model."""
    model.eval()
    preds = []
    probs = []

    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            prob = torch.sigmoid(out).numpy().flatten()
            probs.append(prob)
    probs = np.concatenate(probs)
    preds = (probs > threshold).astype(int)

    return probs, preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    # Load Pytorch dataset
    dataset = torch.load(args.input_path)
    print(f'Loaded dataset with {len(dataset)} samples.')

    # Define hyperparameters
    hparams = {
        'learning_rate': 1e-3,
        'num_epochs': 150,
        'd_h': 32,
        'output_dim': 1,
        'd_v': dataset[0].x.shape[1],
        'd_e': dataset[0].edge_attr.shape[1],
        'batch_size': 8
    }

    # Either load model or train new model
    if args.checkpoint_path: # Loads model from checkpoint and performs predictions
        d_v, d_e, d_h, output_dim = hparams['d_v'], hparams['d_e'], hparams['d_h'], hparams['output_dim']
        model = GraphEncoder(d_v, d_e, d_h, output_dim)
        model.load_state_dict(torch.load(args.checkpoint_path))
        print(f'Loaded model from {args.checkpoint_path}')

        # Make predictions on test set
        print("Making predictions on test set...")
        start_time = time.time()
        test_loader = DataLoader(dataset, batch_size=hparams['batch_size'])
        probs, preds = predict(model, test_loader)
        end_time = time.time()
        print(f'Predictions made in {round(end_time - start_time, 2)} seconds for {len(dataset)} samples.')

        # Save predictions
        df = pd.DataFrame({'SMILES':dataset.smiles, 'Probability': probs, 'Prediction': preds})
        df.to_csv(f'{args.output_path}/predictions.csv', index=False)
        print(f'Saved predictions to {args.output_path}/predictions.csv')

        # Save embeddings
        print("Extracting embeddings...")
        embeddings = extract_embeddings(model, test_loader)
        np.save(f'{args.output_path}/embeddings.npy', embeddings)
        print(f'Saved embeddings to {args.output_path}/embeddings.npy')
    
    else: # Trains a new model and evaluates model based on true labels
        # Split dataset and convert to PyG DataLoaders
        train_loader, val_loader, test_loader = dataset_to_loaders(dataset, batch_size=hparams['batch_size'])

        # Train and evaluate model
        model = train(dataset, args.output_path, train_loader, val_loader, hparams)
        evaluate(model, test_loader, args.output_path)

        # Save model checkpoint
        if args.save_path:
            torch.save(model.state_dict(), f'{args.save_path}/model.pt')


if __name__ == "__main__":
    main()
