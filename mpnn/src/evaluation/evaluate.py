import numpy as np
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, precision_score, recall_score, 
    f1_score, accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
import torch
from utils.plots import plot_roc_curve, plot_pr_curve, plot_score_distribution
from evaluation.metrics_logger import add_to_summary_csv, save_metrics

def evaluate(model, test_loader, output_path, config):
    """Evaluates the model on the test set.

    Args:
        model: A torch.nn.Module object.
        test_loader: A torch.utils.data.DataLoader object.
        output_path: The path to save the evaluation results.
        config: A dictionary containing the configuration of the model.
    """
    model.eval()
    probs, labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            prob = torch.sigmoid(out).numpy().flatten()
            probs.append(prob)
            labels.append(batch.y.numpy())  
    
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    # Compute ROC curve and AUROC
    fpr, tpr, roc_thresholds = roc_curve(labels, probs)
    precisions, recalls, pr_thresholds = precision_recall_curve(labels, probs)
    roc_auc = auc(fpr, tpr)


    # Use ROC threshold for decision
    best_threshold_roc = roc_thresholds[np.argmax(tpr - fpr)]
    preds = (probs > best_threshold_roc).astype(int)

    # F1 threshold??
    
    # Compute metrics
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    print(f'Accuracy: {acc:.2f}, AUROC: {roc_auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    print(f'Best ROC threshold: {best_threshold_roc:.2f}\n Confusion Matrix:\n{cm}')
    
    # Plots
    plot_roc_curve(fpr, tpr, roc_auc, output_path)
    plot_pr_curve(precisions, recalls, output_path)
    plot_score_distribution(probs, best_threshold_roc, output_path)

    metrics = {
        'acc': acc,
        'auroc': roc_auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'best_threshold_roc': best_threshold_roc,
        #'best_threshold_f1': best_threshold_f1,
        'cm': cm,
        'model': config['model']['type'],
        'run_name': config['run_name'],
        'dataset': config['input']['data_path'],
        'hparams': str(config['model']['hparams'])
    }
    add_to_summary_csv(metrics, "output")
    return metrics
