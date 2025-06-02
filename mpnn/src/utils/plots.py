import matplotlib.pyplot as plt
import os

def plot_roc_curve(fpr, tpr, roc_auc, output_path):
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUROC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f'{output_path}/roc_curve_test.png', dpi=300)
    plt.close()

def plot_pr_curve(precisions, recalls, output_path):
    plt.figure()
    plt.plot(recalls, precisions, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(f'{output_path}/pr_curve_test.png', dpi=300)
    plt.close()

def plot_score_distribution(probs, threshold, output_path):
    plt.figure()
    plt.hist(probs, bins=20, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
    plt.xlabel('Predicted Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    plt.savefig(f'{output_path}/score_dist_test.png', dpi=300)
    plt.close()

