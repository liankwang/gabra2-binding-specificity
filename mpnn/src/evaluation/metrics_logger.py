import csv
import os
from datetime import datetime

def save_metrics(metrics, path, model_name="unnamed", config_hash=None):
    os.makedirs(path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{config_hash or 'run'}_{timestamp}.json"
    filepath = os.path.join(path, filename) 

    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f'[INFO] Saved evaluation metrics to {filepath}')


def add_to_summary_csv(metrics, path):
    summary_path = os.path.join(path, "metrics_summary.csv")
    file_exists = os.path.exists(summary_path)

    with open(summary_path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp', 'model', 'run_name', 'dataset', 
            'acc', 'auroc', 'f1', 'precision', 'recall',
            'best_threshold_roc',
            'hparams'
        ])
        if not file_exists:
            writer.writeheader()
        
        row = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'model': metrics.get("model", "unknown"),
            #'config': metrics.get("config", "default"),
            'dataset': metrics.get("dataset", "unspecified"),
            'acc': round(metrics.get("acc", 0), 4),
            'auroc': round(metrics.get("auroc", 0), 4),
            'f1': round(metrics.get("f1", 0), 4),
            'precision': round(metrics.get("precision", 0), 4),
            'recall': round(metrics.get("recall", 0), 4),
            #'best_threshold_f1': round(metrics.get("best_threshold_f1", 0), 4),
            'best_threshold_roc': round(metrics.get("best_threshold_roc", 0), 4),
            'run_name': metrics.get("run_name", "unspecified"),
            'hparams': metrics.get("hparams", "unspecified")
        }
        writer.writerow(row)
    print(f'[INFO] Added metrics to summary CSV: {summary_path}')  