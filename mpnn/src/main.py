import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import yaml
import numpy as np

from data.utils import dataset_to_loaders     
from models.models import DMPNN, GraphEncoder
from evaluation.evaluate import evaluate
from evaluation.predict import predict
from utils.io import save_predictions, save_embeddings
from training.train import train
from data.create_graph_dataset import MolGraphDataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the config.yaml file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    run_name = config['run_name']
    output_path = config['output']['output_path'].format(run_name=run_name)
    checkpoint_path = config['input']['checkpoint_path']
    checkpoint_save_path = config['output']['checkpoint_save_path'].format(run_name=run_name)
    data_path = config['input']['data_path']

    # Load Dataset
    dataset = torch.load(data_path)
    print(f'Loaded dataset with {len(dataset)} samples.')

    # Extract hyperparams
    hparams = config['model']['hparams']
    d_v = dataset[0].x.shape[1]
    d_e = dataset[0].edge_attr.shape[1]

    if config['model']['type'] == "DMPNN":
        model = DMPNN(d_v, d_e, hparams['d_h'], hparams['T'], hparams['output_dim'], task=config['model']['task'])
    elif config['model']['type'] == "GraphEncoder":
        model = GraphEncoder(d_v, d_e, hparams['d_h'], hparams['output_dim'], task=config['model']['task'])
    else:
        raise ValueError(f"Model type {config['model']['type']} not supported")

    # Either load model or train new model
    if checkpoint_path:
        # Load model from checkpoint
        model.load_state_dict(torch.load(checkpoint_path))
        print(f'Loaded model from {checkpoint_path}')

        # Make predictions on test set
        print("Making predictions on test set...")
        test_loader = DataLoader(dataset, batch_size=hparams['batch_size'])
        probs, preds = predict(model, test_loader)
        
        # Save predictions
        save_predictions(preds, probs, dataset.smiles, f"{output_path}/predictions")

        # Save embeddings
        # print("Extracting embeddings...")
        # embeddings = extract_embeddings(model, test_loader)
        # np.save(f'{config["training"]["output_path"]}/embeddings.npy', embeddings)
        # print(f'Saved embeddings to {config["training"]["output_path"]}/embeddings.npy')

    else: # Trains a new model and evaluates model based on true labels
        # Split dataset and convert to PyG DataLoaders
        train_loader, val_loader, test_loader = dataset_to_loaders(dataset, batch_size=hparams['batch_size'])

        # Train and evaluate model
        model = train(model, dataset, output_path, train_loader, val_loader, config)
        evaluate(model, test_loader, output_path, config)

        # Save model checkpoint
        if checkpoint_save_path:
            torch.save(model.state_dict(), checkpoint_save_path)

if __name__ == "__main__":
    main()  