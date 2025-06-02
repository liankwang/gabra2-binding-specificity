import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_dataset(path):
    dataset = torch.load(path)
    print(f'Loaded dataset with {len(dataset)} samples.')
    return dataset

def save_model(model, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), f"{path}/model.pt")
    print(f'Saved model to {path}')

def save_predictions(preds, probs, smiles, path):
    df = pd.DataFrame({
        'SMILES': smiles,
        'Probability': probs,
        'Prediction': preds
    })
    df.to_csv(f"{path}/predictions.csv", index=False)
    print(f'Saved predictions to {path}')

def save_embeddings(embeddings, path):
    np.save(f"{path}/embeddings.npy", embeddings)
    print(f'Saved embeddings to {path}/embeddings.npy')



