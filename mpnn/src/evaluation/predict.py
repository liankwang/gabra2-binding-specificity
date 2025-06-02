import numpy as np
import torch

def predict(model, test_loader, threshold=0.5):
    """ Makes predictions on the test set (with no true labels) using the trained model."""
    model.eval()
    probs = []

    with torch.no_grad():
        for batch in test_loader:   
            out = model(batch)
            prob = torch.sigmoid(out).numpy().flatten()
            probs.append(prob)
    
    probs = np.concatenate(probs)
    preds = (probs > threshold).astype(int)

    return probs, preds

        