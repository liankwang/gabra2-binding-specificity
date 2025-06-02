import matplotlib.pyplot as plt
import os

def plot_losses(train_losses, val_losses, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses') 
    plt.legend()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f'{output_path}/loss_curve.png', dpi=300)
    plt.close()