import torch
import torch.nn as nn
from training.utils import plot_losses
import wandb

def train(model, dataset, output_path, train_loader, val_loader, config):
    """ Trains the model on the given dataset.
    
    Args:
        model: A torch.nn.Module object.
        dataset: The dataset to train on.
        output_path: The path to save the model and training logs.
        train_loader: The training data loader.
        val_loader: The validation data loader.
        hparams: Dictionary of hyperparameters.
    """
    # Initialize wandb
    run_name = config['run_name']
    hparams = config['model']['hparams']
    wandb.init(
        project="gabra2-binding-specificity",
        name=run_name,
        config={
            'learning_rate': hparams['learning_rate'],
            'epochs': hparams['num_epochs'],
            'batch_size': hparams['batch_size'],
            'config': config
        }
    )
    
    # Unpack hyperparams
    lr = hparams['learning_rate']
    num_epochs = hparams['num_epochs']

    # Define loss and optimizer
    if config['model']['task'] == "classification":
        print("Configuring training for classification task")

        # Compute positive sample weight
        labels = dataset.labels
        pos_weight = (labels == 0).float().sum() / (labels == 1).float().sum()
        print(f'Positive sample weight: {pos_weight:.4f}')

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        print("Configuring training for regression task")
        criterion = nn.MSELoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Run training loop
    model, train_losses, val_losses = training_loop(model, 
                                                    train_loader, 
                                                    optimizer, 
                                                    criterion, 
                                                    num_epochs, 
                                                    val_loader)
    
    # Plot losses
    plot_losses(train_losses, val_losses, output_path)

    # Log the final model and losses to WandB
    wandb.log({
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1]
    })

    # Finish the WandB run
    wandb.finish()

    return model
    

def training_loop(model, train_loader, optimizer, criterion, num_epochs, val_loader=None):
    model.train()
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1, 1))
            loss.backward()

            optimizer.step()
            total_train_loss += loss.item()
        
        ave_train_loss = total_train_loss / len(train_loader)
        train_losses.append(ave_train_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {ave_train_loss:.4f}')

        # Log the train loss to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": ave_train_loss
        })

        if val_loader:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    out = model(batch)
                    loss = criterion(out, batch.y.view(-1, 1))
                    total_val_loss += loss.item()
            ave_val_loss = total_val_loss / len(val_loader)
            val_losses.append(ave_val_loss)
            print(f'Validation Loss: {ave_val_loss:.4f}')   
            # Log the validation loss to WandB
            wandb.log({
                "val_loss": ave_val_loss
            })
            model.train()
        
    return model, train_losses, val_losses
                
        
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