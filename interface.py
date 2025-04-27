import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import the_dataloader
from config import FocalLoss
from train import the_trainer
from test import test
from val import validate
from model import TheModel
from config import learning_rate, batchsize, epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model = TheModel().to(device)

# Load existing weights if available
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

start_weights_path = os.path.join(os.getcwd(), "checkpoints", "start_weights.pth")  # Relative path for initial weights
save_weights_path = os.path.join(os.getcwd(), "final_weights_new.pth")  # Relative path for saving weights

if os.path.exists(start_weights_path) and start_weights_path.strip():
    model.load_state_dict(torch.load(start_weights_path, map_location=device))
    print(f"Loaded model weights from {start_weights_path}")
else:
    print("No checkpoint found. Training from scratch.")

# Loss, Optimizer, Scheduler
criterion = FocalLoss()

# Using L2 regularization via weight_decay in Adam optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-5  # L2 regularization (smaller value for regularization)
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',    # minimize validation loss
    factor=0.5,    # reduce LR by half
    patience=5,    # wait 5 epochs of no improvement
)

# Data loaders
train_loader, _ = the_dataloader(
    os.path.join(os.getcwd(), "archive", "train"),  # Relative path to train data
    batch_size=batchsize,
    shuffle=True
)
val_loader, _ = the_dataloader(
    os.path.join(os.getcwd(), "archive", "val"),  # Relative path to validation data
    batch_size=batchsize,
    shuffle=False
)

# Training settings
best_val_acc = 0
patience = 4  # Early stopping patience
patience_counter = 0

# Main Training Loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    # Train for one epoch
    the_trainer(
        model=model,
        train_loader=train_loader,
        loss_fn=criterion,
        optimizer=optimizer,
        device=device,
        num_classes=7,
        save_path=None  # (you are saving manually later, not inside train1)
    )

    # Validation after each epoch
    val_loss, val_acc = validate(model, val_loader, device, num_classes=7)
    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

    # Step the scheduler based on validation loss
    scheduler.step(val_loss)
    print(f"Current learning rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

    # Save best model if accuracy improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), save_weights_path)
        print(f"New best model saved with {best_val_acc:.2f}% accuracy.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Save model every 3 epochs no matter what
    if (epoch + 1) % 3 == 0:
        torch.save(model.state_dict(), save_weights_path)
        print(f"Model also saved at {save_weights_path} (every 3 epochs checkpoint).")

# Final Save
torch.save(model.state_dict(), save_weights_path)
print(f"\nTraining complete. Final model saved to {save_weights_path}")