import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from train_dataset import PairedNPYDataset
from model import create_model


# ============================================================
# PATHS
# ============================================================

NOISY_DIR = r"C:\Users\anish\Downloads\train_data\train\NoisyLR"
GT_DIR = r"C:\Users\anish\Downloads\train_data\train\GT"

MODEL_PATH = r"C:\Users\anish\OneDrive\Desktop\Semicon\models\best_model.pth"

# Create models folder if it doesn't exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# DATASET
# ============================================================

dataset = PairedNPYDataset(
    NOISY_DIR,
    GT_DIR
)

print("Total images:", len(dataset))


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = create_model().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

num_epochs = 10

best_val_loss = float("inf")


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(num_epochs):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for noisy, gt in train_loader:

        noisy = noisy.to(device)
        gt = gt.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(noisy)

        # Calculate loss
        loss = criterion(output, gt)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for noisy, gt in val_loader:

            noisy = noisy.to(device)
            gt = gt.to(device)

            output = model(noisy)

            loss = criterion(output, gt)

            val_loss += loss.item()

    val_loss /= len(val_loader)


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("Best model saved!")


# ============================================================
# COMPLETE
# ============================================================

print("Training completed.")
print("Best model saved at:")
print(MODEL_PATH)