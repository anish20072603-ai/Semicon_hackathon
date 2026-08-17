import os
import torch
import numpy as np
from PIL import Image

from model import create_model
from dataset import NPYImageDataset


# ============================================================
# PATHS
# ============================================================

DATA_DIR = r"C:\Users\anish\Downloads\Test_NoisyLR\NoisyLR"

MODEL_PATH = r"C:\Users\anish\OneDrive\Desktop\Semicon\models\best_model.pth"

OUTPUT_DIR = r"C:\Users\anish\OneDrive\Desktop\Semicon\outputs"


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

model = create_model().to(device)

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model file not found!")
    print(MODEL_PATH)
    exit()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully!")


# ============================================================
# LOAD TEST DATASET
# ============================================================

dataset = NPYImageDataset(DATA_DIR)

print("Found", len(dataset), "images")


# ============================================================
# PREDICTION
# ============================================================

print("Starting prediction...")


with torch.no_grad():

    for i in range(len(dataset)):

        image, filename = dataset[i]

        # Add batch dimension
        image = image.unsqueeze(0).to(device)

        # Model prediction
        output = model(image)

        # Remove batch/channel dimensions
        output = output.squeeze().cpu().numpy()

        # ====================================================
        # NORMALIZE
        # ====================================================

        output = output - output.min()

        if output.max() > 0:
            output = output / output.max()

        output = (output * 255).clip(0, 255).astype(np.uint8)

        # ====================================================
        # SAVE PNG
        # ====================================================

        output_filename = filename.replace(".npy", ".png")

        output_path = os.path.join(
            OUTPUT_DIR,
            output_filename
        )

        Image.fromarray(output).save(output_path)

        print(
            f"{i + 1}/{len(dataset)} -> {output_filename}"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("========================================")
print("Prediction completed!")
print("========================================")
print("Restored images are in:")
print(OUTPUT_DIR)
print("========================================")