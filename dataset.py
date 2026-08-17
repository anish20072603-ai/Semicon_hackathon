import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class NPYImageDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.files = sorted(
            glob.glob(os.path.join(data_dir, "*.npy"))
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No .npy files found in: {data_dir}"
            )

        print(f"Found {len(self.files)} .npy files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file_path = self.files[index]

        image = np.load(file_path)

        image = image.astype(np.float32)

        # Normalize image
        image_min = image.min()
        image_max = image.max()

        if image_max > image_min:
            image = (image - image_min) / (
                image_max - image_min
            )
        else:
            image = np.zeros_like(image)

        # Convert to PyTorch tensor
        image = torch.from_numpy(image)

        # Add channel dimension if needed
        if image.ndim == 2:
            image = image.unsqueeze(0)

        return image, os.path.basename(file_path)


if __name__ == "__main__":

    DATA_DIR = r"C:\Users\anish\Downloads\Test_NoisyLR\NoisyLR"

    dataset = NPYImageDataset(DATA_DIR)

    print("Dataset size:", len(dataset))

    image, filename = dataset[0]

    print("First file:", filename)
    print("Image shape:", image.shape)