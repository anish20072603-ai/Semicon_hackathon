import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class PairedNPYDataset(Dataset):

    def __init__(self, noisy_dir, gt_dir):

        self.noisy_files = sorted(
            glob.glob(os.path.join(noisy_dir, "*.npy"))
        )

        self.gt_files = sorted(
            glob.glob(os.path.join(gt_dir, "*.npy"))
        )

        if len(self.noisy_files) == 0:
            raise RuntimeError("No NoisyLR .npy files found")

        if len(self.gt_files) == 0:
            raise RuntimeError("No GT .npy files found")

        if len(self.noisy_files) != len(self.gt_files):
            raise RuntimeError(
                f"File count mismatch: "
                f"NoisyLR={len(self.noisy_files)}, "
                f"GT={len(self.gt_files)}"
            )

        print(f"Found {len(self.noisy_files)} paired images")

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, index):

        noisy = np.load(self.noisy_files[index]).astype(np.float32)
        gt = np.load(self.gt_files[index]).astype(np.float32)

        # Normalize NoisyLR
        noisy_min = noisy.min()
        noisy_max = noisy.max()

        if noisy_max > noisy_min:
            noisy = (noisy - noisy_min) / (noisy_max - noisy_min)
        else:
            noisy = np.zeros_like(noisy)

        # Normalize GT
        gt_min = gt.min()
        gt_max = gt.max()

        if gt_max > gt_min:
            gt = (gt - gt_min) / (gt_max - gt_min)
        else:
            gt = np.zeros_like(gt)

        # Convert to tensors
        noisy = torch.from_numpy(noisy)
        gt = torch.from_numpy(gt)

        # Add channel dimension
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(0)

        if gt.ndim == 2:
            gt = gt.unsqueeze(0)

        return noisy, gt


if __name__ == "__main__":

    NOISY_DIR = r"C:\Users\anish\Downloads\train_data\train\NoisyLR"
    GT_DIR = r"C:\Users\anish\Downloads\train_data\train\GT"

    dataset = PairedNPYDataset(NOISY_DIR, GT_DIR)

    print("Dataset size:", len(dataset))

    noisy, gt = dataset[0]

    print("Noisy shape:", noisy.shape)
    print("GT shape:", gt.shape)