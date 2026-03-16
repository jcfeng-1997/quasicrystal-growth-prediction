"""
Dataloader

@jcfeng1997
"""

import os
import re
import sys
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from nns.beta_vae import VAE


class CrystalImageDataset(Dataset):
    def __init__(self, npy_dir):
        self.frames = []
        self.global_min = float("inf")
        self.global_max = float("-inf")

        # First pass: compute global min/max
        for file in sorted(os.listdir(npy_dir)):
            if file.endswith(".npy"):
                arr = np.load(os.path.join(npy_dir, file)).astype(np.float32)
                self.global_min = min(self.global_min, arr.min())
                self.global_max = max(self.global_max, arr.max())

        # Second pass: normalize with global min/max and store
        for file in sorted(os.listdir(npy_dir)):
            if file.endswith(".npy"):
                arr = np.load(os.path.join(npy_dir, file)).astype(np.float32)
                arr = (arr - self.global_min) / (
                    self.global_max - self.global_min + 1e-8
                )
                self.frames.extend([arr[i] for i in range(arr.shape[0])])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]  # shape: (256, 256)
        tensor = torch.tensor(frame).unsqueeze(0)  # (1, 256, 256)
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False
        ).squeeze(0)
        return tensor

    def denormalize(self, tensor):
        """Reverse normalization for visualization."""
        return tensor * (self.global_max - self.global_min) + self.global_min


class CrystalImageTimeSeriesDataset(Dataset):
    def __init__(
        self,
        npy_dir,
        encoder,
        input_len=5,
        pred_len=5,
        device="cpu",
        case_filter=None,
        global_min=None,
        global_max=None,
    ):
        self.npy_dir = npy_dir
        self.encoder = encoder.to(device).eval()
        self.input_len = input_len
        self.pred_len = pred_len
        self.device = device
        self.case_filter = case_filter
        self.global_min = global_min
        self.global_max = global_max
        self.samples = []
        self.sample_info = []
        self._prepare_dataset()

    def _extract_alpha_epsilon(self, filename):
        match = re.search(
            r"alpha_([-+]?\d*\.\d+|\d+)_epsilon_([-+]?\d*\.\d+|\d+)", filename
        )
        if match:
            alpha = float(match.group(1))
            epsilon = float(match.group(2))
            return alpha, epsilon
        else:
            raise ValueError(f"Cannot parse alpha and epsilon from {filename}")

    def _compute_global_min_max(self, file_list):
        global_min = float("inf")
        global_max = float("-inf")
        for file in file_list:
            filepath = os.path.join(self.npy_dir, file)
            arr = np.load(filepath).astype(np.float32)
            global_min = min(global_min, arr.min())
            global_max = max(global_max, arr.max())
        return global_min, global_max

    def _prepare_dataset(self):
        npy_files = sorted(
            [file for file in os.listdir(self.npy_dir) if file.endswith(".npy")]
        )
        if self.case_filter is not None:
            npy_files = [
                file for file in npy_files if any(cf in file for cf in self.case_filter)
            ]
        else:
            npy_files = npy_files

        if self.global_min is not None and self.global_max is not None:
            global_min = self.global_min
            global_max = self.global_max
        else:
            global_min, global_max = self._compute_global_min_max(npy_files)
        print(f"Global min: {global_min}, Global max: {global_max}")

        for file in npy_files:
            filepath = os.path.join(self.npy_dir, file)
            arr = np.load(filepath).astype(np.float32)  # (500, 256, 256)

            arr = (arr - global_min) / (global_max - global_min + 1e-8)

            alpha, epsilon = self._extract_alpha_epsilon(file)

            arr_tensor = (
                torch.tensor(arr).unsqueeze(1).float().to(self.device)
            )  # (500, 1, 256, 256)
            with torch.no_grad():
                _, mean, logvariance = self.encoder(arr_tensor)
                latent = self.encoder.sample(mean, logvariance)

            latent = latent.cpu().numpy()

            alpha_epsilon = np.array([alpha, epsilon], dtype=np.float32)
            alpha_epsilon = np.repeat(
                alpha_epsilon[None, :], latent.shape[0], axis=0
            )  # (500, 2)

            latent_augmented = np.concatenate(
                [latent, alpha_epsilon], axis=-1
            )  # (500, latent_dim+2)

            for i in range(len(latent_augmented) - self.input_len - self.pred_len + 1):
                input_seq = latent_augmented[i : i + self.input_len]
                output_seq = latent[
                    i + self.input_len : i + self.input_len + self.pred_len
                ]

                self.samples.append((input_seq, output_seq))
                self.sample_info.append((alpha, epsilon))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, output_seq = self.samples[idx]
        alpha, epsilon = self.sample_info[idx]

        input_tensor = torch.tensor(input_seq).float()
        output_tensor = torch.tensor(output_seq).float()
        info_tensor = torch.tensor([alpha, epsilon], dtype=torch.float32)

        return input_tensor, output_tensor, info_tensor


if __name__ == "__main__":
    latent_dim = 16
    vae_encoder = VAE(latent_dim=latent_dim)
    try:
        vae_encoder.load_state_dict(
            torch.load("/home/jiachenf/quasicrystal/result/beta-vae/vae_model.pth")
        )
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")

    vae_encoder.eval()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    vae_encoder = vae_encoder.to(device)

    dataset = CrystalImageTimeSeriesDataset(
        npy_dir="/home/jiachenf/quasicrystal/outdata/8/npy_files/",
        encoder=vae_encoder,
        input_len=5,
        pred_len=5,
        device=device,
    )
    total_samples = len(dataset)
    print(f"Total number of samples: {total_samples}")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_x, batch_y in loader:
        print("Batch X shape:", batch_x.shape)
        print("Batch Y shape:", batch_y.shape)
        batch_x = batch_x.to(device)

        batch_y = batch_y.to(device)

        decoded_images = vae_encoder.decoder(batch_y)

        decoded_images = decoded_images.squeeze(1)

        for i in range(min(5, decoded_images.size(0))):
            plt.subplot(1, 5, i + 1)
            plt.imshow(decoded_images[i].cpu().detach().numpy(), cmap="gray")
            plt.axis("off")
        plt.show()

        break
