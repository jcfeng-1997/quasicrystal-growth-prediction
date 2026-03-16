import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from lib.data import CrystalImageTimeSeriesDataset
from configs.trans_config import transformer_config as cfg
from nns.transformer import Seq2SeqTransformer
from nns.beta_vae import VAE
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def compute_global_min_max(npy_dir, file_list):
    global_min = float("inf")
    global_max = float("-inf")
    for file in file_list:
        filepath = os.path.join(npy_dir, file)
        arr = np.load(filepath).astype(np.float32)
        global_min = min(global_min, arr.min())
        global_max = max(global_max, arr.max())
    return global_min, global_max


def print_test_param_distribution(test_loader):
    param_counter = defaultdict(int)

    for _, _, batch_info in test_loader:
        for info in batch_info:
            alpha, epsilon = info.tolist()
            alpha = round(alpha, 5)
            epsilon = round(epsilon, 5)
            param_counter[(alpha, epsilon)] += 1

    sorted_params = sorted(param_counter.items(), key=lambda x: (-x[1], x[0]))

    print(f"\nTotal unique (alpha, epsilon) pairs in test set: {len(sorted_params)}")
    print("Top (alpha, epsilon) pairs by frequency in test set:\n")
    print(f"{'alpha':>10} {'epsilon':>10} {'Count':>10}")
    print("-" * 32)
    for (alpha, epsilon), count in sorted_params:
        print(f"{alpha:10.5f} {epsilon:10.5f} {count:10d}")


def train(model, train_loader, optimizer, loss_fn, lr_scheduler, epoch, cfg):
    model.train()
    running_loss = 0.0
    tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", ncols=80)

    for batch_idx, (batch_x, batch_y, _) in enumerate(tqdm_loader):
        batch_x = batch_x.to(cfg.device)  # (batch_size, input_len, 18)
        batch_y = batch_y.to(cfg.device)  # (batch_size, pred_len, 16)

        optimizer.zero_grad()

        out = model(batch_x)
        loss = loss_fn(out, batch_y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        tqdm_loader.set_postfix(loss=running_loss / (batch_idx + 1))

    lr_scheduler.step()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch}/{cfg.epochs}] Training Loss: {avg_loss:.6f}")
    return avg_loss


def test(model, test_loader, loss_fn, cfg, vae_encoder=None, epoch=None, save_dir=None):
    from collections import defaultdict
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    import os, numpy as np

    model.eval()
    running_loss = 0.0

    error_cmap = LinearSegmentedColormap.from_list(
        "custom_red", ["white", (1.0, 0.6, 0.6)], N=256
    )

    case_dict = dict()

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_info) in enumerate(test_loader):
            batch_x = batch_x.to(cfg.device)
            batch_y = batch_y.to(cfg.device)
            out = model(batch_x)

            loss = loss_fn(out, batch_y)
            running_loss += loss.item()

            for j in range(batch_y.size(0)):
                alpha, epsilon = batch_info[j].tolist()
                key = (round(alpha, 4), round(epsilon, 4))
                if key not in case_dict:
                    decoded_true = vae_encoder.decoder(batch_y[j]).cpu().numpy()
                    decoded_pred = vae_encoder.decoder(out[j]).cpu().numpy()
                    case_dict[key] = (decoded_true, decoded_pred)

            if len(case_dict) >= 10:
                break

    all_vals = np.concatenate(
        [np.concatenate([true, pred], axis=0) for true, pred in case_dict.values()],
        axis=0,
    )
    vmin, vmax = all_vals.min(), all_vals.max()
    print(f"[Global Visualization Range] vmin={vmin:.4f}, vmax={vmax:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    for idx, ((alpha, epsilon), (decoded_true, decoded_pred)) in enumerate(
        case_dict.items()
    ):
        diff = np.abs(decoded_true - decoded_pred)
        T = min(cfg.seqLen_out, 5)
        fig, axs = plt.subplots(3, T, figsize=(3 * T, 9))

        for t in range(T):
            axs[0, t].imshow(decoded_true[t, 0], cmap="viridis", vmin=vmin, vmax=vmax)
            axs[0, t].set_title(f"True t+{t+1}")
            axs[0, t].axis("off")

            axs[1, t].imshow(decoded_pred[t, 0], cmap="viridis", vmin=vmin, vmax=vmax)
            axs[1, t].set_title(f"Pred t+{t+1}")
            axs[1, t].axis("off")

            im = axs[2, t].imshow(diff[t, 0], cmap=error_cmap, vmin=0.0, vmax=0.1)
            axs[2, t].set_title(f"Error t+{t+1}")
            axs[2, t].axis("off")

            if t == T - 1:
                cbar = fig.colorbar(im, ax=axs[2, t], fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)

        axs[0, 0].set_ylabel(f"��={alpha:.2f}, ��={epsilon:.2f}", fontsize=11)
        plt.tight_layout()
        save_path = os.path.join(
            save_dir, f"compare_epoch{epoch}_case{idx}_a{alpha:.2f}_e{epsilon:.2f}.png"
        )
        plt.savefig(save_path, dpi=300)
        plt.close()

    avg_loss = running_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.6f}")
    return avg_loss


def plot_comparison(original, predicted, epoch, batch_idx, save_dir):

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original[0].cpu().detach().numpy().transpose(1, 2, 0))
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(predicted[0].cpu().detach().numpy().transpose(1, 2, 0))
    axs[1].set_title("Predicted Image")
    axs[1].axis("off")

    plt.tight_layout()

    save_path = os.path.join(
        save_dir, f"comparison_epoch_{epoch}_batch_{batch_idx}.png"
    )
    plt.savefig(save_path)
    plt.close()


def main(cfg):
    vae_encoder = VAE(latent_dim=cfg.latent_dim)
    try:
        vae_encoder.load_state_dict(
            torch.load("/home/jiachenf/quasicrystal/result/beta-vae/vae_model.pth")
        )
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
    vae_encoder.eval()
    vae_encoder = vae_encoder.to(cfg.device)

    print(
        f"Decoder parameters require grad: {[p.requires_grad for p in vae_encoder.decoder.parameters()]}"
    )

    all_files = sorted([f for f in os.listdir(cfg.root_path) if f.endswith(".npy")])
    np.random.seed(42)
    np.random.shuffle(all_files)
    train_files = all_files[: int(0.8 * len(all_files))]
    test_files = all_files[int(0.8 * len(all_files)) :]

    global_min, global_max = compute_global_min_max(cfg.root_path, all_files)

    train_dataset = CrystalImageTimeSeriesDataset(
        npy_dir=cfg.root_path,
        encoder=vae_encoder,
        input_len=cfg.seqLen_in,
        pred_len=cfg.seqLen_out,
        device=cfg.device,
        case_filter=train_files,
        global_min=global_min,
        global_max=global_max,
    )

    test_dataset = CrystalImageTimeSeriesDataset(
        npy_dir=cfg.root_path,
        encoder=vae_encoder,
        input_len=cfg.seqLen_in,
        pred_len=cfg.seqLen_out,
        device=cfg.device,
        case_filter=test_files,
        global_min=global_min,
        global_max=global_max,
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    print(
        f"Train set: {len(train_dataset)} samples, Test set: {len(test_dataset)} samples"
    )

    print_test_param_distribution(test_loader)

    model = Seq2SeqTransformer(
        d_input=cfg.d_input,
        d_output=cfg.d_output,
        seqLen_in=cfg.seqLen_in,
        seqLen_out=cfg.seqLen_out,
        d_proj=cfg.d_proj,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        num_head=cfg.num_head,
        num_layer=cfg.num_layer,
        dropout=cfg.dropout,  # Dropout
    ).to(cfg.device)

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    loss_fn = nn.HuberLoss(delta=1.0, reduction="mean")

    print(f"Using device: {cfg.device}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train(
            model, train_loader, optimizer, loss_fn, lr_scheduler, epoch, cfg
        )

        if epoch % cfg.save_epochs_steps == 0:
            save_path = os.path.join(cfg.save_path, f"{cfg.name}_epoch_{epoch}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                },
                save_path,
            )
            print(f"Checkpoint saved: {save_path}")

        test_loss = test(
            model,
            test_loader,
            loss_fn,
            cfg,
            vae_encoder=vae_encoder,
            epoch=epoch,
            save_dir=cfg.visual_dir,
        )


if __name__ == "__main__":
    main(cfg)
