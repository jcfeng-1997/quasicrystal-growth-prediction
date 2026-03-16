
import os
import re
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn

# -------- filename parser: alpha/epsilon from name --------
_ALPHA_EPS_RE = re.compile(r'alpha_([-+]?\d*\.?\d+)_epsilon_([-+]?\d*\.?\d+)')

def parse_alpha_eps(name: str) -> Optional[Tuple[float, float]]:
    m = _ALPHA_EPS_RE.search(name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None

# -------- load observation sequence (pixel space) --------
def load_observation_sequence(
    npy_dir: str,
    file_regex: str = r'case_(\d+)\.npy',
    normalize: bool = True,
    global_min: Optional[float] = None,
    global_max: Optional[float] = None,
):

    # single .npy file
    if os.path.isfile(npy_dir) and npy_dir.endswith(".npy"):
        arr = np.load(npy_dir).astype(np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected (T,H,W), got {arr.shape} in {npy_dir}")
        if global_min is None or global_max is None:
            gmin, gmax = float(arr.min()), float(arr.max())
        else:
            gmin, gmax = float(global_min), float(global_max)
        if normalize:
            arr = (arr - gmin) / (gmax - gmin + 1e-8)
        return torch.tensor(arr, dtype=torch.float32), [os.path.basename(npy_dir)], (gmin, gmax)

    # directory
    if not os.path.isdir(npy_dir):
        raise FileNotFoundError(f"Not found: {npy_dir}")
    pat = re.compile(file_regex)
    files = [f for f in os.listdir(npy_dir) if f.endswith(".npy") and pat.search(f)]
    if not files:
        raise FileNotFoundError(f"No .npy matched {file_regex} in {npy_dir}")
    m0 = pat.search(files[0])
    if m0 and m0.groups():
        def _key(f):
            m = pat.search(f); g1 = m.group(1) if m else "0"
            return int(g1) if g1.isdigit() else 0
        files_sorted = sorted(files, key=_key)
    else:
        files_sorted = sorted(files)
    if global_min is None or global_max is None:
        gmin, gmax = float("inf"), float("-inf")
        for f in files_sorted:
            a = np.load(os.path.join(npy_dir, f)).astype(np.float32)
            gmin = min(gmin, float(a.min())); gmax = max(gmax, float(a.max()))
    else:
        gmin, gmax = float(global_min), float(global_max)
    chunks = []
    for f in files_sorted:
        a = np.load(os.path.join(npy_dir, f)).astype(np.float32)
        if a.ndim != 3:
            raise ValueError(f"Expected (T,H,W) in {f}, got {a.shape}")
        if normalize:
            a = (a - gmin) / (gmax - gmin + 1e-8)
        chunks.append(a)
    obs = torch.tensor(np.concatenate(chunks, axis=0), dtype=torch.float32)
    return obs, files_sorted, (gmin, gmax)

# -------- initial ensemble from first frame (optional) --------
def build_initial_ensemble_from_first_frame(
    observation: torch.Tensor,  # (T,H,W)
    N_en: int,
    add_noise_std: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Duplicate t=0 frame N_en times, optionally add small Gaussian noise.
    Returns (N_en, H, W) as float32 numpy array.
    """
    rng = np.random.default_rng(seed)
    first = observation[0].cpu().numpy().astype(np.float32)     # (H,W)
    ens = np.repeat(first[None, ...], N_en, axis=0)             # (N_en,H,W)
    if add_noise_std > 0:
        noise = rng.normal(0.0, add_noise_std, size=ens.shape).astype(np.float32)
        # If inputs are normalized to [0,1], keep in range
        ens = np.clip(ens + noise, 0.0, 1.0)
    return ens

# -------- encode observation frames into latent space --------
@torch.no_grad()
def encode_observation_to_latents(
    encoder: nn.Module,
    observation: torch.Tensor,  # (T,H,W)
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Encode pixel observations to latent vectors z_obs of shape (T, d).

    This is robust to different encoder outputs:
    - tensor -> treat as z
    - tuple/list (e.g., (recon, mu, logvar) or (z, mu, logvar)) -> prefer 'mu' if found,
      otherwise pick the first 2D tensor (B, d)
    - dict -> try keys in ['mu','z','latent'] then fallback to any 2D tensor
    """
    def _extract_latent(out):
        # out may be Tensor / tuple / list / dict
        if torch.is_tensor(out):
            # expect (B, d)
            if out.ndim == 2:
                return out
            # some encoders may return (B, C, H, W) feature maps; reject
            raise ValueError(f"Encoder returned tensor with shape {tuple(out.shape)}, expected (B,d).")

        if isinstance(out, (tuple, list)):
            # Try to identify 'mu' or a (B,d) latent
            # common patterns: (recon, mu, logvar) or (z, mu, logvar)
            # Heuristic: prefer the first 2D tensor; if two 2D tensors exist and one looks like 'mu',
            # pick the second element (many impls put mu at index 1)
            twos = [t for t in out if torch.is_tensor(t) and t.ndim == 2]
            if len(twos) == 0:
                raise ValueError(f"Encoder returned tuple/list but no 2D tensor found: {[type(x) for x in out]}")
            if len(twos) >= 2:
                # very often out[1] is mu
                return twos[1]
            return twos[0]

        if isinstance(out, dict):
            for k in ['mu', 'z', 'latent', 'h']:
                if k in out and torch.is_tensor(out[k]) and out[k].ndim == 2:
                    return out[k]
            # fallback: any 2D tensor in dict values
            for v in out.values():
                if torch.is_tensor(v) and v.ndim == 2:
                    return v
            raise ValueError("Encoder returned dict but no (B,d) tensor found.")

        raise TypeError(f"Unsupported encoder output type: {type(out)}")

    encoder.eval()
    T, H, W = observation.shape
    outs = []
    for s in range(0, T, batch_size):
        x = observation[s: s+batch_size].unsqueeze(1).to(device)  # (b,1,H,W)
        out = encoder(x)
        z  = _extract_latent(out)                                 # (b,d)
        outs.append(z.detach().cpu())
    return torch.cat(outs, dim=0)  # (T,d)


def load_models_from_ckpt(
    vae_cls, trans_cls,
    ckpt_path: str,
    latent_dim: int,
    trans_kwargs: dict,
    device: torch.device,
):
    """
    Returns: encoder, transformer, ae_embed(nn.Module), ae_mean(1,2), ae_std(1,2)
    - If 'ae_embed' is missing in ckpt, we return nn.Identity() as a placeholder.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # VAE
    vae = vae_cls(latent_dim=latent_dim).to(device)
    if 'vae' in ckpt:
        vae.load_state_dict(ckpt['vae'], strict=False)
    else:
        # fallback: in case the whole VAE state_dict was saved
        vae.load_state_dict(ckpt, strict=False)
    vae.eval()
    encoder = vae.encoder if hasattr(vae, "encoder") else vae

    # Transformer
    transformer = trans_cls(**trans_kwargs).to(device)
    if 'transformer' in ckpt:
        transformer.load_state_dict(ckpt['transformer'], strict=False)
    else:
        raise KeyError("Checkpoint does not contain 'transformer' weights.")
    transformer.eval()

    # ae_embed (optional in old checkpoints)
    if 'ae_embed' in ckpt:
        ae_embed = nn.Sequential(nn.Linear(2, latent_dim, bias=True), nn.Tanh()).to(device)
        ae_embed.load_state_dict(ckpt['ae_embed'], strict=False)
        ae_embed.eval()
    else:
        # not used when d_input == latent_dim+2; keep an identity to satisfy API
        ae_embed = nn.Identity()

    # (alpha, epsilon) normalization stats (optional)
    ae_mean = torch.tensor(ckpt.get('ae_mean', [0.0, 0.0]), dtype=torch.float32, device=device).view(1, 2)
    ae_std  = torch.tensor(ckpt.get('ae_std',  [1.0, 1.0]), dtype=torch.float32, device=device).view(1, 2)

    return encoder, transformer, ae_embed, ae_mean, ae_std


class LatentStepModel:
    def __init__(self, transformer, ae_embed, ae_mean, ae_std, seqLen_in, latent_dim, device):
        self.net = transformer
        self.ae_embed = ae_embed         
        self.ae_mean = ae_mean.to(device) # (1,2)
        self.ae_std  = ae_std.to(device)  # (1,2)
        self.seqLen_in = seqLen_in
        self.d = latent_dim               # latent_dim=16
        self.device = device

    @torch.no_grad()
    def step(self, z_hist: torch.Tensor, alpha: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:

        B = z_hist.size(0)


        ae = torch.stack([alpha, epsilon], dim=1)                 # (B,2)
        ae_n = (ae - self.ae_mean) / (self.ae_std + 1e-8)         # (B,2)
        ae_rep = ae_n.unsqueeze(1).repeat(1, self.seqLen_in, 1)   # (B,Tin,2)

        z_in_aug = torch.cat([z_hist, ae_rep], dim=-1)

        y_pred = self.net(z_in_aug)               # Tout = 2

        z_next = y_pred[:, 0, :]                  # (B, d)
        return z_next


