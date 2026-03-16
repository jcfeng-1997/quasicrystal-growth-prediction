import torch


class transformer_config:
    """
    A class of configuration of transformer model

    """

    latent_dim = 16

    d_input = latent_dim + 2

    d_output = latent_dim

    seqLen_in = 4

    seqLen_out = 1

    d_proj = 64

    d_model = 64

    d_ff = 256

    num_head = 4

    num_layer = 4

    batch_size = 64

    lambda_latent = 0.05

    alpha_ssim = 0

    use_ssim = False

    beta = 0.0002

    lr = 1e-3

    weight_decay = 1e-4

    epochs = 800

    dropout = 0.2

    save_epochs_steps = 50

    device = "cuda" if torch.cuda.is_available() else "cpu"

    root_path = "/home/jiachenf/quasicrystal/outdata/8/"

    vae_ckpt = "/home/jiachenf/quasicrystal/result/beta-vae/"

    save_path = "/home/jiachenf/quasicrystal/result/transformer/"

    visual_dir = "/home/jiachenf/quasicrystal/result/transformer/"

    name = "TRANSFORMer"

    freeze_decoder = True

    freeze_encoder = True
