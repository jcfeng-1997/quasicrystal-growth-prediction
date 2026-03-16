import torch

class VAE_config:
    """
    A class of configuration of vae model
    
    """
    
    input_channels = 1
    
    latent_dim  = 32
    
    delta_t     = 1
    
    batch_size  = 256
    
    lr          = 2e-4
    
    lr_end      = 1e-5
    
    epochs      = 900
    
    beta        = 0.0002
    #beta        = 0.0002
    beta_init   = 0.0
    #beta_init   = 0.0

    downsample  = 1 

    beta_warmup = 30
    
    n_test      = 200
    
    encWdecay   = 0
    
    decWdecay   = 0
    
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    data_path = "/home/jiachenf/quasicrystal/outdata/8/npy_files/"
    
    save_path = "/home/jiachenf/quasicrystal/result/beta-vae-32/vae_32.pth"
    

