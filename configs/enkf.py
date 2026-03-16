
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Paths:
    observation_dir: str = "/home/jiachenf/quasicrystal/outdata/8/mini/alpha_0.75_epsilon_0.01_phi_data.npy"         
    ckpt_path: str       = "/home/jiachenf/quasicrystal/result/transformer/model0926.pth/epoch_490.pth"      
    save_dir: str        = "/home/jiachenf/quasicrystal/result/enkf/"             

@dataclass
class ModelCfg:
    latent_dim: int = 16
    d_input: Optional[int] = 18         
    d_output: int = 16                     
    seqLen_in: int = 2                    
    seqLen_out: int = 2                    
    d_proj: int = 64
    d_model: int = 64
    d_ff: int = 256
    num_head: int = 4
    num_layer: int = 4
    dropout: float = 0.1

@dataclass
class EnKF:
    N_en: int = 1000                  
    obs_stride: int = 1               
    sigma_y: float = 0.2        
    sigma_q: float = 0.0001            
    inflation: float = 1.00          
    use_perturbed_obs: bool = True  
    alpha_prior_mean: float = 0
    alpha_prior_var:  float = 0.04
    eps_prior_mean:   float = 0.01
    eps_prior_var:    float = 0.0
    param_sigma_alpha: float = 3e-3
    param_sigma_eps:   float = 0.0
    init_ens_noise: float = 0     
    
@dataclass
class Run:
    device: str = "cuda:5"             
    seed: int = 42
    plot_pdf: str = "alpha_eps_trend_2.pdf"
    verbose: bool = True
    max_steps: Optional[int] = None  

@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    model: ModelCfg = field(default_factory=ModelCfg)
    enkf: EnKF = field(default_factory=EnKF)
    run: Run = field(default_factory=Run)
