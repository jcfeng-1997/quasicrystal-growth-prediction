import torch
import torch.nn as nn


class TimeSpaceEmbedding(nn.Module):


    def __init__(self, d_input, n_mode, d_expand, d_model):
        super(TimeSpaceEmbedding, self).__init__()

        self.spac_proj = nn.Linear(d_input, d_model)              
        self.time_proj = nn.Conv1d(d_input, d_expand, kernel_size=1)  
        self.time_compress = nn.Linear(d_expand, d_model)         
        self.act = nn.Identity()

        nn.init.xavier_uniform_(self.spac_proj.weight)
        nn.init.xavier_uniform_(self.time_proj.weight)
        nn.init.xavier_uniform_(self.time_compress.weight)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T_in, D_in)
        Returns:
            out: Tensor of shape (B, T_in, d_model)
        """
  
        x_spatial = self.spac_proj(x) 

  
        x_temp = x.permute(0, 2, 1)          # (B, D_in, T)
        x_temp = self.time_proj(x_temp)      # (B, d_expand, T)
        x_temp = x_temp.permute(0, 2, 1)      # (B, T, d_expand)

       
        x_time = self.time_compress(x_temp)   # (B, T, d_model)


        out = self.act(x_spatial + x_time)    # (B, T, d_model)

        return out
