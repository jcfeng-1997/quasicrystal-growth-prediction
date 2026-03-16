"""
The transfomer encoders using new embeding layer

"""

import torch
import torch.nn as nn
from nns.embedding import TimeSpaceEmbedding
from nns.attns import MultiHeadAttention
from nns.layers import EncoderLayer

class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_input, d_output, seqLen_in, seqLen_out,
                 d_proj, d_model, d_ff, num_head, num_layer,
                 act_proj="relu", dropout=0.1):
        super().__init__()

        self.seqLen_in = seqLen_in
        self.seqLen_out = seqLen_out

        self.embed = TimeSpaceEmbedding(d_input, seqLen_in, d_proj, d_model)
        self.query_embed = nn.Parameter(torch.randn(seqLen_out, d_model))

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, num_heads=num_head, d_ff=d_ff, dropout=dropout, act_proj=act_proj)
            for _ in range(num_layer)
        ])

        self.proj = nn.Conv1d(d_model, d_output, kernel_size=1)
#         self.proj = nn.Sequential(
#            nn.Conv1d(d_model, d_output, kernel_size=1),
#            nn.ReLU(),
#        )

    def forward(self, src):
        B = src.size(0)
        memory = self.embed(src)
        query = self.query_embed.unsqueeze(0).expand(B, -1, -1)

        for layer in self.encoder_layers:
            query = layer(query, memory)

        query = query.permute(0, 2, 1)
        out = self.proj(query)
        out = out.permute(0, 2, 1)
        return out

