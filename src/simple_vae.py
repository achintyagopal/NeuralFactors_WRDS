import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D

from .base_vae import BaseVAE
from .layer_utils import ResidBlock, RNNModel


class SimpleVAE(BaseVAE):

    def __init__(
        self,
        num_securities,
        num_cont_cols=0,
        hidden_size=64,
        embed_size=4,
        num_blocks=2,
        out_dist="Normal",
        dropout=0.0,
        warmup_period=100,
        k=20,
        lr=1e-4,
        wd=0,
    ):
        # Note that this model assumes a fixed number of securities
        super().__init__(
            embed_size=embed_size,
            warmup_period=warmup_period,
            k=k,
            lr=lr,
            wd=wd,
        )

        self.hidden_size = hidden_size

        self.conditioning_model = nn.Sequential(
            nn.Linear((1 + num_cont_cols) * num_securities, hidden_size),
            RNNModel(
                hidden_size,
                hidden_size,
                num_blocks=num_blocks,
                dropout=dropout,
            ),
        )

        self.encoder = nn.Sequential(
            nn.Linear(num_securities + hidden_size, hidden_size),
            ResidBlock(hidden_size, hidden_size, dropout=dropout),
            ResidBlock(hidden_size, hidden_size, dropout=dropout),
            nn.Linear(hidden_size, 2 * embed_size),
        )

        out_dist = out_dist.lower()
        if out_dist == "normal":
            out_size = num_securities * 2
        elif out_dist == "studentt":
            out_size = num_securities * 3
        else:
            raise ValueError(f"Unrecognized {out_dist=}")

        self.out_dist = out_dist

        self.decoder = nn.Sequential(
            nn.Linear(embed_size + hidden_size, hidden_size),
            ResidBlock(hidden_size, hidden_size, dropout=dropout),
            ResidBlock(hidden_size, hidden_size, dropout=dropout),
            nn.Linear(hidden_size, out_size),
        )

    def _conditioning(self, lookback_returns, lookback_cont_features):
        history = torch.cat([lookback_returns.unsqueeze(-1), lookback_cont_features], -1)
        bsz, num_secs, lookback, num_feats = history.size()
        history = history.permute(0, 2, 1, 3).reshape(bsz, lookback, num_secs * num_feats)
        return self.conditioning_model(history)[..., -1, :]

    def _encoder(self, future_returns, conditioning):
        mu_z, sigma_z = self.encoder(torch.cat([future_returns, conditioning], -1)).chunk(2, -1)
        sigma_z = F.softplus(sigma_z) + 1e-4
        return D.Independent(D.Normal(mu_z, sigma_z), 1)

    def _decoder(self, z, conditioning):
        out = self.decoder(torch.cat([z, conditioning.unsqueeze(0).expand(z.size(0), -1, -1)], -1))
        if self.out_dist == "normal":
            mu_x, sigma_x = out.chunk(2, -1)
            sigma_x = F.softplus(sigma_x) + 1e-4
            out_dist = D.Independent(D.Normal(mu_x, sigma_x), 1)
        elif self.out_dist == "studentt":
            mu_x, sigma_x, df_x = out.chunk(3, -1)
            sigma_x = F.softplus(sigma_x) + 1e-4
            df_x = F.softplus(df_x) + 4
            out_dist = D.Independent(D.StudentT(df_x, mu_x, sigma_x), 1)
        else:
            raise ValueError(f"Unrecognized {self.out_dist=}")
        return out_dist
