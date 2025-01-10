import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D

from .base_vae import BaseVAE
from .layer_utils import ResidBlock, RNNModel


class NeuralFactors(BaseVAE):

    def __init__(
        self,
        num_securities,
        num_cont_cols=0,
        hidden_size=64,
        embed_size=4,
        num_blocks=2,
        out_dist="Normal",
        dropout=0.0,
        include_alpha=False,
        warmup_period=100,
        num_attn_layers=0,
        nhead=4,
        k=20,
        lr=1e-4,
        wd=0,
    ):
        # Note that this model does not assume a fixed number of securities; however, it does assume a fixed
        # number of securities in a single batch.
        super().__init__(
            embed_size=embed_size,
            warmup_period=warmup_period,
            k=k,
            lr=lr,
            wd=wd,
        )

        self.hidden_size = hidden_size

        out_dist = out_dist.lower()
        if out_dist == "normal":
            num_dec_params = 1
        elif out_dist == "studentt":
            num_dec_params = 2
        else:
            raise ValueError(f"Unrecognized {out_dist=}")

        if include_alpha:
            num_dec_params += 1

        self.include_alpha = include_alpha
        self.out_dist = out_dist

        self.conditioning_model = nn.Sequential(
            nn.Linear(1 + num_cont_cols, hidden_size),
            RNNModel(
                hidden_size,
                hidden_size,
                num_blocks=num_blocks,
                dropout=dropout,
            ),
        )
        self.num_attn_layers = num_attn_layers

        if num_attn_layers > 0:
            self.attn = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    dim_feedforward=hidden_size,
                    nhead=nhead,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=num_attn_layers,
            )
        else:
            self.attn = None

        self.out_layer = nn.Linear(hidden_size, embed_size + num_dec_params)

    def _conditioning(self, lookback_returns, lookback_cont_features):
        history = torch.cat([lookback_returns.unsqueeze(-1), lookback_cont_features], -1)
        bsz, num_secs, lookback, num_feats = history.size()
        history = history.reshape(bsz * num_secs, lookback, num_feats)

        h = self.conditioning_model(history)[..., -1, :].reshape(bsz, num_secs, -1)

        if self.attn is not None:
            h = self.attn(h)

        out = self.out_layer(h)
        beta_x, out = out[..., : self.embed_size], out[..., self.embed_size :]
        beta_x = beta_x / np.sqrt(self.embed_size)

        if self.include_alpha:
            alpha_x, out = out[..., :1], out[..., 1:]
        else:
            alpha_x = 0

        if self.out_dist == "normal":
            sigma_x = out
            sigma_x = F.softplus(sigma_x) + 1e-4
            variance_x = sigma_x**2
            return (beta_x, alpha_x, variance_x, sigma_x)
        elif self.out_dist == "studentt":
            sigma_x, df_x = out.chunk(2, -1)
            sigma_x = F.softplus(sigma_x) + 1e-4
            df_x = F.softplus(df_x) + 4
            variance_x = df_x / (df_x - 2) * sigma_x**2
            return (beta_x, alpha_x, variance_x, sigma_x, df_x)
        else:
            raise ValueError(f"Unrecognized {self.out_dist=}")

    def _encoder(self, future_returns, conditioning):
        beta_x, alpha_x, variance_x, *_ = conditioning
        # beta_x: bsz x num_secs x embed_size
        # alpha_x: bsz x num_secs x 1
        # variance_x: bsz x num_secs x 1

        # Even though prior_mu=0, you can compute 'factor returns' by regressing
        # alpha_x against beta_x weighted by variance_x. In essence, measuring how much
        # of alpha_x can be explained by the different factors (beta_x)

        prior_mu = torch.zeros(self.embed_size).to(self.device)
        prior_covariance = torch.eye(self.embed_size).to(self.device)
        prior_precision = prior_covariance  # Because identity

        post_precision = prior_precision + torch.einsum(
            "bsf,bsg->bfg", beta_x, beta_x / variance_x
        )  # Equation 9
        post_mu = torch.linalg.solve(
            post_precision,
            prior_precision @ prior_mu
            + torch.einsum(
                "bsf,bsg->bf", beta_x, (future_returns.unsqueeze(-1) - alpha_x) / variance_x
            ),
        )
        return D.MultivariateNormal(post_mu, precision_matrix=post_precision)

    def _decoder(self, z, conditioning):
        # z: k x bsz x embed_size
        beta_x, alpha_x, _, *dist_args = conditioning

        mu_x = torch.einsum("bsf,kbf->kbs", beta_x, z).unsqueeze(-1) + alpha_x
        if self.out_dist == "normal":
            (sigma_x,) = dist_args
            out_dist = D.Independent(D.Normal(mu_x.squeeze(-1), sigma_x.squeeze(-1)), 1)
        elif self.out_dist == "studentt":
            sigma_x, df_x = dist_args
            out_dist = D.Independent(
                D.StudentT(df_x.squeeze(-1), mu_x.squeeze(-1), sigma_x.squeeze(-1)), 1
            )
        else:
            raise ValueError(f"Unrecognized {self.out_dist=}")
        return out_dist
