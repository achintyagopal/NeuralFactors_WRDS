import numpy as np
import torch
from torch import optim
from torch import distributions as D

import pytorch_lightning as pl


class BaseVAE(pl.LightningModule):

    def __init__(
        self,
        embed_size=4,
        warmup_period=100,
        k=20,
        lr=1e-4,
        wd=0,
    ):
        super().__init__()

        self.embed_size = embed_size

        self.warmup_period = warmup_period
        self.lr = lr
        self.wd = wd
        self.k = k

    def prior_dist(self, device):
        return D.Independent(
            D.Normal(
                torch.zeros(self.embed_size).to(device),
                torch.ones(self.embed_size).to(device),
            ),
            1,
        )

    def _conditioning(self, lookback_returns, lookback_cont_features):
        raise NotImplementedError()

    def _encoder(self, future_returns, conditioning):
        raise NotImplementedError()

    def _decoder(self, z, conditioning):
        raise NotImplementedError()

    def _get_elbo(self, batch, k=None):
        if k is None:
            k = self.k

        conditioning = self._conditioning(
            batch["lookback_returns"],
            batch["lookback_cont_features"],
        )
        post_dist = self._encoder(
            batch["future_return"],
            conditioning,
        )
        z = post_dist.rsample((k,))
        out_dist = self._decoder(
            z,
            conditioning,
        )
        prior_dist = self.prior_dist(self.device)

        log_pxz = out_dist.log_prob(batch["future_return"])
        log_pz = prior_dist.log_prob(z)
        log_qzx = post_dist.log_prob(z)
        elbo = log_pxz + log_pz - log_qzx

        elbo = torch.logsumexp(elbo, 0) - np.log(k)
        return elbo, (post_dist, z, out_dist)

    def _get_stats(self, batch, k=None):
        elbo, (post_dist, z, out_dist) = self._get_elbo(batch, k=k)
        return {
            "loss": -elbo.mean(),
        }

    def training_step(self, batch, batch_nb):
        stats = self._get_stats(batch, k=self.k)
        for k, v in stats.items():
            self.log(f"train_{k}", v, prog_bar=True, on_epoch=True)
        return stats["loss"]

    def validation_step(self, batch, batch_nb):
        stats = self._get_stats(batch, k=self.k)
        for k, v in stats.items():
            self.log(f"val_{k}", v, prog_bar=True, on_epoch=True)

    def _lr_fn(self, step):
        if step < self.warmup_period:
            return step / self.warmup_period
        return 1

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, self._lr_fn)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
