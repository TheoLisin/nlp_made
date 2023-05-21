import pytorch_lightning as pl
from torch import optim, Tensor
from torch.nn import Module
from typing import Callable


class ModelWrapper(pl.LightningModule):
    def __init__(
        self,
        model: Module,
        criterion_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer_fn: Callable[[Module], optim.Optimizer],
    ):
        super().__init__()
        self.model = model
        self.loss = criterion_fn
        self.optimizer_fn = optimizer_fn

    def forward(self, x):
        return self.model(x)
            
    def step(self, batch, batch_idx, metric, prog_bar=False):
        x, y = batch
        translation = self.model(x)
        loss = self.loss(translation, y)
        self.log(metric, loss, prog_bar=prog_bar)
        return loss

    def test_step(self, batch, batch_idx, prog_bar=False):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train_loss")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val_loss", True)
        
    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.model)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]