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
        self.teacher_forcing_ratio = 0.5

    def forward(self, data, batch_idx, **kwargs) -> Tensor:
        if "teacher_forcing_ratio" not in kwargs:
            kwargs["teacher_forcing_ratio"] = self.teacher_forcing_ratio
        return self.model(data, **kwargs)
            
    def step(self, batch, batch_idx, metric, prog_bar=False):
        if "train" in metric:
            self.teacher_forcing_ratio *= 0.9999
            self.log("tfr", self.teacher_forcing_ratio)
        x, y = batch
        batch_size = x.shape[1]
        try:
            translation = self.forward(x, batch_idx)
        except ValueError:
            translation = self.forward(batch, batch_idx)
        # print(translation.view(-1, translation.shape[-1]).shape)
        # print(y.view(-1).shape)
        loss = self.loss(translation, y)
        self.log(metric, loss, prog_bar=prog_bar, batch_size=batch_size)
        return loss

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