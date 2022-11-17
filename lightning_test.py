import logging

import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as pl_loggers
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from common.data import mnist
from common.helper.metrics import auroc, toBinary, rocCurve, rocFigure
from common.helper.tui import modeDialog

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

train_data, test_data, labels = mnist()


class LightningModule(pl.LightningModule):

  def __init__(self) -> None:
    super().__init__()
    self.model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(5, 5), padding=(2, 2)),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        torch.nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(5, 5), padding=(2, 2)),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=7 * 7 * 5, out_features=49),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=49, out_features=10),
    )
    self.example_input_array = torch.unsqueeze(test_data[0][0], dim=0)

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch):
    x, y = batch
    logits = self.model(x)
    loss = torch.nn.functional.cross_entropy(input=logits, target=y)
    self.log('train/loss', loss.item())
    return loss

  def on_validation_epoch_start(self) -> None:
    self.val_probs = []
    self.val_labels = []

  def on_validation_epoch_end(self) -> None:
    probs = torch.vstack(self.val_probs)
    y = torch.cat(self.val_labels)

    for c_ix, (c_label, c_prob) in enumerate(toBinary(y, probs)):
      self.log(f'validate-auroc/{c_ix}-{labels[c_ix]}', auroc(c_label, c_prob))
      if isinstance(self.logger, pl_loggers.TensorBoardLogger):
        logger: SummaryWriter = self.logger.experiment
        logger.add_pr_curve(f'validate-pr/{c_ix}-{labels[c_ix]}',
                            c_label,
                            c_prob,
                            global_step=self.global_step)

  def validation_step(self, batch, batch_ix):
    x, y = batch
    probs = torch.nn.functional.softmax(self.model(x), dim=1)
    loss = torch.nn.functional.cross_entropy(probs, y)

    self.val_probs.append(probs)
    self.val_labels.append(y)

    acc = (probs.argmax(1) == y).type(torch.float).mean()

    self.log('validate/loss', loss.item())
    self.log('validate/acc', acc)

  def on_test_epoch_start(self) -> None:
    self.test_probs = []
    self.test_labels = []

  def on_test_epoch_end(self) -> None:
    probs = torch.vstack(self.test_probs)
    y = torch.cat(self.test_labels)

    for c_ix, (c_label, c_prob) in enumerate(toBinary(y, probs)):
      self.log(f'test-auroc/{c_ix}-{labels[c_ix]}', auroc(c_label, c_prob))
      if isinstance(self.logger, pl_loggers.TensorBoardLogger):
        logger: SummaryWriter = self.logger.experiment
        logger.add_pr_curve(f'test-pr/{c_ix}-{labels[c_ix]}', c_label, c_prob)
        roc = rocFigure(c_label, c_prob, sample_points=40)
        logger.add_figure(tag=f'test-roc/{c_ix}-{labels[c_ix]}', figure=roc)

  def test_step(self, batch, batch_ix):
    x, y = batch
    probs = torch.nn.functional.softmax(self.model(x), dim=1)
    loss = torch.nn.functional.cross_entropy(probs, y)

    self.test_probs.append(probs)
    self.test_labels.append(y)

    acc = (probs.argmax(1) == y).type(torch.float).mean()

    self.log('test/loss', loss.item())
    self.log('test/acc', acc)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters())


train_loader = DataLoader(dataset=train_data, batch_size=60, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=60, shuffle=False, num_workers=4)

config = modeDialog()

logger = pl_loggers.TensorBoardLogger(save_dir=config['log_dir'],
                                      name=config['experiment'],
                                      version=config['version'],
                                      log_graph=True)

es = pl_callbacks.EarlyStopping(monitor='validate/acc',
                                mode='max',
                                min_delta=0.00003,
                                patience=4,
                                strict=True)
trainer = pl.Trainer(max_epochs=5,
                     val_check_interval=100,
                     limit_val_batches=0.4,
                     callbacks=[es],
                     logger=logger)

if config['mode'] == 'train_new':
  m = LightningModule()

  trainer.validate(model=m, dataloaders=test_loader)
  trainer.fit(model=m, train_dataloaders=train_loader, val_dataloaders=test_loader)
  trainer.test(model=m, dataloaders=test_loader)

elif config['mode'] == 'test_checkpoint':
  m = LightningModule.load_from_checkpoint(config['checkpoint'])
  trainer.test(model=m, dataloaders=test_loader)

elif config['mode'] == 'quantize_checkpoint':
  pass
