import pytorch_lightning as pl
import sacred
import torch
from sacred.observers import MongoObserver
from sacred.run import Run
from pytorch_lightning import callbacks as pl_cb
from torch.utils.data import DataLoader, random_split

from common.data import (CombineViews, ComposeTransforms, LabelDtypeTransform, LARa,
                         LARaClassLabelView, LARaOptions, LARaSplitIMUView, NaNToConstTransform,
                         ResampleTransform)
from common.model import CNNIMU
from common.pl_components import MonitorAcc, MonitorWF1, SacredLogger
from common.helper import parseMongoObserverArgs

ex = sacred.Experiment(name='CNN-IMU_LARa')

ex.observers.append(MongoObserver.create(**parseMongoObserverArgs('./config.ini')))


@ex.config
def default_config():
  window = 100
  stride = 20
  sample_frequency = 30
  batch_size = 150
  cnn_imu_blocks = 3
  max_epochs = 15
  loss_patience = 22
  validation_interval = 1 / 10


@ex.automain
def main(window: int, stride: int, sample_frequency: int, batch_size: int, cnn_imu_blocks: int,
         max_epochs: int, loss_patience: int, validation_interval: float, _run: Run):
  # Setup datasets #################################################################################
  data = LARa(download=True,
              window=window,
              stride=stride,
              opts=[LARaOptions.ALL_RUNS, LARaOptions.ALL_SUBJECTS],
              transform=ComposeTransforms([
                  NaNToConstTransform(batch_constant=0, label_constant=7),
                  ResampleTransform(freq_in=100, freq_out=sample_frequency),
                  LabelDtypeTransform(dtype=torch.int64)
              ]),
              view=CombineViews(
                  batch_view=LARaSplitIMUView(locations=LARaSplitIMUView.allLocations()),
                  labels_view=LARaClassLabelView()))

  train_data, validation_data, test_data = random_split(dataset=data, lengths=[0.8, 0.12, 0.08])

  # Setup data loaders #############################################################################
  train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
  validation_loader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

  # Setup model ####################################################################################
  imu_sizes = [segment.shape[1] for segment in train_data[0][0]]
  model = CNNIMU(n_blocks=cnn_imu_blocks, imu_sizes=imu_sizes, sample_length=window, n_classes=8)
  trainer = pl.Trainer(max_epochs=max_epochs,
                       accelerator='auto',
                       callbacks=[
                           pl_cb.DeviceStatsMonitor(),
                           pl_cb.LearningRateMonitor(),
                           pl_cb.ModelSummary(max_depth=2),
                           pl_cb.EarlyStopping(monitor='validation/loss',
                                               min_delta=0.001,
                                               patience=loss_patience,
                                               mode='min'),
                           MonitorWF1(),
                           MonitorAcc()
                       ],
                       logger=SacredLogger(_run=_run),
                       val_check_interval=validation_interval,
                       enable_checkpointing=False)
  trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
  trainer.test(model=model, dataloaders=test_loader)
