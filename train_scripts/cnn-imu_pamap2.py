import pytorch_lightning as pl
import sacred
import torch
from pytorch_lightning import callbacks as pl_cb
from sacred.observers import MongoObserver
from sacred.run import Run
from torch.utils.data import DataLoader

from common.data import (BatchAdditiveGaussianNoise, ComposeTransforms, LabelDtypeTransform,
                         MeanVarianceNormalize, Pamap2, Pamap2FilterRowsByLabel,
                         Pamap2InterpolateHeartrate, Pamap2Options, Pamap2SplitIMUView,
                         RemoveNanRows, ResampleTransform)
from common.helper import getRunCheckpointDirectory, parseMongoConfig
from common.model import CNNIMU
from common.pl_components import (MonitorAcc, MonitorBatchTime, MonitorWF1, SacredLogger)

ex = sacred.Experiment(name='CNN-IMU_Pamap2(activity_labels)')

ex.observers.append(MongoObserver.create(**parseMongoConfig('./config.ini')))


@ex.config
def default_config():
  window = 100
  stride = 22
  sample_frequency = 30
  use_transient_class = True
  batch_size = 64
  cnn_imu_blocks = 2
  cnn_imu_channels = 64
  cnn_imu_fc_features = 256
  max_epochs = 50
  loss_patience = 32
  validation_interval = 1 / 5
  optimizer = 'Adam'
  cnn_imu_weight_initialization = 'orthogonal'

  if optimizer == 'Adam':
    lr = 1e-3
    betas = (0.9, 0.999)
    weight_decay = 0
  elif optimizer == 'RMSProp':
    lr = 1e-2
    alpha = 0.99
    weight_decay = 0.95
    momentum = 0


@ex.automain
def main(window: int, stride: int, sample_frequency: int, batch_size: int, cnn_imu_blocks: int,
         cnn_imu_channels: int, cnn_imu_fc_features: int, max_epochs: int, loss_patience: int,
         validation_interval: float, use_transient_class: bool, cnn_imu_weight_initialization: str,
         _run: Run, _config):
  # Setup datasets #################################################################################
  dynamic_transform = ComposeTransforms([BatchAdditiveGaussianNoise(mu=0, sigma=0.01)])
  static_transform = ComposeTransforms([
      Pamap2InterpolateHeartrate(),
      RemoveNanRows(),
      Pamap2FilterRowsByLabel(keep_labels=[0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 24]
                              if use_transient_class else [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 24],
                              remap=True),
      ResampleTransform(freq_in=100, freq_out=sample_frequency),
      LabelDtypeTransform(dtype=torch.int64),
      MeanVarianceNormalize(mean=0.5, variance=1)
  ])
  view = Pamap2SplitIMUView(locations=Pamap2SplitIMUView.allLocations())
  train_data = Pamap2(download=True,
                      window=window,
                      stride=stride,
                      opts=[
                          Pamap2Options.SUBJECT1, Pamap2Options.SUBJECT2, Pamap2Options.SUBJECT3,
                          Pamap2Options.SUBJECT4, Pamap2Options.SUBJECT7, Pamap2Options.SUBJECT8,
                          Pamap2Options.SUBJECT9, Pamap2Options.OPTIONAL1, Pamap2Options.OPTIONAL8,
                          Pamap2Options.OPTIONAL9
                      ],
                      static_transform=static_transform,
                      dynamic_transform=dynamic_transform,
                      view=view)
  validation_data = Pamap2(download=True,
                           window=window,
                           stride=stride,
                           opts=[Pamap2Options.SUBJECT5, Pamap2Options.OPTIONAL5],
                           static_transform=static_transform,
                           dynamic_transform=dynamic_transform,
                           view=view)
  test_data = Pamap2(download=True,
                     window=window,
                     stride=stride,
                     opts=[Pamap2Options.SUBJECT6, Pamap2Options.OPTIONAL6],
                     static_transform=static_transform,
                     dynamic_transform=dynamic_transform,
                     view=view)

  # Setup data loaders #############################################################################
  train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
  validation_loader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

  # Checkpointing ##################################################################################
  best_acc_ckpt = pl_cb.ModelCheckpoint(
      dirpath=getRunCheckpointDirectory(root='logs/checkpoints', _run=_run),
      filename=
      'BEST_ACC-e={epoch}-s={step}-loss={validation/loss}-acc={validation/acc}-wf1={validation/wf1}',
      auto_insert_metric_name=False,
      monitor='validation/acc',
      mode='max',
  )
  best_wf1_ckpt = pl_cb.ModelCheckpoint(
      dirpath=getRunCheckpointDirectory(root='logs/checkpoints', _run=_run),
      filename=
      'BEST_WF1-e={epoch}-s={step}-loss={validation/loss}-acc={validation/acc}-wf1={validation/wf1}',
      auto_insert_metric_name=False,
      monitor='validation/wf1',
      mode='max',
  )
  best_loss_ckpt = pl_cb.ModelCheckpoint(
      dirpath=getRunCheckpointDirectory(root='logs/checkpoints', _run=_run),
      filename=
      'BEST_LOSS-e={epoch}-s={step}-loss={validation/loss}-acc={validation/acc}-wf1={validation/wf1}',
      auto_insert_metric_name=False,
      monitor='validation/loss',
      mode='min',
  )

  # Setup model ####################################################################################
  imu_sizes = [segment.shape[1] for segment in train_data[0][0]]
  model = CNNIMU(n_blocks=cnn_imu_blocks,
                 imu_sizes=imu_sizes,
                 sample_length=window,
                 fc_features=cnn_imu_fc_features,
                 conv_channels=cnn_imu_channels,
                 n_classes=12 if use_transient_class else 11,
                 initialization=cnn_imu_weight_initialization,
                 **_config)
  trainer = pl.Trainer(max_epochs=max_epochs,
                       accelerator='auto',
                       callbacks=[
                           pl_cb.LearningRateMonitor(),
                           pl_cb.ModelSummary(max_depth=2),
                           pl_cb.EarlyStopping(monitor='validation/loss',
                                               min_delta=0.001,
                                               patience=loss_patience,
                                               mode='min'),
                           MonitorWF1(),
                           MonitorAcc(),
                           MonitorBatchTime(), best_acc_ckpt, best_wf1_ckpt, best_loss_ckpt
                       ],
                       logger=SacredLogger(_run=_run),
                       val_check_interval=validation_interval)
  trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
  trainer.test(ckpt_path=best_loss_ckpt.best_model_path, dataloaders=test_loader)
  trainer.test(ckpt_path=best_acc_ckpt.best_model_path, dataloaders=test_loader)
  trainer.test(ckpt_path=best_wf1_ckpt.best_model_path, dataloaders=test_loader)
