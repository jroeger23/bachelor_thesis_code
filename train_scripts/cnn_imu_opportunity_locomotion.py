import pytorch_lightning as pl
import sacred
import torch
from pytorch_lightning import callbacks as pl_cb
from sacred.observers import MongoObserver
from sacred.run import Run
from torch.utils.data import DataLoader

from common.data import (BatchAdditiveGaussianNoise, CombineViews, ComposeTransforms,
                         LabelDtypeTransform, MeanVarianceNormalize, Opportunity,
                         OpportunityHumanSensorUnitsView, OpportunityLocomotionLabelAdjustMissing3,
                         OpportunityLocomotionLabelView, OpportunityOptions,
                         OpportunityRemoveHumanSensorUnitNaNRows, BlankInvalidColumns)
from common.helper import getRunCheckpointDirectory, parseMongoConfig
from common.model import CNNIMU
from common.pl_components import (MonitorAcc, MonitorBatchTime, MonitorWF1, SacredLogger)

ex = sacred.Experiment(name='CNN-IMU_Opportunity-Locomotion')

ex.observers.append(MongoObserver.create(**parseMongoConfig('./config.ini')))


@ex.config
def default_config():
  window = 24
  stride = 12
  batch_size = 64
  cnn_imu_blocks = 3
  cnn_imu_channels = 64
  cnn_imu_fc_features = 512
  max_epochs = 50
  loss_patience = 32
  validation_interval = 1 / 5
  optimizer = 'Adam'
  cnn_imu_weight_initialization = 'orthogonal'
  blank_invalid_columns = True

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
def main(window: int, stride: int, batch_size: int, cnn_imu_blocks: int, cnn_imu_channels: int,
         cnn_imu_fc_features: int, max_epochs: int, loss_patience: int, validation_interval: float,
         cnn_imu_weight_initialization: str, blank_invalid_columns: bool, _run: Run, _config):
  # Setup datasets #################################################################################
  static_transform = ComposeTransforms([
      BlankInvalidColumns(sample_thresh=0.5 if blank_invalid_columns else None, label_thresh=None),
      OpportunityRemoveHumanSensorUnitNaNRows(),
      LabelDtypeTransform(dtype=torch.int64),
      OpportunityLocomotionLabelAdjustMissing3(),
      MeanVarianceNormalize(mean=0.5, variance=1),
  ])
  dynamic_transform = ComposeTransforms([BatchAdditiveGaussianNoise(mu=0, sigma=0.01)])
  view = CombineViews(sample_view=OpportunityHumanSensorUnitsView(),
                      label_view=OpportunityLocomotionLabelView())
  train_data = Opportunity(download=True,
                           window=window,
                           stride=stride,
                           opts=[OpportunityOptions.DEFAULT_TRAIN],
                           static_transform=static_transform,
                           dynamic_transform=dynamic_transform,
                           view=view)
  validation_data = Opportunity(download=True,
                                window=window,
                                stride=stride,
                                opts=[OpportunityOptions.DEFAULT_VALIDATION],
                                static_transform=static_transform,
                                dynamic_transform=dynamic_transform,
                                view=view)
  test_data = Opportunity(download=True,
                          window=window,
                          stride=stride,
                          opts=[OpportunityOptions.DEFAULT_TEST],
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
                 n_classes=5,
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
                       log_every_n_steps=1,
                       logger=SacredLogger(_run=_run),
                       val_check_interval=validation_interval)
  trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
  trainer.test(ckpt_path=best_loss_ckpt.best_model_path, dataloaders=test_loader)
  trainer.test(ckpt_path=best_acc_ckpt.best_model_path, dataloaders=test_loader)
  trainer.test(ckpt_path=best_wf1_ckpt.best_model_path, dataloaders=test_loader)
