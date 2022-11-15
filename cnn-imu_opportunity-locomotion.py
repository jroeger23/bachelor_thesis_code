import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks as pl_cb
from pytorch_lightning import loggers as pl_log
from torch.utils.data import DataLoader, random_split

from common.data import (CombineViews, ComposeTransforms, LabelDtypeTransform, NaNToConstTransform,
                         Opportunity, OpportunityHumanSensorUnitsView,
                         OpportunityLocomotionLabelAdjustMissing3, OpportunityLocomotionLabelView,
                         OpportunityOptions)
from common.model import CNNIMU
from common.pl_components import ModelProfiler, MonitorAcc, MonitorWF1


def main():
  # Load Dataset ###################################################################################
  data = Opportunity(window=24,
                     stride=12,
                     transform=ComposeTransforms([
                         NaNToConstTransform(batch_constant=0, label_constant=0),
                         LabelDtypeTransform(dtype=torch.int64),
                         OpportunityLocomotionLabelAdjustMissing3()
                     ]),
                     view=CombineViews(batch_view=OpportunityHumanSensorUnitsView(),
                                       labels_view=OpportunityLocomotionLabelView()),
                     download=True,
                     opts=[OpportunityOptions.ALL_SUBJECTS, OpportunityOptions.ALL_ADL])

  # Split into train, validation and test data
  train_data, validation_data, test_data = random_split(dataset=data, lengths=[0.8, 0.12, 0.08])

  # Setup Dataloaders
  train_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=True)
  validation_loader = DataLoader(dataset=validation_data, batch_size=100, shuffle=False)
  test_loader = DataLoader(dataset=test_data, batch_size=100, shuffle=False)

  # Setup model ####################################################################################
  imu_sizes = [imu_tensor.shape[1] for imu_tensor in train_data[0][0]]
  model = CNNIMU(n_blocks=2, imu_sizes=imu_sizes, sample_length=24, n_classes=5)

  # Setup Training #################################################################################
  pl_logger = pl_log.TensorBoardLogger(save_dir='logs', name='CNNIMU-Opportunity-Locomotion')
  trainer = pl.Trainer(max_epochs=15,
                       accelerator='auto',
                       callbacks=[
                           pl_cb.DeviceStatsMonitor(),
                           pl_cb.LearningRateMonitor(),
                           pl_cb.ModelSummary(max_depth=2),
                           pl_cb.EarlyStopping(monitor='validation/loss',
                                               min_delta=0.001,
                                               patience=22,
                                               mode='min'),
                           MonitorWF1(),
                           MonitorAcc()
                       ],
                       val_check_interval=1 / 10,
                       enable_checkpointing=True,
                       profiler=ModelProfiler(dirpath=pl_logger.log_dir, filename="perf_logs"),
                       logger=pl_logger)

  trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
  trainer.test(model=model, dataloaders=test_loader)


if __name__ == '__main__':
  main()