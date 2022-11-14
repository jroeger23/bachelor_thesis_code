import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (DeviceStatsMonitor, EarlyStopping, LearningRateMonitor)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler
from torch.utils.data import DataLoader, random_split

from common.data import (ComposeTransforms, LabelDtypeTransform, NaNToConstTransform, Pamap2,
                         Pamap2Options, Pamap2SplitIMUView, ResampleTransform)
from common.model import CNNIMU
from common.pl_components import ModelProfiler, MonitorAcc, MonitorWF1


def main() -> None:
  # Load datasets ##################################################################################
  view = Pamap2SplitIMUView(locations=Pamap2SplitIMUView.allLocations())
  train_data = Pamap2(opts=[Pamap2Options.ALL_SUBJECTS, Pamap2Options.ALL_OPTIONAL],
                      window=100,
                      stride=33,
                      transform=ComposeTransforms([
                          NaNToConstTransform(batch_constant=0, label_constant=0),
                          ResampleTransform(freq_in=100, freq_out=30),
                          LabelDtypeTransform(dtype=torch.int64)
                      ]),
                      view=Pamap2SplitIMUView(locations=Pamap2SplitIMUView.allLocations()),
                      download=True)

  # randomly draw some training data as validation data
  train_data, validation_data, test_data = random_split(dataset=train_data,
                                                        lengths=[
                                                            round(len(train_data) * 0.8),
                                                            round(len(train_data) * 0.12),
                                                            round(len(train_data) * 0.08)
                                                        ])

  # Setup data loaders #############################################################################
  train_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=True)
  validation_loader = DataLoader(dataset=validation_data, batch_size=100, shuffle=False)
  test_loader = DataLoader(dataset=test_data, batch_size=100, shuffle=False)

  # Configure CNN-IMU Model ########################################################################
  imu_sizes = [segment.shape[1] for segment in train_data[0][0]]
  model = CNNIMU(n_blocks=3, imu_sizes=imu_sizes, sample_length=100, n_classes=25)

  # Training / Evaluation ##########################################################################
  logger = TensorBoardLogger(save_dir='logs', name='CNNIMU-Pamap2')
  trainer = pl.Trainer(max_epochs=15,
                       accelerator='auto',
                       callbacks=[
                           DeviceStatsMonitor(),
                           LearningRateMonitor(),
                           EarlyStopping(monitor='validation/loss',
                                         min_delta=0.001,
                                         patience=22,
                                         mode='min'),
                           MonitorWF1(),
                           MonitorAcc()
                       ],
                       val_check_interval=1 / 10,
                       enable_checkpointing=True,
                       profiler=ModelProfiler(dirpath=logger.log_dir, filename="perf_logs"),
                       logger=logger)
  trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
  trainer.test(model=model, dataloaders=test_loader)


if __name__ == '__main__':
  main()
