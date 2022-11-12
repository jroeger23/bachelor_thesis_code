import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (DeviceStatsMonitor, EarlyStopping, LearningRateMonitor)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler
from torch.utils.data import DataLoader, random_split

from common.data import (ComposeTransforms, LabelDtypeTransform, NaNToConstTransform, Pamap2,
                         Pamap2Options, Pamap2SplitIMUView, ResampleTransform)
from common.model import CNNIMU
from common.pl_callbacks import MonitorWF1
from common.pl_callbacks.monitors import MonitorAcc


def main() -> None:
  # Load datasets ##################################################################################
  view = Pamap2SplitIMUView(locations=Pamap2SplitIMUView.allLocations())
  train_data = Pamap2(opts=[
      Pamap2Options.SUBJECT1, Pamap2Options.SUBJECT2, Pamap2Options.SUBJECT3,
      Pamap2Options.SUBJECT4, Pamap2Options.SUBJECT5, Pamap2Options.SUBJECT6,
      Pamap2Options.OPTIONAL1, Pamap2Options.OPTIONAL5, Pamap2Options.OPTIONAL6,
      Pamap2Options.OPTIONAL8, Pamap2Options.OPTIONAL9
  ],
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
  train_data, validation_data = random_split(
      dataset=train_data, lengths=[round(len(train_data) * 0.9),
                                   round(len(train_data) * 0.1)])

  test_data = Pamap2(opts=[Pamap2Options.SUBJECT7, Pamap2Options.SUBJECT8, Pamap2Options.SUBJECT9],
                     window=100,
                     stride=33,
                     transform=ComposeTransforms([
                         NaNToConstTransform(batch_constant=0, label_constant=0),
                         ResampleTransform(freq_in=100, freq_out=30),
                         LabelDtypeTransform(dtype=torch.int64)
                     ]),
                     view=view,
                     download=True)

  # Setup data loaders #############################################################################
  train_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=True)
  validation_loader = DataLoader(dataset=validation_data, batch_size=100, shuffle=False)
  test_loader = DataLoader(dataset=test_data, batch_size=100, shuffle=False)

  # Configure CNN-IMU Model ########################################################################
  imu_sizes = [segment.shape[1] for segment in train_data[0][0]]
  model = CNNIMU(n_blocks=3, imu_sizes=imu_sizes, sample_length=100, n_classes=25)

  # Training / Evaluation ##########################################################################
  trainer = pl.Trainer(max_epochs=3,
                       accelerator='auto',
                       callbacks=[
                           DeviceStatsMonitor(),
                           LearningRateMonitor(),
                           EarlyStopping(monitor='validation/loss',
                                         min_delta=0.001,
                                         patience=7,
                                         mode='min'),
                           MonitorWF1(),
                           MonitorAcc()
                       ],
                       val_check_interval=1 / 5,
                       enable_checkpointing=True,
                       profiler=AdvancedProfiler(dirpath="logs/CNNIMU-Pamap2",
                                                 filename="perf_logs"),
                       logger=TensorBoardLogger(save_dir='logs', name='CNNIMU-Pamap2'))
  trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
  trainer.test(model=model, dataloaders=test_loader)


if __name__ == '__main__':
  main()
