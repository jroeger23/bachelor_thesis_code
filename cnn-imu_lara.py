from common.data import LARa, LARaSplitIMUView, LabelDtypeTransform, ResampleTransform, ComposeTransforms, NaNToConstTransform, LARaOptions, CombineViews, LARaClassLabelView
from common.model import CNNIMU
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_log
from pytorch_lightning import callbacks as pl_cb
from pytorch_lightning import profilers as pl_prof
from common.pl_callbacks import MonitorAcc, MonitorWF1


def main():
  # Setup datasets #################################################################################
  view = CombineViews(batch_view=LARaSplitIMUView(locations=LARaSplitIMUView.allLocations()),
                      labels_view=LARaClassLabelView())

  train_data = LARa(download=True,
                    window=100,
                    stride=3,
                    opts=[
                        LARaOptions.ALL_SUBJECTS, LARaOptions.RUN01, LARaOptions.RUN02,
                        LARaOptions.RUN03, LARaOptions.RUN04, LARaOptions.RUN05, LARaOptions.RUN06,
                        LARaOptions.RUN07, LARaOptions.RUN08, LARaOptions.RUN11, LARaOptions.RUN12,
                        LARaOptions.RUN13, LARaOptions.RUN14, LARaOptions.RUN15, LARaOptions.RUN16,
                        LARaOptions.RUN17, LARaOptions.RUN18, LARaOptions.RUN21, LARaOptions.RUN22,
                        LARaOptions.RUN23, LARaOptions.RUN24, LARaOptions.RUN25, LARaOptions.RUN26,
                        LARaOptions.RUN27, LARaOptions.RUN28
                    ],
                    transform=ComposeTransforms([
                        NaNToConstTransform(batch_constant=0, label_constant=7),
                        ResampleTransform(freq_in=100, freq_out=30),
                        LabelDtypeTransform(dtype=torch.int64)
                    ]),
                    view=view)

  test_data = LARa(download=True,
                   window=100,
                   stride=3,
                   opts=[
                       LARaOptions.ALL_SUBJECTS, LARaOptions.RUN09, LARaOptions.RUN10,
                       LARaOptions.RUN19, LARaOptions.RUN20, LARaOptions.RUN29, LARaOptions.RUN30
                   ],
                   transform=ComposeTransforms([
                       NaNToConstTransform(batch_constant=0, label_constant=7),
                       ResampleTransform(freq_in=100, freq_out=30),
                       LabelDtypeTransform(dtype=torch.int64)
                   ]),
                   view=view)

  train_data, validation_data = random_split(
      dataset=train_data, lengths=[round(len(train_data) * 0.9),
                                   round(len(train_data) * 0.1)])

  # Setup data loaders #############################################################################
  train_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=True)
  validation_loader = DataLoader(dataset=validation_data, batch_size=100, shuffle=False)
  test_loader = DataLoader(dataset=test_data, batch_size=100, shuffle=False)

  # Setup model ####################################################################################
  imu_sizes = [segment.shape[1] for segment in train_data[0][0]]
  model = CNNIMU(n_blocks=3, imu_sizes=imu_sizes, sample_length=100, n_classes=8)
  trainer = pl.Trainer(max_epochs=3,
                       accelerator='auto',
                       callbacks=[
                           pl_cb.DeviceStatsMonitor(),
                           pl_cb.LearningRateMonitor(),
                           pl_cb.EarlyStopping(monitor='validation/loss',
                                               min_delta=0.001,
                                               patience=7,
                                               mode='min'),
                           MonitorWF1(),
                           MonitorAcc()
                       ],
                       val_check_interval=1 / 5,
                       enable_checkpointing=True,
                       profiler=pl_prof.AdvancedProfiler(dirpath="logs/CNNIMU-LARa",
                                                         filename="perf_logs"),
                       logger=pl_log.TensorBoardLogger(save_dir='logs', name='CNNIMU-LARa'))
  trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
  trainer.test(model=model, dataloaders=test_loader)


if __name__ == '__main__':
  main()