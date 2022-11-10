from common.model import CNNIMU
from common.data import Pamap2, Pamap2Options, Pamap2SplitIMUView, NaNToConstTransform, LabelDtypeTransform, ComposeTransforms, ResampleTransform
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import multiprocessing
import torch
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor

view = Pamap2SplitIMUView(locations=Pamap2SplitIMUView.allLocations())

train_data = Pamap2(opts=[Pamap2Options.ALL_SUBJECTS],
                    window=100,
                    stride=33,
                    transform=ComposeTransforms([
                        NaNToConstTransform(batch_constant=0, label_constant=0),
                        ResampleTransform(freq_in=100, freq_out=30),
                        LabelDtypeTransform(dtype=torch.int64)
                    ]),
                    view=view,
                    download=True)

imu_sizes = [segment.shape[1] for segment in train_data[0][0]]

train_loader = DataLoader(dataset=train_data,
                          batch_size=60,
                          shuffle=True,
                          num_workers=multiprocessing.cpu_count())
trainer = pl.Trainer(max_epochs=1,
                     accelerator='auto',
                     callbacks=[DeviceStatsMonitor(), LearningRateMonitor()],
                     profiler='simple')
model = CNNIMU(n_blocks=2, imu_sizes=imu_sizes, sample_length=100, n_classes=25)
trainer.fit(model=model, train_dataloaders=train_loader)