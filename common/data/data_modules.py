import pytorch_lightning as pl

from .lara import LARa, LARaSplitIMUView, LARaClassLabelView, LARaOptions
from .common import Transform, View, ComposeTransforms, BatchAdditiveGaussianNoise, RemoveNanRows, ResampleTransform, LabelDtypeTransform, MeanVarianceNormalize, ClipSampleRange, CombineViews

import torch
from torch.utils.data import DataLoader, Dataset

from typing import Optional

default_lara_dynamic_transform = ComposeTransforms([BatchAdditiveGaussianNoise(mu=0, sigma=0.01)])
default_lara_view = CombineViews(
    sample_view=LARaSplitIMUView(locations=LARaSplitIMUView.allLocations()),
    label_view=LARaClassLabelView())


def default_lara_static_transform(sample_frequency: int) -> Transform:
  return ComposeTransforms([
      RemoveNanRows(),
      ResampleTransform(freq_in=100, freq_out=sample_frequency),
      LabelDtypeTransform(dtype=torch.int64),
      MeanVarianceNormalize(mean=0.5, variance=1),
      ClipSampleRange(range_min=0, range_max=1)
  ])


class LARaDataModule(pl.LightningDataModule):

  def __init__(self,
               batch_size: int,
               window: int,
               stride: int,
               sample_frequency: Optional[int],
               static_transform: Optional[Transform] = None,
               dynamic_transform: Optional[Transform] = None,
               view: Optional[View] = None) -> None:
    super().__init__()
    if static_transform is None and sample_frequency is None:
      raise ValueError(f'Using default static_transform, but sample_frequency is None')

    self.batch_size = batch_size
    self.window = window
    self.stride = stride

    self.static_transform = static_transform or default_lara_static_transform(sample_frequency or
                                                                              100)
    self.dynamic_transform = dynamic_transform or default_lara_dynamic_transform
    self.view = view or default_lara_view

    self._train_set: Optional[Dataset] = None
    self._val_set: Optional[Dataset] = None
    self._test_set: Optional[Dataset] = None

  @property
  def train_set(self) -> Dataset:
    assert self._train_set is not None
    return self._train_set

  @train_set.setter
  def train_set(self, value) -> None:
    self._train_set = value

  @property
  def val_set(self) -> Dataset:
    assert self._val_set is not None
    return self._val_set

  @val_set.setter
  def val_set(self, value) -> None:
    self._val_set = value

  @property
  def test_set(self) -> Dataset:
    assert self._test_set is not None
    return self._test_set

  @test_set.setter
  def test_set(self, value) -> None:
    self._test_set = value

  def setup(self, stage: str) -> None:
    if stage == 'fit':
      self.train_set = LARa(download=True,
                            window=self.window,
                            stride=self.stride,
                            opts=[LARaOptions.DEFAULT_TRAIN_SET],
                            static_transform=self.static_transform,
                            dynamic_transform=self.dynamic_transform,
                            view=self.view)
      self.val_set = LARa(download=True,
                          window=self.window,
                          stride=self.stride,
                          opts=[LARaOptions.DEFAULT_VALIDATION_SET],
                          static_transform=self.static_transform,
                          dynamic_transform=self.dynamic_transform,
                          view=self.view)
    elif stage == 'test':
      self.test_set = LARa(download=True,
                           window=self.window,
                           stride=self.stride,
                           opts=[LARaOptions.DEFAULT_TEST_SET],
                           static_transform=self.static_transform,
                           dynamic_transform=self.dynamic_transform,
                           view=self.view)

  def train_dataloader(self, **dataloader_kwargs) -> DataLoader:
    kwargs = {'shuffle': True} | dataloader_kwargs
    return DataLoader(dataset=self.train_set, batch_size=self.batch_size, **kwargs)

  def val_dataloader(self, **dataloader_kwargs) -> DataLoader:
    kwargs = {'shuffle': False} | dataloader_kwargs
    return DataLoader(dataset=self.val_set, batch_size=self.batch_size, **kwargs)

  def test_dataloader(self, **dataloader_kwargs) -> DataLoader:
    kwargs = {'shuffle': False} | dataloader_kwargs
    return DataLoader(dataset=self.test_set, batch_size=self.batch_size, **kwargs)
