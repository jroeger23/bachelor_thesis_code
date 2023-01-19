from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .common import (BatchAdditiveGaussianNoise, BlankInvalidColumns, ClipSampleRange, CombineViews,
                     ComposeTransforms, LabelDtypeTransform, MeanVarianceNormalize, RemoveNanRows,
                     ResampleTransform, Transform, View)
from .lara import LARa, LARaClassLabelView, LARaOptions, LARaSplitIMUView
from .opportunity import (Opportunity, OpportunityHumanSensorUnitsView,
                          OpportunityLocomotionLabelAdjustMissing3, OpportunityLocomotionLabelView,
                          OpportunityOptions, OpportunityRemoveHumanSensorUnitNaNRows)
from .pamap2 import (Pamap2, Pamap2FilterRowsByLabel, Pamap2InterpolateHeartrate, Pamap2Options,
                     Pamap2SplitIMUView)

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


default_pamap2_dynamic_transform = ComposeTransforms([BatchAdditiveGaussianNoise(mu=0, sigma=0.01)])
default_pamap2_view = Pamap2SplitIMUView(locations=Pamap2SplitIMUView.allLocations())


def default_pamap2_static_transform(use_transient_class: bool, sample_frequency: int) -> Transform:
  return ComposeTransforms([
      Pamap2InterpolateHeartrate(),
      RemoveNanRows(),
      Pamap2FilterRowsByLabel(keep_labels=[0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 24]
                              if use_transient_class else [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 24],
                              remap=True),
      ResampleTransform(freq_in=100, freq_out=sample_frequency),
      LabelDtypeTransform(dtype=torch.int64),
      MeanVarianceNormalize(mean=0.5, variance=1)
  ])


def default_opportunity_static_transform(blank_invalid_columns: bool) -> Transform:
  return ComposeTransforms([
      BlankInvalidColumns(sample_thresh=0.5 if blank_invalid_columns else None, label_thresh=None),
      OpportunityRemoveHumanSensorUnitNaNRows(),
      LabelDtypeTransform(dtype=torch.int64),
      OpportunityLocomotionLabelAdjustMissing3(),
      MeanVarianceNormalize(mean=0.5, variance=1),
  ])


default_opportunity_dynamic_transform = ComposeTransforms(
    [BatchAdditiveGaussianNoise(mu=0, sigma=0.01)])
default_opportunity_view = CombineViews(sample_view=OpportunityHumanSensorUnitsView(),
                                        label_view=OpportunityLocomotionLabelView())


class Pamap2DataModule(pl.LightningDataModule):

  def __init__(self,
               batch_size: int,
               window: int,
               stride: int,
               sample_frequency: Optional[int],
               use_transient_class: Optional[bool],
               static_transform: Optional[Transform] = None,
               dynamic_transform: Optional[Transform] = None,
               view: Optional[View] = None) -> None:
    super().__init__()
    if static_transform is None and sample_frequency is None:
      raise ValueError(f'Using default static_transform, but sample_frequency is None')
    if static_transform is None and use_transient_class is None:
      raise ValueError(f'Using default static_transform, but use_transient_class is None')

    self.batch_size = batch_size
    self.window = window
    self.stride = stride

    self.static_transform = static_transform or default_pamap2_static_transform(
        use_transient_class or False, sample_frequency or 100)
    self.dynamic_transform = dynamic_transform or default_pamap2_dynamic_transform
    self.view = view or default_pamap2_view

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
      self.train_set = Pamap2(download=True,
                              window=self.window,
                              stride=self.stride,
                              opts=[
                                  Pamap2Options.SUBJECT1, Pamap2Options.SUBJECT2,
                                  Pamap2Options.SUBJECT3, Pamap2Options.SUBJECT4,
                                  Pamap2Options.SUBJECT7, Pamap2Options.SUBJECT8,
                                  Pamap2Options.SUBJECT9, Pamap2Options.OPTIONAL1,
                                  Pamap2Options.OPTIONAL8, Pamap2Options.OPTIONAL9
                              ],
                              static_transform=self.static_transform,
                              dynamic_transform=self.dynamic_transform,
                              view=self.view)
    elif stage == 'validate':
      self.val_set = Pamap2(download=True,
                            window=self.window,
                            stride=self.stride,
                            opts=[Pamap2Options.SUBJECT5, Pamap2Options.OPTIONAL5],
                            static_transform=self.static_transform,
                            dynamic_transform=self.dynamic_transform,
                            view=self.view)
    elif stage == 'test':
      self.test_set = Pamap2(download=True,
                             window=self.window,
                             stride=self.stride,
                             opts=[Pamap2Options.SUBJECT6, Pamap2Options.OPTIONAL6],
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
    elif stage == 'validate':
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


class OpportunityDataModule(pl.LightningDataModule):

  def __init__(self,
               batch_size: int,
               window: int,
               stride: int,
               blank_invalid_columns: Optional[bool],
               static_transform: Optional[Transform] = None,
               dynamic_transform: Optional[Transform] = None,
               view: Optional[View] = None) -> None:
    super().__init__()
    if static_transform is None and blank_invalid_columns is None:
      raise ValueError(f'Using default static_transform, but blank_invalid_columns is None')

    self.batch_size = batch_size
    self.window = window
    self.stride = stride

    self.static_transform = static_transform or default_opportunity_static_transform(
        blank_invalid_columns or True)
    self.dynamic_transform = dynamic_transform or default_opportunity_dynamic_transform
    self.view = view or default_opportunity_view

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
      self.train_set = Opportunity(download=True,
                                   window=self.window,
                                   stride=self.stride,
                                   opts=[OpportunityOptions.DEFAULT_TRAIN],
                                   static_transform=self.static_transform,
                                   dynamic_transform=self.dynamic_transform,
                                   view=self.view)
    elif stage == 'validate':
      self.val_set = Opportunity(download=True,
                                 window=self.window,
                                 stride=self.stride,
                                 opts=[OpportunityOptions.DEFAULT_VALIDATION],
                                 static_transform=self.static_transform,
                                 dynamic_transform=self.dynamic_transform,
                                 view=self.view)
    elif stage == 'test':
      self.test_set = Opportunity(download=True,
                                  window=self.window,
                                  stride=self.stride,
                                  opts=[OpportunityOptions.DEFAULT_TEST],
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