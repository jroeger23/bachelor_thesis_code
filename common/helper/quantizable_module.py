import enum
import re
from typing import Dict, List, Optional, Pattern, Type, Union

import torch
import torch.ao.quantization as tq
import torch.ao.nn.quantized as tqnn
import copy

from .qconfig_factory import QConfigFactory


class QuantizationType(enum.Enum):
  PTQ = "PTQ"
  QAT = "QAT"
  NONE = "NONE"
  FUSE_ONLY = "FUSE_ONLY"


class QuantizationState(enum.Enum):
  TRAIN_CALIBRATE = "TRAIN/CALIBRATE"
  QUANTIZED = "QUANTIZED"


class QuantizationMode():

  def __init__(self,
               has_quantized_input: bool,
               has_quantized_output: bool,
               quantization_type: QuantizationType,
               operator_fuse_list: Union[List[List[str]], List[str]] = [],
               qconfig_factory: Optional[QConfigFactory] = None,
               propagate_qconfig: bool = True) -> None:
    self._has_quantized_input: bool = has_quantized_input
    self._has_quantized_output: bool = has_quantized_output
    self._quantization_type: QuantizationType = quantization_type
    self._operator_fuse_list: Union[List[List[str]], List[str]] = operator_fuse_list
    self._qconfig_factory: Optional[QConfigFactory] = qconfig_factory
    self._propagate_qconfig: bool = propagate_qconfig

  @classmethod
  def ptq(cls, has_quantized_input: bool, has_quantized_output: bool,
          operator_fuse_list: Union[List[List[str]], List[str]], qconfig_factory: QConfigFactory,
          propagate_qconfig: bool):
    return cls(has_quantized_input, has_quantized_output, QuantizationType.PTQ, operator_fuse_list,
               qconfig_factory, propagate_qconfig)

  @classmethod
  def qat(cls, has_quantized_input: bool, has_quantized_output: bool,
          operator_fuse_list: Union[List[List[str]], List[str]], qconfig_factory: QConfigFactory,
          propagate_qconfig: bool):
    return cls(has_quantized_input, has_quantized_output, QuantizationType.QAT, operator_fuse_list,
               qconfig_factory, propagate_qconfig)

  @classmethod
  def fuse_only(cls, operator_fuse_list: Union[List[List[str]], List[str]]):
    return cls(False, False, QuantizationType.FUSE_ONLY, operator_fuse_list, None)

  @classmethod
  def none(cls, has_quantized_input: bool, has_quantized_output: bool,
           qconfig_factory: QConfigFactory, propagate_qconfig: bool):
    return cls(has_quantized_input, has_quantized_output, QuantizationType.NONE, [],
               qconfig_factory, propagate_qconfig)

  @property
  def has_quantized_input(self) -> bool:
    return self._has_quantized_input

  @property
  def has_quantized_output(self) -> bool:
    return self._has_quantized_output

  @property
  def qconfig(self) -> tq.QConfig:
    if self._qconfig_factory is not None:
      return self._qconfig_factory.getQConfig()
    else:
      raise ValueError('Property qconfig not set.')

  @property
  def operator_fuse_list(self) -> Union[List[List[str]], List[str]]:
    return self._operator_fuse_list

  @property
  def quantization_type(self) -> QuantizationType:
    return self._quantization_type

  @property
  def is_quantized(self) -> bool:
    return self._quantization_type in (QuantizationType.PTQ, QuantizationType.QAT)

  @property
  def is_ptq(self) -> bool:
    return self._quantization_type == QuantizationType.PTQ

  @property
  def is_qat(self) -> bool:
    return self._quantization_type == QuantizationType.QAT

  @property
  def is_fuse_only(self) -> bool:
    return self._quantization_type == QuantizationType.FUSE_ONLY

  @property
  def needs_fusion(self) -> bool:
    return self.is_fuse_only or (self.is_quantized and len(self.operator_fuse_list) > 0)

  @property
  def needs_input_quantize(self) -> bool:
    return self.is_quantized and not self.has_quantized_input

  @property
  def needs_input_dequantize(self) -> bool:
    return not self.is_quantized and self.has_quantized_input

  @property
  def needs_output_quantize(self) -> bool:
    return not self.is_quantized and self.has_quantized_output

  @property
  def needs_output_dequantize(self) -> bool:
    return self.is_quantized and not self.has_quantized_output

  @property
  def needs_quant_wrapper(self) -> bool:
    return any((
        self.needs_input_quantize,
        self.needs_input_dequantize,
        self.needs_output_quantize,
        self.needs_output_dequantize,
    ))


class QuantizationModeMapping():

  def __init__(self) -> None:
    self._global: Optional[QuantizationMode] = None
    self._type_mapping: Dict[Type, QuantizationMode] = {}
    self._name_mapping: Dict[str, QuantizationMode] = {}
    self._regex_mapping: Dict[Pattern, QuantizationMode] = {}

  def setGlobal(self, new_global: Optional[QuantizationMode]):
    self._global = new_global
    return self

  def addTypeMapping(self, type: Type, mode: QuantizationMode):
    self._type_mapping[type] = mode
    return self

  def addNameMapping(self, name: str, mode: QuantizationMode):
    self._name_mapping[name] = mode
    return self

  def addRegexMapping(self, regex_or_pattern: Union[str, Pattern], mode: QuantizationMode):
    if isinstance(regex_or_pattern, str):
      regex_or_pattern = re.compile(regex_or_pattern)

    self._regex_mapping[regex_or_pattern] = mode

    return self

  def _regexMatch(self, name: str) -> Optional[QuantizationMode]:
    for pattern, q_mode in self._regex_mapping.items():
      if pattern.match(name):
        return q_mode

    return None

  def forModule(self, name: Optional[str], type: Optional[Type]) -> Optional[QuantizationMode]:
    type_mapped_value = self._type_mapping.get(type, None) if type is not None else None

    name_mapped_value = self._name_mapping.get(name,
                                               self._regexMatch(name)) if name is not None else None

    return name_mapped_value or type_mapped_value or self._global


class QuantizationWrapper(torch.nn.Module):

  def __init__(self,
               module: torch.nn.Module,
               qconfig: tq.QConfig,
               input_quantize: bool = False,
               input_dequantize: bool = False,
               output_quantize: bool = False,
               output_dequantize: bool = False) -> None:
    super(QuantizationWrapper, self).__init__()

    if input_quantize and input_dequantize:
      raise ValueError(f'Invalid combination of {input_quantize=} and {input_dequantize=}')
    if output_quantize and output_dequantize:
      raise ValueError(f'Invalid combination of {output_quantize=} and {output_dequantize=}')

    self.module = module

    if input_quantize:
      self.input_quant = tq.QuantStub(qconfig)
    if input_dequantize:
      self.input_quant = tq.DeQuantStub(qconfig)
    if output_quantize:
      self.output_quant = tq.QuantStub(qconfig)
    if output_dequantize:
      self.output_quant = tq.DeQuantStub(qconfig)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if hasattr(self, 'input_quant'):
      x = self.input_quant(x)

    x = self.module(x)

    if hasattr(self, 'output_quant'):
      x = self.output_quant(x)

    return x


def applyQuantizationModePreparations(module: torch.nn.Module,
                                      quantization_mode: QuantizationMode) -> torch.nn.Module:
  # Install qconfig
  if quantization_mode.is_quantized:
    setattr(module, 'qconfig', quantization_mode.qconfig)
  else:
    setattr(module, 'qconfig', None)

  # Wrap if needed
  if quantization_mode.needs_quant_wrapper:
    module = QuantizationWrapper(
        module=module,
        qconfig=quantization_mode.qconfig,
        input_quantize=quantization_mode.needs_input_quantize,
        input_dequantize=quantization_mode.needs_input_dequantize,
        output_quantize=quantization_mode.needs_output_quantize,
        output_dequantize=quantization_mode.needs_output_dequantize,
    )

  # apply special QAT conversions
  if quantization_mode.is_qat:
    tq.convert(module=module, mapping=tq.get_default_qat_module_mappings(), inplace=True)

  return module


def applyQuantizationModePreAttach(module: torch.nn.Module, quantization_mode: QuantizationMode):
  # Apply fusion
  if quantization_mode.needs_fusion:
    tq.fuse_modules(model=module,
                    modules_to_fuse=quantization_mode.operator_fuse_list,
                    inplace=True)


def _attachQuantizationModes(module: torch.nn.Module,
                             quantization_mode_mapping: QuantizationModeMapping):
  for name, child in module.named_modules():
    mode = quantization_mode_mapping.forModule(name=name, type=type(child))
    if mode is not None:
      applyQuantizationModePreAttach(child, mode)
      setattr(child, 'quantization_mode', mode)


def _removeQuantizationModes(module: torch.nn.Module):
  for _, child in module.named_modules():
    if hasattr(child, 'quantization_mode'):
      delattr(child, 'quantization_mode')


def applyQuantizationModeMapping(module: torch.nn.Module,
                                 quantization_mode_mapping: QuantizationModeMapping,
                                 inplace: bool = False):
  if not inplace:
    module = copy.deepcopy(module)

  _attachQuantizationModes(module, quantization_mode_mapping)

  queue = [module]

  while len(queue) > 0:
    m = queue.pop(0)

    reassign = {}
    for name, child in m.named_children():
      print(name)

      if hasattr(child, 'quantization_mode'):
        assert isinstance(child.quantization_mode, QuantizationMode)
        reassign[name] = applyQuantizationModePreparations(child, child.quantization_mode)

      queue.append(child)  # child is still reffered to inside the newly reassigned module

    for key in reassign:
      m._modules[key] = reassign[key]

  _removeQuantizationModes(module=module)

  setattr(module, 'quantization_state', QuantizationState.TRAIN_CALIBRATE)

  return tq.prepare(model=module,
                    observer_non_leaf_module_list=set(
                        tq.get_default_qat_module_mappings().values()),
                    inplace=True)


def applyConversionAfterModeMapping(module: torch.nn.Module, inplace: bool = False):
  module = tq.convert(module=module, inplace=inplace)
  setattr(module, 'quantization_state', QuantizationState.QUANTIZED)
  return module