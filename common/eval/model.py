import torch
import torch.nn.quantized.modules as tqnn
import torch.nn.intrinsic.quantized.modules.conv_relu as tnniqc
import torch.nn.intrinsic.quantized.modules.linear_relu as tnniql
from typing import Dict, Any
from torch.nn.utils.parametrize import type_before_parametrizations


def _fpLinearEstim(module: torch.nn.Linear, _: int) -> int:
  return (module.weight.numel() + module.bias.numel()) * 32


def _fpConv2dEstim(module: torch.nn.Conv2d, _: int) -> int:
  return (module.weight.numel() + module.bias.numel() if module.bias is not None else 0) * 32


def _qLinearEstim(module: tqnn.Linear, n_bits: int) -> int:
  return (module.weight().numel() + module.bias().numel()) * n_bits


def _qConv2dEstim(module: tqnn.Conv2d, n_bits: int) -> int:
  return (module.weight().numel() + module.bias().numel()) * n_bits


ESTIM_BY_MODULE: Dict[Any, Any] = {
    torch.nn.Linear: _fpLinearEstim,
    torch.nn.Conv2d: _fpConv2dEstim,
    tqnn.Linear: _qLinearEstim,
    tqnn.Conv2d: _qConv2dEstim,
    tnniql.LinearReLU: _qLinearEstim,
    tnniqc.ConvReLU2d: _qConv2dEstim,
}


def estimateModuleSize(module: torch.nn.Module, n_bits: int) -> int:
  memory = 0
  for _, m in module.named_modules():
    if type_before_parametrizations(m) in ESTIM_BY_MODULE:
      estim = ESTIM_BY_MODULE[type_before_parametrizations(m)]
      memory += estim(m, n_bits)
  return memory