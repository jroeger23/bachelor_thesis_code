from typing import Any, Dict

import torch.ao.quantization as tq


class GlobalPlaceholder():
  """Pickleable global variable/class factory
  """

  def __init__(self, global_name, **kwargs) -> None:
    """Create a new GlobalPlaceholder with a class definition and kwargs.
    The class is stored as a fully quallified string, which will be imported
    when the class is requested.

    Args:
        global_name (_type_): the class or global name (either fully qualified string or class variable)
                              needs to implement with_args, if kwargs are given
    """
    self._global_name = global_name if isinstance(
        global_name, str) else f'{global_name.__module__}.{global_name.__qualname__}'
    self._kwargs = kwargs

  @property
  def global_name(self):
    return self._global_name

  @staticmethod
  def _reconstructKwargs(argument_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v.getItemWithArgs() if isinstance(v, GlobalPlaceholder) else v
        for k, v in argument_dict.items()
    }

  def getItem(self) -> Any:
    """Get the stored global item

    Returns:
        Any: item or class factory
    """
    components = self._global_name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
      mod = getattr(mod, comp)
    return mod

  def getKwargs(self) -> Dict[str, Any]:
    """Get the kwargs dict of for the item

    Returns:
        Dict[str, Any]: kwargs
    """
    return GlobalPlaceholder._reconstructKwargs(self._kwargs)

  def getItemWithArgs(self) -> Any:
    """Bind the kwargs to the observer and return the factory

    Returns:
        Any: item or class factory
    """
    item = self.getItem()
    kwargs = self.getKwargs()

    # Restore arguments with _PartialWrapper of torch.ao.quantization.observer
    if kwargs:
      item = item.with_args(**kwargs)

    return item


class QConfigFactory():
  """A pickleable factory for qconfigs
  """

  def __init__(self, activation_quantizer: GlobalPlaceholder,
               weight_quantizer: GlobalPlaceholder) -> None:
    """Create a new QConfigFactory

    Args:
        activation_quantizer (ObserverPlaceholder): The observer factory to use for activations
        weight_quantizer (ObserverPlaceholder): The observer factory to use for weights
    """
    self._activation_quantizer = activation_quantizer
    self._weight_quantizer = weight_quantizer

  def getQConfig(self) -> tq.QConfig:
    """Construct a qconfig with the specified observers

    Returns:
        tq.QConfig: the new qconfig
    """
    act_cls = self._activation_quantizer.getItemWithArgs()
    wgt_cls = self._weight_quantizer.getItemWithArgs()

    return tq.QConfig(activation=act_cls, weight=wgt_cls)
