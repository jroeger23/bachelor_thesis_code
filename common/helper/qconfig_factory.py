from typing import Any, Dict

import torch.ao.quantization as tq


class ObserverPlaceholder():
  """Pickleable Observer factory
  """

  def __init__(self, klass, **kwargs) -> None:
    """Create a new ObserverPlaceholder with a class definition and kwargs.
    The class is stored as a fully quallified string, which will be imported
    when the class is requested.

    Args:
        klass (_type_): the observer class (either fully qualified string or class variable)
                        needs to implement with_args, if kwargs are given
    """
    self._class_name = klass if isinstance(klass,
                                           str) else f'{klass.__module__}.{klass.__qualname__}'
    self._kwargs = kwargs

  @staticmethod
  def _reconstructKwargs(argument_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v.getClassWithArgs() if isinstance(v, ObserverPlaceholder) else v
        for k, v in argument_dict.items()
    }

  def getClass(self) -> Any:
    """Get the class of the Observer

    Returns:
        Any: class variable
    """
    components = self._class_name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
      mod = getattr(mod, comp)
    return mod

  def getKwargs(self) -> Dict[str, Any]:
    """Get the kwargs dict of for the observer

    Returns:
        Dict[str, Any]: kwargs
    """
    return ObserverPlaceholder._reconstructKwargs(self._kwargs)

  def getClassWithArgs(self) -> Any:
    """Bind the kwargs to the observer and return the factory

    Returns:
        Any: observer factory
    """
    klass = self.getClass()
    kwargs = self.getKwargs()

    # Restore arguments with _PartialWrapper of torch.ao.quantization.observer
    if kwargs:
      klass = klass.with_args(**kwargs)

    return klass


class QConfigFactory():
  """A pickleable factory for qconfigs
  """

  def __init__(self, activation_quantizer: ObserverPlaceholder,
               weight_quantizer: ObserverPlaceholder) -> None:
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
    act_cls = self._activation_quantizer.getClassWithArgs()
    wgt_cls = self._weight_quantizer.getClassWithArgs()

    return tq.QConfig(activation=act_cls, weight=wgt_cls)
