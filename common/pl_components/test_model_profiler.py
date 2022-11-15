from .model_profiler import actionMethod, actionIsLightningModule, Timer
import pytest


def test_actionMedhod():
  data = [
      ('name.method1', 'method1'),
      ('name.something.method 2', 'method 2'),
      ('method  3', 'method  3'),
  ]

  for i, o in data:
    assert actionMethod(i) == o


def test_actionIsLightningModule():
  data = [
      ('[LightningModule]', True),
      ('[NoLightningModule]', False),
      ('[LightningModule]Something', True),
      ('[NoLightningModule]Something', False),
      ('a[LightningModule]Something', False),
      ('a[NoLightningModule]Something', False),
      ('[LightningModule]  Something', True),
      ('[NoLightningModule]   Something', False),
  ]

  for i, o in data:
    assert actionIsLightningModule(i) == o


def test_timer():
  timer = Timer()
  pytest.raises(ValueError, timer.stop)
  timer.start()
  pytest.raises(ValueError, timer.start)
  assert timer.stop() > 0
