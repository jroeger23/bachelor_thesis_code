import os
import re
import typing as t

import inquirer


def _autoExperimentName(inquiry_dict: t.Dict):
  path = inquiry_dict['log_dir']

  m_times = [
      (node.stat().st_mtime_ns, node.name) for node in os.scandir(path=path) if node.is_dir()
  ] if os.path.exists(path=path) else []

  name = max(m_times)[1] if m_times != [] else 'experiment'
  name = inquirer.Text(name='name', message='Choose a new Experiment name', default=name)
  name = inquirer.prompt([name])['name']

  inquiry_dict['experiment'] = name


def _autoVersion(inquiry_dict: t.Dict):
  path = os.path.join(inquiry_dict['log_dir'], inquiry_dict['experiment'])

  numbers = [
      int(node.name.removeprefix(f'version_'))
      for node in os.scandir(path=path)
      if node.is_file() and re.match('version_[0-9]+')
  ] if os.path.exists(path=path) else []

  new_number = max(numbers) + 1 if numbers != [] else 0

  inquiry_dict['version'] = f'version_{new_number}'


def modeDialog(log_dir='logs'):
  inquiry_dict = {'log_dir': log_dir}

  modes = ['quantize_checkpoint', 'test_checkpoint', 'train_new']

  mode = inquirer.List(name='mode', message='Select mode', choices=modes)
  mode = inquirer.prompt([mode])['mode']

  inquiry_dict['mode'] = mode

  if mode == modes[0] or mode == modes[1]:
    experimentLoadDialog(inquiry_dict)
    versionLoadDialog(inquiry_dict)
    checkpointLoadDialog(inquiry_dict)

  elif mode == modes[2]:
    _autoExperimentName(inquiry_dict)
    _autoVersion(inquiry_dict)

  if mode == modes[0]:
    inquiry_dict['version'] = inquiry_dict['version'] + '_q'

  return inquiry_dict


def experimentLoadDialog(inquiry_dict: t.Dict):
  path = inquiry_dict['log_dir']

  experiments = [os.path.basename(node) for node in os.scandir(path=path) if node.is_dir()]
  experiment = inquirer.List(name='experiment', message='Experiment to load', choices=experiments)
  experiment = inquirer.prompt([experiment])['experiment']

  inquiry_dict['experiment'] = experiment


def versionLoadDialog(inquiry_dict: t.Dict):
  path = os.path.join(inquiry_dict['log_dir'], inquiry_dict['experiment'])

  versions = [os.path.basename(node) for node in os.scandir(path=path) if node.is_dir()]
  version = inquirer.List(name='version', message='Version to load', choices=versions)
  version = inquirer.prompt([version])['version']

  inquiry_dict['version'] = version


def checkpointLoadDialog(inquiry_dict: t.Dict):
  path = os.path.join(inquiry_dict['log_dir'], inquiry_dict['experiment'], inquiry_dict['version'],
                      "checkpoints")

  checkpoints = [os.path.basename(node) for node in os.scandir(path=path)]
  checkpoint = inquirer.List(name='checkpoint', message='Checkpoint to load', choices=checkpoints)
  checkpoint = inquirer.prompt([checkpoint])['checkpoint']

  inquiry_dict['checkpoint'] = os.path.join(path, checkpoint)


if __name__ == '__main__':
  print(f'Result <{modeDialog()}>')