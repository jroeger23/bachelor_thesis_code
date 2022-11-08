import logging

import torchvision


def mnist():
  logger = logging.getLogger(__name__)

  transform = torchvision.transforms.ToTensor()

  train = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transform)
  test = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transform)

  labels = [
      'zero',
      'one',
      'two',
      'three',
      'four',
      'five',
      'six',
      'seven',
      'eight',
      'nine',
  ]

  logger.info('Using MNIST Dataset')
  logger.debug(f'  #Train = {len(train)}, #Test = {len(test)}')
  logger.debug(f'  ImageShape = {train.data[0].shape}')
  logger.debug(f'  Labels = {labels}')

  return train, test, labels
