from common.data import LARa, LARaOptions
import logging

logging.getLogger().setLevel(logging.DEBUG)

validation_data = LARa(
  root='/home/jonas/Stuff/Datasets',
  window=300,
  stride=300,
  opts=[LARaOptions.ALL_RUNS, LARaOptions.ALL_SUBJECTS]
  )
