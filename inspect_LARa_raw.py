from torch.utils.data import DataLoader

from common.data import (ComposeTransforms, LARa, LARaClassLabelView, LARaIMUView, LARaOptions,
                         describeLARaLabels)

validation_data = LARa(window=300,
                       stride=300,
                       transform=ComposeTransforms(LARaClassLabelView(), LARaIMUView(['N'])),
                       opts=[LARaOptions.ALL_RUNS, LARaOptions.ALL_SUBJECTS])

loader = DataLoader(dataset=validation_data, batch_size=64, shuffle=True)

for batch, labels in loader:
  print(labels.shape)
  print(batch.shape)
  print(describeLARaLabels(labels))
  break