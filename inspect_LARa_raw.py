from torch.utils.data import DataLoader

from common.data import (BatchAdditiveGaussianNoise, LARa, LARaClassLabelView, LARaIMUView,
                         LARaOptions, describeLARaLabels, CombineViews)

validation_data = LARa(window=300,
                       stride=300,
                       view=CombineViews(batch_view=LARaIMUView(['N']),
                                         labels_view=LARaClassLabelView()),
                       dynamic_transform=BatchAdditiveGaussianNoise(mu=0, sigma=1e-2),
                       opts=[LARaOptions.ALL_RUNS, LARaOptions.ALL_SUBJECTS])

loader = DataLoader(dataset=validation_data, batch_size=64, shuffle=True)

for batch, labels in loader:
  print(labels.shape)
  print(batch.shape)
  print(describeLARaLabels(labels))
  break