from common.data import Pamap2DataModule, OpportunityDataModule, LARaDataModule
from common.data.data_modules import default_pamap2_static_transform
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

pamap2 = Pamap2DataModule(batch_size=1,
                          window=100,
                          stride=33,
                          sample_frequency=30,
                          use_transient_class=False)
pamap2.setup('fit')
pamap2.setup('validate')
pamap2.setup('test')

opportunity = OpportunityDataModule(batch_size=1, window=24, stride=12, blank_invalid_columns=True)
opportunity.setup('fit')
opportunity.setup('validate')
opportunity.setup('test')

lara = LARaDataModule(batch_size=1, window=100, stride=12, sample_frequency=30)
lara.setup('fit')
lara.setup('validate')
lara.setup('test')


def classCount(dataset, n_classes):
  bins = defaultdict(lambda: 0)
  for _, label in dataset:
    bins[label.item()] += 1

  return np.array(list(map(lambda i: bins[i], range(n_classes))))


pamap2_names = [
    #'Übergangsaktivitäten',
    'Liegen',
    'Sitzen',
    'Stehen',
    'Laufen',
    'Rennen',
    'Fahrrad fahren',
    'Nordic Walking',
    'Treppen hinaufgehen',
    'Treppen hinabgehen',
    'Staubsaugen',
    'Seilspringen',
]
opportunity_names = [
    'Andere',
    'Stehen',
    'Laufen',
    'Sitzen',
    'Liegen',
]
lara_names = [
    'Stehen',
    'Laufen',
    'Wagen schieben',
    'Bedienen (oben)',
    'Bedienen (mittig)',
    'Bedienen (unten)',
    'Synchronisation',
    'Andere',
]

pamap2_counts = {
    'Training': classCount(pamap2.train_set, len(pamap2_names)),
    'Validierung': classCount(pamap2.val_set, len(pamap2_names)),
    'Test': classCount(pamap2.test_set, len(pamap2_names)),
}

opportunity_counts = {
    'Training': classCount(opportunity.train_set, len(opportunity_names)),
    'Validierung': classCount(opportunity.val_set, len(opportunity_names)),
    'Test': classCount(opportunity.test_set, len(opportunity_names)),
}

lara_counts = {
    'Training': classCount(lara.train_set, len(lara_names)),
    'Validierung': classCount(lara.val_set, len(lara_names)),
    'Test': classCount(lara.test_set, len(lara_names)),
}


def mkPlot(names, counts):
  fig, ax = plt.subplots()
  fig.set_dpi(300)
  fig.set_figwidth(fig.get_figwidth() * 1.5)

  btm = None
  for mode, cnt in counts.items():
    btm = np.zeros(cnt.shape) if btm is None else btm
    ax.bar(x=names, height=cnt, bottom=btm, label=mode)
    btm += cnt

  ax.set_ylabel('# Segmente')
  ax.legend()
  fig.autofmt_xdate()

  return fig


mkPlot(pamap2_names, pamap2_counts)
mkPlot(opportunity_names, opportunity_counts)
mkPlot(lara_names, lara_counts)
plt.show()
