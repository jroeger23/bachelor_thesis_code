import torch
import common.data
import logging
import tqdm
from torch.utils.data import DataLoader

logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

train_data, test_data, labels = common.data.mnist()


m = torch.nn.Sequential(
  torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(5,5), padding=(2,2)),
  torch.nn.MaxPool2d(kernel_size=(2,2), stride=2),
  torch.nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(5,5), padding=(2,2)),
  torch.nn.MaxPool2d(kernel_size=(2,2), stride=2),
  torch.nn.Flatten(),
  torch.nn.Linear(in_features=7*7*5, out_features=49),
  torch.nn.ReLU(),
  torch.nn.Linear(in_features=49, out_features=10),
)


logger.debug(f'Model: {m}')

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=m.parameters())

with tqdm.tqdm(desc='Training', unit='epoch', iterable=range(10)) as e_bar:
  for epoch in e_bar:
    train_loader = DataLoader(dataset=train_data, batch_size=60, shuffle=True)
    validation_loader = DataLoader(dataset=test_data, batch_size=60, shuffle=True)

    with tqdm.tqdm(desc='Epoch', unit='batch', iterable=train_loader, leave=False) as b_bar:
      for i, (data, labels) in enumerate(b_bar):
        optimizer.zero_grad()
        logits = m(data)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        if i%50 == 0:
          b_bar.set_postfix_str(f'loss={loss.item():.02e}')


    hit = 0
    cnt = 0

    with torch.no_grad():
      for x, y in tqdm.tqdm(desc="Validating", iterable=validation_loader, unit="batch", leave=False):
        logits = m(x)
        hit += (logits.argmax(1) == y).type(torch.float).sum().item()
        cnt += len(x)

    e_bar.set_postfix_str(f'acc={hit/cnt*100:.02f}%')

