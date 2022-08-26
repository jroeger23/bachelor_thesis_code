import torch
import common.data
import logging
import tqdm
from itertools import islice
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
writer = SummaryWriter("test_experiment")
batch_cnt = 0

with tqdm.tqdm(desc='Training', unit='epoch', iterable=range(3)) as e_bar:
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

        batch_cnt += 1
        writer.add_scalar(tag="train/loss", scalar_value=loss.item(), global_step=batch_cnt)

        if i%50 == 0:
          b_bar.set_postfix_str(f'loss={loss.item():.02e}')

          hit = 0
          cnt = 0

          test_probs = []
          test_labels = []
          with torch.no_grad():
            for x, y in islice(validation_loader, 50):
              logits = m(x)
              test_labels.append(torch.eye(10)[y])
              test_probs.append(torch.nn.functional.softmax(logits, dim=1))
              writer.add_scalar(tag="test/loss", scalar_value=loss_fn(logits, y), global_step=batch_cnt)
              hit += (logits.argmax(1) == y).type(torch.float).sum().item()
              cnt += len(x)

          test_probs = torch.row_stack(test_probs)
          test_labels = torch.row_stack(test_labels)

          for label in range(10):
            probs = test_probs[:, label]
            labels = test_labels[:, label]
            writer.add_pr_curve(tag=f'pr_curve_{label}', labels=labels, predictions=probs, global_step=batch_cnt)

          acc = hit/cnt

          e_bar.set_postfix_str(f'acc={acc*100:.02f}%')
          writer.add_scalar(tag="test/acc", scalar_value=acc, global_step=batch_cnt)