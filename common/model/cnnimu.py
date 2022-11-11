import pytorch_lightning as pl
import torch
import typing as t
from common.metrics import wF1Score


class CNNIMUBlock(pl.LightningModule):
  """ A CNN-IMU Block with two 5x1 convolutional layers and one 2x1 max-pooling layer for 64 channels.
  """

  def __init__(self, first_block: bool = False):
    """ Create a CNNIMU Block

    Args:
        first_block (bool, optional): Marks this block as the first block in a Convolution Stack,
                                      meaning it has one input channel instead of 64.
                                      Defaults to False.
    """
    super().__init__()
    channels = 64
    in_channels = 1 if first_block else channels
    self.conv1 = torch.nn.Conv2d(in_channels=in_channels,
                                 out_channels=channels,
                                 kernel_size=(5, 1),
                                 stride=1,
                                 padding=(2, 0))
    self.conv2 = torch.nn.Conv2d(in_channels=channels,
                                 out_channels=channels,
                                 kernel_size=(5, 1),
                                 stride=1,
                                 padding=(2, 0))
    self.pool = torch.nn.MaxPool2d(kernel_size=(2, 1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """ Forward pass an 2D column-wise (T x D x C) IMU feature matrix

    Args:
        x (torch.Tensor): The IMU feature matrix (TxDxC)

    Returns:
        torch.Tensor: The output features (T//2 x D x C)
    """
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.pool(x)
    return x

  def shape_transform(self, shape: t.Tuple[int, int]) -> t.Tuple[int, int]:
    """ Transform a shape, like forward() would do.

    Args:
        shape (t.Tuple[int, int]): The Shape (T x D) to transform

    Returns:
        t.Tuple[int, int]: (T//2 x D)
    """
    t, d = shape
    return t // 2, d


class CNNIMUPipeline(pl.LightningModule):
  """A CNN-IMU IMU convolution stack consisting of n CNN-IMU blocks
  """

  def __init__(self, n_blocks: int) -> None:
    """Create a new CNN-IMU colvolution stack

    Args:
        n_blocks (int): the number of CNN-IMU blocks to chain
    """
    super().__init__()
    self.blocks = [CNNIMUBlock(first_block=False) for _ in range(n_blocks - 1)]
    self.blocks.insert(0, CNNIMUBlock(first_block=True))
    self.blocks = torch.nn.ModuleList(self.blocks)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass 2D column-wise (T x D) IMU data to extract features

    Args:
        x (torch.Tensor): 2D column-wise (T x D) IMU data

    Returns:
        torch.Tensor: IMU feature maps (T x D//2^n x 64)
    """
    for block in self.blocks:
      x = block(x)

    return x

  def shape_transform(self, shape: t.Tuple[int, int]) -> t.Tuple[int, int]:
    """Transform a shape, as if it would be transformed with forward()

    Args:
        shape (t.Tuple[int, int]): (T x D)

    Returns:
        t.Tuple[int, int]: (T//2^n x D)
    """
    for block in self.blocks:
      if not isinstance(block, CNNIMUBlock):
        raise TypeError("Block is not of type {CNNIMUBlock}")

      shape = block.shape_transform(shape)

    return shape


class FuseModule(pl.LightningModule):
  """ NN Module to fuse a list of feature maps into a flat tensor (preserves batches)
  """

  def __init__(self) -> None:
    super().__init__()
    self.flatten = torch.nn.Flatten()

  def forward(self, feature_maps: t.List[torch.Tensor]) -> torch.Tensor:
    """Fuse a list of feature maps into a flat tensor (preserves batches)

    Args:
        feature_maps (t.List[torch.Tensor]): the list of feature maps (batched)

    Returns:
        torch.Tensor: 1D Tensor (batched)
    """
    flat_features = [self.flatten(features) for features in feature_maps]
    return torch.column_stack(flat_features)


class CNNIMU(pl.LightningModule):
  """The CNN-IMU Model. There are CNN-IMU colvolutional stacks for each IMU, which are eventually
  classified by a fully-connected softmax classifier.
  """

  def __init__(self, n_blocks: int, imu_sizes: t.List[int], sample_length: int,
               n_classes: int) -> None:
    """Create a new CNN-IMU. With a given block depth, IMU data columns and a sample length

    Args:
        n_blocks (int): the number of CNN-IMU blocks per IMU
        imu_sizes (t.List[int]): A List of all IMU column sizes (D) corresponding to the order of the
                                 forward() parameter
        sample_length (int): the length of all samples (T)
        n_classes (int): the number of output classes
    """
    super().__init__()
    pipelines = [CNNIMUPipeline(n_blocks=n_blocks) for _ in imu_sizes]

    pipe_output_shapes = [
        p.shape_transform((sample_length, d)) for p, d in zip(pipelines, imu_sizes)
    ]
    fc_in = sum([t * d for t, d in pipe_output_shapes]) * 64
    fc_width = 512

    self.pipelines = torch.nn.ModuleList(pipelines)
    self.fuse = FuseModule()
    self.fc = torch.nn.Sequential(
        torch.nn.Dropout(),
        torch.nn.Linear(in_features=fc_in, out_features=fc_width),
        torch.nn.Dropout(),
        torch.nn.Linear(in_features=fc_width, out_features=fc_width),
        torch.nn.Linear(in_features=fc_width, out_features=n_classes),
    )

  def forward(self, imu_x: t.List[torch.Tensor]) -> torch.Tensor:
    """Forward pass a list of IMU data batches

    Args:
        imu_x (t.List[torch.Tensor]): a list of imu data batches (bs x T x D)

    Returns:
        torch.Tensor: the prediction logits for each class
    """
    pipe_outputs = [p(x[:, None, :, :]) for p, x in zip(self.pipelines, imu_x)]
    combined = self.fuse(pipe_outputs)
    y = self.fc(combined)
    return y

  def training_step(self, batch: t.Tuple[t.List[torch.Tensor], torch.Tensor],
                    batch_ix) -> torch.Tensor:
    imu_x, labels = batch

    y_logits = self(imu_x)
    loss = torch.nn.functional.cross_entropy(input=y_logits, target=labels)

    self.log('train/loss', loss)

    return loss

  def on_validation_epoch_start(self) -> None:
    self.validation_labels = []
    self.validation_probs = []

  def validation_step(self, batch: t.Tuple[t.List[torch.Tensor], torch.Tensor],
                      batch_ix) -> torch.Tensor:
    imu_x, labels = batch

    y_logits: torch.Tensor = self(imu_x)
    loss = torch.nn.functional.cross_entropy(input=y_logits, target=labels)
    probs = torch.nn.functional.softmax(input=y_logits, dim=1)

    hits = (y_logits.argmax(dim=1) == labels).type(torch.float).sum()

    self.log('validation/loss', loss, prog_bar=True)
    self.log('validation/acc', hits / len(y_logits))
    self.validation_labels.append(labels)
    self.validation_probs.append(probs)

    return loss

  def on_validation_epoch_end(self) -> None:
    validation_labels = torch.concat(self.validation_labels)
    validation_probs = torch.row_stack(self.validation_probs)
    n_classes = validation_probs.shape[1]

    self.log('validation/wf1',
             wF1Score(validation_labels,
                      torch.eye(n_classes)[validation_probs.argmax(dim=1)]))

  def on_test_epoch_start(self) -> None:
    self.test_labels = []
    self.test_probs = []

  def test_step(self, batch: t.Tuple[t.List[torch.Tensor], torch.Tensor], batch_ix) -> torch.Tensor:
    imu_x, labels = batch

    y_logits: torch.Tensor = self(imu_x)
    probs = torch.nn.functional.softmax(input=y_logits, dim=1)
    loss = torch.nn.functional.cross_entropy(input=y_logits, target=labels)

    hits = (y_logits.argmax(dim=1) == labels).type(torch.float).sum()

    self.log('test/loss', loss)
    self.log('test/acc', hits / len(y_logits))

    self.test_probs.append(probs)
    self.test_labels.append(labels)

    return loss

  def on_test_epoch_end(self) -> None:
    test_probs = torch.row_stack(self.test_probs)
    test_labels = torch.concat(self.test_labels)
    n_classes = test_probs.shape[1]

    self.log('test/wf1', wF1Score(test_labels, torch.eye(n_classes)[test_probs.argmax(dim=1)]))

  def predict_step(self, batch: t.Tuple[t.List[torch.Tensor], torch.Tensor],
                   batch_ix) -> torch.Tensor:
    imu_x, _ = batch
    return self(imu_x)

  def configure_optimizers(self):
    return torch.optim.Adam(params=self.parameters())
