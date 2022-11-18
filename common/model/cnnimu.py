import pytorch_lightning as pl
import torch
import typing as t
import logging

logger = logging.getLogger(__name__)


class CNNIMUBlock(pl.LightningModule):
  """ A CNN-IMU Block with two 5x1 convolutional layers and one 2x1 max-pooling layer for 64 channels.
  """

  def __init__(self, first_block: bool = False, conv_channels: int = 64):
    """ Create a CNNIMU Block

    Args:
        first_block (bool, optional): Marks this block as the first block in a Convolution Stack,
                                      meaning it has one input channel instead of 64.
                                      Defaults to False.
        conv_channels (int, optional): The number of convolition channels. Defaults to 64.
    """
    super().__init__()
    channels = conv_channels
    in_channels = 1 if first_block else channels
    self.conv1 = torch.nn.Conv2d(in_channels=in_channels,
                                 out_channels=channels,
                                 kernel_size=(5, 1),
                                 stride=1,
                                 padding=(2, 0))
    self.relu1 = torch.nn.ReLU()
    self.conv2 = torch.nn.Conv2d(in_channels=channels,
                                 out_channels=channels,
                                 kernel_size=(5, 1),
                                 stride=1,
                                 padding=(2, 0))
    self.relu2 = torch.nn.ReLU()
    self.pool = torch.nn.MaxPool2d(kernel_size=(2, 1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """ Forward pass an 2D column-wise (T x D x C) IMU feature matrix

    Args:
        x (torch.Tensor): The IMU feature matrix (TxDxC)

    Returns:
        torch.Tensor: The output features (T//2 x D x C)
    """
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.conv2(x)
    x = self.relu2(x)
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

  def __init__(self, n_blocks: int, conv_channels: int) -> None:
    """Create a new CNN-IMU colvolution stack

    Args:
        n_blocks (int): the number of CNN-IMU blocks to chain
        conv_channels (int): the number of inner channels
    """
    super().__init__()
    self.blocks = [
        CNNIMUBlock(first_block=False, conv_channels=conv_channels) for _ in range(n_blocks - 1)
    ]
    self.blocks.insert(0, CNNIMUBlock(first_block=True, conv_channels=conv_channels))
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

  def __init__(self,
               n_blocks: int,
               imu_sizes: t.List[int],
               sample_length: int,
               n_classes: int,
               conv_channels: int = 64,
               fc_features: int = 512,
               **extra_hyper_params) -> None:
    """Create a new CNN-IMU. With a given block depth, IMU data columns, sample length,
       number of convolution channels and a number of fully-connected features per layer

    Args:
        n_blocks (int): the number of CNN-IMU blocks per IMU
        imu_sizes (t.List[int]): A List of all IMU column sizes (D) corresponding to the order of the
                                 forward() parameter
        sample_length (int): the length of all samples (T)
        n_classes (int): the number of output classes
        conv_channels (int, optional): The number of conv channles to use in each block. Defaults to 64.
        fc_features (int, optional): Number of fc_featues at each imu output and inside the classification
                                     stack. Defaults to 512.
    """
    super().__init__()
    self.extra_hyper_params = extra_hyper_params

    # All IMU conv pipelines
    pipelines = [CNNIMUPipeline(n_blocks=n_blocks, conv_channels=conv_channels) for _ in imu_sizes]

    # Their output shapes
    pipe_output_shapes = [
        p.shape_transform((sample_length, d)) for p, d in zip(pipelines, imu_sizes)
    ]

    # The number of features
    pipe_output_features = [t * d * conv_channels for t, d in pipe_output_shapes]

    # The number of total features
    total_features = len(imu_sizes) * fc_features

    # Torch modules in order
    self.pipelines = torch.nn.ModuleList(pipelines)
    self.pipe_fc = torch.nn.ModuleList([
        torch.nn.Sequential(torch.nn.Flatten(),
                            torch.nn.Linear(in_features=i, out_features=fc_features),
                            torch.nn.ReLU()) for i in pipe_output_features
    ])
    self.fuse = FuseModule()
    self.fc = torch.nn.Sequential(
        torch.nn.Dropout(),
        torch.nn.Linear(in_features=total_features, out_features=fc_features),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(in_features=fc_features, out_features=fc_features),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=fc_features, out_features=n_classes),
    )

    logger.info(f'Set up CNNIMU for {n_classes} classes with convolution setup:')
    for ix, (i_d, (o_t, o_d)) in enumerate(zip(imu_sizes, pipe_output_shapes)):
      logger.info(
          f'  - IMU{ix} (T={sample_length}, D={i_d}) -> {n_blocks} blocks -> (T={o_t}, D={o_d}, C={conv_channels})'
      )
    logger.info(f'And a fully-connected feature width of {fc_features}')

  def forward(self, imu_x: t.List[torch.Tensor]) -> torch.Tensor:
    """Forward pass a list of IMU data batches

    Args:
        imu_x (t.List[torch.Tensor]): a list of imu data batches (bs x T x D)

    Returns:
        torch.Tensor: the prediction logits for each class
    """
    pipe_outputs = [p(x[:, None, :, :]) for p, x in zip(self.pipelines, imu_x)]
    pipe_fc_outputs = [p_fc(p_o) for p_fc, p_o in zip(self.pipe_fc, pipe_outputs)]
    combined = self.fuse(pipe_fc_outputs)
    y = self.fc(combined)
    return y

  def training_step(self, batch: t.Tuple[t.List[torch.Tensor], torch.Tensor],
                    batch_ix) -> torch.Tensor:
    imu_x, labels = batch

    y_logits = self(imu_x)
    loss = torch.nn.functional.cross_entropy(input=y_logits, target=labels)

    self.log('train/loss', loss)

    return loss

  def validation_step(self, batch: t.Tuple[t.List[torch.Tensor], torch.Tensor],
                      batch_ix) -> torch.Tensor:
    imu_x, labels = batch

    y_logits: torch.Tensor = self(imu_x)
    loss = torch.nn.functional.cross_entropy(input=y_logits, target=labels)
    probs = torch.nn.functional.softmax(input=y_logits, dim=1)

    self.log('validation/loss', loss, prog_bar=True)

    if hasattr(self, 'validation_labels') and isinstance(self.validation_labels, list):
      self.validation_labels.append(labels)
    if hasattr(self, 'validation_probs') and isinstance(self.validation_probs, list):
      self.validation_probs.append(probs)

    return loss

  def test_step(self, batch: t.Tuple[t.List[torch.Tensor], torch.Tensor], batch_ix) -> torch.Tensor:
    imu_x, labels = batch

    y_logits: torch.Tensor = self(imu_x)
    probs = torch.nn.functional.softmax(input=y_logits, dim=1)
    loss = torch.nn.functional.cross_entropy(input=y_logits, target=labels)

    hits = (y_logits.argmax(dim=1) == labels).type(torch.float).sum()

    self.log('test/loss', loss)

    if hasattr(self, 'test_labels') and isinstance(self.test_labels, list):
      self.test_labels.append(labels)
    if hasattr(self, 'test_probs') and isinstance(self.test_probs, list):
      self.test_probs.append(probs)

    return loss

  def predict_step(self, batch: t.Tuple[t.List[torch.Tensor], torch.Tensor],
                   batch_ix) -> torch.Tensor:
    imu_x, _ = batch
    return self(imu_x)

  def configure_optimizers(self):
    if 'optimizer' not in self.extra_hyper_params:
      return torch.optim.Adam(params=self.parameters())

    if self.extra_hyper_params['optimizer'] == 'Adam':
      adapt = {
          k: self.extra_hyper_params[k]
          for k in ('lr', 'betas', 'momentum')
          if k in self.extra_hyper_params
      }
      return torch.optim.Adam(params=self.parameters(), **adapt)
    elif self.extra_hyper_params['optimizer'] == 'RMSProp':
      adapt = {
          k: self.extra_hyper_params[k]
          for k in ('lr', 'alpha', 'weight_decay', 'momentum')
          if k in self.extra_hyper_params
      }
      return torch.optim.RMSprop(params=self.parameters(), **adapt)