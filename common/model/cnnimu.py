import pytorch_lightning as pl
import torch
import typing as t


class CNNIMUBlock(pl.LightningModule):

  def __init__(self, first_block: bool = False):
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
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.pool(x)
    return x

  def shape_transform(self, shape: t.Tuple[int, int]) -> t.Tuple[int, int]:
    t, d = shape
    return t // 2, d


class CNNIMUPipeline(pl.LightningModule):

  def __init__(self, n_blocks: int) -> None:
    super().__init__()
    self.blocks = [CNNIMUBlock(first_block=False) for _ in range(n_blocks - 1)]
    self.blocks.insert(0, CNNIMUBlock(first_block=True))
    self.blocks = torch.nn.ModuleList(self.blocks)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for block in self.blocks:
      x = block(x)

    return x

  def shape_transform(self, shape: t.Tuple[int, int]) -> t.Tuple[int, int]:
    for block in self.blocks:
      shape = block.shape_transform(shape)

    return shape


class FuseModule(pl.LightningModule):

  def __init__(self) -> None:
    super().__init__()
    self.flatten = torch.nn.Flatten()

  def forward(self, feature_maps: t.List[torch.Tensor]) -> torch.Tensor:
    combined_features = torch.stack(feature_maps, dim=-1)
    return self.flatten(combined_features)


class CNNIMU(pl.LightningModule):

  def __init__(self, n_blocks: int, imu_sizes: t.List[int], sample_length: int,
               n_classes: int) -> None:
    super().__init__()
    self.pipelines = torch.nn.ModuleList([CNNIMUPipeline(n_blocks=n_blocks) for _ in imu_sizes])

    pipe_output_shapes = [
        p.shape_transform((sample_length, d)) for p, d in zip(self.pipelines, imu_sizes)
    ]
    fc_in = sum([t * d for t, d in pipe_output_shapes]) * 64
    fc_width = 512

    self.fuse = FuseModule()

    self.fc = torch.nn.Sequential(
        torch.nn.Dropout(),
        torch.nn.Linear(in_features=fc_in, out_features=fc_width),
        torch.nn.Dropout(),
        torch.nn.Linear(in_features=fc_width, out_features=fc_width),
        torch.nn.Linear(in_features=fc_width, out_features=n_classes),
    )

  def forward(self, imu_x: t.List[torch.Tensor]) -> torch.Tensor:
    pipe_outputs = [p(x[:, None, :, :]) for p, x in zip(self.pipelines, imu_x)]
    combined = self.fuse(pipe_outputs)
    y = self.fc(combined)
    return y

  def training_step(self, batch: t.Tuple[t.List[torch.Tensor], torch.Tensor],
                    batch_ix) -> torch.Tensor:
    imu_x, labels = batch

    y_logits = self(imu_x)
    loss = torch.nn.functional.cross_entropy(input=y_logits, target=labels)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(params=self.parameters())
