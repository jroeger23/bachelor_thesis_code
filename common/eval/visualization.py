import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from typing import List
from collections import defaultdict


def quantizedWeightHistogram(ax: plt.Axes,
                             q_weights: torch.Tensor,
                             q_min: int,
                             q_max: int,
                             cmap=cm.get_cmap('tab10')) -> None:
  assert q_weights.is_quantized, 'q_weights must be quantized'
  i_weights = q_weights.int_repr().flatten()
  mean_weight = i_weights.mean(dtype=torch.float)
  dist, _, _ = ax.hist(x=i_weights, bins=range(q_min, q_max + 1), color=cmap(0))
  ax.vlines(x=[mean_weight], ymin=0, ymax=max(dist), color=cmap(1))


def quantizationErrorHistorgram(ax: plt.Axes,
                                q_error: torch.Tensor,
                                n_bins: int,
                                cmap=cm.get_cmap('tab10')) -> None:

  hist, bins = q_error.histogram(bins=n_bins)
  width = (bins[-1] - bins[0]) / (len(bins) - 1)

  ax.bar(x=bins[:-1], height=hist, width=width)
  ax.vlines(x=[q_error.mean().item()], ymin=0, ymax=max(hist), color=cmap(1))
  ax.set_title(f'MSE={q_error.square().mean():.02e}')


def autoSortedRowTable(items: list, content_fn, row_id_fn, row_order_desc, col_id_fn, col_ord_fn, col_ord_desc,
                       reduce_fn, n_cols) -> List[list]:
  table = defaultdict(lambda: defaultdict(list))

  # put items into rows
  for i in items:
    table[row_id_fn(i)][col_id_fn(i)].append(i)

  # reduce to best representative
  for rkey in table:
    for ckey in table[rkey]:
      table[rkey][ckey] = reduce_fn(table[rkey][ckey])

  # order table
  ret_table = []
  for rkey in sorted(table.keys(), reverse=row_order_desc):
    row = [rkey]
    row.extend( map(content_fn, sorted(table[rkey].values(), key=col_ord_fn, reverse=col_ord_desc)[:n_cols]))
    ret_table.append(row)

  return ret_table