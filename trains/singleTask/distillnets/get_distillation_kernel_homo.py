"""Graph distillation for homo GD"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from ..utils import distance_metric, min_cosine

class DistillationKernel(nn.Module):
  """Graph Distillation kernel.

  Calculate the edge weights e_{j->k} for each j. Modality k is specified by
  to_idx, and the other modalities are specified by from_idx.
  """

  def __init__(self, n_classes, hidden_size, gd_size, to_idx, from_idx,
               gd_prior, gd_reg, w_losses, metric, alpha, hyp_params):
    super(DistillationKernel, self).__init__()
    self.W_logit = nn.Linear(n_classes, gd_size)
    self.W_repr = nn.Linear(hidden_size, gd_size)
    self.W_edge = nn.Linear(gd_size * 4, 1)

    self.gd_size = gd_size
    self.to_idx = to_idx
    self.from_idx = from_idx
    self.alpha = alpha
    self.gd_prior = Variable(torch.FloatTensor(gd_prior).cuda())
    self.gd_reg = gd_reg
    self.w_losses = w_losses
    self.metric = metric
    self.hyp_params = hyp_params


  def forward(self, logits, reprs):
    """
    Args:
      logits: (n_modalities, batch_size, n_classes)
      reprs: (n_modalities, batch_siz`, hidden_size)
    Return:
      edges: weights e_{j->k} (n_modalities_from, batch_size)
    """
    n_modalities, batch_size = logits.size()[:2]
    z_logits = self.W_logit(logits.view(n_modalities * batch_size, -1))
    z_reprs = self.W_repr(reprs.view(n_modalities * batch_size, -1))
    z = torch.cat(
        (z_logits, z_reprs), dim=1).view(n_modalities, batch_size,
                                         self.gd_size * 2)


    edges = []
    for j in self.to_idx:
      for i in self.from_idx:
        if i == j:
          continue
        else:
          # To calculate e_{j->k}, concatenate z^j, z^k
          e = self.W_edge(torch.cat((z[j], z[i]), dim=1))
          edges.append(e)
    edges = torch.cat(edges, dim=1)
    edges_origin = edges.sum(0).unsqueeze(0).transpose(0, 1)  # original value of edges
    edges = F.softmax(edges * self.alpha, dim=1).transpose(0, 1)  # normalized value of edges
    return edges, edges_origin


  def distillation_loss(self, logits, reprs, edges):
    """Calculate graph distillation losses, which include:
    regularization loss, loss for logits, and loss for representation.
    """
    loss_reg = (edges.mean(1) - self.gd_prior).pow(2).sum() * self.gd_reg


    loss_logit, loss_repr = 0, 0
    x = 0
    for j in self.to_idx:
      for i, idx in enumerate(self.from_idx):
        if i == j:
          continue
        else:
          w_distill = edges[x] + self.gd_prior[x]
          # print(edges.sum(1), w_distill.sum(0))
          loss_logit += self.w_losses[0] * distance_metric(
            logits[j], logits[idx], self.metric, w_distill)
          loss_repr += self.w_losses[1] * distance_metric(
            reprs[j], reprs[idx], self.metric, w_distill)
          x = x + 1
    return loss_reg, loss_logit, loss_repr


def get_distillation_kernel(n_classes,
                            hidden_size,
                            gd_size,
                            to_idx,
                            from_idx,
                            gd_prior,
                            gd_reg,
                            w_losses,
                            metric,
                            alpha=1 / 8):
  return DistillationKernel(n_classes, hidden_size, gd_size, to_idx, from_idx,
                            gd_prior, gd_reg, w_losses, metric, alpha)
