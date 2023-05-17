import warnings
from abc import ABC
from typing import Optional

import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy, cross_entropy, kl_div
from torch_geometric.nn import MessagePassing


class NeighborEmbedding(MessagePassing, ABC):
    def __init__(self, hidden_channels, num_rbf, cutoff_lower, cutoff_upper):
        super(NeighborEmbedding, self).__init__(aggr="add")
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = x
        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(GaussianSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(self.cutoff_lower, self.cutoff_upper, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                    torch.cos(
                        math.pi
                        * (
                                2
                                * (distances - self.cutoff_lower)
                                / (self.cutoff_upper - self.cutoff_lower)
                                + 1.0
                        )
                    )
                    + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class Distance(nn.Module):
    def __init__(
            self,
            cutoff_lower,
            cutoff_upper,
            return_vecs=False,
            loop=False,
    ):
        super(Distance, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.return_vecs = return_vecs
        self.loop = loop

    def forward(self, pos, edge_index):
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        mask: Optional[torch.Tensor] = None
        if self.loop:
            # mask out self loops when computing distances because
            # the norm of 0 produces NaN gradients
            # NOTE: might influence force predictions as self loop gradients are ignored
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device, dtype=edge_vec.dtype)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        lower_mask = edge_weight >= self.cutoff_lower
        if self.loop and mask is not None:
            # keep self loops even though they might be below the lower cutoff
            lower_mask = lower_mask | ~mask
        edge_index = edge_index[:, lower_mask]
        edge_weight = edge_weight[lower_mask]

        if self.return_vecs:
            edge_vec = edge_vec[lower_mask]
            return edge_index, edge_weight, edge_vec
        # TODO: return only `edge_index` and `edge_weight` once
        # Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
        return edge_index, edge_weight, None


rbf_class_mapping = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}


class AbsTanh(nn.Module):
    def __init__(self):
        super(AbsTanh, self).__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(torch.tanh(x))


class Tanh2(nn.Module):
    def __init__(self):
        super(Tanh2, self).__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.square(torch.tanh(x))


act_class_mapping = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "pass": nn.Identity,
    "abs_tanh": AbsTanh,
    "tanh2": Tanh2,
    "softmax": nn.Softmax,
    "gelu": nn.GELU,
}


def cosin_contrastive_loss(input, target, margin=0):
    if target.ndim == 1:
        target = target.unsqueeze(1)
    if input.shape[0] == 1:
        return torch.tensor(0, dtype=input.dtype, device=input.device)
    # calculate distance of input
    dist = F.cosine_similarity(input.unsqueeze(1), input.unsqueeze(0), dim=2)
    # calculate similarity matrix
    sim = torch.eq(target, target.T)
    # change similarity matrix to -1 and 1
    sim = sim.float() * 2 - 1
    # calculate loss, but only for the upper triangle of the similarity matrix
    loss = - dist * sim + (sim + 1) / 2 + (sim - 1) * margin / 2
    # mean over all pairs
    loss = torch.clamp(loss.triu(diagonal=1), min=0).sum() / (target.shape[0] * (target.shape[0] - 1) / 2)
    return loss


def euclid_contrastive_loss(input, target):
    if target.ndim == 1:
        target = target.unsqueeze(1)
    if input.shape[0] == 1:
        return torch.tensor(0, dtype=input.dtype, device=input.device)
    # margin is set according to input dimension
    margin = 10 * input.shape[1]
    # calculate distance of input
    dist = torch.cdist(input, input)
    # calculate similarity matrix
    sim = torch.eq(target, target.T)
    # change similarity matrix to -1 and 1
    sim = sim.float() * 2 - 1
    # calculate loss, but only for the upper triangle of the similarity matrix
    mask = (dist > margin).float() * (sim == -1).float()
    loss = dist * sim * (1 - mask)
    # mean over all pairs
    loss = loss.triu(diagonal=1).sum() / (target.shape[0] * (target.shape[0] - 1) / 2)
    return loss


loss_fn_mapping = {
    "mse_loss": mse_loss,
    "l1_loss": l1_loss,
    "binary_cross_entropy": binary_cross_entropy,
    "cross_entropy": cross_entropy,
    "kl_div": kl_div,
    "cosin_contrastive_loss": cosin_contrastive_loss,
    "euclid_contrastive_loss": euclid_contrastive_loss,
}


def get_template_fn(template):
    if template == 'plain-distance':
        return plain_distance, 1
    elif template == 'exp-normal-smearing-distance':
        return exp_normal_smearing_distance, 50

def plain_distance(pos):
    eps=1e-10
    CA = pos[..., 3, :]  # [b, n_res, 5, 3] -> [b, n_res, 3]
    d = (eps + (CA[..., None, :, :] - CA[..., :, None, :]).pow(2).sum(dim=-1, keepdims=True)) ** 0.5
    return d

def exp_normal_smearing_distance(pos, cutoff_upper=100, cutoff_lower=0, num_rbf=50):
    alpha = 5.0 / (cutoff_upper - cutoff_lower)
    start_value = torch.exp(
        torch.scalar_tensor(-cutoff_upper + cutoff_lower)
    ).to(pos.device)
    means = torch.linspace(start_value, 1, num_rbf).to(pos.device)
    betas = torch.tensor(
        [(2 / num_rbf * (1 - start_value)) ** -2] * num_rbf
    ).to(pos.device)
    dist = plain_distance(pos)
    cutoffs = 0.5 * (torch.cos(dist * math.pi / cutoff_upper).to(pos.device) + 1.0)
    # remove contributions beyond the cutoff radius
    cutoffs = cutoffs * (dist < cutoff_upper).float()
    return cutoffs * torch.exp(
        -betas * (torch.exp(alpha * (-dist + cutoff_lower)) - means) ** 2
    )

