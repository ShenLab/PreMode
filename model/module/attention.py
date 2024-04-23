from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_sparse import SparseTensor
import loralib as lora
from esm.multihead_attention import MultiheadAttention
import math
from torch import _dynamo
_dynamo.config.suppress_errors = True
from ..module.utils import (
    CosineCutoff,
    act_class_mapping,
    get_template_fn,
    gelu
)


# original torchmd-net attention layer
class EquivariantMultiHeadAttention(MessagePassing):
    """Equivariant multi-head attention layer."""

    def __init__(
            self,
            x_channels,
            x_hidden_channels,
            vec_channels,
            vec_hidden_channels,
            share_kv,
            edge_attr_channels,
            distance_influence,
            num_heads,
            activation,
            attn_activation,
            cutoff_lower,
            cutoff_upper,
            use_lora=None,
    ):
        super(EquivariantMultiHeadAttention, self).__init__(
            aggr="mean", node_dim=0)
        assert x_hidden_channels % num_heads == 0 \
            and vec_channels % num_heads == 0, (
                f"The number of hidden channels x_hidden_channels ({x_hidden_channels}) "
                f"and vec_channels ({vec_channels}) "
                f"must be evenly divisible by the number of "
                f"attention heads ({num_heads})"
            )
        assert vec_hidden_channels == x_channels, (
            f"The number of hidden channels x_channels ({x_channels}) "
            f"and vec_hidden_channels ({vec_hidden_channels}) "
            f"must be equal"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.x_channels = x_channels
        self.x_hidden_channels = x_hidden_channels
        self.x_head_dim = x_hidden_channels // num_heads
        self.vec_channels = vec_channels
        self.vec_hidden_channels = vec_hidden_channels
        # important, not vec_hidden_channels // num_heads
        self.vec_head_dim = vec_channels // num_heads
        self.share_kv = share_kv
        self.layernorm = nn.LayerNorm(x_channels)
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        if use_lora is not None:
            self.q_proj = lora.Linear(x_channels, x_hidden_channels, r=use_lora)
            self.k_proj = lora.Linear(x_channels, x_hidden_channels, r=use_lora) if not share_kv else None
            self.v_proj = lora.Linear(
                x_channels, x_hidden_channels + vec_channels * 2, r=use_lora)
            self.o_proj = lora.Linear(
                x_hidden_channels, x_channels * 2 + vec_channels, r=use_lora)
            self.vec_proj = lora.Linear(
                vec_channels, vec_hidden_channels * 2 + vec_channels, bias=False, r=use_lora)
        else:
            self.q_proj = nn.Linear(x_channels, x_hidden_channels)
            self.k_proj = nn.Linear(x_channels, x_hidden_channels) if not share_kv else None
            self.v_proj = nn.Linear(
                x_channels, x_hidden_channels + vec_channels * 2)
            self.o_proj = nn.Linear(
                x_hidden_channels, x_channels * 2 + vec_channels)
            self.vec_proj = nn.Linear(
            vec_channels, vec_hidden_channels * 2 + vec_channels, bias=False)

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            if use_lora is not None:
                self.dk_proj = lora.Linear(edge_attr_channels, x_hidden_channels, r=use_lora)
            else:
                self.dk_proj = nn.Linear(edge_attr_channels, x_hidden_channels)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            if use_lora is not None:
                self.dv_proj = lora.Linear(edge_attr_channels, x_hidden_channels + vec_channels * 2, r=use_lora)
            else:
                self.dv_proj = nn.Linear(edge_attr_channels, x_hidden_channels + vec_channels * 2)

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij, return_attn=False):
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads,
                                   self.x_head_dim + self.vec_head_dim * 2)
        if self.share_kv:
            k = v[:, :, :self.x_head_dim]
        else:
            k = self.k_proj(x).reshape(-1, self.num_heads, self.x_head_dim)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec),
                                       [self.vec_hidden_channels, self.vec_hidden_channels, self.vec_channels], dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.vec_head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1,
                                                 self.num_heads, self.x_head_dim)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads,
                                                 self.x_head_dim + self.vec_head_dim * 2)
            if self.dv_proj is not None
            else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor,
        # d_ij: Tensor)
        x, vec, attn = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        x = x.reshape(-1, self.x_hidden_channels)
        vec = vec.reshape(-1, 3, self.vec_channels)

        o1, o2, o3 = torch.split(self.o_proj(
            x), [self.vec_channels, self.x_channels, self.x_channels], dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec
        if return_attn:
            return dx, dvec, torch.concat((edge_index.T, attn), dim=1)
        else:
            return dx, dvec, None

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:  # TODO: consider add or multiply dk
            attn = (q_i * k_j * dk).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        # value pathway
        if dv is not None:
            v_j = v_j * dv
        x, vec1, vec2 = torch.split(
            v_j, [self.x_head_dim, self.vec_head_dim, self.vec_head_dim], dim=2)

        # update scalar features
        x = x * attn.unsqueeze(2)
        # update vector features
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * \
            d_ij.unsqueeze(2).unsqueeze(3)
        return x, vec, attn

    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, vec, attn = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec, attn

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    def edge_update(self) -> Tensor:
        pass


# ESM multi-head attention layer, added LoRA
class ESMMultiheadAttention(MultiheadAttention):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
        use_rotary_embeddings: bool = False,
    ):
        super().__init__(embed_dim, num_heads, kdim, vdim, dropout, bias, add_bias_kv, add_zero_attn, self_attention,
                         encoder_decoder_attention, use_rotary_embeddings)
        # change the projection to LoRA
        self.k_proj = lora.Linear(self.kdim, embed_dim, bias=bias, r=16)
        self.v_proj = lora.Linear(self.vdim, embed_dim, bias=bias, r=16)
        self.q_proj = lora.Linear(embed_dim, embed_dim, bias=bias, r=16)
        self.out_proj = lora.Linear(embed_dim, embed_dim, bias=bias, r=16)


# original torchmd-net attention layer, add pair-wise confidence of PAE
class EquivariantPAEMultiHeadAttention(EquivariantMultiHeadAttention):
    """Equivariant multi-head attention layer."""

    def __init__(
            self,
            x_channels,
            x_hidden_channels,
            vec_channels,
            vec_hidden_channels,
            share_kv,
            edge_attr_channels,
            edge_attr_dist_channels,
            distance_influence,
            num_heads,
            activation,
            attn_activation,
            cutoff_lower,
            cutoff_upper,
            use_lora=None,
    ):
        super(EquivariantPAEMultiHeadAttention, self).__init__(
            x_channels=x_channels,
            x_hidden_channels=x_hidden_channels,
            vec_channels=vec_channels,
            vec_hidden_channels=vec_hidden_channels,
            share_kv=share_kv,
            edge_attr_channels=edge_attr_channels,
            distance_influence=distance_influence,
            num_heads=num_heads,
            activation=activation,
            attn_activation=attn_activation,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            use_lora=use_lora)
        # we cancel the cutoff function
        self.cutoff = None
        # we set separate projection for distance influence
        self.dk_dist_proj = None
        if distance_influence in ["keys", "both"]:
            if use_lora is not None:
                self.dk_dist_proj = lora.Linear(edge_attr_dist_channels, x_hidden_channels, r=use_lora)
            else:
                self.dk_dist_proj = nn.Linear(edge_attr_dist_channels, x_hidden_channels)
        self.dv_dist_proj = None
        if distance_influence in ["values", "both"]:
            if use_lora is not None:
                self.dv_dist_proj = lora.Linear(edge_attr_dist_channels, x_hidden_channels + vec_channels * 2, r=use_lora)
            else:
                self.dv_dist_proj = nn.Linear(edge_attr_dist_channels, x_hidden_channels + vec_channels * 2)
        if self.dk_dist_proj:
            nn.init.xavier_uniform_(self.dk_dist_proj.weight)
            self.dk_dist_proj.bias.data.fill_(0)
        if self.dv_dist_proj:
            nn.init.xavier_uniform_(self.dv_dist_proj.weight)
            self.dv_dist_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, w_ij, f_dist_ij, f_ij, d_ij, plddt, return_attn=False):
        # we replaced r_ij to w_ij as pair-wise confidence
        # we add plddt as position-wise confidence
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads,
                                   self.x_head_dim + self.vec_head_dim * 2)
        if self.share_kv:
            k = v[:, :, :self.x_head_dim]
        else:
            k = self.k_proj(x).reshape(-1, self.num_heads, self.x_head_dim)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec),
                                       [self.vec_hidden_channels, self.vec_hidden_channels, self.vec_channels], dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.vec_head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.x_head_dim)
            if self.dk_proj is not None
            else None
        )
        dk_dist = (
            self.act(self.dk_dist_proj(f_dist_ij)).reshape(-1, self.num_heads, self.x_head_dim)
            if self.dk_dist_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.x_head_dim + self.vec_head_dim * 2)
            if self.dv_proj is not None
            else None
        )
        dv_dist = (
            self.act(self.dv_dist_proj(f_dist_ij)).reshape(-1, self.num_heads, self.x_head_dim + self.vec_head_dim * 2)
            if self.dv_dist_proj is not None
            else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor,
        # d_ij: Tensor)
        x, vec, attn = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dk_dist=dk_dist,
            dv=dv,
            dv_dist=dv_dist,
            d_ij=d_ij,
            w_ij=w_ij,
            size=None,
        )
        x = x.reshape(-1, self.x_hidden_channels)
        vec = vec.reshape(-1, 3, self.vec_channels)

        o1, o2, o3 = torch.split(self.o_proj(
            x), [self.vec_channels, self.x_channels, self.x_channels], dim=1)
        dx = vec_dot * o2 * plddt.unsqueeze(1) + o3
        dvec = vec3 * o1.unsqueeze(1) * plddt.unsqueeze(1).unsqueeze(2) + vec
        if return_attn:
            return dx, dvec, torch.concat((edge_index.T, attn), dim=1)
        else:
            return dx, dvec, None

    def message(self, q_i, k_j, v_j, vec_j, dk, dk_dist, dv, dv_dist, d_ij, w_ij):
        # attention mechanism
        attn = (q_i * k_j)
        if dk is not None:
            attn += dk
        if dk_dist is not None:
            attn += dk_dist * w_ij.unsqueeze(1).unsqueeze(2)
        attn = attn.sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn)

        # value pathway, add dv, but apply w_ij to dv
        if dv is not None:
            v_j += dv
        if dv_dist is not None:
            v_j += dv_dist * w_ij.unsqueeze(1).unsqueeze(2)
        x, vec1, vec2 = torch.split(
            v_j, [self.x_head_dim, self.vec_head_dim, self.vec_head_dim], dim=2)

        # update scalar features
        x = x * attn.unsqueeze(2)
        # update vector features
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * \
            d_ij.unsqueeze(2).unsqueeze(3)
        return x, vec, attn


# original torchmd-net attention layer, add pair-wise confidence of PAE
class EquivariantWeightedPAEMultiHeadAttention(EquivariantMultiHeadAttention):
    """Equivariant multi-head attention layer."""

    def __init__(
            self,
            x_channels,
            x_hidden_channels,
            vec_channels,
            vec_hidden_channels,
            share_kv,
            edge_attr_channels,
            edge_attr_dist_channels,
            distance_influence,
            num_heads,
            activation,
            attn_activation,
            cutoff_lower,
            cutoff_upper,
            use_lora=None,
    ):
        super(EquivariantWeightedPAEMultiHeadAttention, self).__init__(
            x_channels=x_channels,
            x_hidden_channels=x_hidden_channels,
            vec_channels=vec_channels,
            vec_hidden_channels=vec_hidden_channels,
            share_kv=share_kv,
            edge_attr_channels=edge_attr_channels,
            distance_influence=distance_influence,
            num_heads=num_heads,
            activation=activation,
            attn_activation=attn_activation,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            use_lora=use_lora)
        # we cancel the cutoff function
        self.cutoff = None
        # we set a separate weight for distance influence
        self.pae_weight = nn.Linear(1, 1, bias=True)
        self.pae_weight.weight.data.fill_(-0.5)
        self.pae_weight.bias.data.fill_(7.5)
        # we set separate projection for distance influence
        self.dk_dist_proj = None
        if distance_influence in ["keys", "both"]:
            if use_lora is not None:
                self.dk_dist_proj = lora.Linear(edge_attr_dist_channels, x_hidden_channels, r=use_lora)
            else:
                self.dk_dist_proj = nn.Linear(edge_attr_dist_channels, x_hidden_channels)
        self.dv_dist_proj = None
        if distance_influence in ["values", "both"]:
            if use_lora is not None:
                self.dv_dist_proj = lora.Linear(edge_attr_dist_channels, x_hidden_channels + vec_channels * 2, r=use_lora)
            else:
                self.dv_dist_proj = nn.Linear(edge_attr_dist_channels, x_hidden_channels + vec_channels * 2)
        if self.dk_dist_proj:
            nn.init.xavier_uniform_(self.dk_dist_proj.weight)
            self.dk_dist_proj.bias.data.fill_(0)
        if self.dv_dist_proj:
            nn.init.xavier_uniform_(self.dv_dist_proj.weight)
            self.dv_dist_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, w_ij, f_dist_ij, f_ij, d_ij, plddt, return_attn=False):
        # we replaced r_ij to w_ij as pair-wise confidence
        # we add plddt as position-wise confidence
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads,
                                   self.x_head_dim + self.vec_head_dim * 2)
        if self.share_kv:
            k = v[:, :, :self.x_head_dim]
        else:
            k = self.k_proj(x).reshape(-1, self.num_heads, self.x_head_dim)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec),
                                       [self.vec_hidden_channels, self.vec_hidden_channels, self.vec_channels], dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.vec_head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.x_head_dim)
            if self.dk_proj is not None
            else None
        )
        dk_dist = (
            self.act(self.dk_dist_proj(f_dist_ij)).reshape(-1, self.num_heads, self.x_head_dim)
            if self.dk_dist_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.x_head_dim + self.vec_head_dim * 2)
            if self.dv_proj is not None
            else None
        )
        dv_dist = (
            self.act(self.dv_dist_proj(f_dist_ij)).reshape(-1, self.num_heads, self.x_head_dim + self.vec_head_dim * 2)
            if self.dv_dist_proj is not None
            else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor,
        # d_ij: Tensor)
        x, vec, attn = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dk_dist=dk_dist,
            dv=dv,
            dv_dist=dv_dist,
            d_ij=d_ij,
            w_ij=nn.functional.sigmoid(self.pae_weight(w_ij.unsqueeze(-1)).squeeze(-1)),
            size=None,
        )
        x = x.reshape(-1, self.x_hidden_channels)
        vec = vec.reshape(-1, 3, self.vec_channels)

        o1, o2, o3 = torch.split(self.o_proj(
            x), [self.vec_channels, self.x_channels, self.x_channels], dim=1)
        dx = vec_dot * o2 * plddt.unsqueeze(1) + o3
        dvec = vec3 * o1.unsqueeze(1) * plddt.unsqueeze(1).unsqueeze(2) + vec
        if return_attn:
            return dx, dvec, torch.concat((edge_index.T, attn), dim=1)
        else:
            return dx, dvec, None

    def message(self, q_i, k_j, v_j, vec_j, dk, dk_dist, dv, dv_dist, d_ij, w_ij):
        # attention mechanism
        attn = (q_i * k_j)
        if dk_dist is not None:
            if dk is not None:
                attn *= (dk + dk_dist * w_ij.unsqueeze(1).unsqueeze(2))
            else:
                attn *= dk_dist * w_ij
        else:
            if dk is not None:
                attn *= dk
        attn = attn.sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn)

        # value pathway, add dv, but apply w_ij to dv
        if dv is not None:
            v_j += dv
        if dv_dist is not None:
            v_j += dv_dist * w_ij.unsqueeze(1).unsqueeze(2)
        x, vec1, vec2 = torch.split(
            v_j, [self.x_head_dim, self.vec_head_dim, self.vec_head_dim], dim=2)

        # update scalar features
        x = x * attn.unsqueeze(2)
        # update vector features
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * \
            d_ij.unsqueeze(2).unsqueeze(3)
        return x, vec, attn


class EquivariantPAEMultiHeadAttentionSoftMaxFullGraph(nn.Module):
    """Equivariant multi-head attention layer with softmax, apply attention on full graph by default"""
    def __init__(
            self,
            x_channels,
            x_hidden_channels,
            vec_channels,
            vec_hidden_channels,
            share_kv,
            edge_attr_channels,
            edge_attr_dist_channels,
            distance_influence,
            num_heads,
            activation,
            attn_activation,
            cutoff_lower,
            cutoff_upper,
            use_lora=None,
    ):
        # same as EquivariantPAEMultiHeadAttentionSoftMax, but apply attention on full graph by default
        super(EquivariantPAEMultiHeadAttentionSoftMaxFullGraph, self).__init__()
        assert x_hidden_channels % num_heads == 0 \
            and vec_channels % num_heads == 0, (
                f"The number of hidden channels x_hidden_channels ({x_hidden_channels}) "
                f"and vec_channels ({vec_channels}) "
                f"must be evenly divisible by the number of "
                f"attention heads ({num_heads})"
            )
        assert vec_hidden_channels == x_channels, (
            f"The number of hidden channels x_channels ({x_channels}) "
            f"and vec_hidden_channels ({vec_hidden_channels}) "
            f"must be equal"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.x_channels = x_channels
        self.x_hidden_channels = x_hidden_channels
        self.x_head_dim = x_hidden_channels // num_heads
        self.vec_channels = vec_channels
        self.vec_hidden_channels = vec_hidden_channels
        # important, not vec_hidden_channels // num_heads
        self.vec_head_dim = vec_channels // num_heads
        self.share_kv = share_kv
        self.layernorm = nn.LayerNorm(x_channels)
        self.act = activation()
        self.cutoff = None
        self.scaling = self.x_head_dim**-0.5
        if use_lora is not None:
            self.q_proj = lora.Linear(x_channels, x_hidden_channels, r=use_lora)
            self.k_proj = lora.Linear(x_channels, x_hidden_channels, r=use_lora) if not share_kv else None
            self.v_proj = lora.Linear(x_channels, x_hidden_channels + vec_channels * 2, r=use_lora)
            self.o_proj = lora.Linear(x_hidden_channels, x_channels * 2 + vec_channels, r=use_lora)
            self.vec_proj = lora.Linear(vec_channels, vec_hidden_channels * 2 + vec_channels, bias=False, r=use_lora)
        else:
            self.q_proj = nn.Linear(x_channels, x_hidden_channels)
            self.k_proj = nn.Linear(x_channels, x_hidden_channels) if not share_kv else None
            self.v_proj = nn.Linear(x_channels, x_hidden_channels + vec_channels * 2)
            self.o_proj = nn.Linear(x_hidden_channels, x_channels * 2 + vec_channels)
            self.vec_proj = nn.Linear(vec_channels, vec_hidden_channels * 2 + vec_channels, bias=False)

        self.dk_proj = None
        self.dk_dist_proj = None
        self.dv_proj = None
        self.dv_dist_proj = None
        if distance_influence in ["keys", "both"]:
            if use_lora is not None:
                self.dk_proj = lora.Linear(edge_attr_channels, x_hidden_channels, r=use_lora)
                self.dk_dist_proj = lora.Linear(edge_attr_dist_channels, x_hidden_channels, r=use_lora)
            else:
                self.dk_proj = nn.Linear(edge_attr_channels, x_hidden_channels)
                self.dk_dist_proj = nn.Linear(edge_attr_dist_channels, x_hidden_channels)

        if distance_influence in ["values", "both"]:
            if use_lora is not None:
                self.dv_proj = lora.Linear(edge_attr_channels, x_hidden_channels + vec_channels * 2, r=use_lora)
                self.dv_dist_proj = lora.Linear(edge_attr_dist_channels, x_hidden_channels + vec_channels * 2, r=use_lora)
            else:
                self.dv_proj = nn.Linear(edge_attr_channels, x_hidden_channels + vec_channels * 2)
                self.dv_dist_proj = nn.Linear(edge_attr_dist_channels, x_hidden_channels + vec_channels * 2)
        # set PAE weight as a learnable parameter, basiclly a sigmoid function
        self.pae_weight = nn.Linear(1, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        self.pae_weight.weight.data.fill_(-0.5)
        self.pae_weight.bias.data.fill_(7.5)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)
        if self.dk_dist_proj:
            nn.init.xavier_uniform_(self.dk_dist_proj.weight)
            self.dk_dist_proj.bias.data.fill_(0)
        if self.dv_dist_proj:
            nn.init.xavier_uniform_(self.dv_dist_proj.weight)
            self.dv_dist_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, w_ij, f_dist_ij, f_ij, d_ij, plddt, key_padding_mask, return_attn=False):
        # we replaced r_ij to w_ij as pair-wise confidence
        # we add plddt as position-wise confidence
        # edge_index is unused
        x = self.layernorm(x)
        q = self.q_proj(x) * self.scaling
        v = self.v_proj(x)
        # if self.share_kv:
        #     k = v[:, :, :self.x_head_dim]
        # else:
        k = self.k_proj(x)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec),
                                       [self.vec_hidden_channels, self.vec_hidden_channels, self.vec_channels], dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=-2)

        dk = self.act(self.dk_proj(f_ij)) 
        dk_dist = self.act(self.dk_dist_proj(f_dist_ij)) 
        dv = self.act(self.dv_proj(f_ij))  
        dv_dist = self.act(self.dv_dist_proj(f_dist_ij)) 

        # full graph attention
        x, vec, attn = self.attention(
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dk_dist=dk_dist,
            dv=dv,
            dv_dist=dv_dist,
            d_ij=d_ij,
            w_ij=nn.functional.sigmoid(self.pae_weight(w_ij.unsqueeze(-1)).squeeze(-1)),
            key_padding_mask=key_padding_mask,
        )
        o1, o2, o3 = torch.split(self.o_proj(x), [self.vec_channels, self.x_channels, self.x_channels], dim=-1)
        dx = vec_dot * o2 * plddt.unsqueeze(-1) + o3
        dvec = vec3 * o1.unsqueeze(-2) * plddt.unsqueeze(-1).unsqueeze(-2) + vec
        # apply key_padding_mask to dx
        dx = dx.masked_fill(key_padding_mask.unsqueeze(-1), 0)
        dvec = dvec.masked_fill(key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0)
        if return_attn:
            return dx, dvec, attn
        else:
            return dx, dvec, None

    def attention(self, q, k, v, vec, dk, dk_dist, dv, dv_dist, d_ij, w_ij, key_padding_mask=None, need_head_weights=False):
        # note that q is of shape (bsz, tgt_len, num_heads * head_dim)
        # k, v is of shape (bsz, src_len, num_heads * head_dim)
        # vec is of shape (bsz, src_len, 3, num_heads * head_dim)
        # dk, dk_dist, dv, dv_dist is of shape (bsz, tgt_len, src_len, num_heads * head_dim)
        # d_ij is of shape (bsz, tgt_len, src_len, 3)
        # w_ij is of shape (bsz, tgt_len, src_len)
        # key_padding_mask is of shape (bsz, src_len)
        bsz, tgt_len, _ = q.size()
        src_len = k.size(1)
        # change q size to (bsz * num_heads, tgt_len, head_dim)
        # change k,v size to (bsz * num_heads, src_len, head_dim)
        q = q.transpose(0, 1).reshape(tgt_len, bsz * self.num_heads, self.x_head_dim).transpose(0, 1).contiguous()
        k = k.transpose(0, 1).reshape(src_len, bsz * self.num_heads, self.x_head_dim).transpose(0, 1).contiguous()
        v = v.transpose(0, 1).reshape(src_len, bsz * self.num_heads, self.x_head_dim + 2 * self.vec_head_dim).transpose(0, 1).contiguous()
        # change vec to (bsz * num_heads, src_len, 3, head_dim)
        vec = vec.permute(1, 2, 0, 3).reshape(src_len, 3, bsz * self.num_heads, self.vec_head_dim).permute(2, 0, 1, 3).contiguous()
        # dk size is (bsz, tgt_len, src_len, num_heads * head_dim)
        # if dk is not None:
        # change dk to (bsz * num_heads, tgt_len, src_len, head_dim)
        dk = dk.permute(1, 2, 0, 3).reshape(tgt_len, src_len, bsz * self.num_heads, self.x_head_dim).permute(2, 0, 1, 3).contiguous()
        # if dk_dist is not None:
        # change dk_dist to (bsz * num_heads, tgt_len, src_len, head_dim)
        dk_dist = dk_dist.permute(1, 2, 0, 3).reshape(tgt_len, src_len, bsz * self.num_heads, self.x_head_dim).permute(2, 0, 1, 3).contiguous()
        # dv size is (bsz, tgt_len, src_len, num_heads * head_dim)
        # if dv is not None:
        # change dv to (bsz * num_heads, tgt_len, src_len, head_dim)
        dv = dv.permute(1, 2, 0, 3).reshape(tgt_len, src_len, bsz * self.num_heads, self.x_head_dim + 2 * self.vec_head_dim).permute(2, 0, 1, 3).contiguous()
        # if dv_dist is not None:
        # change dv_dist to (bsz * num_heads, tgt_len, src_len, head_dim)
        dv_dist = dv_dist.permute(1, 2, 0, 3).reshape(tgt_len, src_len, bsz * self.num_heads, self.x_head_dim + 2 * self.vec_head_dim).permute(2, 0, 1, 3).contiguous()
        # if key_padding_mask is not None:
        # key_padding_mask should be (bsz, src_len)
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len
        # attn_weights size is (bsz * num_heads, tgt_len, src_len, head_dim)
        attn_weights = torch.multiply(q[:, :, None, :], k[:, None, :, :])
        # w_ij is PAE confidence
        # w_ij size is (bsz, tgt_len, src_len)
        # change dimension of w_ij to (bsz * num_heads, tgt_len, src_len, head_dim)
        # if dk_dist is not None:
        assert w_ij is not None
        #     if dk is not None:
        attn_weights *= (dk + dk_dist * w_ij[:, :, :, None].repeat(self.num_heads, 1, 1, self.x_head_dim))
        # add dv and dv_dist
        v = v.unsqueeze(1) + dv + dv_dist * w_ij[:, :, :, None].repeat(self.num_heads, 1, 1, self.x_head_dim + 2 * self.vec_head_dim)
        #     else:
        #         attn_weights *= dk_dist * w_ij
        # else:
        #     if dk is not None:
        #         attn_weights *= dk
        # attn_weights size is (bsz * num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=-1)
        # apply key_padding_mask to attn_weights
        # if key_padding_mask is not None:
        # don't attend to padding symbols
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len).contiguous()
        attn_weights = attn_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
        )
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len).contiguous()
        # apply softmax to attn_weights
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        # x, vec1, vec2 are of shape (bsz * num_heads, src_len, head_dim)
        x, vec1, vec2 = torch.split(v, [self.x_head_dim, self.vec_head_dim, self.vec_head_dim], dim=-1)
        # first get invariant feature outputs x, size is (bsz * num_heads, tgt_len, head_dim)
        x_out = torch.einsum('bts,btsh->bth', attn_weights, x)
        # next get equivariant feature outputs vec_out_1, size is (bsz * num_heads, tgt_len, 3, head_dim)
        vec_out_1 = torch.einsum('bsih,btsh->btih', vec, vec1)
        # next get equivariant feature outputs vec_out_2, size is (bsz * num_heads, tgt_len, src_len, 3, head_dim)
        vec_out_2 = torch.einsum('btsi,btsh->btih', d_ij, vec2)
        # adds up vec_out_1 and vec_out_2, get vec_out, size is (bsz * num_heads, tgt_len, 3, head_dim)
        vec_out = vec_out_1 + vec_out_2
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
        # if not need_head_weights:
        # average attention weights over heads
        attn_weights = attn_weights.mean(dim=0)
        # reshape x_out to (bsz, tgt_len, num_heads * head_dim)
        x_out = x_out.transpose(1, 0).reshape(tgt_len, bsz, self.num_heads * self.x_head_dim).transpose(1, 0).contiguous()
        # reshape vec_out to (bsz, tgt_len, 3, num_heads * head_dim)
        vec_out = vec_out.permute(1, 2, 0, 3).reshape(tgt_len, 3, bsz, self.num_heads * self.vec_head_dim).permute(2, 0, 1, 3).contiguous()
        return x_out, vec_out, attn_weights


class MultiHeadAttentionSoftMaxFullGraph(nn.Module):
    """
    Multi-head attention layer with softmax, apply attention on full graph by default
    No equivariant property, but can take structure information as input, just didn't use it
    """
    def __init__(
            self,
            x_channels,
            x_hidden_channels,
            vec_channels,
            vec_hidden_channels,
            share_kv,
            edge_attr_channels,
            edge_attr_dist_channels,
            distance_influence,
            num_heads,
            activation,
            attn_activation,
            cutoff_lower,
            cutoff_upper,
            use_lora=None,
    ):
        # same as EquivariantPAEMultiHeadAttentionSoftMax, but apply attention on full graph by default
        super(MultiHeadAttentionSoftMaxFullGraph, self).__init__()
        assert x_hidden_channels % num_heads == 0 \
            and vec_channels % num_heads == 0, (
                f"The number of hidden channels x_hidden_channels ({x_hidden_channels}) "
                f"and vec_channels ({vec_channels}) "
                f"must be evenly divisible by the number of "
                f"attention heads ({num_heads})"
            )
        assert vec_hidden_channels == x_channels, (
            f"The number of hidden channels x_channels ({x_channels}) "
            f"and vec_hidden_channels ({vec_hidden_channels}) "
            f"must be equal"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.x_channels = x_channels
        self.x_hidden_channels = x_hidden_channels
        self.x_head_dim = x_hidden_channels // num_heads
        self.vec_channels = vec_channels
        self.vec_hidden_channels = vec_hidden_channels
        # important, not vec_hidden_channels // num_heads
        self.vec_head_dim = vec_channels // num_heads
        self.share_kv = share_kv
        self.layernorm = nn.LayerNorm(x_channels)
        self.act = activation()
        self.cutoff = None
        self.scaling = self.x_head_dim**-0.5
        if use_lora is not None:
            self.q_proj = lora.Linear(x_channels, x_hidden_channels, r=use_lora)
            self.k_proj = lora.Linear(x_channels, x_hidden_channels, r=use_lora) if not share_kv else None
            self.v_proj = lora.Linear(x_channels, x_hidden_channels, r=use_lora)
            self.o_proj = lora.Linear(x_hidden_channels, x_channels, r=use_lora)
            # self.vec_proj = lora.Linear(vec_channels, vec_hidden_channels * 2 + vec_channels, bias=False, r=use_lora)
        else:
            self.q_proj = nn.Linear(x_channels, x_hidden_channels)
            self.k_proj = nn.Linear(x_channels, x_hidden_channels) if not share_kv else None
            self.v_proj = nn.Linear(x_channels, x_hidden_channels)
            self.o_proj = nn.Linear(x_hidden_channels, x_channels)
            # self.vec_proj = nn.Linear(vec_channels, vec_hidden_channels * 2 + vec_channels, bias=False)

        self.dk_proj = None
        self.dk_dist_proj = None
        self.dv_proj = None
        self.dv_dist_proj = None
        if distance_influence in ["keys", "both"]:
            if use_lora is not None:
                self.dk_proj = lora.Linear(edge_attr_channels, x_hidden_channels, r=use_lora)
                # self.dk_dist_proj = lora.Linear(edge_attr_dist_channels, x_hidden_channels, r=use_lora)
            else:
                self.dk_proj = nn.Linear(edge_attr_channels, x_hidden_channels)
                # self.dk_dist_proj = nn.Linear(edge_attr_dist_channels, x_hidden_channels)

        if distance_influence in ["values", "both"]:
            if use_lora is not None:
                self.dv_proj = lora.Linear(edge_attr_channels, x_hidden_channels, r=use_lora)
                # self.dv_dist_proj = lora.Linear(edge_attr_dist_channels, x_hidden_channels + vec_channels * 2, r=use_lora)
            else:
                self.dv_proj = nn.Linear(edge_attr_channels, x_hidden_channels)
                # self.dv_dist_proj = nn.Linear(edge_attr_dist_channels, x_hidden_channels + vec_channels * 2)
        # set PAE weight as a learnable parameter, basiclly a sigmoid function
        # self.pae_weight = nn.Linear(1, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.vec_proj.weight)
        # self.pae_weight.weight.data.fill_(-0.5)
        # self.pae_weight.bias.data.fill_(7.5)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, w_ij, f_dist_ij, f_ij, d_ij, plddt, key_padding_mask, return_attn=False):
        # we replaced r_ij to w_ij as pair-wise confidence
        # we add plddt as position-wise confidence
        # edge_index is unused
        x = self.layernorm(x)
        q = self.q_proj(x) * self.scaling
        v = self.v_proj(x)
        # if self.share_kv:
        #     k = v[:, :, :self.x_head_dim]
        # else:
        k = self.k_proj(x)

        # vec1, vec2, vec3 = torch.split(self.vec_proj(vec),
        #                                [self.vec_hidden_channels, self.vec_hidden_channels, self.vec_channels], dim=-1)
        # vec_dot = (vec1 * vec2).sum(dim=-2)

        dk = self.act(self.dk_proj(f_ij)) 
        # dk_dist = self.act(self.dk_dist_proj(f_dist_ij)) 
        dv = self.act(self.dv_proj(f_ij))  
        # dv_dist = self.act(self.dv_dist_proj(f_dist_ij)) 

        # full graph attention
        x, vec, attn = self.attention(
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            # dk_dist=dk_dist,
            dv=dv,
            # dv_dist=dv_dist,
            # d_ij=d_ij,
            # w_ij=nn.functional.sigmoid(self.pae_weight(w_ij.unsqueeze(-1)).squeeze(-1)),
            key_padding_mask=key_padding_mask,
        )
        # o1, o2, o3 = torch.split(self.o_proj(x), [self.vec_channels, self.x_channels, self.x_channels], dim=-1)
        # dx = vec_dot * o2 * plddt.unsqueeze(-1) + o3
        dx = self.o_proj(x)
        # dvec = vec3 * o1.unsqueeze(-2) * plddt.unsqueeze(-1).unsqueeze(-2) + vec
        # apply key_padding_mask to dx
        dx = dx.masked_fill(key_padding_mask.unsqueeze(-1), 0)
        # dvec = dvec.masked_fill(key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0)
        if return_attn:
            return dx, vec, attn
        else:
            return dx, vec, None

    def attention(self, q, k, v, vec, dk, dv, key_padding_mask=None, need_head_weights=False):
        # note that q is of shape (bsz, tgt_len, num_heads * head_dim)
        # k, v is of shape (bsz, src_len, num_heads * head_dim)
        # vec is of shape (bsz, src_len, 3, num_heads * head_dim)
        # dk, dk_dist, dv, dv_dist is of shape (bsz, tgt_len, src_len, num_heads * head_dim)
        # d_ij is of shape (bsz, tgt_len, src_len, 3)
        # w_ij is of shape (bsz, tgt_len, src_len)
        # key_padding_mask is of shape (bsz, src_len)
        bsz, tgt_len, _ = q.size()
        src_len = k.size(1)
        # change q size to (bsz * num_heads, tgt_len, head_dim)
        # change k,v size to (bsz * num_heads, src_len, head_dim)
        q = q.transpose(0, 1).reshape(tgt_len, bsz * self.num_heads, self.x_head_dim).transpose(0, 1).contiguous()
        k = k.transpose(0, 1).reshape(src_len, bsz * self.num_heads, self.x_head_dim).transpose(0, 1).contiguous()
        v = v.transpose(0, 1).reshape(src_len, bsz * self.num_heads, self.x_head_dim).transpose(0, 1).contiguous()
        # change vec to (bsz * num_heads, src_len, 3, head_dim)
        # vec = vec.permute(1, 2, 0, 3).reshape(src_len, 3, bsz * self.num_heads, self.vec_head_dim).permute(2, 0, 1, 3).contiguous()
        # dk size is (bsz, tgt_len, src_len, num_heads * head_dim)
        # if dk is not None:
        # change dk to (bsz * num_heads, tgt_len, src_len, head_dim)
        dk = dk.permute(1, 2, 0, 3).reshape(tgt_len, src_len, bsz * self.num_heads, self.x_head_dim).permute(2, 0, 1, 3).contiguous()
        # if dk_dist is not None:
        # change dk_dist to (bsz * num_heads, tgt_len, src_len, head_dim)
        # dk_dist = dk_dist.permute(1, 2, 0, 3).reshape(tgt_len, src_len, bsz * self.num_heads, self.x_head_dim).permute(2, 0, 1, 3).contiguous()
        # dv size is (bsz, tgt_len, src_len, num_heads * head_dim)
        # if dv is not None:
        # change dv to (bsz * num_heads, tgt_len, src_len, head_dim)
        dv = dv.permute(1, 2, 0, 3).reshape(tgt_len, src_len, bsz * self.num_heads, self.x_head_dim).permute(2, 0, 1, 3).contiguous()
        # if dv_dist is not None:
        # change dv_dist to (bsz * num_heads, tgt_len, src_len, head_dim)
        # dv_dist = dv_dist.permute(1, 2, 0, 3).reshape(tgt_len, src_len, bsz * self.num_heads, self.x_head_dim + 2 * self.vec_head_dim).permute(2, 0, 1, 3).contiguous()
        # if key_padding_mask is not None:
        # key_padding_mask should be (bsz, src_len)
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len
        # attn_weights size is (bsz * num_heads, tgt_len, src_len, head_dim)
        attn_weights = torch.multiply(q[:, :, None, :], k[:, None, :, :])
        # w_ij is PAE confidence
        # w_ij size is (bsz, tgt_len, src_len)
        # change dimension of w_ij to (bsz * num_heads, tgt_len, src_len, head_dim)
        # if dk_dist is not None:
        # assert w_ij is not None
        #     if dk is not None:
        attn_weights *= dk
        # add dv and dv_dist
        v = v.unsqueeze(1) + dv
        #     else:
        #         attn_weights *= dk_dist * w_ij
        # else:
        #     if dk is not None:
        #         attn_weights *= dk
        # attn_weights size is (bsz * num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=-1)
        # apply key_padding_mask to attn_weights
        # if key_padding_mask is not None:
        # don't attend to padding symbols
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len).contiguous()
        attn_weights = attn_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
        )
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len).contiguous()
        # apply softmax to attn_weights
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        # x, vec1, vec2 are of shape (bsz * num_heads, src_len, head_dim)
        # x, vec1, vec2 = torch.split(v, [self.x_head_dim, self.vec_head_dim, self.vec_head_dim], dim=-1)
        # first get invariant feature outputs x, size is (bsz * num_heads, tgt_len, head_dim)
        x_out = torch.einsum('bts,btsh->bth', attn_weights, v)
        # next get equivariant feature outputs vec_out_1, size is (bsz * num_heads, tgt_len, 3, head_dim)
        # vec_out_1 = torch.einsum('bsih,btsh->btih', vec, vec1)
        # next get equivariant feature outputs vec_out_2, size is (bsz * num_heads, tgt_len, src_len, 3, head_dim)
        # vec_out_2 = torch.einsum('btsi,btsh->btih', d_ij, vec2)
        # adds up vec_out_1 and vec_out_2, get vec_out, size is (bsz * num_heads, tgt_len, 3, head_dim)
        # vec_out = vec_out_1 + vec_out_2
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
        # if not need_head_weights:
        # average attention weights over heads
        attn_weights = attn_weights.mean(dim=0)
        # reshape x_out to (bsz, tgt_len, num_heads * head_dim)
        x_out = x_out.transpose(1, 0).reshape(tgt_len, bsz, self.num_heads * self.x_head_dim).transpose(1, 0).contiguous()
        # reshape vec_out to (bsz, tgt_len, 3, num_heads * head_dim)
        # vec_out = vec_out.permute(1, 2, 0, 3).reshape(tgt_len, 3, bsz, self.num_heads * self.vec_head_dim).permute(2, 0, 1, 3).contiguous()
        return x_out, vec, attn_weights


class PAEMultiHeadAttentionSoftMaxStarGraph(nn.Module):
    """Equivariant multi-head attention layer with softmax, apply attention on full graph by default"""
    def __init__(
            self,
            x_channels,
            x_hidden_channels,
            vec_channels,
            vec_hidden_channels,
            share_kv,
            edge_attr_channels,
            edge_attr_dist_channels,
            distance_influence,
            num_heads,
            activation,
            cutoff_lower,
            cutoff_upper,
            use_lora=None,
    ):
        # same as EquivariantPAEMultiHeadAttentionSoftMax, but apply attention on full graph by default
        super(PAEMultiHeadAttentionSoftMaxStarGraph, self).__init__()
        assert x_hidden_channels % num_heads == 0 \
            and vec_channels % num_heads == 0, (
                f"The number of hidden channels x_hidden_channels ({x_hidden_channels}) "
                f"and vec_channels ({vec_channels}) "
                f"must be evenly divisible by the number of "
                f"attention heads ({num_heads})"
            )
        assert vec_hidden_channels == x_channels, (
            f"The number of hidden channels x_channels ({x_channels}) "
            f"and vec_hidden_channels ({vec_hidden_channels}) "
            f"must be equal"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.x_channels = x_channels
        self.x_hidden_channels = x_hidden_channels
        self.x_head_dim = x_hidden_channels // num_heads
        self.vec_channels = vec_channels
        self.vec_hidden_channels = vec_hidden_channels
        # important, not vec_hidden_channels // num_heads
        self.vec_head_dim = vec_channels // num_heads
        self.share_kv = share_kv
        self.layernorm = nn.LayerNorm(x_channels)
        self.act = activation()
        self.cutoff = None
        self.scaling = self.x_head_dim**-0.5
        if use_lora is not None:
            self.q_proj = lora.Linear(x_channels, x_hidden_channels, r=use_lora)
            self.k_proj = lora.Linear(x_channels, x_hidden_channels, r=use_lora) if not share_kv else None
            self.v_proj = lora.Linear(x_channels, x_hidden_channels + vec_channels * 2, r=use_lora)
        else:
            self.q_proj = nn.Linear(x_channels, x_hidden_channels)
            self.k_proj = nn.Linear(x_channels, x_hidden_channels) if not share_kv else None
            self.v_proj = nn.Linear(x_channels, x_hidden_channels)

        self.dk_proj = None
        self.dk_dist_proj = None
        self.dv_proj = None
        self.dv_dist_proj = None
        if distance_influence in ["keys", "both"]:
            if use_lora is not None:
                self.dk_proj = lora.Linear(edge_attr_channels, x_hidden_channels, r=use_lora)
                self.dk_dist_proj = lora.Linear(edge_attr_dist_channels, x_hidden_channels, r=use_lora)
            else:
                self.dk_proj = nn.Linear(edge_attr_channels, x_hidden_channels)
                self.dk_dist_proj = nn.Linear(edge_attr_dist_channels, x_hidden_channels)

        if distance_influence in ["values", "both"]:
            if use_lora is not None:
                self.dv_proj = lora.Linear(edge_attr_channels, x_hidden_channels + vec_channels * 2, r=use_lora)
                self.dv_dist_proj = lora.Linear(edge_attr_dist_channels, x_hidden_channels + vec_channels * 2, r=use_lora)
            else:
                self.dv_proj = nn.Linear(edge_attr_channels, x_hidden_channels + vec_channels * 2)
                self.dv_dist_proj = nn.Linear(edge_attr_dist_channels, x_hidden_channels + vec_channels * 2)
        # set PAE weight as a learnable parameter, basiclly a sigmoid function
        self.pae_weight = nn.Linear(1, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        self.pae_weight.weight.data.fill_(-0.5)
        self.pae_weight.bias.data.fill_(7.5)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)
        if self.dk_dist_proj:
            nn.init.xavier_uniform_(self.dk_dist_proj.weight)
            self.dk_dist_proj.bias.data.fill_(0)
        if self.dv_dist_proj:
            nn.init.xavier_uniform_(self.dv_dist_proj.weight)
            self.dv_dist_proj.bias.data.fill_(0)

    def forward(self, x, x_center_index, w_ij, f_dist_ij, f_ij, key_padding_mask, return_attn=False):
        # we replaced r_ij to w_ij as pair-wise confidence
        # we add plddt as position-wise confidence
        # edge_index is unused
        x = self.layernorm(x)
        q = self.q_proj(x[x_center_index].unsqueeze(1)) * self.scaling
        v = self.v_proj(x)
        # if self.share_kv:
        #     k = v[:, :, :self.x_head_dim]
        # else:
        k = self.k_proj(x)

        dk = self.act(self.dk_proj(f_ij)) 
        dk_dist = self.act(self.dk_dist_proj(f_dist_ij)) 
        dv = self.act(self.dv_proj(f_ij)) 
        dv_dist = self.act(self.dv_dist_proj(f_dist_ij)) 

        # full graph attention
        x, attn = self.attention(
            q=q,
            k=k,
            v=v,
            dk=dk,
            dk_dist=dk_dist,
            dv=dv,
            dv_dist=dv_dist,
            w_ij=nn.functional.sigmoid(self.pae_weight(w_ij.unsqueeze(-1)).squeeze(-1)),
            key_padding_mask=key_padding_mask,
        )
        if return_attn:
            return x, attn
        else:
            return x, None

    def attention(self, q, k, v, dk, dk_dist, dv, dv_dist, w_ij, key_padding_mask=None, need_head_weights=False):
        # note that q is of shape (bsz, tgt_len, num_heads * head_dim)
        # k, v is of shape (bsz, src_len, num_heads * head_dim)
        # vec is of shape (bsz, src_len, 3, num_heads * head_dim)
        # dk, dk_dist, dv, dv_dist is of shape (bsz, tgt_len, src_len, num_heads * head_dim)
        # d_ij is of shape (bsz, tgt_len, src_len, 3)
        # w_ij is of shape (bsz, tgt_len, src_len)
        # key_padding_mask is of shape (bsz, src_len)
        bsz, tgt_len, _ = q.size()
        src_len = k.size(1)
        # change q size to (bsz * num_heads, tgt_len, head_dim)
        # change k,v size to (bsz * num_heads, src_len, head_dim)
        q = q.transpose(0, 1).reshape(tgt_len, bsz * self.num_heads, self.x_head_dim).transpose(0, 1).contiguous()
        k = k.transpose(0, 1).reshape(src_len, bsz * self.num_heads, self.x_head_dim).transpose(0, 1).contiguous()
        v = v.transpose(0, 1).reshape(src_len, bsz * self.num_heads, self.x_head_dim).transpose(0, 1).contiguous()
        # dk size is (bsz, tgt_len, src_len, num_heads * head_dim)
        # if dk is not None:
        # change dk to (bsz * num_heads, tgt_len, src_len, head_dim)
        dk = dk.permute(1, 2, 0, 3).reshape(tgt_len, src_len, bsz * self.num_heads, self.x_head_dim).permute(2, 0, 1, 3).contiguous()
        # if dk_dist is not None:
        # change dk_dist to (bsz * num_heads, tgt_len, src_len, head_dim)
        dk_dist = dk_dist.permute(1, 2, 0, 3).reshape(tgt_len, src_len, bsz * self.num_heads, self.x_head_dim).permute(2, 0, 1, 3).contiguous()
        # dv size is (bsz, tgt_len, src_len, num_heads * head_dim)
        # if dv is not None:
        # change dv to (bsz * num_heads, tgt_len, src_len, head_dim)
        dv = dv.permute(1, 2, 0, 3).reshape(tgt_len, src_len, bsz * self.num_heads, self.x_head_dim + 2 * self.vec_head_dim).permute(2, 0, 1, 3).contiguous()
        # if dv_dist is not None:
        # change dv_dist to (bsz * num_heads, tgt_len, src_len, head_dim)
        dv_dist = dv_dist.permute(1, 2, 0, 3).reshape(tgt_len, src_len, bsz * self.num_heads, self.x_head_dim + 2 * self.vec_head_dim).permute(2, 0, 1, 3).contiguous()
        # if key_padding_mask is not None:
        # key_padding_mask should be (bsz, src_len)
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len
        # attn_weights size is (bsz * num_heads, tgt_len, src_len, head_dim)
        attn_weights = torch.multiply(q[:, :, None, :], k[:, None, :, :])
        # w_ij is PAE confidence
        # w_ij size is (bsz, tgt_len, src_len)
        # change dimension of w_ij to (bsz * num_heads, tgt_len, src_len, head_dim)
        # if dk_dist is not None:
        assert w_ij is not None
        #     if dk is not None:
        attn_weights *= (dk + dk_dist * w_ij[:, :, :, None].repeat(self.num_heads, 1, 1, self.x_head_dim))
        # add dv and dv_dist
        v = v.unsqueeze(1) + dv + dv_dist * w_ij[:, :, :, None].repeat(self.num_heads, 1, 1, self.x_head_dim + 2 * self.vec_head_dim)
        #     else:
        #         attn_weights *= dk_dist * w_ij
        # else:
        #     if dk is not None:
        #         attn_weights *= dk
        # attn_weights size is (bsz * num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=-1)
        # apply key_padding_mask to attn_weights
        # if key_padding_mask is not None:
        # don't attend to padding symbols
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len).contiguous()
        attn_weights = attn_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
        )
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len).contiguous()
        # apply softmax to attn_weights
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        # first get invariant feature outputs x, size is (bsz * num_heads, tgt_len, head_dim)
        x_out = torch.einsum('bts,btsh->bth', attn_weights, v)
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
        # if not need_head_weights:
        # average attention weights over heads
        attn_weights = attn_weights.mean(dim=0)
        # reshape x_out to (bsz, tgt_len, num_heads * head_dim)
        x_out = x_out.transpose(1, 0).reshape(tgt_len, bsz, self.num_heads * self.x_head_dim).transpose(1, 0).contiguous()
        return x_out, attn_weights


class MultiHeadAttentionSoftMaxStarGraph(nn.Module):
    """Equivariant multi-head attention layer with softmax, apply attention on full graph by default"""
    def __init__(
            self,
            x_channels,
            x_hidden_channels,
            vec_channels,
            vec_hidden_channels,
            share_kv,
            edge_attr_channels,
            edge_attr_dist_channels,
            distance_influence,
            num_heads,
            activation,
            cutoff_lower,
            cutoff_upper,
            use_lora=None,
    ):
        # same as EquivariantPAEMultiHeadAttentionSoftMax, but apply attention on full graph by default
        super(MultiHeadAttentionSoftMaxStarGraph, self).__init__()
        assert x_hidden_channels % num_heads == 0 \
            and vec_channels % num_heads == 0, (
                f"The number of hidden channels x_hidden_channels ({x_hidden_channels}) "
                f"and vec_channels ({vec_channels}) "
                f"must be evenly divisible by the number of "
                f"attention heads ({num_heads})"
            )
        assert vec_hidden_channels == x_channels, (
            f"The number of hidden channels x_channels ({x_channels}) "
            f"and vec_hidden_channels ({vec_hidden_channels}) "
            f"must be equal"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.x_channels = x_channels
        self.x_hidden_channels = x_hidden_channels
        self.x_head_dim = x_hidden_channels // num_heads
        self.vec_channels = vec_channels
        self.vec_hidden_channels = vec_hidden_channels
        # important, not vec_hidden_channels // num_heads
        self.vec_head_dim = vec_channels // num_heads
        self.share_kv = share_kv
        self.layernorm = nn.LayerNorm(x_channels)
        self.act = activation()
        self.cutoff = None
        self.scaling = self.x_head_dim**-0.5
        if use_lora is not None:
            self.q_proj = lora.Linear(x_channels, x_hidden_channels, r=use_lora)
            self.k_proj = lora.Linear(x_channels, x_hidden_channels, r=use_lora) if not share_kv else None
            self.v_proj = lora.Linear(x_channels, x_hidden_channels, r=use_lora)
        else:
            self.q_proj = nn.Linear(x_channels, x_hidden_channels)
            self.k_proj = nn.Linear(x_channels, x_hidden_channels) if not share_kv else None
            self.v_proj = nn.Linear(x_channels, x_hidden_channels)

        self.dk_proj = None
        # self.dk_dist_proj = None
        self.dv_proj = None
        # self.dv_dist_proj = None
        if distance_influence in ["keys", "both"]:
            if use_lora is not None:
                self.dk_proj = lora.Linear(edge_attr_channels, x_hidden_channels, r=use_lora)
                # self.dk_dist_proj = lora.Linear(edge_attr_dist_channels, x_hidden_channels, r=use_lora)
            else:
                self.dk_proj = nn.Linear(edge_attr_channels, x_hidden_channels)
                # self.dk_dist_proj = nn.Linear(edge_attr_dist_channels, x_hidden_channels)

        if distance_influence in ["values", "both"]:
            if use_lora is not None:
                self.dv_proj = lora.Linear(edge_attr_channels, x_hidden_channels, r=use_lora)
                # self.dv_dist_proj = lora.Linear(edge_attr_dist_channels, x_hidden_channels + vec_channels * 2, r=use_lora)
            else:
                self.dv_proj = nn.Linear(edge_attr_channels, x_hidden_channels)
                # self.dv_dist_proj = nn.Linear(edge_attr_dist_channels, x_hidden_channels + vec_channels * 2)
        # set PAE weight as a learnable parameter, basiclly a sigmoid function
        # self.pae_weight = nn.Linear(1, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        # self.pae_weight.weight.data.fill_(-0.5)
        # self.pae_weight.bias.data.fill_(7.5)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)
        # if self.dk_dist_proj:
        #     nn.init.xavier_uniform_(self.dk_dist_proj.weight)
        #     self.dk_dist_proj.bias.data.fill_(0)
        # if self.dv_dist_proj:
        #     nn.init.xavier_uniform_(self.dv_dist_proj.weight)
        #     self.dv_dist_proj.bias.data.fill_(0)

    def forward(self, x, x_center_index, w_ij, f_dist_ij, f_ij, key_padding_mask, return_attn=False):
        # we replaced r_ij to w_ij as pair-wise confidence
        # we add plddt as position-wise confidence
        # edge_index is unused
        x = self.layernorm(x)
        q = self.q_proj(x[x_center_index].unsqueeze(1)) * self.scaling
        v = self.v_proj(x)
        # if self.share_kv:
        #     k = v[:, :, :self.x_head_dim]
        # else:
        k = self.k_proj(x)

        dk = self.act(self.dk_proj(f_ij)) 
        # dk_dist = self.act(self.dk_dist_proj(f_dist_ij)) 
        dv = self.act(self.dv_proj(f_ij)) 
        # dv_dist = self.act(self.dv_dist_proj(f_dist_ij)) 

        # full graph attention
        x, attn = self.attention(
            q=q,
            k=k,
            v=v,
            dk=dk,
            # dk_dist=dk_dist,
            dv=dv,
            # dv_dist=dv_dist,
            # w_ij=nn.functional.sigmoid(self.pae_weight(w_ij.unsqueeze(-1)).squeeze(-1)),
            key_padding_mask=key_padding_mask,
        )
        if return_attn:
            return x, attn
        else:
            return x, None

    def attention(self, q, k, v, dk, dv, key_padding_mask=None, need_head_weights=False):
        # note that q is of shape (bsz, tgt_len, num_heads * head_dim)
        # k, v is of shape (bsz, src_len, num_heads * head_dim)
        # vec is of shape (bsz, src_len, 3, num_heads * head_dim)
        # dk, dk_dist, dv, dv_dist is of shape (bsz, tgt_len, src_len, num_heads * head_dim)
        # d_ij is of shape (bsz, tgt_len, src_len, 3)
        # w_ij is of shape (bsz, tgt_len, src_len)
        # key_padding_mask is of shape (bsz, src_len)
        bsz, tgt_len, _ = q.size()
        src_len = k.size(1)
        # change q size to (bsz * num_heads, tgt_len, head_dim)
        # change k,v size to (bsz * num_heads, src_len, head_dim)
        q = q.transpose(0, 1).reshape(tgt_len, bsz * self.num_heads, self.x_head_dim).transpose(0, 1).contiguous()
        k = k.transpose(0, 1).reshape(src_len, bsz * self.num_heads, self.x_head_dim).transpose(0, 1).contiguous()
        v = v.transpose(0, 1).reshape(src_len, bsz * self.num_heads, self.x_head_dim).transpose(0, 1).contiguous()
        # dk size is (bsz, tgt_len, src_len, num_heads * head_dim)
        # if dk is not None:
        # change dk to (bsz * num_heads, tgt_len, src_len, head_dim)
        dk = dk.permute(1, 2, 0, 3).reshape(tgt_len, src_len, bsz * self.num_heads, self.x_head_dim).permute(2, 0, 1, 3).contiguous()
        # if dk_dist is not None:
        # change dk_dist to (bsz * num_heads, tgt_len, src_len, head_dim)
        # dk_dist = dk_dist.permute(1, 2, 0, 3).reshape(tgt_len, src_len, bsz * self.num_heads, self.x_head_dim).permute(2, 0, 1, 3).contiguous()
        # dv size is (bsz, tgt_len, src_len, num_heads * head_dim)
        # if dv is not None:
        # change dv to (bsz * num_heads, tgt_len, src_len, head_dim)
        dv = dv.permute(1, 2, 0, 3).reshape(tgt_len, src_len, bsz * self.num_heads, self.x_head_dim).permute(2, 0, 1, 3).contiguous()
        # if dv_dist is not None:
        # change dv_dist to (bsz * num_heads, tgt_len, src_len, head_dim)
        # dv_dist = dv_dist.permute(1, 2, 0, 3).reshape(tgt_len, src_len, bsz * self.num_heads, self.x_head_dim + 2 * self.vec_head_dim).permute(2, 0, 1, 3).contiguous()
        # if key_padding_mask is not None:
        # key_padding_mask should be (bsz, src_len)
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len
        # attn_weights size is (bsz * num_heads, tgt_len, src_len, head_dim)
        attn_weights = torch.multiply(q[:, :, None, :], k[:, None, :, :])
        # w_ij is PAE confidence
        # w_ij size is (bsz, tgt_len, src_len)
        # change dimension of w_ij to (bsz * num_heads, tgt_len, src_len, head_dim)
        # if dk_dist is not None:
        # assert w_ij is not None
        #     if dk is not None:
        attn_weights *= dk
        # add dv and dv_dist
        v = v.unsqueeze(1) + dv
        #     else:
        #         attn_weights *= dk_dist * w_ij
        # else:
        #     if dk is not None:
        #         attn_weights *= dk
        # attn_weights size is (bsz * num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=-1)
        # apply key_padding_mask to attn_weights
        # if key_padding_mask is not None:
        # don't attend to padding symbols
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len).contiguous()
        attn_weights = attn_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
        )
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len).contiguous()
        # apply softmax to attn_weights
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        # first get invariant feature outputs x, size is (bsz * num_heads, tgt_len, head_dim)
        x_out = torch.einsum('bts,btsh->bth', attn_weights, v)
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
        # if not need_head_weights:
        # average attention weights over heads
        attn_weights = attn_weights.mean(dim=0)
        # reshape x_out to (bsz, tgt_len, num_heads * head_dim)
        x_out = x_out.transpose(1, 0).reshape(tgt_len, bsz, self.num_heads * self.x_head_dim).transpose(1, 0).contiguous()
        return x_out, attn_weights


# original torchmd-net attention layer, let k, v share the same projection
class EquivariantProMultiHeadAttention(MessagePassing):
    """Equivariant multi-head attention layer."""

    def __init__(
            self,
            x_channels,
            x_hidden_channels,
            vec_channels,
            vec_hidden_channels,
            edge_attr_channels,
            distance_influence,
            num_heads,
            activation,
            attn_activation,
            cutoff_lower,
            cutoff_upper,
    ):
        super(EquivariantMultiHeadAttention, self).__init__(
            aggr="mean", node_dim=0)
        assert x_hidden_channels % num_heads == 0 \
            and vec_channels % num_heads == 0, (
                f"The number of hidden channels x_hidden_channels ({x_hidden_channels}) "
                f"and vec_channels ({vec_channels}) "
                f"must be evenly divisible by the number of "
                f"attention heads ({num_heads})"
            )
        assert vec_hidden_channels == x_channels, (
            f"The number of hidden channels x_channels ({x_channels}) "
            f"and vec_hidden_channels ({vec_hidden_channels}) "
            f"must be equal"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.x_channels = x_channels
        self.x_hidden_channels = x_hidden_channels
        self.x_head_dim = x_hidden_channels // num_heads
        self.vec_channels = vec_channels
        self.vec_hidden_channels = vec_hidden_channels
        # important, not vec_hidden_channels // num_heads
        self.vec_head_dim = vec_channels // num_heads

        self.layernorm = nn.LayerNorm(x_channels)
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(x_channels, x_hidden_channels)
        # self.k_proj = nn.Linear(x_channels, x_hidden_channels)
        self.kv_proj = nn.Linear(
            x_channels, x_hidden_channels + vec_channels * 2)
        self.o_proj = nn.Linear(
            x_hidden_channels, x_channels * 2 + vec_channels)

        self.vec_proj = nn.Linear(
            vec_channels, vec_hidden_channels * 2 + vec_channels, bias=False)

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            self.dk_proj = nn.Linear(edge_attr_channels, x_hidden_channels)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            self.dv_proj = nn.Linear(
                edge_attr_channels, x_hidden_channels + vec_channels * 2)

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.k_proj.weight)
        # self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij, return_attn=False):
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        # k = self.k_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        v = self.kv_proj(x).reshape(-1, self.num_heads,
                                   self.x_head_dim + self.vec_head_dim * 2)
        k = v[:, :, :self.x_head_dim]

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec),
                                       [self.vec_hidden_channels, self.vec_hidden_channels, self.vec_channels], dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.vec_head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.x_head_dim)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.x_head_dim + self.vec_head_dim * 2)
            if self.dv_proj is not None
            else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor,
        # d_ij: Tensor)
        x, vec, attn = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        x = x.reshape(-1, self.x_hidden_channels)
        vec = vec.reshape(-1, 3, self.vec_channels)

        o1, o2, o3 = torch.split(self.o_proj(
            x), [self.vec_channels, self.x_channels, self.x_channels], dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec
        if return_attn:
            return dx, dvec, torch.concat((edge_index.T, attn), dim=1)
        else:
            return dx, dvec, None

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:  # TODO: consider add or multiply dk
            attn = (q_i * k_j * dk).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        # value pathway
        if dv is not None:
            v_j = v_j * dv
        x, vec1, vec2 = torch.split(
            v_j, [self.x_head_dim, self.vec_head_dim, self.vec_head_dim], dim=2)

        # update scalar features
        x = x * attn.unsqueeze(2)
        # update vector features
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * \
            d_ij.unsqueeze(2).unsqueeze(3)
        return x, vec, attn

    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, vec, attn = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec, attn

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    def edge_update(self) -> Tensor:
        pass


# softmax version of torchmd-net attention layer
class EquivariantMultiHeadAttentionSoftMax(EquivariantMultiHeadAttention):
    """Equivariant multi-head attention layer with softmax"""

    def __init__(
            self,
            x_channels,
            x_hidden_channels,
            vec_channels,
            vec_hidden_channels,
            share_kv,
            edge_attr_channels,
            distance_influence,
            num_heads,
            activation,
            attn_activation,
            cutoff_lower,
            cutoff_upper,
            use_lora=None,
    ):
        super(EquivariantMultiHeadAttentionSoftMax, self).__init__(x_channels=x_channels,
                                                                   x_hidden_channels=x_hidden_channels,
                                                                   vec_channels=vec_channels,
                                                                   vec_hidden_channels=vec_hidden_channels,
                                                                   share_kv=share_kv,
                                                                   edge_attr_channels=edge_attr_channels,
                                                                   distance_influence=distance_influence,
                                                                   num_heads=num_heads,
                                                                   activation=activation,
                                                                   attn_activation=attn_activation,
                                                                   cutoff_lower=cutoff_lower,
                                                                   cutoff_upper=cutoff_upper,
                                                                   use_lora=use_lora)
        self.attn_activation = nn.LeakyReLU(0.2)

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij,
                index: Tensor,
                ptr: Optional[Tensor],
                size_i: Optional[int]):
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:  # TODO: consider add or multiply dk
            attn = (q_i * k_j * dk).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)
        attn = softmax(attn, index, ptr, size_i)
        # TODO: consider drop out attn or not.
        # attn = F.dropout(attn, p=self.dropout, training=self.training)
        # value pathway
        if dv is not None:
            v_j = v_j * dv
        x, vec1, vec2 = torch.split(
            v_j, [self.x_head_dim, self.vec_head_dim, self.vec_head_dim], dim=2)

        # update scalar features
        x = x * attn.unsqueeze(2)
        # update vector features
        vec = (vec1.unsqueeze(1) * vec_j + vec2.unsqueeze(1) * d_ij.unsqueeze(2).unsqueeze(3)) \
            * attn.unsqueeze(1).unsqueeze(3)
        return x, vec, attn


# softmax version of torchmd-net attention layer, add pair-wise confidence of PAE
class EquivariantPAEMultiHeadAttentionSoftMax(EquivariantPAEMultiHeadAttention):
    """Equivariant multi-head attention layer with softmax"""

    def __init__(
            self,
            x_channels,
            x_hidden_channels,
            vec_channels,
            vec_hidden_channels,
            share_kv,
            edge_attr_channels,
            edge_attr_dist_channels,
            distance_influence,
            num_heads,
            activation,
            attn_activation,
            cutoff_lower,
            cutoff_upper,
            use_lora=None,
    ):
        super(EquivariantPAEMultiHeadAttentionSoftMax, self).__init__(
            x_channels=x_channels,
            x_hidden_channels=x_hidden_channels,
            vec_channels=vec_channels,
            vec_hidden_channels=vec_hidden_channels,
            share_kv=share_kv,
            edge_attr_channels=edge_attr_channels,
            edge_attr_dist_channels=edge_attr_dist_channels,
            distance_influence=distance_influence,
            num_heads=num_heads,
            activation=activation,
            attn_activation=attn_activation,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            use_lora=use_lora)
        self.attn_activation = nn.LeakyReLU(0.2)

    def message(self, q_i, k_j, v_j, vec_j, dk, dk_dist, dv, dv_dist, d_ij, w_ij,
                index: Tensor,
                ptr: Optional[Tensor],
                size_i: Optional[int]):
        # attention mechanism
        attn = (q_i * k_j)
        if dk is not None:
            attn += dk
        if dk_dist is not None:
            attn += dk_dist * w_ij.unsqueeze(1).unsqueeze(2)
        attn = attn.sum(dim=-1)
        # attention activation function
        attn = self.attn_activation(attn)
        attn = softmax(attn, index, ptr, size_i)
        # TODO: consider drop out attn or not.
        # attn = F.dropout(attn, p=self.dropout, training=self.training)
        # value pathway
        if dv is not None:
            v_j += dv
        if dv_dist is not None:
            v_j += dv_dist * w_ij.unsqueeze(1).unsqueeze(2)
        x, vec1, vec2 = torch.split(
            v_j, [self.x_head_dim, self.vec_head_dim, self.vec_head_dim], dim=2)

        # update scalar features
        x = x * attn.unsqueeze(2)
        # update vector features
        vec = (vec1.unsqueeze(1) * vec_j + vec2.unsqueeze(1) * d_ij.unsqueeze(2).unsqueeze(3)) \
            * attn.unsqueeze(1).unsqueeze(3)
        return x, vec, attn

# softmax version of torchmd-net attention layer, add pair-wise confidence of PAE
class EquivariantWeightedPAEMultiHeadAttentionSoftMax(EquivariantWeightedPAEMultiHeadAttention):
    """Equivariant multi-head attention layer with softmax"""

    def __init__(
            self,
            x_channels,
            x_hidden_channels,
            vec_channels,
            vec_hidden_channels,
            share_kv,
            edge_attr_channels,
            edge_attr_dist_channels,
            distance_influence,
            num_heads,
            activation,
            attn_activation,
            cutoff_lower,
            cutoff_upper,
            use_lora=None,
    ):
        super(EquivariantWeightedPAEMultiHeadAttentionSoftMax, self).__init__(
            x_channels=x_channels,
            x_hidden_channels=x_hidden_channels,
            vec_channels=vec_channels,
            vec_hidden_channels=vec_hidden_channels,
            share_kv=share_kv,
            edge_attr_channels=edge_attr_channels,
            edge_attr_dist_channels=edge_attr_dist_channels,
            distance_influence=distance_influence,
            num_heads=num_heads,
            activation=activation,
            attn_activation=attn_activation,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            use_lora=use_lora)
        self.attn_activation = nn.LeakyReLU(0.2)

    def message(self, q_i, k_j, v_j, vec_j, dk, dk_dist, dv, dv_dist, d_ij, w_ij,
                index: Tensor,
                ptr: Optional[Tensor],
                size_i: Optional[int]):
        # attention mechanism
        attn = (q_i * k_j)
        if dk_dist is not None:
            if dk is not None:
                attn *= (dk + dk_dist * w_ij.unsqueeze(1).unsqueeze(2))
            else:
                attn *= dk_dist * w_ij
        else:
            if dk is not None:
                attn *= dk
        attn = attn.sum(dim=-1)
        # attention activation function
        attn = self.attn_activation(attn)
        attn = softmax(attn, index, ptr, size_i)
        # TODO: consider drop out attn or not.
        # attn = F.dropout(attn, p=self.dropout, training=self.training)
        # value pathway
        if dv is not None:
            v_j += dv
        if dv_dist is not None:
            v_j += dv_dist * w_ij.unsqueeze(1).unsqueeze(2)
        x, vec1, vec2 = torch.split(
            v_j, [self.x_head_dim, self.vec_head_dim, self.vec_head_dim], dim=2)

        # update scalar features
        x = x * attn.unsqueeze(2)
        # update vector features
        vec = (vec1.unsqueeze(1) * vec_j + vec2.unsqueeze(1) * d_ij.unsqueeze(2).unsqueeze(3)) \
            * attn.unsqueeze(1).unsqueeze(3)
        return x, vec, attn


# MSA encoder adapted from gMVP
class MSAEncoder(nn.Module):
    def __init__(self, num_species, pairwise_type, weighting_schema):
        """[summary]

        Args:
            num_species (int): Number of species to use from MSA. [1,200] // 200 used in default gMVP
            pairwise_type ([str]): method for calculating pairwise coevolution. only "cov" supported
            weighting_schema ([str]): species weighting type; "spe" -> use dense layer to weight speices 
                                        "none" -> uniformly weight species 

        Raises:
            NotImplementedError: [description]
        """        
        super(MSAEncoder, self).__init__()
        self.num_species = num_species
        self.pairwise_type = pairwise_type
        self.weighting_schema = weighting_schema
        if self.weighting_schema == 'spe':
            self.W = nn.parameter.Parameter(
                torch.zeros((1, num_species)), 
                requires_grad=True)

        elif self.weighting_schema  == 'none':
            self.W = torch.tensor(1.0 / self.num_species).repeat(self.num_species)
        else:
            raise NotImplementedError
       
    def forward(self, x, edge_index):
        # x: L nodes x N num_species
        shape  = x.shape
        L, N = shape[0], shape[1]
        E = edge_index.shape[1]
        
        A = 21 # number of amino acids 
        x = x[:, :self.num_species]
        if self.weighting_schema == 'spe':
            sm = torch.nn.Softmax(dim=-1)
            W = sm(self.W)
        else:
            W = self.W
        x = nn.functional.one_hot(x.type(torch.int64), A).type(torch.float32) # L x N x A
        x1 = torch.matmul(W[:, None], x) # L x 1 x A

        if self.pairwise_type  == 'fre':
            x2 = torch.matmul(x[edge_index[0], :, :, None], x[edge_index[1], :, None, :]) # E x N x A x A
            x2 = x2.reshape((E, N, A * A)) # E x N x (A x A)
            x2 = (W[:, :, None] * x2).sum(dim=1) # E x (A x A)
        elif self.pairwise_type == 'cov':
            #numerical stability
            x2 = torch.matmul(x[edge_index[0], :, :, None], x[edge_index[1], :, None, :]) # E x N x A x A
            x2 = (W[:, :, None, None] * x2).sum(dim=1) # E x A x A
            x2_t = x1[edge_index[0], 0, :, None] * x1[edge_index[1], 0, None, :] # E x A x A
            x2 = (x2 - x2_t).reshape(E, A * A) # E x (A x A)
            x2 = x2.reshape(E, A * A) # E x (A x A)
            norm = torch.sqrt(torch.sum(torch.square(x2), dim=-1, keepdim=True) + 1e-12)
            x2 = torch.cat([x2, norm], dim=-1) # E x (A x A + 1)
        elif self.pairwise_type == 'cov_all':
            print('cov_all not implemented in EvolEncoder2')
            raise NotImplementedError
        elif self.pairwise_type == 'inv_cov':
            print('in_cov not implemented in EvolEncoder2')
            raise NotImplementedError
        elif self.pairwise_type == 'none':
            x2 = None
        else:
            raise NotImplementedError(
                f'pairwise_type {self.pairwise_type} not implemented')

        x1 = torch.squeeze(x1, dim=1) # L x A

        return x1, x2


# MSA encoder adapted from gMVP
class MSAEncoderFullGraph(nn.Module):
    def __init__(self, num_species, pairwise_type, weighting_schema):
        """[summary]

        Args:
            num_species (int): Number of species to use from MSA. [1,200] // 200 used in default gMVP
            pairwise_type ([str]): method for calculating pairwise coevolution. only "cov" supported
            weighting_schema ([str]): species weighting type; "spe" -> use dense layer to weight speices 
                                        "none" -> uniformly weight species 

        Raises:
            NotImplementedError: [description]
        """        
        super(MSAEncoderFullGraph, self).__init__()
        self.num_species = num_species
        self.pairwise_type = pairwise_type
        self.weighting_schema = weighting_schema
        if self.weighting_schema == 'spe':
            self.W = nn.parameter.Parameter(
                torch.zeros((num_species)), 
                requires_grad=True)

        elif self.weighting_schema  == 'none':
            self.W = torch.tensor(1.0 / self.num_species).repeat(self.num_species)
        else:
            raise NotImplementedError
       
    def forward(self, x):
        # x: B batch size x L lenth x N num_species
        shape  = x.shape
        B, L, N = shape[0], shape[1], shape[2]
        A = 21 # number of amino acids 
        x = x[:, :, :self.num_species]
        if self.weighting_schema == 'spe':
            W = torch.nn.functional.softmax(self.W, dim=-1)
        else:
            W = self.W
        x = nn.functional.one_hot(x.type(torch.int64), A).type(torch.float32) # B x L x N x A
        x1 = torch.einsum('blna,n->bla', x, W) # B x L x A

        if self.pairwise_type == 'cov':
            #numerical stability
            # x2 = torch.einsum('bLnA,blna,n->bLlAa', x, x, W) # B x L x L x A x A, check if ram supports this
            # x2_t = x1[:, :, None, :, None] * x1[:, None, :, None, :] # B x L x L x A x A
            # x2 = (x2 - x2_t).reshape(B, L, L, A * A) # B x L x L x (A x A)
            # complete that in one line to save memory
            x2 = (torch.einsum('bLnA,blna,n->bLlAa', x, x, W) - x1[:, :, None, :, None] * x1[:, None, :, None, :]).reshape(B, L, L, A * A)
            norm = torch.sqrt(torch.sum(torch.square(x2), dim=-1, keepdim=True) + 1e-12) # B x L x L x 1
            x2 = torch.cat([x2, norm], dim=-1) # B x L x L x (A x A + 1)
        elif self.pairwise_type == 'cov_all':
            print('cov_all not implemented in EvolEncoder2')
            raise NotImplementedError
        elif self.pairwise_type == 'inv_cov':
            print('in_cov not implemented in EvolEncoder2')
            raise NotImplementedError
        elif self.pairwise_type == 'none':
            x2 = None
        else:
            raise NotImplementedError(
                f'pairwise_type {self.pairwise_type} not implemented')
        return x1, x2


class NodeToEdgeAttr(nn.Module):
    def __init__(self, node_channel, hidden_channel, edge_attr_channel, use_lora=None, layer_norm=False):
        super().__init__()
        self.layer_norm = layer_norm
        if layer_norm:
            self.layernorm = nn.LayerNorm(node_channel)
        if use_lora is not None:
            self.proj = lora.Linear(node_channel, hidden_channel * 2, bias=True, r=use_lora)
            self.o_proj = lora.Linear(2 * hidden_channel, edge_attr_channel, r=use_lora)
        else:
            self.proj = nn.Linear(node_channel, hidden_channel * 2, bias=True)
            self.o_proj = nn.Linear(2 * hidden_channel, edge_attr_channel, bias=True)

        torch.nn.init.zeros_(self.proj.bias)
        torch.nn.init.zeros_(self.o_proj.bias)

    def forward(self, x, edge_index):
        """
        Inputs:
          x: N x sequence_state_dim

        Output:
          edge_attr: edge_index.shape[0] x pairwise_state_dim

        Intermediate state:
          B x L x L x 2*inner_dim
        """
        x = self.layernorm(x) if self.layer_norm else x
        q, k = self.proj(x).chunk(2, dim=-1)

        prod = q[edge_index[0], :] * k[edge_index[1], :]
        diff = q[edge_index[0], :] - k[edge_index[1], :]

        edge_attr = torch.cat([prod, diff], dim=-1)
        edge_attr = self.o_proj(edge_attr)

        return edge_attr


class MultiplicativeUpdate(MessagePassing):
    def __init__(self, vec_in_channel, hidden_channel, hidden_vec_channel, ee_channels=None, use_lora=None, layer_norm=True) -> None:
        super(MultiplicativeUpdate, self).__init__(aggr="mean")
        self.vec_in_channel = vec_in_channel
        self.hidden_channel = hidden_channel
        self.hidden_vec_channel = hidden_vec_channel

        if use_lora is not None:
            self.linear_a_p = lora.Linear(self.vec_in_channel, self.hidden_vec_channel, bias=False, r=use_lora)
            self.linear_b_p = lora.Linear(self.vec_in_channel, self.hidden_vec_channel, bias=False, r=use_lora)
            self.linear_g = lora.Linear(self.hidden_vec_channel, self.hidden_channel, r=use_lora)
        else:
            self.linear_a_p = nn.Linear(self.vec_in_channel, self.hidden_vec_channel, bias=False)
            self.linear_b_p = nn.Linear(self.vec_in_channel, self.hidden_vec_channel, bias=False)
            self.linear_g = nn.Linear(self.hidden_vec_channel, self.hidden_channel)
        if ee_channels is not None:
            if use_lora is not None:
                self.linear_ee = lora.Linear(ee_channels, self.hidden_channel, r=use_lora)
            else:
                self.linear_ee = nn.Linear(ee_channels, self.hidden_channel)
        else:
            self.linear_ee = None
        self.layer_norm = layer_norm
        if layer_norm:
            self.layer_norm_in = nn.LayerNorm(self.hidden_channel)
            self.layer_norm_out = nn.LayerNorm(self.hidden_channel)

        self.sigmoid = nn.Sigmoid()

    def forward(self,
                edge_attr: torch.Tensor,
                edge_vec: torch.Tensor,
                edge_edge_index: torch.Tensor,
                edge_edge_attr: torch.Tensor,
                ) -> torch.Tensor:
        """
        Args:
            edge_vec:
                [*, 3, in_channel] input tensor
            edge_attr:
                [*, hidden_channel] input mask
        Returns:
            [*, hidden_channel] output tensor
        """
        if self.layer_norm:
            x = self.layer_norm_in(edge_attr)
        x = self.propagate(edge_index=edge_edge_index, 
                           a=self.linear_a_p(edge_vec).reshape(edge_attr.shape[0], -1),
                           b=self.linear_b_p(edge_vec).reshape(edge_attr.shape[0], -1),
                           edge_attr=x,
                           ee_ij=edge_edge_attr, )
        if self.layer_norm:
            x = self.layer_norm_out(x)
        edge_attr = edge_attr + x
        return edge_attr

    def message(self, a_i: Tensor, b_j: Tensor, edge_attr_j: Tensor, ee_ij: Tensor,) -> Tensor:
        # a: [*, 3, hidden_channel]
        # b: [*, 3, hidden_channel]
        s = (a_i.reshape(-1, 3, self.hidden_vec_channel).permute(0, 2, 1) \
             * b_j.reshape(-1, 3, self.hidden_vec_channel).permute(0, 2, 1)).sum(dim=-1)
        if ee_ij is not None and self.linear_ee is not None:
            s = self.sigmoid(self.linear_ee(ee_ij) + self.linear_g(s))
        else:
            s = self.sigmoid(self.linear_g(s))
        return s * edge_attr_j


# let k v share the same weight
class EquivariantTriAngularMultiHeadAttention(MessagePassing):
    """Equivariant multi-head attention layer. Add Triangular update between edges."""

    def __init__(
            self,
            x_channels,
            x_hidden_channels,
            vec_channels,
            vec_hidden_channels,
            edge_attr_channels,
            distance_influence,
            num_heads,
            activation,
            attn_activation,
            cutoff_lower,
            cutoff_upper,
            triangular_update=False,
            ee_channels=None,
    ):
        super(EquivariantTriAngularMultiHeadAttention, self).__init__(aggr="mean", node_dim=0)

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.x_channels = x_channels
        self.x_hidden_channels = x_hidden_channels
        self.x_head_dim = x_hidden_channels // num_heads
        self.vec_channels = vec_channels
        self.vec_hidden_channels = vec_hidden_channels
        self.ee_channels = ee_channels
        # important, not vec_hidden_channels // num_heads

        self.layernorm_in = nn.LayerNorm(x_channels)
        self.layernorm_out = nn.LayerNorm(x_hidden_channels)

        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()

        self.q_proj = nn.Linear(x_channels, x_hidden_channels)
        self.kv_proj = nn.Linear(x_channels, x_hidden_channels)
        # self.v_proj = nn.Linear(x_channels, x_hidden_channels)
        self.o_proj = nn.Linear(x_hidden_channels, x_hidden_channels)
        self.out = nn.Linear(x_hidden_channels, x_channels)
        # add residue to x
        # self.residue_hidden = nn.Linear(x_channels, x_hidden_channels)

        self.dk_proj = nn.Linear(edge_attr_channels, x_hidden_channels)
        self.triangular_update = triangular_update
        if self.triangular_update:
            self.edge_triangle_start_update = MultiplicativeUpdate(vec_in_channel=vec_channels,
                                                                hidden_channel=edge_attr_channels,
                                                                hidden_vec_channel=vec_hidden_channels,
                                                                ee_channels=ee_channels, )
            self.edge_triangle_end_update = MultiplicativeUpdate(vec_in_channel=vec_channels,
                                                                hidden_channel=edge_attr_channels,
                                                                hidden_vec_channel=vec_hidden_channels,
                                                                ee_channels=ee_channels, )
            self.node_to_edge_attr = NodeToEdgeAttr(node_channel=x_channels,
                                                    hidden_channel=x_hidden_channels,
                                                    edge_attr_channel=edge_attr_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm_in.reset_parameters()
        self.layernorm_out.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.v_proj.weight)
        # self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)

    def get_start_index(self, edge_index):
        edge_start_index = []
        start_node_count = edge_index[0].unique(return_counts=True)
        start_nodes = start_node_count[0][start_node_count[1] > 1]
        for i in start_nodes:
            node_start_index = torch.where(edge_index[0] == i)[0]
            candidates = torch.combinations(node_start_index, r=2).T
            edge_start_index.append(torch.cat([candidates, candidates.flip(0)], dim=1))
        edge_start_index = torch.concat(edge_start_index, dim=1)
        edge_start_index = edge_start_index[:, edge_start_index[0] != edge_start_index[1]]
        return edge_start_index

    def get_end_index(self, edge_index):
        edge_end_index = []
        end_node_count = edge_index[1].unique(return_counts=True)
        end_nodes = end_node_count[0][end_node_count[1] > 1]
        for i in end_nodes:
            node_end_index = torch.where(edge_index[1] == i)[0]
            candidates = torch.combinations(node_end_index, r=2).T
            edge_end_index.append(torch.cat([candidates, candidates.flip(0)], dim=1))
        edge_end_index = torch.concat(edge_end_index, dim=1)
        edge_end_index = edge_end_index[:, edge_end_index[0] != edge_end_index[1]]
        return edge_end_index

    def forward(self, x, coords, edge_index, edge_attr, edge_vec, return_attn=False):
        residue = x
        x = self.layernorm_in(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        k = self.kv_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        v = k

        # point ettr to edge_attr
        if self.triangular_update:
            edge_attr += self.node_to_edge_attr(x, edge_index)

            # Triangular edge update
            # TODO: Add drop out layers here
            edge_edge_index = self.get_start_index(edge_index)
            if self.ee_channels is not None:
                edge_edge_attr = coords[edge_index[1][edge_edge_index[0]], :, [0]] - coords[edge_index[1][edge_edge_index[1]], :, [0]]
                edge_edge_attr = torch.norm(edge_edge_attr, dim=-1, keepdim=True)
            else:
                edge_edge_attr = None
            edge_attr = self.edge_triangle_start_update(
                edge_attr, edge_vec, 
                edge_edge_index,
                edge_edge_attr
            )
            edge_edge_index = self.get_end_index(edge_index)
            if self.ee_channels is not None:
                edge_edge_attr = coords[edge_index[0][edge_edge_index[0]], :, [0]] - coords[edge_index[0][edge_edge_index[1]], :, [0]]
                edge_edge_attr = torch.norm(edge_edge_attr, dim=-1, keepdim=True)
            else:
                edge_edge_attr = None
            edge_attr = self.edge_triangle_end_update(
                edge_attr, edge_vec, 
                edge_edge_index,
                edge_edge_attr
            )
            del edge_edge_attr, edge_edge_index

        dk = (
            self.act(self.dk_proj(edge_attr)).reshape(-1, self.num_heads, self.x_head_dim) 
            if self.dk_proj is not None else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor,
        # d_ij: Tensor)
        x, attn = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            size=None,
        )
        x = x.reshape(-1, self.x_hidden_channels)
        x = residue + x
        x = self.layernorm_out(x)
        x = gelu(self.o_proj(x))
        x = self.out(x)
        del residue, q, k, v, dk
        if return_attn:
            return x, edge_attr, torch.concat((edge_index.T, attn), dim=1)
        else:
            return x, edge_attr, None

    def message(self, q_i, k_j, v_j, dk):
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:  # TODO: consider add or multiply dk
            attn = (q_i * k_j * dk).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn)

        # update scalar features
        x = v_j * attn.unsqueeze(2)
        return x, attn

    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, attn = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x, attn

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    def edge_update(self) -> Tensor:
        pass


# let k v share the same weight, dropout attention weights, with option LoRA
class EquivariantTriAngularDropMultiHeadAttention(MessagePassing):
    """Equivariant multi-head attention layer. Add Triangular update between edges."""

    def __init__(
            self,
            x_channels,
            x_hidden_channels,
            vec_channels,
            vec_hidden_channels,
            edge_attr_channels,
            distance_influence,
            num_heads,
            activation,
            attn_activation,
            rbf_channels, 
            triangular_update=False,
            ee_channels=None,
            drop_out_rate=0.0,
            use_lora=None,
            layer_norm=True,
    ):
        super(EquivariantTriAngularDropMultiHeadAttention, self).__init__(aggr="mean", node_dim=0)

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.x_channels = x_channels
        self.x_hidden_channels = x_hidden_channels
        self.x_head_dim = x_hidden_channels // num_heads
        self.vec_channels = vec_channels
        self.vec_hidden_channels = vec_hidden_channels
        self.ee_channels = ee_channels
        self.rbf_channels = rbf_channels
        self.layer_norm = layer_norm
        # important, not vec_hidden_channels // num_heads
        if layer_norm:
            self.layernorm_in = nn.LayerNorm(x_channels)
            self.layernorm_out = nn.LayerNorm(x_hidden_channels)

        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()

        if use_lora is not None:
            self.q_proj = lora.Linear(x_channels, x_hidden_channels, r=use_lora)
            self.kv_proj = lora.Linear(x_channels, x_hidden_channels, r=use_lora)
            self.dk_proj = lora.Linear(edge_attr_channels, x_hidden_channels, r=use_lora)
            self.o_proj = lora.Linear(x_hidden_channels, x_hidden_channels, r=use_lora)
        else:
            self.q_proj = nn.Linear(x_channels, x_hidden_channels)
            self.kv_proj = nn.Linear(x_channels, x_hidden_channels)
            self.dk_proj = nn.Linear(edge_attr_channels, x_hidden_channels)
            self.o_proj = nn.Linear(x_hidden_channels, x_hidden_channels)

        self.triangular_drop = nn.Dropout(drop_out_rate)
        self.rbf_drop = nn.Dropout(drop_out_rate)
        self.dense_drop = nn.Dropout(drop_out_rate)
        self.dropout = nn.Dropout(drop_out_rate)
        self.triangular_update = triangular_update
        if self.triangular_update:
            self.edge_triangle_end_update = MultiplicativeUpdate(vec_in_channel=vec_channels,
                                                                 hidden_channel=edge_attr_channels,
                                                                 hidden_vec_channel=vec_hidden_channels,
                                                                 ee_channels=ee_channels, 
                                                                 layer_norm=layer_norm,
                                                                 use_lora=use_lora)
            self.node_to_edge_attr = NodeToEdgeAttr(node_channel=x_channels,
                                                    hidden_channel=x_hidden_channels,
                                                    edge_attr_channel=edge_attr_channels,
                                                    use_lora=use_lora)
            self.triangle_update_dropout = nn.Dropout(0.5)
        self.reset_parameters()

    def reset_parameters(self):
        if self.layer_norm:
            self.layernorm_in.reset_parameters()
            self.layernorm_out.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.v_proj.weight)
        # self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)

    def get_start_index(self, edge_index):
        edge_start_index = []
        start_node_count = edge_index[0].unique(return_counts=True)
        start_nodes = start_node_count[0][start_node_count[1] > 1]
        for i in start_nodes:
            node_start_index = torch.where(edge_index[0] == i)[0]
            candidates = torch.combinations(node_start_index, r=2).T
            edge_start_index.append(torch.cat([candidates, candidates.flip(0)], dim=1))
        edge_start_index = torch.concat(edge_start_index, dim=1)
        edge_start_index = edge_start_index[:, edge_start_index[0] != edge_start_index[1]]
        return edge_start_index

    def get_end_index(self, edge_index):
        edge_end_index = []
        end_node_count = edge_index[1].unique(return_counts=True)
        end_nodes = end_node_count[0][end_node_count[1] > 1]
        for i in end_nodes:
            node_end_index = torch.where(edge_index[1] == i)[0]
            candidates = torch.combinations(node_end_index, r=2).T
            edge_end_index.append(torch.cat([candidates, candidates.flip(0)], dim=1))
        edge_end_index = torch.concat(edge_end_index, dim=1)
        edge_end_index = edge_end_index[:, edge_end_index[0] != edge_end_index[1]]
        return edge_end_index

    def forward(self, x, coords, edge_index, edge_attr, edge_vec, return_attn=False):
        residue = x
        if self.layer_norm:
            x = self.layernorm_in(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        k = self.kv_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        v = k

        # point ettr to edge_attr
        if self.triangular_update:
            edge_attr += self.node_to_edge_attr(x, edge_index)
            # Triangular edge update
            # TODO: Add drop out layers here
            edge_edge_index = self.get_end_index(edge_index)
            edge_edge_index = edge_edge_index[:, self.triangular_drop(
                torch.ones(edge_edge_index.shape[1], device=edge_edge_index.device)
                ).to(torch.bool)]
            if self.ee_channels is not None:
                edge_edge_attr = coords[edge_index[0][edge_edge_index[0]], :, [0]] - coords[edge_index[0][edge_edge_index[1]], :, [0]]
                edge_edge_attr = torch.norm(edge_edge_attr, dim=-1, keepdim=True)
            else:
                edge_edge_attr = None
            edge_attr = self.edge_triangle_end_update(
                edge_attr, edge_vec, 
                edge_edge_index,
                edge_edge_attr
            )
            del edge_edge_attr, edge_edge_index

        # drop rbfs
        edge_attr = torch.cat((edge_attr[:, :-self.rbf_channels],
                              self.rbf_drop(edge_attr[:, -self.rbf_channels:])), 
                              dim=-1)

        dk = (
            self.act(self.dk_proj(edge_attr)).reshape(-1, self.num_heads, self.x_head_dim) 
            if self.dk_proj is not None else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor,
        # d_ij: Tensor)
        x, attn = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            size=None,
        )
        x = x.reshape(-1, self.x_hidden_channels)
        if self.layer_norm:
            x = self.layernorm_out(x)
        x = self.dense_drop(x)
        x = residue + gelu(x)
        x = self.o_proj(x)
        x = self.dropout(x)
        del residue, q, k, v, dk
        if return_attn:
            return x, edge_attr, torch.concat((edge_index.T, attn), dim=1)
        else:
            return x, edge_attr, None

    def message(self, q_i, k_j, v_j, dk):
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:  # TODO: consider add or multiply dk
            attn = (q_i * k_j * dk).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn)

        # update scalar features
        x = v_j * attn.unsqueeze(2)
        return x, attn

    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, attn = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x, attn

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    def edge_update(self) -> Tensor:
        pass


# let k v share the same weight
class EquivariantTriAngularStarMultiHeadAttention(MessagePassing):
    """
    Equivariant multi-head attention layer. Add Triangular update between edges. Only update the center node.
    """

    def __init__(
            self,
            x_channels,
            x_hidden_channels,
            vec_channels,
            vec_hidden_channels,
            edge_attr_channels,
            distance_influence,
            num_heads,
            activation,
            attn_activation,
            cutoff_lower,
            cutoff_upper,
            triangular_update=False,
            ee_channels=None,
    ):
        super(EquivariantTriAngularStarMultiHeadAttention, self).__init__(aggr="mean", node_dim=0)

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.x_channels = x_channels
        self.x_hidden_channels = x_hidden_channels
        self.x_head_dim = x_hidden_channels // num_heads
        self.vec_channels = vec_channels
        self.vec_hidden_channels = vec_hidden_channels
        self.ee_channels = ee_channels
        # important, not vec_hidden_channels // num_heads

        # self.layernorm_in = nn.LayerNorm(x_channels)
        self.layernorm_out = nn.LayerNorm(x_hidden_channels)

        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()

        self.q_proj = nn.Linear(x_channels, x_hidden_channels)
        self.kv_proj = nn.Linear(x_channels, x_hidden_channels)
        # self.v_proj = nn.Linear(x_channels, x_hidden_channels)
        # self.o_proj = nn.Linear(x_hidden_channels, x_hidden_channels)
        # self.out = nn.Linear(x_hidden_channels, x_channels)
        # add residue to x
        # self.residue_hidden = nn.Linear(x_channels, x_hidden_channels)
        self.gru = nn.GRUCell(x_channels, x_channels)
        self.dk_proj = nn.Linear(edge_attr_channels, x_hidden_channels)
        self.triangular_update = triangular_update
        if self.triangular_update:
            # self.edge_triangle_start_update = MultiplicativeUpdate(vec_in_channel=vec_channels,
            #                                                     hidden_channel=edge_attr_channels,
            #                                                     hidden_vec_channel=vec_hidden_channels,
            #                                                     ee_channels=ee_channels, )
            self.edge_triangle_end_update = MultiplicativeUpdate(vec_in_channel=vec_channels,
                                                                hidden_channel=edge_attr_channels,
                                                                hidden_vec_channel=vec_hidden_channels,
                                                                ee_channels=ee_channels, )
            self.node_to_edge_attr = NodeToEdgeAttr(node_channel=x_channels,
                                                    hidden_channel=x_hidden_channels,
                                                    edge_attr_channel=edge_attr_channels)

        self.reset_parameters()

    def reset_parameters(self):
        # self.layernorm_in.reset_parameters()
        self.layernorm_out.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.v_proj.weight)
        # self.v_proj.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.o_proj.weight)
        # self.o_proj.bias.data.fill_(0)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)

    def get_start_index(self, edge_index):
        edge_start_index = []
        start_node_count = edge_index[0].unique(return_counts=True)
        start_nodes = start_node_count[0][start_node_count[1] > 1]
        for i in start_nodes:
            node_start_index = torch.where(edge_index[0] == i)[0]
            candidates = torch.combinations(node_start_index, r=2).T
            edge_start_index.append(torch.cat([candidates, candidates.flip(0)], dim=1))
        edge_start_index = torch.concat(edge_start_index, dim=1)
        edge_start_index = edge_start_index[:, edge_start_index[0] != edge_start_index[1]]
        return edge_start_index

    def get_end_index(self, edge_index):
        edge_end_index = []
        end_node_count = edge_index[1].unique(return_counts=True)
        end_nodes = end_node_count[0][end_node_count[1] > 1]
        for i in end_nodes:
            node_end_index = torch.where(edge_index[1] == i)[0]
            candidates = torch.combinations(node_end_index, r=2).T
            edge_end_index.append(torch.cat([candidates, candidates.flip(0)], dim=1))
        edge_end_index = torch.concat(edge_end_index, dim=1)
        edge_end_index = edge_end_index[:, edge_end_index[0] != edge_end_index[1]]
        return edge_end_index

    def forward(self, x, coords, edge_index, edge_attr, edge_vec, return_attn=False):
        # perform topK pooling
        end_node_count = edge_index[1].unique(return_counts=True)
        center_nodes = end_node_count[0][end_node_count[1] > 1]
        other_nodes = end_node_count[0][end_node_count[1] <= 1]
        residue = x[center_nodes] # batch_size * x_channels
        # filter edge_index and edge_attr to from context to center only
        edge_attr = edge_attr[torch.isin(edge_index[1], center_nodes), :]
        edge_vec = edge_vec[torch.isin(edge_index[1], center_nodes), :]
        edge_index = edge_index[:, torch.isin(edge_index[1], center_nodes)]
        # x itself is q, k and v
        q = self.q_proj(residue).reshape(-1, self.num_heads, self.x_head_dim)
        kv = self.kv_proj(x[other_nodes]).reshape(-1, self.num_heads, self.x_head_dim)
        qkv = torch.zeros(x.shape[0], self.num_heads, self.x_head_dim).to(x.device, non_blocking=True)
        qkv[center_nodes] = q
        qkv[other_nodes] = kv
        # point ettr to edge_attr
        if self.triangular_update:
            edge_attr += self.node_to_edge_attr(x, edge_index)
            # Triangular edge update
            # TODO: Add drop out layers here
            edge_edge_index = self.get_end_index(edge_index)
            if self.ee_channels is not None:
                edge_edge_attr = coords[edge_index[0][edge_edge_index[0]], :, [0]] - coords[edge_index[0][edge_edge_index[1]], :, [0]]
                edge_edge_attr = torch.norm(edge_edge_attr, dim=-1, keepdim=True)
            else:
                edge_edge_attr = None
            edge_attr = self.edge_triangle_end_update(
                edge_attr, edge_vec, 
                edge_edge_index,
                edge_edge_attr
            )
            del edge_edge_attr, edge_edge_index

        dk = (
            self.act(self.dk_proj(edge_attr)).reshape(-1, self.num_heads, self.x_head_dim) 
            if self.dk_proj is not None else None
        ) # TODO: check self.act

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor,
        # d_ij: Tensor)
        x, attn = self.propagate(
            edge_index,
            q=qkv,
            k=qkv,
            v=qkv,
            dk=dk,
            size=None,
        )
        x = x.reshape(-1, self.x_hidden_channels)
        # only get the center nodes
        x = x[center_nodes]
        x = self.layernorm_out(x)
        x = self.gru(residue, x)
        del residue, dk
        if return_attn:
            return x, edge_attr, torch.concat((edge_index.T, attn), dim=1)
        else:
            return x, edge_attr, None

    def message(self, q_i, k_j, v_j, dk):
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:  # TODO: consider add or multiply dk
            attn = (q_i * k_j + dk).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn) / self.x_head_dim

        # update scalar features
        x = v_j * attn.unsqueeze(2)
        return x, attn

    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, attn = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x, attn

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    def edge_update(self) -> Tensor:
        pass


# let k v share the same weight, dropout attention weights, with option LoRA
class EquivariantTriAngularStarDropMultiHeadAttention(MessagePassing):
    """
    Equivariant multi-head attention layer. Add Triangular update between edges. Only update the center node.
    """

    def __init__(
            self,
            x_channels,
            x_hidden_channels,
            vec_channels,
            vec_hidden_channels,
            edge_attr_channels,
            distance_influence,
            num_heads,
            activation,
            attn_activation,
            rbf_channels,
            triangular_update=False,
            ee_channels=None,
            drop_out_rate=0.0,
            use_lora=None,
    ):
        super(EquivariantTriAngularStarDropMultiHeadAttention, self).__init__(aggr="mean", node_dim=0)

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.x_channels = x_channels
        self.x_hidden_channels = x_hidden_channels
        self.x_head_dim = x_hidden_channels // num_heads
        self.vec_channels = vec_channels
        self.vec_hidden_channels = vec_hidden_channels
        self.ee_channels = ee_channels
        self.rbf_channels = rbf_channels
        # important, not vec_hidden_channels // num_heads

        # self.layernorm_in = nn.LayerNorm(x_channels)
        self.layernorm_out = nn.LayerNorm(x_hidden_channels)

        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()

        if use_lora is not None:
            self.q_proj = lora.Linear(x_channels, x_hidden_channels, r=use_lora)
            self.kv_proj = lora.Linear(x_channels, x_hidden_channels, r=use_lora)
            self.dk_proj = lora.Linear(edge_attr_channels, x_hidden_channels, r=use_lora)
        else:
            self.q_proj = nn.Linear(x_channels, x_hidden_channels)
            self.kv_proj = nn.Linear(x_channels, x_hidden_channels)
            self.dk_proj = nn.Linear(edge_attr_channels, x_hidden_channels)
        # self.v_proj = nn.Linear(x_channels, x_hidden_channels)
        # self.o_proj = nn.Linear(x_hidden_channels, x_hidden_channels)
        # self.out = nn.Linear(x_hidden_channels, x_channels)
        # add residue to x
        # self.residue_hidden = nn.Linear(x_channels, x_hidden_channels)
        self.gru = nn.GRUCell(x_channels, x_channels)
        
        self.triangular_drop = nn.Dropout(drop_out_rate)
        self.rbf_drop = nn.Dropout(drop_out_rate)
        self.dense_drop = nn.Dropout(drop_out_rate)
        self.dropout = nn.Dropout(drop_out_rate)
        self.triangular_update = triangular_update
        if self.triangular_update:
            self.edge_triangle_end_update = MultiplicativeUpdate(vec_in_channel=vec_channels,
                                                                 hidden_channel=edge_attr_channels,
                                                                 hidden_vec_channel=vec_hidden_channels,
                                                                 ee_channels=ee_channels, 
                                                                 use_lora=use_lora)
            self.node_to_edge_attr = NodeToEdgeAttr(node_channel=x_channels,
                                                    hidden_channel=x_hidden_channels,
                                                    edge_attr_channel=edge_attr_channels,
                                                    use_lora=use_lora)
            self.triangle_update_dropout = nn.Dropout(0.5)

        self.reset_parameters()

    def reset_parameters(self):
        # self.layernorm_in.reset_parameters()
        self.layernorm_out.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.v_proj.weight)
        # self.v_proj.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.o_proj.weight)
        # self.o_proj.bias.data.fill_(0)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)

    def get_start_index(self, edge_index):
        edge_start_index = []
        start_node_count = edge_index[0].unique(return_counts=True)
        start_nodes = start_node_count[0][start_node_count[1] > 1]
        for i in start_nodes:
            node_start_index = torch.where(edge_index[0] == i)[0]
            candidates = torch.combinations(node_start_index, r=2).T
            edge_start_index.append(torch.cat([candidates, candidates.flip(0)], dim=1))
        edge_start_index = torch.concat(edge_start_index, dim=1)
        edge_start_index = edge_start_index[:, edge_start_index[0] != edge_start_index[1]]
        return edge_start_index

    def get_end_index(self, edge_index):
        edge_end_index = []
        end_node_count = edge_index[1].unique(return_counts=True)
        end_nodes = end_node_count[0][end_node_count[1] > 1]
        for i in end_nodes:
            node_end_index = torch.where(edge_index[1] == i)[0]
            candidates = torch.combinations(node_end_index, r=2).T
            edge_end_index.append(torch.cat([candidates, candidates.flip(0)], dim=1))
        edge_end_index = torch.concat(edge_end_index, dim=1)
        edge_end_index = edge_end_index[:, edge_end_index[0] != edge_end_index[1]]
        return edge_end_index

    def forward(self, x, coords, edge_index, edge_attr, edge_vec, return_attn=False):
        # perform topK pooling
        end_node_count = edge_index[1].unique(return_counts=True)
        center_nodes = end_node_count[0][end_node_count[1] > 1]
        other_nodes = end_node_count[0][end_node_count[1] <= 1]
        residue = x[center_nodes] # batch_size * x_channels
        # filter edge_index and edge_attr to from context to center only
        edge_attr = edge_attr[torch.isin(edge_index[1], center_nodes), :]
        edge_vec = edge_vec[torch.isin(edge_index[1], center_nodes), :]
        edge_index = edge_index[:, torch.isin(edge_index[1], center_nodes)]
        # x itself is q, k and v
        q = self.q_proj(residue).reshape(-1, self.num_heads, self.x_head_dim)
        kv = self.kv_proj(x[other_nodes]).reshape(-1, self.num_heads, self.x_head_dim)
        qkv = torch.zeros(x.shape[0], self.num_heads, self.x_head_dim).to(x.device, non_blocking=True)
        qkv[center_nodes] = q
        qkv[other_nodes] = kv
        # point ettr to edge_attr
        if self.triangular_update:
            edge_attr += self.node_to_edge_attr(x, edge_index)
            # Triangular edge update
            # TODO: Add drop out layers here
            edge_edge_index = self.get_end_index(edge_index)
            edge_edge_index = edge_edge_index[:, self.triangular_drop(
                torch.ones(edge_edge_index.shape[1], device=edge_edge_index.device)
                ).to(torch.bool)]
            if self.ee_channels is not None:
                edge_edge_attr = coords[edge_index[0][edge_edge_index[0]], :, [0]] - coords[edge_index[0][edge_edge_index[1]], :, [0]]
                edge_edge_attr = torch.norm(edge_edge_attr, dim=-1, keepdim=True)
            else:
                edge_edge_attr = None
            edge_attr = self.edge_triangle_end_update(
                edge_attr, edge_vec, 
                edge_edge_index,
                edge_edge_attr
            )
            del edge_edge_attr, edge_edge_index
        
        # drop rbfs
        edge_attr = torch.cat((edge_attr[:, :-self.rbf_channels],
                              self.rbf_drop(edge_attr[:, -self.rbf_channels:])), 
                              dim=-1)

        dk = (
            self.act(self.dk_proj(edge_attr)).reshape(-1, self.num_heads, self.x_head_dim) 
            if self.dk_proj is not None else None
        ) # TODO: check self.act

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor,
        # d_ij: Tensor)
        x, attn = self.propagate(
            edge_index,
            q=qkv,
            k=qkv,
            v=qkv,
            dk=dk,
            size=None,
        )
        x = x.reshape(-1, self.x_hidden_channels)
        # only get the center nodes
        x = x[center_nodes]
        x = self.layernorm_out(x)
        x = self.dense_drop(x)
        x = self.gru(residue, x)
        x = self.dropout(x)
        del residue, dk
        if return_attn:
            return x, edge_attr, torch.concat((edge_index.T, attn), dim=1)
        else:
            return x, edge_attr, None

    def message(self, q_i, k_j, v_j, dk):
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:  # TODO: consider add or multiply dk
            attn = (q_i * k_j + dk).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn) / self.x_head_dim

        # update scalar features
        x = v_j * attn.unsqueeze(2)
        return x, attn

    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, attn = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x, attn

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    def edge_update(self) -> Tensor:
        pass


# Transform sequence, structure, and relative position into a pair feature
class PairFeatureNet(nn.Module):

    def __init__(self, c_s, c_p, relpos_k=32, template_type="exp-normal-smearing-distance"):
        super(PairFeatureNet, self).__init__()

        self.c_s = c_s
        self.c_p = c_p

        self.linear_s_p_i = nn.Linear(c_s, c_p)
        self.linear_s_p_j = nn.Linear(c_s, c_p)

        self.relpos_k = relpos_k
        self.n_bin = 2 * relpos_k + 1
        self.linear_relpos = nn.Linear(self.n_bin, c_p)

        # TODO: implement structure to pairwise feature function
        self.template_fn, c_template = get_template_fn(template_type)
        self.linear_template = nn.Linear(c_template, c_p)

    def relpos(self, r):
        # AlphaFold 2 Algorithm 4 & 5
        # Based on OpenFold utils/tensor_utils.py
        # Input: [b, n_res]

        # [b, n_res, n_res]
        d = r[:, :, None] - r[:, None, :]

        # [n_bin]
        v = torch.arange(-self.relpos_k, self.relpos_k + 1).to(r.device, non_blocking=True)

        # [1, 1, 1, n_bin]
        v_reshaped = v.view(*((1,) * len(d.shape) + (len(v),)))

        # [b, n_res, n_res]
        b = torch.argmin(torch.abs(d[:, :, :, None] - v_reshaped), dim=-1)

        # [b, n_res, n_res, n_bin]
        oh = nn.functional.one_hot(b, num_classes=len(v)).float()

        # [b, n_res, n_res, c_p]
        p = self.linear_relpos(oh)

        return p

    def template(self, t):
        return self.linear_template(self.template_fn(t))

    def forward(self, s, t, r, mask):
        # Input: [b, n_res, c_s]
        p_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        # [b, n_res, c_p]
        p_i = self.linear_s_p_i(s)
        p_j = self.linear_s_p_j(s)

        # [b, n_res, n_res, c_p]
        p = p_i[:, :, None, :] + p_j[:, None, :, :]

        # [b, n_res, n_res, c_p]
        p += self.relpos(r)  # upper bond is 64 AA
        p += self.template(t)  # upper bond is 100 A

        # [b, n_res, n_res, c_p]
        p *= p_mask.unsqueeze(-1)

        return p


# AF2's TriangularSelfAttentionBlock, but I removed the pairwise attention because of memory issues.
# In genie they are doing the same.
class TriangularSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        sequence_state_dim,
        pairwise_state_dim,
        sequence_head_width,
        pairwise_head_width,
        dropout=0,
        **__kwargs,
    ):
        super().__init__()
        from openfold.model.triangular_multiplicative_update import (
            TriangleMultiplicationIncoming,
            TriangleMultiplicationOutgoing,
        )
        from esm.esmfold.v1.misc import (
            Attention,
            Dropout,
            PairToSequence,
            ResidueMLP,
            SequenceToPair,
        )
        assert sequence_state_dim % sequence_head_width == 0
        assert pairwise_state_dim % pairwise_head_width == 0
        sequence_num_heads = sequence_state_dim // sequence_head_width
        pairwise_num_heads = pairwise_state_dim // pairwise_head_width
        assert sequence_state_dim == sequence_num_heads * sequence_head_width
        assert pairwise_state_dim == pairwise_num_heads * pairwise_head_width
        assert pairwise_state_dim % 2 == 0

        self.sequence_state_dim = sequence_state_dim
        self.pairwise_state_dim = pairwise_state_dim

        self.layernorm_1 = nn.LayerNorm(sequence_state_dim)

        self.sequence_to_pair = SequenceToPair(
            sequence_state_dim, pairwise_state_dim // 2, pairwise_state_dim
        )
        self.pair_to_sequence = PairToSequence(
            pairwise_state_dim, sequence_num_heads)

        self.seq_attention = Attention(
            sequence_state_dim, sequence_num_heads, sequence_head_width, gated=True
        )
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            pairwise_state_dim,
            pairwise_state_dim,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            pairwise_state_dim,
            pairwise_state_dim,
        )

        self.mlp_seq = ResidueMLP(
            sequence_state_dim, 4 * sequence_state_dim, dropout=dropout)
        self.mlp_pair = ResidueMLP(
            pairwise_state_dim, 4 * pairwise_state_dim, dropout=dropout)

        assert dropout < 0.4
        self.drop = nn.Dropout(dropout)
        self.row_drop = Dropout(dropout * 2, 2)
        self.col_drop = Dropout(dropout * 2, 1)

        torch.nn.init.zeros_(self.tri_mul_in.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_in.linear_z.bias)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.bias)

        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.weight)
        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.bias)
        torch.nn.init.zeros_(self.pair_to_sequence.linear.weight)
        torch.nn.init.zeros_(self.seq_attention.o_proj.weight)
        torch.nn.init.zeros_(self.seq_attention.o_proj.bias)
        torch.nn.init.zeros_(self.mlp_seq.mlp[-2].weight)
        torch.nn.init.zeros_(self.mlp_seq.mlp[-2].bias)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-2].weight)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-2].bias)

    def forward(self, sequence_state, pairwise_state, mask=None, chunk_size=None, **__kwargs):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim
          pairwise_state: B x L x L x pairwise_state_dim
          mask: B x L boolean tensor of valid positions

        Output:
          sequence_state: B x L x sequence_state_dim
          pairwise_state: B x L x L x pairwise_state_dim
        """
        assert len(sequence_state.shape) == 3
        assert len(pairwise_state.shape) == 4
        if mask is not None:
            assert len(mask.shape) == 2

        batch_dim, seq_dim, sequence_state_dim = sequence_state.shape
        pairwise_state_dim = pairwise_state.shape[3]
        assert sequence_state_dim == self.sequence_state_dim
        assert pairwise_state_dim == self.pairwise_state_dim
        assert batch_dim == pairwise_state.shape[0]
        assert seq_dim == pairwise_state.shape[1]
        assert seq_dim == pairwise_state.shape[2]

        # Update sequence state
        bias = self.pair_to_sequence(pairwise_state)

        # Self attention with bias + mlp.
        y = self.layernorm_1(sequence_state)
        y, _ = self.seq_attention(y, mask=mask, bias=bias)
        sequence_state = sequence_state + self.drop(y)
        sequence_state = self.mlp_seq(sequence_state)

        # Update pairwise state
        pairwise_state = pairwise_state + self.sequence_to_pair(sequence_state)

        # Axial attention with triangular bias.
        tri_mask = mask.unsqueeze(
            2) * mask.unsqueeze(1) if mask is not None else None
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_mul_out(pairwise_state, mask=tri_mask)
        )
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_mul_in(pairwise_state, mask=tri_mask)
        )

        # MLP over pairs.
        pairwise_state = self.mlp_pair(pairwise_state)

        return sequence_state, pairwise_state


# A Self-Attention Pooling Block
class SeqPairAttentionOutput(nn.Module):
    def __init__(self, seq_state_dim, pairwise_state_dim, num_heads, output_dim, dropout):
        super(SeqPairAttentionOutput, self).__init__()
        from esm.esmfold.v1.misc import (
            Attention,
            PairToSequence,
            ResidueMLP,
        )
        self.seq_state_dim = seq_state_dim
        self.pairwise_state_dim = pairwise_state_dim
        self.output_dim = output_dim
        seq_head_width = seq_state_dim // num_heads

        self.layernorm = nn.LayerNorm(seq_state_dim)
        self.seq_attention = Attention(
            seq_state_dim, num_heads, seq_head_width, gated=True
        )
        self.pair_to_sequence = PairToSequence(pairwise_state_dim, num_heads)
        self.mlp_seq = ResidueMLP(
            seq_state_dim, 4 * seq_state_dim, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, sequence_state, pairwise_state, mask=None):
        # Update sequence state
        bias = self.pair_to_sequence(pairwise_state)

        # Self attention with bias + mlp.
        y = self.layernorm(sequence_state)
        y, _ = self.seq_attention(y, mask=mask, bias=bias)
        sequence_state = sequence_state + self.drop(y)
        sequence_state = self.mlp_seq(sequence_state)

        return sequence_state
