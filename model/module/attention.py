from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_sparse import SparseTensor
import loralib as lora
# from openfold.model.triangular_multiplicative_update import (
#     TriangleMultiplicationIncoming,
#     TriangleMultiplicationOutgoing,
# )
# from esm.esmfold.v1.misc import (
#     Attention,
#     Dropout,
#     PairToSequence,
#     ResidueMLP,
#     SequenceToPair,
# )
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
            sm = torch.nn.Softmax()
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


class EquivariantTriAngularMultiHeadAttention_archive(MessagePassing):
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
        self.k_proj = nn.Linear(x_channels, x_hidden_channels)
        self.v_proj = nn.Linear(x_channels, x_hidden_channels)
        self.o_proj = nn.Linear(x_hidden_channels, x_hidden_channels)
        self.out = nn.Linear(x_hidden_channels, x_channels)
        # add residue to x
        self.residue_hidden = nn.Linear(x_channels, x_hidden_channels)

        self.dk_proj = nn.Linear(edge_attr_channels, x_hidden_channels)

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
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
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
        k = self.k_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.x_head_dim)

        # point ettr to edge_attr
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
        x = self.residue_hidden(residue) + x
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
        qkv = torch.zeros(x.shape[0], self.num_heads, self.x_head_dim).to(x.device)
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
        qkv = torch.zeros(x.shape[0], self.num_heads, self.x_head_dim).to(x.device)
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
        v = torch.arange(-self.relpos_k, self.relpos_k + 1).to(r.device)

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
