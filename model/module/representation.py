from typing import Optional, Tuple, List
import math
import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_sparse import SparseTensor
from openfold.model.triangular_attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
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
from ..module.utils import (
    NeighborEmbedding,
    CosineCutoff,
    Distance,
    rbf_class_mapping,
    act_class_mapping,
    get_template_fn,
    plain_distance,
    exp_normal_smearing_distance,
)

# A fake model, do nothing and just past the input, serve as a baseline
class PassForward(nn.Module):
    def __init__(
            self,
            x_in_channels=None,
            x_channels=5120,
            x_hidden_channels=1280,
            vec_in_channels=4,
            vec_channels=128,
            vec_hidden_channels=5120,
            num_layers=6,
            num_edge_attr=145,
            num_rbf=50,
            rbf_type="expnorm",
            trainable_rbf=True,
            activation="silu",
            attn_activation="silu",
            neighbor_embedding=True,
            num_heads=8,
            distance_influence="both",
            cutoff_lower=0.0,
            cutoff_upper=5.0,
            x_in_embedding_type="Linear",
            drop_out_rate=0, # new feature
    ):
        super(PassForward, self).__init__()
        
    def reset_parameters(self):
        pass

    def forward(
            self,
            x: Tensor,
            pos: Tensor,
            batch: Tensor,
            edge_index: Tensor,
            edge_index_star: Tensor = None,  # unused
            edge_attr: Tensor = None,
            edge_attr_star: Tensor = None,  # unused
            node_vec_attr: Tensor = None,
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List]:
        # pass input to output directly, serve as a baseline
        vec = node_vec_attr 
        attn_weight_layers = []
        return x, vec, pos, edge_attr, batch, attn_weight_layers


# original torchmd-net, 2-layers of full graph
class eqTransformer(nn.Module):
    """The equivariant Transformer architecture.

    Args:
        x_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_layers (int, optional): The number of attention layers.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        attn_activation (string, optional): The type of activation function to use
            inside the attention mechanism. (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`8`)
        distance_influence (string, optional): Where distance information is used inside
            the attention mechanism. (default: :obj:`"both"`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`5.0`)
    """

    def __init__(
            self,
            x_in_channels=None,
            x_channels=5120,
            x_hidden_channels=1280,
            vec_in_channels=4,
            vec_channels=128,
            vec_hidden_channels=5120,
            num_layers=6,
            num_edge_attr=145,
            num_rbf=50,
            rbf_type="expnorm",
            trainable_rbf=True,
            activation="silu",
            attn_activation="silu",
            neighbor_embedding=True,
            num_heads=8,
            distance_influence="both",
            cutoff_lower=0.0,
            cutoff_upper=5.0,
            x_in_embedding_type="Linear",
            drop_out_rate=0, # new feature
    ):
        super(eqTransformer, self).__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert attn_activation in act_class_mapping, (
            f'Unknown attention activation function "{attn_activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        self.x_in_channels = x_in_channels
        self.x_channels = x_channels
        self.vec_in_channels = vec_in_channels
        self.vec_channels = vec_channels
        self.x_hidden_channels = x_hidden_channels
        self.vec_hidden_channels = vec_hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.num_edge_attr = num_edge_attr
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

        self.distance = Distance(
            cutoff_lower,
            cutoff_upper,
            return_vecs=True,
            loop=True,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                x_channels, num_rbf + num_edge_attr, cutoff_lower, cutoff_upper,
            )
            if neighbor_embedding
            else None
        )

        self.node_x_proj = None
        if x_in_channels is not None:
            self.node_x_proj = nn.Linear(x_in_channels, x_channels) if x_in_embedding_type == "Linear" \
                else nn.Embedding(x_in_channels, x_channels)
        self.node_vec_proj = nn.Linear(
            vec_in_channels, vec_channels, bias=False)

        self.attention_layers = nn.ModuleList()
        self._set_attn_layers()
        self.drop = nn.Dropout(drop_out_rate)
        self.out_norm = nn.LayerNorm(x_channels)

        self.reset_parameters()

    def _set_attn_layers(self):
        for _ in range(self.num_layers):
            layer = EquivariantMultiHeadAttention(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_channels,
                vec_hidden_channels=self.vec_hidden_channels,
                edge_attr_channels=self.num_rbf + self.num_edge_attr,
                distance_influence=self.distance_influence,
                num_heads=self.num_heads,
                activation=act_class_mapping[self.activation],
                attn_activation=self.attn_activation,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
            )
            self.attention_layers.append(layer)

    def reset_parameters(self):
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(
            self,
            x: Tensor,
            pos: Tensor,
            batch: Tensor,
            edge_index: Tensor,
            edge_index_star: Tensor = None,  # unused
            edge_attr: Tensor = None,
            edge_attr_star: Tensor = None,  # unused
            node_vec_attr: Tensor = None,
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List]:

        edge_index, edge_weight, edge_vec = self.distance(pos, edge_index)
        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"
        # get distance expansion edge attributes
        edge_attr_distance = self.distance_expansion(
            edge_weight)  # [E, num_rbf]
        # concatenate edge attributes
        # [E, num_rbf + 145 = 64 + 145 = 209]
        edge_attr = torch.cat([edge_attr, edge_attr_distance], dim=-1)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / \
            torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        # apply embedding of x if necessary
        x = self.node_x_proj(x) if self.node_x_proj is not None else x

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(x, edge_index, edge_weight, edge_attr)
        # apply embedding of vec if necessary
        vec = self.node_vec_proj(node_vec_attr) if node_vec_attr is not None \
            else torch.zeros(x.size(0), 3, self.vec_channels, device=x.device)

        attn_weight_layers = []
        for attn in self.attention_layers:
            dx, dvec, attn_weight = attn(
                x, vec, edge_index, edge_weight, edge_attr, edge_vec)
            x = x + self.drop(dx)
            vec = vec + self.drop(dvec)
            if return_attn:
                attn_weight_layers.append(attn_weight)
        x = self.out_norm(x)

        return x, vec, pos, edge_attr, batch, attn_weight_layers

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"x_channels={self.x_channels}, "
            f"x_hidden_channels={self.x_hidden_channels}, "
            f"vec_in_channels={self.vec_in_channels}, "
            f"vec_channels={self.vec_channels}, "
            f"vec_hidden_channels={self.vec_hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper})"
        )


# original torchmd-net, 1 layer of star graph, 1 layer of full graph
class eqStarTransformer(eqTransformer):
    """The equivariant Transformer architecture.
    First Layer is Star Graph, next layer is full graph

    Args:
        x_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_layers (int, optional): The number of attention layers.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        attn_activation (string, optional): The type of activation function to use
            inside the attention mechanism. (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`8`)
        distance_influence (string, optional): Where distance information is used inside
            the attention mechanism. (default: :obj:`"both"`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`5.0`)
    """

    def __init__(
            self,
            x_in_channels=None,
            x_channels=5120,
            x_hidden_channels=1280,
            vec_in_channels=4,
            vec_channels=128,
            vec_hidden_channels=5120,
            num_layers=6,
            num_edge_attr=145,
            num_rbf=50,
            rbf_type="expnorm",
            trainable_rbf=True,
            activation="silu",
            attn_activation="silu",
            neighbor_embedding=True,
            num_heads=8,
            distance_influence="both",
            cutoff_lower=0.0,
            cutoff_upper=5.0,
            x_in_embedding_type="Linear",
            drop_out_rate=0, # new feature
    ):
        super(eqStarTransformer, self).__init__(x_in_channels=x_in_channels,
                                                x_channels=x_channels,
                                                x_hidden_channels=x_hidden_channels,
                                                vec_in_channels=vec_in_channels,
                                                vec_channels=vec_channels,
                                                vec_hidden_channels=vec_hidden_channels,
                                                num_layers=num_layers,
                                                num_edge_attr=num_edge_attr,
                                                num_rbf=num_rbf,
                                                rbf_type=rbf_type,
                                                trainable_rbf=trainable_rbf,
                                                activation=activation,
                                                attn_activation=attn_activation,
                                                neighbor_embedding=neighbor_embedding,
                                                num_heads=num_heads,
                                                distance_influence=distance_influence,
                                                cutoff_lower=cutoff_lower,
                                                cutoff_upper=cutoff_upper,
                                                x_in_embedding_type=x_in_embedding_type,
                                                drop_out_rate=drop_out_rate)

    def forward(
            self,
            x: Tensor,
            pos: Tensor,
            batch: Tensor,
            edge_index: Tensor,
            edge_index_star: Tensor = None,
            edge_attr: Tensor = None,
            edge_attr_star: Tensor = None,
            node_vec_attr: Tensor = None,
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List]:
        edge_index, edge_weight, edge_vec = self.distance(pos, edge_index)
        edge_index_star, edge_weight_star, edge_vec_star = self.distance(
            pos, edge_index_star)

        assert (
            edge_vec is not None and edge_vec_star is not None
        ), "Distance module did not return directional information"
        # get distance expansion edge attributes
        edge_attr_distance = self.distance_expansion(
            edge_weight)  # [E, num_rbf]
        edge_attr_distance_star = self.distance_expansion(
            edge_weight_star)  # [E, num_rbf]
        # concatenate edge attributes
        if edge_attr is not None:
            # [E, num_rbf + 145 = 64 + 145 = 209]
            edge_attr = torch.cat([edge_attr, edge_attr_distance], dim=-1)
        else:
            edge_attr = edge_attr_distance
        if edge_attr_star is not None:
            edge_attr_star = torch.cat(
                [edge_attr_star, edge_attr_distance_star], dim=-1)
        else:
            edge_attr_star = edge_attr_distance_star
        # cancel edge mask
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / \
            torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        mask = edge_index_star[0] != edge_index_star[1]
        edge_vec_star[mask] = edge_vec_star[mask] / \
            torch.norm(edge_vec_star[mask], dim=1).unsqueeze(1)
        # apply x embedding if necessary
        x = self.node_x_proj(x) if self.node_x_proj is not None else x
        if self.neighbor_embedding is not None:
            # neighbor embedding is star graph
            x = self.neighbor_embedding(
                x, edge_index_star, edge_weight_star, edge_attr_star)
        # apply vec embedding if necessary
        vec = self.node_vec_proj(node_vec_attr) if node_vec_attr is not None \
            else torch.zeros(x.size(0), 3, self.vec_channels, device=x.device)

        attn_weight_layers = []
        for i, attn in enumerate(self.attention_layers):
            # first layer is star graph, next layers are normal graph
            if i == 0:
                dx, dvec, attn_weight = attn(x, vec,
                                             edge_index_star, edge_weight_star, edge_attr_star, edge_vec_star,
                                             return_attn=return_attn)
            else:
                dx, dvec, attn_weight = attn(x, vec,
                                             edge_index, edge_weight, edge_attr, edge_vec,
                                             return_attn=return_attn)
            x = x + self.drop(dx)
            vec = vec + self.drop(dvec)
            if return_attn:
                attn_weight_layers.append(attn_weight)
        x = self.out_norm(x)

        return x, vec, pos, edge_attr, batch, attn_weight_layers


# Softmax version of torchmd-net, 2-layer of full graph
class eqTransformerSoftMax(eqTransformer):
    """The equivariant Transformer architecture.

    Args:
        x_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_layers (int, optional): The number of attention layers.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        attn_activation (string, optional): The type of activation function to use
            inside the attention mechanism. (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`8`)
        distance_influence (string, optional): Where distance information is used inside
            the attention mechanism. (default: :obj:`"both"`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`5.0`)
    """

    def __init__(
            self,
            x_in_channels=None,
            x_channels=5120,
            x_hidden_channels=1280,
            vec_in_channels=4,
            vec_channels=128,
            vec_hidden_channels=5120,
            num_layers=6,
            num_edge_attr=145,
            num_rbf=50,
            rbf_type="expnorm",
            trainable_rbf=True,
            activation="silu",
            attn_activation="silu",
            neighbor_embedding=True,
            num_heads=8,
            distance_influence="both",
            cutoff_lower=0.0,
            cutoff_upper=5.0,
            x_in_embedding_type="Linear",
            drop_out_rate=0, # new feature
    ):
        super(eqTransformerSoftMax, self).__init__(x_in_channels=x_in_channels,
                                                   x_channels=x_channels,
                                                   x_hidden_channels=x_hidden_channels,
                                                   vec_in_channels=vec_in_channels,
                                                   vec_channels=vec_channels,
                                                   vec_hidden_channels=vec_hidden_channels,
                                                   num_layers=num_layers,
                                                   num_edge_attr=num_edge_attr,
                                                   num_rbf=num_rbf,
                                                   rbf_type=rbf_type,
                                                   trainable_rbf=trainable_rbf,
                                                   activation=activation,
                                                   attn_activation=attn_activation,
                                                   neighbor_embedding=neighbor_embedding,
                                                   num_heads=num_heads,
                                                   distance_influence=distance_influence,
                                                   cutoff_lower=cutoff_lower,
                                                   cutoff_upper=cutoff_upper,
                                                   x_in_embedding_type=x_in_embedding_type,
                                                   drop_out_rate=drop_out_rate,)

    def _set_attn_layers(self):
        for _ in range(self.num_layers):
            layer = EquivariantMultiHeadAttentionSoftMax(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_channels,
                vec_hidden_channels=self.vec_hidden_channels,
                edge_attr_channels=self.num_rbf + self.num_edge_attr,
                distance_influence=self.distance_influence,
                num_heads=self.num_heads,
                activation=act_class_mapping[self.activation],
                attn_activation=self.attn_activation,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
            )
            self.attention_layers.append(layer)


# Softmax version of torchmd-net, 1 layer of star graph, 1 layer of full graph
class eqStarTransformerSoftMax(eqStarTransformer):
    """The equivariant Transformer architecture.
    First Layer is Star Graph, next layer is full graph

    Args:
        x_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_layers (int, optional): The number of attention layers.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        attn_activation (string, optional): The type of activation function to use
            inside the attention mechanism. (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`8`)
        distance_influence (string, optional): Where distance information is used inside
            the attention mechanism. (default: :obj:`"both"`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`5.0`)
    """

    def __init__(
            self,
            x_in_channels=None,
            x_channels=5120,
            x_hidden_channels=1280,
            vec_in_channels=4,
            vec_channels=128,
            vec_hidden_channels=5120,
            num_layers=6,
            num_edge_attr=145,
            num_rbf=50,
            rbf_type="expnorm",
            trainable_rbf=True,
            activation="silu",
            attn_activation="silu",
            neighbor_embedding=True,
            num_heads=8,
            distance_influence="both",
            cutoff_lower=0.0,
            cutoff_upper=5.0,
            x_in_embedding_type="Linear",
            drop_out_rate=0, # new feature
    ):
        super(eqStarTransformerSoftMax, self).__init__(x_in_channels=x_in_channels,
                                                       x_channels=x_channels,
                                                       x_hidden_channels=x_hidden_channels,
                                                       vec_in_channels=vec_in_channels,
                                                       vec_channels=vec_channels,
                                                       vec_hidden_channels=vec_hidden_channels,
                                                       num_layers=num_layers,
                                                       num_edge_attr=num_edge_attr,
                                                       num_rbf=num_rbf,
                                                       rbf_type=rbf_type,
                                                       trainable_rbf=trainable_rbf,
                                                       activation=activation,
                                                       attn_activation=attn_activation,
                                                       neighbor_embedding=neighbor_embedding,
                                                       num_heads=num_heads,
                                                       distance_influence=distance_influence,
                                                       cutoff_lower=cutoff_lower,
                                                       cutoff_upper=cutoff_upper,
                                                       x_in_embedding_type=x_in_embedding_type,
                                                       drop_out_rate=drop_out_rate)

    def _set_attn_layers(self):
        for _ in range(self.num_layers):
            layer = EquivariantMultiHeadAttentionSoftMax(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_channels,
                vec_hidden_channels=self.vec_hidden_channels,
                edge_attr_channels=self.num_rbf + self.num_edge_attr,
                distance_influence=self.distance_influence,
                num_heads=self.num_heads,
                activation=act_class_mapping[self.activation],
                attn_activation=self.attn_activation,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
            )
            self.attention_layers.append(layer)


class eqStar2TransformerSoftMax(eqStarTransformer):
    """The equivariant Transformer architecture.
    First Layer is Star Graph, next layer is full graph

    Args:
        x_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_layers (int, optional): The number of attention layers.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        attn_activation (string, optional): The type of activation function to use
            inside the attention mechanism. (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`8`)
        distance_influence (string, optional): Where distance information is used inside
            the attention mechanism. (default: :obj:`"both"`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`5.0`)
    """

    def __init__(
            self,
            x_in_channels=None,
            x_channels=5120,
            x_hidden_channels=1280,
            vec_in_channels=4,
            vec_channels=128,
            vec_hidden_channels=5120,
            num_layers=6,
            num_edge_attr=145,
            num_rbf=50,
            rbf_type="expnorm",
            trainable_rbf=True,
            activation="silu",
            attn_activation="silu",
            neighbor_embedding=True,
            num_heads=8,
            distance_influence="both",
            cutoff_lower=0.0,
            cutoff_upper=5.0,
            x_in_embedding_type="Linear",
            drop_out_rate=0, # new feature
    ):
        super(eqStar2TransformerSoftMax, self).__init__(x_in_channels=x_in_channels,
                                                       x_channels=x_channels,
                                                       x_hidden_channels=x_hidden_channels,
                                                       vec_in_channels=vec_in_channels,
                                                       vec_channels=vec_channels,
                                                       vec_hidden_channels=vec_hidden_channels,
                                                       num_layers=num_layers,
                                                       num_edge_attr=num_edge_attr,
                                                       num_rbf=num_rbf,
                                                       rbf_type=rbf_type,
                                                       trainable_rbf=trainable_rbf,
                                                       activation=activation,
                                                       attn_activation=attn_activation,
                                                       neighbor_embedding=neighbor_embedding,
                                                       num_heads=num_heads,
                                                       distance_influence=distance_influence,
                                                       cutoff_lower=cutoff_lower,
                                                       cutoff_upper=cutoff_upper,
                                                       x_in_embedding_type=x_in_embedding_type,
                                                       drop_out_rate=drop_out_rate)

    def _set_attn_layers(self):
        assert self.num_layers > 0, "num_layers must be greater than 0"
        # first star graph layer does not have softmax
        self.attention_layers.append(
            EquivariantMultiHeadAttention(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_channels,
                vec_hidden_channels=self.vec_hidden_channels,
                edge_attr_channels=self.num_rbf + self.num_edge_attr,
                distance_influence=self.distance_influence,
                num_heads=self.num_heads,
                activation=act_class_mapping[self.activation],
                attn_activation=self.attn_activation,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
            )
        )
        for _ in range(self.num_layers - 1):
            layer = EquivariantMultiHeadAttentionSoftMax(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_channels,
                vec_hidden_channels=self.vec_hidden_channels,
                edge_attr_channels=self.num_rbf + self.num_edge_attr,
                distance_influence=self.distance_influence,
                num_heads=self.num_heads,
                activation=act_class_mapping[self.activation],
                attn_activation=self.attn_activation,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
            )
            self.attention_layers.append(layer)


# original torchmd-net attention layer
class EquivariantMultiHeadAttention(MessagePassing):
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
        self.k_proj = nn.Linear(x_channels, x_hidden_channels)
        self.v_proj = nn.Linear(
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
        k = self.k_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads,
                                   self.x_head_dim + self.vec_head_dim * 2)

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


# softmax version of torchmd-net attention layer
class EquivariantMultiHeadAttentionSoftMax(EquivariantMultiHeadAttention):
    """Equivariant multi-head attention layer with softmax"""

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
        super(EquivariantMultiHeadAttentionSoftMax, self).__init__(x_channels=x_channels,
                                                                   x_hidden_channels=x_hidden_channels,
                                                                   vec_channels=vec_channels,
                                                                   vec_hidden_channels=vec_hidden_channels,
                                                                   edge_attr_channels=edge_attr_channels,
                                                                   distance_influence=distance_influence,
                                                                   num_heads=num_heads,
                                                                   activation=activation,
                                                                   attn_activation=attn_activation,
                                                                   cutoff_lower=cutoff_lower,
                                                                   cutoff_upper=cutoff_upper)
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
        self.pair_to_sequence = PairToSequence(pairwise_state_dim, sequence_num_heads)

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

        self.mlp_seq = ResidueMLP(sequence_state_dim, 4 * sequence_state_dim, dropout=dropout)
        self.mlp_pair = ResidueMLP(pairwise_state_dim, 4 * pairwise_state_dim, dropout=dropout)

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
        tri_mask = mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
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
        self.mlp_seq = ResidueMLP(seq_state_dim, 4 * seq_state_dim, dropout=dropout)
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


# A new representation using AlphaFold's Triangular Attention mechanism
class eqTriAttnTransformer(nn.Module):
    """
    Input a sequence representation and structure, output a new sequence representation and structure
    """

    def __init__(self,
                 x_in_channels=None,
                 x_channels=1280,
                 pairwise_state_dim=128,
                 num_layers=4,
                 num_heads=8,
                 x_in_embedding_type="Embedding",
                 drop_out_rate=0.1,
                 x_hidden_channels=None,  # unused
                 vec_channels=None,  # unused
                 vec_in_channels=None,  # unused
                 vec_hidden_channels=None,  # unused
                 num_edge_attr=None,  # unused
                 num_rbf=None,  # unused
                 rbf_type=None,  # unused
                 trainable_rbf=None,  # unused
                 activation=None,  # unused
                 neighbor_embedding=None,  # unused
                 cutoff_lower=None,  # unused
                 cutoff_upper=None,  # unused
                 ):
        super(eqTriAttnTransformer, self).__init__()
        if x_in_channels is not None:
            self.node_x_proj = nn.Linear(x_in_channels, x_channels) if x_in_embedding_type == "Linear" \
                else nn.Embedding(x_in_channels, x_channels)
        else:
            self.node_x_proj = None
        assert x_channels % num_heads == 0 \
            and pairwise_state_dim % num_heads == 0, (
                f"The number of hidden channels x_channels ({x_channels}) "
                f"and pair-wise channels ({pairwise_state_dim}) "
                f"must be evenly divisible by the number of "
                f"attention heads ({num_heads})"
            )
        sequence_head_width = x_channels // num_heads
        pairwise_head_width = pairwise_state_dim // num_heads
        self.tri_attn_block = nn.ModuleList(
            [
                TriangularSelfAttentionBlock(
                    sequence_state_dim=x_channels,
                    pairwise_state_dim=pairwise_state_dim,
                    sequence_head_width=sequence_head_width,
                    pairwise_head_width=pairwise_head_width,
                    dropout=drop_out_rate,
                )
                for _ in range(num_layers)
            ]
        )
        self.seq_struct_to_pair = PairFeatureNet(
            x_channels, pairwise_state_dim)
        # self.max_recycles = max_recycles
        # TODO: implement sequence & pair representation to output net
        self.seq_pair_to_output = SeqPairAttentionOutput(seq_state_dim=x_channels,
                                                         pairwise_state_dim=pairwise_state_dim, 
                                                         num_heads=num_heads, 
                                                         output_dim=x_channels,
                                                         dropout=drop_out_rate)

    def reset_parameters(self):
        pass

    def forward(self, 
                x: Tensor, 
                pos: Tensor,
                residx: Tensor = None,
                mask: Tensor = None,
                batch: Tensor = None,
                edge_index: Tensor = None,
                edge_index_star: Tensor = None,
                edge_attr: Tensor = None,
                edge_attr_star: Tensor = None,
                node_vec_attr: Tensor = None,
                return_attn: bool = False,
                ):
        """
        Inputs:
          x:     B x L x C            tensor of sequence features
          pos:    B x L x 4 x 3        tensor of [CA, CB, N, O] coordinates
          residx:        B x L                long tensor giving the position in the sequence
          mask:          B x L                boolean tensor indicating valid residues

        Output:
          predicted_structure: B x L x (num_atoms_per_residue * 3) tensor wrapped in a Coordinates object
        """
        
        if residx is None:
            residx = torch.arange(x.shape[1], device=x.device).repeat(x.shape[0], 1)
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
        # apply x embedding if necessary
        x = self.node_x_proj(x) if self.node_x_proj is not None else x
        # pair-wise features, include seq-wise feature, Distance(struct_features), torsion angle, reative position
        pair_feats = self.seq_struct_to_pair(x, pos, residx, mask)

        s_s = x
        s_z = pair_feats

        for block in self.tri_attn_block:
            s_s, s_z = block(sequence_state=s_s,
                             pairwise_state=s_z, 
                             mask=mask.to(torch.float32))

        s_s = self.seq_pair_to_output(sequence_state=s_s, pairwise_state=s_z, mask=mask.to(torch.float32))
        # s_out = self.seq_pair_to_output(s_s, s_z, residx, mask)
        # to output and make it look like previous transformers
        # x, vec, pos, edge_attr, batch, attn_weight_layers
        return s_s, s_z, pos, None, None, None
