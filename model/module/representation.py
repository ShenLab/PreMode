from typing import Tuple, List
import torch
from torch import _dynamo
_dynamo.config.suppress_errors = True
from torch import Tensor, nn
import loralib as lora
import math
import esm
from ..module.utils import (
    NeighborEmbedding,
    Distance,
    DistanceV2,
    rbf_class_mapping,
    act_class_mapping
)
from ..module.attention import (
    EquivariantMultiHeadAttention,
    EquivariantMultiHeadAttentionSoftMax,
    EquivariantPAEMultiHeadAttention,
    EquivariantPAEMultiHeadAttentionSoftMax,
    EquivariantWeightedPAEMultiHeadAttention,
    EquivariantWeightedPAEMultiHeadAttentionSoftMax,
    EquivariantPAEMultiHeadAttentionSoftMaxFullGraph,
    MultiHeadAttentionSoftMaxFullGraph,
    MSAEncoderFullGraph,
    EquivariantTriAngularMultiHeadAttention,
    EquivariantTriAngularStarMultiHeadAttention,
    EquivariantTriAngularStarDropMultiHeadAttention,
    EquivariantTriAngularDropMultiHeadAttention,
    PairFeatureNet,
    TriangularSelfAttentionBlock,
    SeqPairAttentionOutput,
    MSAEncoder,
    ESMMultiheadAttention
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
            x_use_msa=False,
            drop_out_rate=0,  # new feature
            use_lora=None,
    ):
        super(PassForward, self).__init__()
        self.x_in_channels = x_in_channels
        self.x_channels = x_channels

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
            edge_vec: Tensor = None,
            edge_vec_star: Tensor = None,  # unused
            node_vec_attr: Tensor = None,
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List]:
        # pass input to output directly, serve as a baseline
        vec = node_vec_attr
        attn_weight_layers = []
        return x, vec, pos, edge_attr, batch, attn_weight_layers

# Transformer Layer copied from ESM2, added LoRA, used for tuning ESM2
class ESMTransformerLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        add_bias_kv=True,
        use_esm1b_layer_norm=False,
        use_rotary_embeddings: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv, use_esm1b_layer_norm)

    def _init_submodules(self, add_bias_kv, use_esm1b_layer_norm):
        BertLayerNorm = nn.LayerNorm

        self.self_attn = ESMMultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = lora.Linear(self.embed_dim, self.ffn_embed_dim, r=16)
        self.fc2 = lora.Linear(self.ffn_embed_dim, self.embed_dim, r=16)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(
        self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn

# Use LoRA to tune ESM2 
class LoRAESM2(nn.Module):
    def __init__(
            self,
            x_in_channels=None,
            x_channels=5120, # not used
            x_hidden_channels=1280,
            vec_in_channels=4,
            vec_channels=128,
            vec_hidden_channels=5120,
            num_layers=6, # not used
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
            x_use_msa=False,
            drop_out_rate=0,  # new feature
            use_lora=None,
    ):
        super(LoRAESM2, self).__init__()
        self.x_in_channels = x_in_channels
        self.x_channels = 1280
        self.num_layers = 33
        self.embed_dim = 1280
        self.attention_heads = 20
        self.embed_scale = 1
        _, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = True

        # set ESM2 model with LoRA
        self.embed_tokens = lora.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
            r=16,
        )
        self.layers = nn.ModuleList(
            [
                ESMTransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.emb_layer_norm_after = nn.LayerNorm(self.embed_dim)

    def reset_parameters(self):
        # assign esm2 model weights to LoRA model
        esm_weights, _ = esm.pretrained.esm2_t33_650M_UR50D()
        self.load_state_dict(esm_weights.state_dict(), strict=False)
        
    def forward(
            self,
            x: Tensor,
            pos: Tensor,
            batch: Tensor,
            edge_index: Tensor, # unused
            edge_index_star: Tensor = None,  # unused
            edge_attr: Tensor = None,
            edge_attr_star: Tensor = None,  # unused
            edge_vec: Tensor = None,
            edge_vec_star: Tensor = None,  # unused
            node_vec_attr: Tensor = None,
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List]:
        # pass input to output directly, serve as a baseline
        vec = node_vec_attr
        attn_weight_layers = []
        tokens = x
        # tokens should be B x L, where each element is an integer in [0, ESM_ALPHABET_SIZE]
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for _, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=False,
            )
            attn_weight_layers.append(attn)

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

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
            share_kv=False,
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
            x_use_msa=False,
            drop_out_rate=0,  # new feature
            use_lora=None,
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
        self.share_kv = share_kv
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
        self.use_lora = use_lora
        self.use_msa = x_use_msa

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
        self.msa_encoder = MSAEncoder(
            num_species=199, 
            weighting_schema='spe',
            pairwise_type='cov',
        ) if x_use_msa else None

        self.node_x_proj = None
        if x_in_channels is not None:
            if x_in_embedding_type == "Linear":
                self.node_x_proj = nn.Linear(x_in_channels, x_channels)
            elif x_in_embedding_type == "Linear_gelu":
                self.node_x_proj = nn.Sequential(
                    nn.Linear(x_in_channels, x_channels),
                    nn.GELU(),
                )
            else:
                self.node_x_proj = nn.Embedding(x_in_channels, x_channels)
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
                share_kv=self.share_kv,
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
            edge_vec: Tensor = None,
            edge_vec_star: Tensor = None,  # unused
            node_vec_attr: Tensor = None,
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List]:
        if edge_vec is None:
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
        # add MSA to edge attributes
        if (self.x_in_channels is not None and x.shape[1] > self.x_in_channels) or x.shape[1] > self.x_channels:
            if self.node_x_proj is not None:
                x, x_msa = x[:, :self.x_in_channels], x[:, self.x_in_channels:]
            else:
                x, x_msa = x[:, :self.x_channels], x[:, self.x_channels:]
        else:
            x_msa = None
        # MSA channels by defaule are 200
        # embed msa into edge features
        if self.msa_encoder is not None and x_msa is not None:
            _, msa_edge_attr_star = self.msa_encoder(x_msa, edge_index_star)
            edge_attr_star = torch.cat([edge_attr_star, msa_edge_attr_star], dim=-1)
            _, msa_edge_attr = self.msa_encoder(x_msa, edge_index)
            edge_attr = torch.cat([edge_attr, msa_edge_attr], dim=-1)
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
            share_kv=False,
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
            x_use_msa=False,
            drop_out_rate=0,  # new feature
            use_lora=None,
    ):
        super(eqStarTransformer, self).__init__(x_in_channels=x_in_channels,
                                                x_channels=x_channels,
                                                x_hidden_channels=x_hidden_channels,
                                                vec_in_channels=vec_in_channels,
                                                vec_channels=vec_channels,
                                                vec_hidden_channels=vec_hidden_channels,
                                                share_kv=share_kv,
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
                                                x_use_msa=x_use_msa,
                                                drop_out_rate=drop_out_rate,
                                                use_lora=use_lora)

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
        # add MSA to edge attributes
        if self.node_x_proj is not None:
            if x.shape[1] > self.x_in_channels:
                x, x_msa = x[:, :self.x_in_channels], x[:, self.x_in_channels:]
            else:
                x_msa = None
        elif x.shape[1] > self.x_channels:
            x, x_msa = x[:, :self.x_channels], x[:, self.x_channels:]
        else:
            x_msa = None
        # MSA channels by defaule are 200
        # embed msa into edge features
        if self.msa_encoder is not None and x_msa is not None:
            _, msa_edge_attr_star = self.msa_encoder(x_msa, edge_index_star)
            edge_attr_star = torch.cat([edge_attr_star, msa_edge_attr_star], dim=-1)
            # msa can only be added to edge_attr_star
            # _, msa_edge_attr = self.msa_encoder(x_msa, edge_index)
            # edge_attr = torch.cat([edge_attr, msa_edge_attr], dim=-1)
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
        # if self.use_msa:
        #     # if use msa, means edge_attr is updated, then we return the edge_attr_star
        #     return x, vec, pos, edge_attr_star, batch, attn_weight_layers
        # else:
        #     return x, vec, pos, edge_attr, batch, attn_weight_layers
        return x, vec, pos, edge_attr_star, batch, attn_weight_layers


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
            x_use_msa=False,
            drop_out_rate=0,  # new feature
            use_lora=None,
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
                                                   x_use_msa=x_use_msa,
                                                   drop_out_rate=drop_out_rate,
                                                   use_lora=use_lora)

    def _set_attn_layers(self):
        for _ in range(self.num_layers):
            layer = EquivariantMultiHeadAttentionSoftMax(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_channels,
                vec_hidden_channels=self.vec_hidden_channels,
                share_kv=self.share_kv,
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
            share_kv=False,
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
            x_use_msa=False,
            drop_out_rate=0,  # new feature
            use_lora=None,
    ):
        super(eqStarTransformerSoftMax, self).__init__(x_in_channels=x_in_channels,
                                                       x_channels=x_channels,
                                                       x_hidden_channels=x_hidden_channels,
                                                       vec_in_channels=vec_in_channels,
                                                       vec_channels=vec_channels,
                                                       vec_hidden_channels=vec_hidden_channels,
                                                       share_kv=share_kv,
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
                                                       x_use_msa=x_use_msa,
                                                       drop_out_rate=drop_out_rate,
                                                       use_lora=use_lora)

    def _set_attn_layers(self):
        for _ in range(self.num_layers):
            layer = EquivariantMultiHeadAttentionSoftMax(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_channels,
                vec_hidden_channels=self.vec_hidden_channels,
                share_kv=self.share_kv,
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
            share_kv=False,
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
            x_use_msa=False,
            drop_out_rate=0,  # new feature
            use_lora=None,
    ):
        super(eqStar2TransformerSoftMax, self).__init__(x_in_channels=x_in_channels,
                                                        x_channels=x_channels,
                                                        x_hidden_channels=x_hidden_channels,
                                                        vec_in_channels=vec_in_channels,
                                                        vec_channels=vec_channels,
                                                        vec_hidden_channels=vec_hidden_channels,
                                                        share_kv=share_kv,
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
                                                        x_use_msa=x_use_msa,
                                                        drop_out_rate=drop_out_rate,
                                                        use_lora=use_lora)

    def _set_attn_layers(self):
        assert self.num_layers > 0, "num_layers must be greater than 0"
        # first star graph layer does not have softmax, can have msa
        self.attention_layers.append(
            EquivariantMultiHeadAttention(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_channels,
                vec_hidden_channels=self.vec_hidden_channels,
                share_kv=self.share_kv,
                edge_attr_channels=self.num_rbf + self.num_edge_attr,
                distance_influence=self.distance_influence,
                num_heads=self.num_heads,
                activation=act_class_mapping[self.activation],
                attn_activation=self.attn_activation,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                use_lora=self.use_lora,
            )
        )
        # following layers are full graph layers, have softmax, no msa
        for _ in range(self.num_layers - 1):
            layer = EquivariantMultiHeadAttentionSoftMax(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_channels,
                vec_hidden_channels=self.vec_hidden_channels,
                share_kv=self.share_kv,
                edge_attr_channels=self.num_rbf + self.num_edge_attr - 442 if self.use_msa else self.num_rbf + self.num_edge_attr,
                distance_influence=self.distance_influence,
                num_heads=self.num_heads,
                activation=act_class_mapping[self.activation],
                attn_activation=self.attn_activation,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                use_lora=self.use_lora,
            )
            self.attention_layers.append(layer)


class eqStar2PAETransformerSoftMax(eqStar2TransformerSoftMax):
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
            share_kv=False,
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
            x_use_msa=False,
            drop_out_rate=0,  # new feature
            use_lora=None,
    ):
        super(eqStar2PAETransformerSoftMax, self).__init__(x_in_channels=x_in_channels,
                                                        x_channels=x_channels,
                                                        x_hidden_channels=x_hidden_channels,
                                                        vec_in_channels=vec_in_channels,
                                                        vec_channels=vec_channels,
                                                        vec_hidden_channels=vec_hidden_channels,
                                                        share_kv=share_kv,
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
                                                        x_use_msa=x_use_msa,
                                                        drop_out_rate=drop_out_rate,
                                                        use_lora=use_lora)
        # reformat neighbor embedding
        self.neighbor_embedding = (
            NeighborEmbedding(
                x_channels, num_edge_attr,
                cutoff_lower, cutoff_upper,
            )
            if neighbor_embedding
            else None
        )
        self.neighbor_embedding.reset_parameters()

    def _set_attn_layers(self):
        assert self.num_layers > 0, "num_layers must be greater than 0"
        # first star graph layer does not have softmax, can have msa
        self.attention_layers.append(
            EquivariantPAEMultiHeadAttention(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_channels,
                vec_hidden_channels=self.vec_hidden_channels,
                share_kv=self.share_kv,
                edge_attr_dist_channels=self.num_rbf,
                edge_attr_channels=self.num_edge_attr,
                distance_influence=self.distance_influence,
                num_heads=self.num_heads,
                activation=act_class_mapping[self.activation],
                attn_activation=self.attn_activation,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                use_lora=self.use_lora,
            )
        )
        # following layers are full graph layers, have softmax, no msa
        for _ in range(self.num_layers - 1):
            layer = EquivariantPAEMultiHeadAttentionSoftMax(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_channels,
                vec_hidden_channels=self.vec_hidden_channels,
                share_kv=self.share_kv,
                edge_attr_dist_channels=self.num_rbf,
                edge_attr_channels=self.num_edge_attr - 442 if self.use_msa else self.num_edge_attr,
                distance_influence=self.distance_influence,
                num_heads=self.num_heads,
                activation=act_class_mapping[self.activation],
                attn_activation=self.attn_activation,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                use_lora=self.use_lora,
            )
            self.attention_layers.append(layer)

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
            plddt: Tensor = None,  # required for PAE
            edge_confidence: Tensor = None,  # required for PAE
            edge_confidence_star: Tensor = None,  # required for PAE
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List]:
        edge_index, edge_weight, edge_vec = self.distance(pos, edge_index)
        edge_index_star, edge_weight_star, edge_vec_star = self.distance(pos, edge_index_star)

        assert (
            edge_vec is not None and edge_vec_star is not None
        ), "Distance module did not return directional information"
        # get distance expansion edge attributes
        edge_attr_distance = self.distance_expansion(
            edge_weight)  # [E, num_rbf]
        edge_attr_distance_star = self.distance_expansion(
            edge_weight_star)  # [E, num_rbf]
        # concatenate edge attributes, keep the original edge attributes
        # if edge_attr is not None:
        #     # [E, num_rbf + 145 = 64 + 145 = 209]
        #     edge_attr = torch.cat([edge_attr, edge_attr_distance], dim=-1)
        # else:
        #     edge_attr = edge_attr_distance
        # if edge_attr_star is not None:
        #     edge_attr_star = torch.cat(
        #         [edge_attr_star, edge_attr_distance_star], dim=-1)
        # else:
        #     edge_attr_star = edge_attr_distance_star
        # add MSA to edge attributes
        if self.node_x_proj is not None:
            if x.shape[1] > self.x_in_channels:
                x, x_msa = x[:, :self.x_in_channels], x[:, self.x_in_channels:]
            else:
                x_msa = None
        elif x.shape[1] > self.x_channels:
            x, x_msa = x[:, :self.x_channels], x[:, self.x_channels:]
        else:
            x_msa = None
        # MSA channels by defaule are 200
        # embed msa into edge features
        if self.msa_encoder is not None and x_msa is not None:
            _, msa_edge_attr_star = self.msa_encoder(x_msa, edge_index_star)
            if edge_attr_star is not None:
                edge_attr_star = torch.cat([edge_attr_star, msa_edge_attr_star], dim=-1)
            else:
                edge_attr_star = msa_edge_attr_star
            # _, msa_edge_attr = self.msa_encoder(x_msa, edge_index)
            # if edge_attr is not None:
            #     edge_attr = torch.cat([edge_attr, msa_edge_attr], dim=-1)
            # else:
            #     edge_attr = msa_edge_attr
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
                                             edge_index_star, edge_confidence_star, 
                                             edge_attr_distance_star, edge_attr_star, 
                                             edge_vec_star, plddt,
                                             return_attn=return_attn)
            else:
                dx, dvec, attn_weight = attn(x, vec,
                                             edge_index, edge_confidence, 
                                             edge_attr_distance, edge_attr, 
                                             edge_vec, plddt,
                                             return_attn=return_attn)
            x = x + self.drop(dx)
            vec = vec + self.drop(dvec)
            if return_attn:
                attn_weight_layers.append(attn_weight)
        x = self.out_norm(x)
        return x, vec, pos, edge_attr_star, batch, attn_weight_layers


class eqStar2FullGraphPAETransformerSoftMax(nn.Module):
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
            share_kv=False,
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
            x_use_msa=False,
            drop_out_rate=0,  # new feature
            use_lora=None,
    ):
        super(eqStar2FullGraphPAETransformerSoftMax, self).__init__()

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
        self.share_kv = share_kv
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
        self.use_lora = use_lora
        self.use_msa = x_use_msa

        self.distance = Distance(
            cutoff_lower,
            cutoff_upper,
            return_vecs=True,
            loop=True,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = None
        self.msa_encoder = MSAEncoderFullGraph(
            num_species=199, 
            weighting_schema='spe',
            pairwise_type='cov',
        ) if x_use_msa else None

        self.node_x_proj = None
        if x_in_channels is not None:
            if x_in_embedding_type == "Linear":
                self.node_x_proj = nn.Linear(x_in_channels, x_channels)
            elif x_in_embedding_type == "Linear_gelu":
                self.node_x_proj = nn.Sequential(
                    nn.Linear(x_in_channels, x_channels),
                    nn.GELU(),
                )
            else:
                self.node_x_proj = nn.Embedding(x_in_channels, x_channels)
        self.node_vec_proj = nn.Linear(
            vec_in_channels, vec_channels, bias=False)

        self.attention_layers = nn.ModuleList()
        self._set_attn_layers()
        self.drop = nn.Dropout(drop_out_rate)
        self.out_norm = nn.LayerNorm(x_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()

    def _set_attn_layers(self):
        assert self.num_layers > 0, "num_layers must be greater than 0"
        # first star graph layer does not have softmax, can have msa
        # following layers are full graph layers, have softmax, no msa
        input_dic = {
            "x_channels": self.x_channels,
            "x_hidden_channels": self.x_hidden_channels,
            "vec_channels": self.vec_channels,
            "vec_hidden_channels": self.vec_hidden_channels,
            "share_kv": self.share_kv,
            "edge_attr_dist_channels": self.num_rbf,
            "edge_attr_channels": self.num_edge_attr,
            "distance_influence": self.distance_influence,
            "num_heads": self.num_heads,
            "activation": act_class_mapping[self.activation],
            "attn_activation": self.attn_activation,
            "cutoff_lower": self.cutoff_lower,
            "cutoff_upper": self.cutoff_upper,
            "use_lora": self.use_lora
        }
        for _ in range(self.num_layers):
            layer = EquivariantPAEMultiHeadAttentionSoftMaxFullGraph(**input_dic)
            self.attention_layers.append(layer)

    def forward(
            self,
            x: Tensor,
            pos: Tensor,
            batch: Tensor = None,
            x_padding_mask: Tensor = None,
            edge_index: Tensor = None,
            edge_index_star: Tensor = None,
            edge_attr: Tensor = None,
            edge_attr_star: Tensor = None,
            node_vec_attr: Tensor = None,
            plddt: Tensor = None,  # required for PAE
            edge_confidence: Tensor = None,  # required for PAE
            edge_confidence_star: Tensor = None,  # required for PAE
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List]:
        edge_vec = pos[:, :, None, :] - pos[:, None, :, :]
        edge_weight = torch.norm(edge_vec, dim=-1)
        
        # get distance expansion edge attributes
        edge_attr_distance = self.distance_expansion(edge_weight)  # [E, num_rbf]
        # if self.node_x_proj is not None:
        #     if x.shape[-1] > self.x_in_channels:
        x, x_msa = x[..., :self.x_in_channels], x[..., self.x_in_channels:]
        #     else:
        #         x_msa = None
        # elif x.shape[-1] > self.x_channels:
        #     x, x_msa = x[..., :self.x_channels], x[..., self.x_channels:]
        # else:
        #     x_msa = None
        # MSA channels by defaule are 200
        # embed msa into edge features
        # if self.msa_encoder is not None and x_msa is not None:
            # _, msa_edge_attr_star = self.msa_encoder(x_msa, edge_index_star)
            # if edge_attr_star is not None:
            #     edge_attr_star = torch.cat([edge_attr_star, msa_edge_attr_star], dim=-1)
            # else:
            #     edge_attr_star = msa_edge_attr_star
        _, msa_edge_attr = self.msa_encoder(x_msa)
            # if edge_attr is not None:
        edge_attr = torch.cat([edge_attr, msa_edge_attr], dim=-1)
            # else:
            #     edge_attr = msa_edge_attr
        # cancel edge mask
        mask = torch.ones((edge_vec.shape[0], edge_vec.shape[1], edge_vec.shape[2]),  device=edge_vec.device, dtype=torch.bool)^torch.eye(edge_vec.shape[1], device=edge_vec.device, dtype=torch.bool).unsqueeze(0)
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask] + 1e-12, dim=-1).unsqueeze(-1)
        # apply x embedding if necessary
        x = self.node_x_proj(x) if self.node_x_proj is not None else x
        # apply vec embedding if necessary
        vec = self.node_vec_proj(node_vec_attr) if node_vec_attr is not None \
            else torch.zeros(x.size(0), 3, self.vec_channels, device=x.device)

        attn_weight_layers = []
        for i, attn in enumerate(self.attention_layers):
            # first layer is star graph, next layers are normal graph
            dx, dvec, attn_weight = attn(x, vec,
                                        edge_index, edge_confidence, 
                                        edge_attr_distance, edge_attr, 
                                        edge_vec, plddt, x_padding_mask,
                                        return_attn=return_attn)
            x = x + self.drop(dx)
            vec = vec + self.drop(dvec)
            if return_attn:
                attn_weight_layers.append(attn_weight)
        x = self.out_norm(x)
        return x, vec, pos, [edge_confidence, edge_attr_distance, edge_attr, plddt], batch, attn_weight_layers


class FullGraphPAETransformerSoftMax(eqStar2FullGraphPAETransformerSoftMax):
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
            share_kv=False,
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
            x_use_msa=False,
            drop_out_rate=0,  # new feature
            use_lora=None,
    ):
        super(FullGraphPAETransformerSoftMax, self).__init__(x_in_channels=x_in_channels,
                                                            x_channels=x_channels,
                                                            x_hidden_channels=x_hidden_channels,
                                                            vec_in_channels=vec_in_channels,
                                                            vec_channels=vec_channels,
                                                            vec_hidden_channels=vec_hidden_channels,
                                                            share_kv=share_kv,
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
                                                            x_use_msa=x_use_msa,
                                                            drop_out_rate=drop_out_rate,
                                                            use_lora=use_lora)

    def _set_attn_layers(self):
        assert self.num_layers > 0, "num_layers must be greater than 0"
        # first star graph layer does not have softmax, can have msa
        # following layers are full graph layers, have softmax, no msa
        input_dic = {
            "x_channels": self.x_channels,
            "x_hidden_channels": self.x_hidden_channels,
            "vec_channels": self.vec_channels,
            "vec_hidden_channels": self.vec_hidden_channels,
            "share_kv": self.share_kv,
            "edge_attr_dist_channels": self.num_rbf,
            "edge_attr_channels": self.num_edge_attr,
            "distance_influence": self.distance_influence,
            "num_heads": self.num_heads,
            "activation": act_class_mapping[self.activation],
            "attn_activation": self.attn_activation,
            "cutoff_lower": self.cutoff_lower,
            "cutoff_upper": self.cutoff_upper,
            "use_lora": self.use_lora
        }
        for _ in range(self.num_layers):
            layer = MultiHeadAttentionSoftMaxFullGraph(**input_dic)
            self.attention_layers.append(layer)


class eqStar2WeightedPAETransformerSoftMax(eqStar2PAETransformerSoftMax):
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
            share_kv=False,
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
            x_use_msa=False,
            drop_out_rate=0,  # new feature
            use_lora=None,
    ):
        super(eqStar2WeightedPAETransformerSoftMax, self).__init__(x_in_channels=x_in_channels,
                                                        x_channels=x_channels,
                                                        x_hidden_channels=x_hidden_channels,
                                                        vec_in_channels=vec_in_channels,
                                                        vec_channels=vec_channels,
                                                        vec_hidden_channels=vec_hidden_channels,
                                                        share_kv=share_kv,
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
                                                        x_use_msa=x_use_msa,
                                                        drop_out_rate=drop_out_rate,
                                                        use_lora=use_lora)
        
    def _set_attn_layers(self):
        assert self.num_layers > 0, "num_layers must be greater than 0"
        # first star graph layer does not have softmax, can have msa
        self.attention_layers.append(
            EquivariantWeightedPAEMultiHeadAttention(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_channels,
                vec_hidden_channels=self.vec_hidden_channels,
                share_kv=self.share_kv,
                edge_attr_dist_channels=self.num_rbf,
                edge_attr_channels=self.num_edge_attr,
                distance_influence=self.distance_influence,
                num_heads=self.num_heads,
                activation=act_class_mapping[self.activation],
                attn_activation=self.attn_activation,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                use_lora=self.use_lora,
            )
        )
        # following layers are full graph layers, have softmax, no msa
        for _ in range(self.num_layers - 1):
            layer = EquivariantWeightedPAEMultiHeadAttentionSoftMax(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_channels,
                vec_hidden_channels=self.vec_hidden_channels,
                share_kv=self.share_kv,
                edge_attr_dist_channels=self.num_rbf,
                edge_attr_channels=self.num_edge_attr - 442 if self.use_msa else self.num_edge_attr,
                distance_influence=self.distance_influence,
                num_heads=self.num_heads,
                activation=act_class_mapping[self.activation],
                attn_activation=self.attn_activation,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                use_lora=self.use_lora,
            )
            self.attention_layers.append(layer)


class eqTriStarTransformer(nn.Module):
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
            vec_in_channels=4,  # Now changed to the edge_vec_channels
            vec_channels=128,   # Now changed to the edge_vec_hidden_channels
            vec_hidden_channels=5120,
            num_layers=6,
            num_edge_attr=145,
            num_rbf=50,
            rbf_type="expnormunlim",
            trainable_rbf=True,
            activation="silu",
            attn_activation="silu",
            neighbor_embedding=False,
            num_heads=8,
            distance_influence="both",
            cutoff_lower=0.0,
            cutoff_upper=5.0,
            x_in_embedding_type="Linear",
            x_use_msa=False,
            drop_out_rate=0,  # new feature
            use_lora=None,
    ):
        super(eqTriStarTransformer, self).__init__()

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

        self.distance = DistanceV2(
            return_vecs=True,
            loop=True,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        
        self.node_x_proj = None
        if x_in_channels is not None:
            self.node_x_proj = nn.Linear(x_in_channels, x_channels) if x_in_embedding_type == "Linear" \
                else nn.Embedding(x_in_channels, x_channels)

        self.attention_layers = nn.ModuleList()
        self._set_attn_layers()
        self.drop = nn.Dropout(drop_out_rate)
        self.out_norm = nn.LayerNorm(x_channels)

        self.reset_parameters()

    def _set_attn_layers(self):
        for _ in range(self.num_layers):
            layer = EquivariantTriAngularMultiHeadAttention(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_in_channels,
                vec_hidden_channels=self.vec_channels,
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
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()

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
        coords = node_vec_attr + pos.unsqueeze(2)
        edge_index, edge_weight, edge_vec = self.distance(pos, coords, edge_index)
        edge_index_star, edge_weight_star, edge_vec_star = self.distance(pos, coords, edge_index_star)
        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"
        # get distance expansion edge attributes
        # edge_attr_distance =   # [E, num_rbf]
        # edge_attr_distance_star =   # [E, num_rbf]
        # concatenate edge attributes
        # TODO: ADD MSA HERE
        # [E, num_rbf + 145 = 64 + 145 = 209]
        edge_attr = torch.cat([edge_attr, self.distance_expansion(edge_weight)], dim=-1)
        edge_attr_star = torch.cat([edge_attr_star, self.distance_expansion(edge_weight_star)], dim=-1)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        mask = edge_index_star[0] != edge_index_star[1]
        edge_vec_star[mask] = edge_vec_star[mask] / torch.norm(edge_vec_star[mask], dim=1).unsqueeze(1)
        del mask, edge_weight, edge_weight_star
        # apply embedding of x if necessary
        x = self.node_x_proj(x) if self.node_x_proj is not None else x

        attn_weight_layers = []
        for i, attn in enumerate(self.attention_layers):
            if i == 0:
                dx, edge_attr_star, attn_weight = attn(
                    x, edge_index_star, edge_attr_star, edge_vec_star)
            else:
                dx, edge_attr, attn_weight = attn(
                    x, edge_index, edge_attr, edge_vec)
            x = x + self.drop(dx)
            if return_attn:
                attn_weight_layers.append(attn_weight)
        x = self.out_norm(x)
        return x, None, pos, edge_attr, batch, attn_weight_layers

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


class eqMSATriStarTransformer(nn.Module):
    """The equivariant Transformer architecture. Edge attributes are MSA weights.

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
            vec_in_channels=4,  # Now changed to the edge_vec_channels
            vec_channels=128,   # Now changed to the edge_vec_hidden_channels
            vec_hidden_channels=5120,
            num_layers=6,
            num_edge_attr=145,
            num_rbf=50,
            rbf_type="expnormunlim",
            trainable_rbf=True,
            activation="silu",
            attn_activation="silu",
            neighbor_embedding=False,
            num_heads=8,
            distance_influence="both",
            cutoff_lower=0.0,
            cutoff_upper=5.0,
            x_in_embedding_type="Linear",
            x_use_msa=True,
            triangular_update=True,
            ee_channels=None,  # new feature
            drop_out_rate=0,  # new feature
            use_lora=None,
    ):
        super(eqMSATriStarTransformer, self).__init__()

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
        self.triangular_update = triangular_update

        self.distance = DistanceV2(
            return_vecs=True,
            loop=True,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.msa_encoder = MSAEncoder(
            num_species=199, 
            weighting_schema='spe',
            pairwise_type='cov',
        ) if x_use_msa else None

        self.node_x_proj = None
        if x_in_channels is not None:
            if x_in_embedding_type == "Linear":
                self.node_x_proj = nn.Linear(x_in_channels, x_channels)
            elif x_in_embedding_type == "Linear_gelu":
                self.node_x_proj = nn.Sequential(
                    nn.Linear(x_in_channels, x_channels),
                    nn.GELU(),
                )
            else:
                nn.Embedding(x_in_channels, x_channels)
        self.ee_channels = ee_channels
        self.attention_layers = nn.ModuleList()
        self._set_attn_layers()
        self.drop = nn.Dropout(drop_out_rate)
        self.out_norm = nn.LayerNorm(x_channels)

        self.reset_parameters()

    def _set_attn_layers(self):
        for _ in range(self.num_layers):
            layer = EquivariantTriAngularMultiHeadAttention(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_in_channels,
                vec_hidden_channels=self.vec_channels,
                edge_attr_channels=self.num_rbf + self.num_edge_attr,
                distance_influence=self.distance_influence,
                num_heads=self.num_heads,
                activation=act_class_mapping[self.activation],
                attn_activation=self.attn_activation,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                ee_channels=self.ee_channels,
                triangular_update=self.triangular_update,
            )
            self.attention_layers.append(layer)

    def reset_parameters(self):
        self.distance_expansion.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()

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
        coords = node_vec_attr + pos.unsqueeze(2)
        # edge_index, edge_weight, edge_vec = self.distance(pos, coords, edge_index)
        edge_index_star, edge_weight_star, edge_vec_star = self.distance(pos, coords, edge_index_star)
        # split MSA features in x
        if (self.x_in_channels is not None and x.shape[1] > self.x_in_channels) or x.shape[1] > self.x_channels:
            if self.node_x_proj is not None:
                x, x_msa = x[:, :self.x_in_channels], x[:, self.x_in_channels:]
            else:
                x, x_msa = x[:, :self.x_channels], x[:, self.x_channels:]
        else:
            x_msa = None
        # MSA channels by defaule are 200
        # assert (
        #     edge_vec is not None
        # ), "Distance module did not return directional information"
        # embed msa into edge features
        if self.msa_encoder is not None and x_msa is not None:
            _, msa_edge_attr_star = self.msa_encoder(x_msa, edge_index_star)
            edge_attr_star = torch.cat([edge_attr_star, msa_edge_attr_star], dim=-1)
            # No edge attr to save RAM
            # msa_edge_attr = self.msa_encoder(x_msa, edge_index)
            # edge_attr = torch.cat([edge_attr, msa_edge_attr], dim=-1)
        # get distance expansion edge attributes
        # edge_attr = torch.cat([edge_attr, self.distance_expansion(edge_weight)], dim=-1)
        del edge_attr
        edge_attr_star = torch.cat([edge_attr_star, self.distance_expansion(edge_weight_star)], dim=-1)
        # mask = edge_index[0] != edge_index[1]
        # edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        mask = edge_index_star[0] != edge_index_star[1]
        edge_vec_star[mask] = edge_vec_star[mask] / torch.norm(edge_vec_star[mask], dim=1).unsqueeze(1)
        del mask, edge_weight_star
        # apply embedding of x if necessary
        x = self.node_x_proj(x) if self.node_x_proj is not None else x

        attn_weight_layers = []
        for i, attn in enumerate(self.attention_layers):
            if i == 0:
                dx, edge_attr_star, attn_weight = attn(
                    x, coords, edge_index_star, edge_attr_star, edge_vec_star)
            else:
                dx = 0
            x = x + self.drop(dx)
            if return_attn:
                attn_weight_layers.append(attn_weight)
        x = self.out_norm(x)
        return x, None, pos, edge_attr_star, batch, attn_weight_layers

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


class eqMSATriStarGRUTransformer(nn.Module):
    """The equivariant Transformer architecture. Edge attributes are MSA weights.

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
            vec_in_channels=4,  # Now changed to the edge_vec_channels
            vec_channels=128,   # Now changed to the edge_vec_hidden_channels
            vec_hidden_channels=5120,
            num_layers=6,
            num_edge_attr=145,
            num_rbf=50,
            rbf_type="expnormunlim",
            trainable_rbf=True,
            activation="silu",
            attn_activation="silu",
            neighbor_embedding=False,
            num_heads=8,
            distance_influence="both",
            cutoff_lower=0.0,
            cutoff_upper=5.0,
            x_in_embedding_type="Linear",
            x_use_msa=True,
            triangular_update=True,
            ee_channels=None,  # new feature
            drop_out_rate=0,  # new feature
            use_lora=None,
    ):
        super(eqMSATriStarGRUTransformer, self).__init__()

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
        self.triangular_update = triangular_update

        self.distance = DistanceV2(
            return_vecs=True,
            loop=True,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.msa_encoder = MSAEncoder(
            num_species=199, 
            weighting_schema='spe',
            pairwise_type='cov',
        ) if x_use_msa else None

        self.node_x_proj = None
        if x_in_channels is not None:
            if x_in_embedding_type == "Linear":
                self.node_x_proj = nn.Linear(x_in_channels, x_channels)
            elif x_in_embedding_type == "Linear_gelu":
                self.node_x_proj = nn.Sequential(
                    nn.Linear(x_in_channels, x_channels),
                    nn.GELU(),
                )
            else:
                nn.Embedding(x_in_channels, x_channels)
        self.ee_channels = ee_channels
        self.attention_layers = nn.ModuleList()
        self._set_attn_layers()
        self.drop = nn.Dropout(drop_out_rate)
        # self.out_norm = nn.LayerNorm(x_channels)

        self.reset_parameters()

    def _set_attn_layers(self):
        for _ in range(self.num_layers):
            layer = EquivariantTriAngularStarMultiHeadAttention(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_in_channels,
                vec_hidden_channels=self.vec_channels,
                edge_attr_channels=self.num_rbf + self.num_edge_attr,
                distance_influence=self.distance_influence,
                num_heads=self.num_heads,
                activation=act_class_mapping[self.activation],
                attn_activation=self.attn_activation,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                ee_channels=self.ee_channels,
                triangular_update=self.triangular_update,
            )
            self.attention_layers.append(layer)

    def reset_parameters(self):
        self.distance_expansion.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        # self.out_norm.reset_parameters()

    def forward(
            self,
            x: Tensor,
            x_center: Tensor,
            x_mask: Tensor,
            pos: Tensor,
            batch: Tensor,
            edge_index: Tensor,
            edge_index_star: Tensor = None,  
            edge_attr: Tensor = None,
            edge_attr_star: Tensor = None, 
            node_vec_attr: Tensor = None,
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List]:
        coords = node_vec_attr + pos.unsqueeze(2)
        # edge_index, edge_weight, edge_vec = self.distance(pos, coords, edge_index)
        edge_index_star, edge_weight_star, edge_vec_star = self.distance(pos, coords, edge_index_star)
        # split MSA features in x
        if (self.x_in_channels is not None and x.shape[1] > self.x_in_channels) or x.shape[1] > self.x_channels:
            if self.node_x_proj is not None:
                x, x_msa = x[:, :self.x_in_channels], x[:, self.x_in_channels:]
            else:
                x, x_msa = x[:, :self.x_channels], x[:, self.x_channels:]
        else:
            x_msa = None
        # MSA channels by defaule are 200
        # assert (
        #     edge_vec is not None
        # ), "Distance module did not return directional information"
        # embed msa into edge features
        if self.msa_encoder is not None and x_msa is not None:
            _, msa_edge_attr_star = self.msa_encoder(x_msa, edge_index_star)
            edge_attr_star = torch.cat([edge_attr_star, msa_edge_attr_star], dim=-1)
            # No edge attr to save RAM
            # msa_edge_attr = self.msa_encoder(x_msa, edge_index)
            # edge_attr = torch.cat([edge_attr, msa_edge_attr], dim=-1)
        # get distance expansion edge attributes
        # edge_attr = torch.cat([edge_attr, self.distance_expansion(edge_weight)], dim=-1)
        del edge_attr
        edge_attr_star = torch.cat([edge_attr_star, self.distance_expansion(edge_weight_star)], dim=-1)
        # mask = edge_index[0] != edge_index[1]
        # edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        mask = edge_index_star[0] != edge_index_star[1]
        edge_vec_star[mask] = edge_vec_star[mask] / torch.norm(edge_vec_star[mask], dim=1).unsqueeze(1)
        del mask, edge_weight_star
        # apply embedding of x if necessary
        x = self.node_x_proj(x) if self.node_x_proj is not None else x
        x = x * x_mask.unsqueeze(1) + x_center * (~x_mask).unsqueeze(1)

        attn_weight_layers = []
        for _, attn in enumerate(self.attention_layers):
            x, edge_attr_star, attn_weight = attn(
                x, coords, edge_index_star, edge_attr_star, edge_vec_star)
            if return_attn:
                attn_weight_layers.append(attn_weight)
        x = self.drop(x)
        # x = self.out_norm(x)
        batch = batch[~x_mask]
        return x, None, pos, edge_attr_star, batch, attn_weight_layers

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


class eqMSATriStarDropTransformer(nn.Module):
    """The equivariant Transformer architecture. Edge attributes are MSA weights, distances and drop out is applied.

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
            vec_in_channels=4,  # Now changed to the edge_vec_channels
            vec_channels=128,   # Now changed to the edge_vec_hidden_channels
            vec_hidden_channels=5120,
            num_layers=6,
            num_edge_attr=145,
            num_rbf=50,
            rbf_type="expnormunlim",
            trainable_rbf=True,
            activation="silu",
            attn_activation="silu",
            neighbor_embedding=False,
            num_heads=8,
            distance_influence="both",
            cutoff_lower=0.0,
            cutoff_upper=5.0,
            x_in_embedding_type="Linear",
            x_use_msa=True,
            triangular_update=True,
            ee_channels=None,  # new feature
            drop_out_rate=0,  # new feature
            use_lora=None,
            layer_norm=True,
    ):
        super(eqMSATriStarDropTransformer, self).__init__()

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
        self.triangular_update = triangular_update
        self.use_lora = use_lora
        self.layer_norm = layer_norm

        self.distance = DistanceV2(
            return_vecs=True,
            loop=True,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.msa_encoder = MSAEncoder(
            num_species=199, 
            weighting_schema='spe',
            pairwise_type='cov',
        ) if x_use_msa else None

        self.node_x_proj = None
        if x_in_channels is not None:
            if x_in_embedding_type == "Linear":
                if use_lora is not None:
                    self.node_x_proj = lora.Linear(x_in_channels, x_channels, r=use_lora)
                else:
                    self.node_x_proj = nn.Linear(x_in_channels, x_channels)
            elif x_in_embedding_type == "Linear_gelu":
                self.node_x_proj = nn.Sequential(
                    lora.Linear(x_in_channels, x_channels, r=use_lora) if use_lora is not None else nn.Linear(x_in_channels, x_channels),
                    nn.GELU(),
                )
            else:
                nn.Embedding(x_in_channels, x_channels) if use_lora is None else lora.Embedding(x_in_channels, x_channels, r=use_lora)
        self.ee_channels = ee_channels
        self.attention_layers = nn.ModuleList()
        # self.drop = nn.Dropout(drop_out_rate)
        self.drop_out_rate = drop_out_rate
        self._set_attn_layers()
        # self.out_norm = nn.LayerNorm(x_channels)

        self.reset_parameters()

    def _set_attn_layers(self):
        for _ in range(self.num_layers):
            layer = EquivariantTriAngularDropMultiHeadAttention(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_in_channels,
                vec_hidden_channels=self.vec_channels,
                edge_attr_channels=self.num_rbf + self.num_edge_attr,
                distance_influence=self.distance_influence,
                num_heads=self.num_heads,
                activation=act_class_mapping[self.activation],
                attn_activation=self.attn_activation,
                ee_channels=self.ee_channels,
                rbf_channels=self.num_rbf,
                triangular_update=self.triangular_update,
                drop_out_rate=self.drop_out_rate,
                use_lora=self.use_lora,
                layer_norm=self.layer_norm,
            )
            self.attention_layers.append(layer)

    def reset_parameters(self):
        self.distance_expansion.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        # self.out_norm.reset_parameters()

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
        coords = node_vec_attr + pos.unsqueeze(2)
        # edge_index, edge_weight, edge_vec = self.distance(pos, coords, edge_index)
        edge_index_star, edge_weight_star, edge_vec_star = self.distance(pos, coords, edge_index_star)
        # split MSA features in x
        if (self.x_in_channels is not None and x.shape[1] > self.x_in_channels) or x.shape[1] > self.x_channels:
            if self.node_x_proj is not None:
                x, x_msa = x[:, :self.x_in_channels], x[:, self.x_in_channels:]
            else:
                x, x_msa = x[:, :self.x_channels], x[:, self.x_channels:]
        else:
            x_msa = None
        # MSA channels by defaule are 200
        # assert (
        #     edge_vec is not None
        # ), "Distance module did not return directional information"
        # embed msa into edge features
        if self.msa_encoder is not None and x_msa is not None:
            _, msa_edge_attr_star = self.msa_encoder(x_msa, edge_index_star)
            edge_attr_star = torch.cat([edge_attr_star, msa_edge_attr_star], dim=-1)
            # No edge attr to save RAM
            # msa_edge_attr = self.msa_encoder(x_msa, edge_index)
            # edge_attr = torch.cat([edge_attr, msa_edge_attr], dim=-1)
        # get distance expansion edge attributes
        # edge_attr = torch.cat([edge_attr, self.distance_expansion(edge_weight)], dim=-1)
        del edge_attr
        edge_attr_star = torch.cat([edge_attr_star, self.distance_expansion(edge_weight_star)], dim=-1)
        # mask = edge_index[0] != edge_index[1]
        # edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        mask = edge_index_star[0] != edge_index_star[1]
        edge_vec_star[mask] = edge_vec_star[mask] / torch.norm(edge_vec_star[mask], dim=1).unsqueeze(1)
        del mask, edge_weight_star
        # apply embedding of x if necessary
        x = self.node_x_proj(x) if self.node_x_proj is not None else x
        # x = x * x_mask.unsqueeze(1) + x_center * (~x_mask).unsqueeze(1)

        attn_weight_layers = []
        for _, attn in enumerate(self.attention_layers):
            x, edge_attr_star, attn_weight = attn(
                x, coords, edge_index_star, edge_attr_star, edge_vec_star)
            if return_attn:
                attn_weight_layers.append(attn_weight)
        # x = self.drop(x)
        # x = self.out_norm(x)
        # batch = batch[~x_mask]
        return x, None, pos, edge_attr_star, batch, attn_weight_layers

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


class eqMSATriStarDropGRUTransformer(nn.Module):
    """The equivariant Transformer architecture. Edge attributes are MSA weights, distances and drop out is applied.

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
            vec_in_channels=4,  # Now changed to the edge_vec_channels
            vec_channels=128,   # Now changed to the edge_vec_hidden_channels
            vec_hidden_channels=5120,
            num_layers=6,
            num_edge_attr=145,
            num_rbf=50,
            rbf_type="expnormunlim",
            trainable_rbf=True,
            activation="silu",
            attn_activation="silu",
            neighbor_embedding=False,
            num_heads=8,
            distance_influence="both",
            cutoff_lower=0.0,
            cutoff_upper=5.0,
            x_in_embedding_type="Linear",
            x_use_msa=True,
            triangular_update=True,
            ee_channels=None,  # new feature
            drop_out_rate=0,  # new feature
            use_lora=None,
    ):
        super(eqMSATriStarDropGRUTransformer, self).__init__()

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
        self.triangular_update = triangular_update
        self.use_lora = use_lora

        self.distance = DistanceV2(
            return_vecs=True,
            loop=True,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.msa_encoder = MSAEncoder(
            num_species=199, 
            weighting_schema='spe',
            pairwise_type='cov',
        ) if x_use_msa else None

        self.node_x_proj = None
        if x_in_channels is not None:
            if x_in_embedding_type == "Linear":
                if use_lora is not None:
                    self.node_x_proj = lora.Linear(x_in_channels, x_channels, r=use_lora)
                else:
                    self.node_x_proj = nn.Linear(x_in_channels, x_channels)
            elif x_in_embedding_type == "Linear_gelu":
                self.node_x_proj = nn.Sequential(
                    lora.Linear(x_in_channels, x_channels, r=use_lora) if use_lora is not None else nn.Linear(x_in_channels, x_channels),
                    nn.GELU(),
                )
            else:
                nn.Embedding(x_in_channels, x_channels) if use_lora is None else lora.Embedding(x_in_channels, x_channels, r=use_lora)
        self.ee_channels = ee_channels
        self.attention_layers = nn.ModuleList()
        # self.drop = nn.Dropout(drop_out_rate)
        self.drop_out_rate = drop_out_rate
        self._set_attn_layers()
        # self.out_norm = nn.LayerNorm(x_channels)

        self.reset_parameters()

    def _set_attn_layers(self):
        for _ in range(self.num_layers):
            layer = EquivariantTriAngularStarDropMultiHeadAttention(
                x_channels=self.x_channels,
                x_hidden_channels=self.x_hidden_channels,
                vec_channels=self.vec_in_channels,
                vec_hidden_channels=self.vec_channels,
                edge_attr_channels=self.num_rbf + self.num_edge_attr,
                distance_influence=self.distance_influence,
                num_heads=self.num_heads,
                activation=act_class_mapping[self.activation],
                attn_activation=self.attn_activation,
                ee_channels=self.ee_channels,
                rbf_channels=self.num_rbf,
                triangular_update=self.triangular_update,
                drop_out_rate=self.drop_out_rate,
                use_lora=self.use_lora,
            )
            self.attention_layers.append(layer)

    def reset_parameters(self):
        self.distance_expansion.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        # self.out_norm.reset_parameters()

    def forward(
            self,
            x: Tensor,
            x_center: Tensor,
            x_mask: Tensor,
            pos: Tensor,
            batch: Tensor,
            edge_index: Tensor,
            edge_index_star: Tensor = None,  
            edge_attr: Tensor = None,
            edge_attr_star: Tensor = None, 
            node_vec_attr: Tensor = None,
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List]:
        coords = node_vec_attr + pos.unsqueeze(2)
        # edge_index, edge_weight, edge_vec = self.distance(pos, coords, edge_index)
        edge_index_star, edge_weight_star, edge_vec_star = self.distance(pos, coords, edge_index_star)
        # split MSA features in x
        if (self.x_in_channels is not None and x.shape[1] > self.x_in_channels) or x.shape[1] > self.x_channels:
            if self.node_x_proj is not None:
                x, x_msa = x[:, :self.x_in_channels], x[:, self.x_in_channels:]
            else:
                x, x_msa = x[:, :self.x_channels], x[:, self.x_channels:]
        else:
            x_msa = None
        # MSA channels by defaule are 200
        # assert (
        #     edge_vec is not None
        # ), "Distance module did not return directional information"
        # embed msa into edge features
        if self.msa_encoder is not None and x_msa is not None:
            _, msa_edge_attr_star = self.msa_encoder(x_msa, edge_index_star)
            edge_attr_star = torch.cat([edge_attr_star, msa_edge_attr_star], dim=-1)
            # No edge attr to save RAM
            # msa_edge_attr = self.msa_encoder(x_msa, edge_index)
            # edge_attr = torch.cat([edge_attr, msa_edge_attr], dim=-1)
        # get distance expansion edge attributes
        # edge_attr = torch.cat([edge_attr, self.distance_expansion(edge_weight)], dim=-1)
        del edge_attr
        edge_attr_star = torch.cat([edge_attr_star, self.distance_expansion(edge_weight_star)], dim=-1)
        # mask = edge_index[0] != edge_index[1]
        # edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        mask = edge_index_star[0] != edge_index_star[1]
        edge_vec_star[mask] = edge_vec_star[mask] / torch.norm(edge_vec_star[mask], dim=1).unsqueeze(1)
        del mask, edge_weight_star
        # apply embedding of x if necessary
        x = self.node_x_proj(x) if self.node_x_proj is not None else x
        x = x * x_mask.unsqueeze(1) + x_center * (~x_mask).unsqueeze(1)

        attn_weight_layers = []
        for _, attn in enumerate(self.attention_layers):
            x, edge_attr_star, attn_weight = attn(
                x, coords, edge_index_star, edge_attr_star, edge_vec_star)
            if return_attn:
                attn_weight_layers.append(attn_weight)
        # x = self.drop(x)
        # x = self.out_norm(x)
        batch = batch[~x_mask]
        return x, None, pos, edge_attr_star, batch, attn_weight_layers

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
                 x_use_msa=False,
                 use_lora=None,
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
            residx = torch.arange(
                x.shape[1], device=x.device).repeat(x.shape[0], 1)
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[1]),
                              dtype=torch.bool, device=x.device)
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

        s_s = self.seq_pair_to_output(
            sequence_state=s_s, pairwise_state=s_z, mask=mask.to(torch.float32))
        # s_out = self.seq_pair_to_output(s_s, s_z, residx, mask)
        # to output and make it look like previous transformers
        # x, vec, pos, edge_attr, batch, attn_weight_layers
        return s_s, s_z, pos, None, None, None
