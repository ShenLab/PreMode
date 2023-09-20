from abc import abstractmethod, ABCMeta
from typing import Optional, Tuple
import math
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_scatter import scatter
import loralib as lora

from model.module.utils import act_class_mapping
from .utils import CosineCutoff


class OutputModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, allow_prior_model, reduce_op):
        super(OutputModel, self).__init__()
        self.allow_prior_model = allow_prior_model
        self.reduce_op = reduce_op

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, pos, batch):
        return

    def reduce(self, x, edge_index, edge_attr, batch):
        return scatter(x, batch, dim=0, reduce=self.reduce_op), None

    def post_reduce(self, x):
        return x


class StarPoolOutputModel(OutputModel, metaclass=ABCMeta):
    def __init__(
            self,
            args,
            activation="pass", # default is regression task, no activation
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        out_channels = args["output_dim"]
        reduce_op = args["reduce_op"]
        super(StarPoolOutputModel, self).__init__(
            allow_prior_model=allow_prior_model, reduce_op=reduce_op
        )
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(x_channels, out_channels),
            act_class(),
        )
        self.reset_parameters()
        # apply two layers of SAGPool
        self.StarPoolLayers = nn.ModuleList([
            StarPool(hidden_channels=x_channels,
                     edge_channels=args["num_rbf"] + args["num_edge_attr"],
                     cutoff_lower=args["cutoff_lower"],
                     cutoff_upper=args["cutoff_upper"],
                     use_lora=args["use_lora"],
                     drop_out=args["drop_out"],
                     ratio=0.5),
        ])

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], pos, batch):
        return x

    def post_reduce(self, x):
        return self.output_network(x)

    def reduce(self, x, edge_index, edge_attr, batch):
        for layer in self.StarPoolLayers:
            x, edge_index, edge_attr, batch, perm = layer(x, edge_index, edge_attr, batch)
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)
        return out, None


class EquivariantBinaryClassificationNoGraphScalar(OutputModel):
    def __init__(
            self,
            args,
            activation="sigmoid",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        out_channels = args["output_dim"]
        reduce_op = args["reduce_op"]
        super(EquivariantBinaryClassificationNoGraphScalar, self).__init__(
            allow_prior_model=allow_prior_model, reduce_op=reduce_op
        )
        act_class = act_class_mapping[activation]
        self.layer_norm = nn.LayerNorm(x_channels)
        self.output_network = nn.Sequential(
            nn.Linear(x_channels, out_channels),
            act_class(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], pos, batch):
        return x

    def reduce(self, x, edge_index, edge_attr, batch):
        return x.sum(axis=-2, keepdim=False), None

    def post_reduce(self, x):
        x = self.layer_norm(x)
        return self.output_network(x)


class EquivariantBinaryClassificationNoGraphTanhScalar(EquivariantBinaryClassificationNoGraphScalar):
    def __init__(
            self,
            args,
            activation="tanh",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        out_channels = args["output_dim"]
        super(EquivariantBinaryClassificationNoGraphTanhScalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )
        self.output_network = nn.Sequential(
            nn.Linear(x_channels, out_channels, bias=False),
            act_class_mapping[activation](),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)


class EquivariantBinaryClassificationScalar(OutputModel):
    def __init__(
            self,
            args,
            activation="sigmoid",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        if args["model"] == "pass-forward":
            x_channels = args["x_in_channels"] if args["x_in_channels"] is not None else args["x_channels"]
            if args["add_msa"]:
                x_channels += 199
        out_channels = args["output_dim"]
        reduce_op = args["reduce_op"]
        super(EquivariantBinaryClassificationScalar, self).__init__(
            allow_prior_model=allow_prior_model, reduce_op=reduce_op
        )
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(x_channels, out_channels),
            act_class(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.output_network[0].weight)
        nn.init.constant_(self.output_network[0].weight, 0)
        self.output_network[0].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], pos, batch):
        return x

    def post_reduce(self, x):
        return self.output_network(x)


class EquivariantBinaryClassificationSAGPoolScalar(EquivariantBinaryClassificationScalar):
    def __init__(
            self,
            args,
            activation="sigmoid",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        super(EquivariantBinaryClassificationSAGPoolScalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )
        # apply two layers of SAGPool
        self.SAGPoolLayers = nn.ModuleList([
            SAGPool(hidden_channels=x_channels,
                    edge_channels=args["num_rbf"] + args["num_edge_attr"],
                    cutoff_lower=args["cutoff_lower"],
                    cutoff_upper=args["cutoff_upper"],
                    ratio=0.5),
            SAGPool(hidden_channels=x_channels,
                    edge_channels=args["num_rbf"] + args["num_edge_attr"],
                    cutoff_lower=args["cutoff_lower"],
                    cutoff_upper=args["cutoff_upper"],
                    ratio=0.5),
        ])

    def reduce(self, x, edge_index, edge_attr, batch):
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)
        for layer in self.SAGPoolLayers:
            x, edge_index, edge_attr, batch, perm = layer(x, edge_index, edge_attr, batch)
            out += scatter(x, batch, dim=0, reduce=self.reduce_op)
        return out, None


class EquivariantBinaryClassificationStarPoolScalar(EquivariantBinaryClassificationScalar):
    def __init__(
            self,
            args,
            activation="sigmoid",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        super(EquivariantBinaryClassificationStarPoolScalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )
        # apply two layers of StarPool
        if args["loss_fn"] == "weighted_combined_loss" or args["loss_fn"] == "combined_loss":
            use_lora = args["use_lora"]
        else:
            use_lora = None
        self.StarPoolLayers = nn.ModuleList([
            StarPool(hidden_channels=x_channels,
                     edge_channels=args["num_rbf"] + args["num_edge_attr"],
                     cutoff_lower=args["cutoff_lower"],
                     cutoff_upper=args["cutoff_upper"],
                     use_lora=use_lora,
                     drop_out=args["drop_out"],
                     ratio=0.5),
        ])

    def reduce(self, x, edge_index, edge_attr, batch):
        for layer in self.StarPoolLayers:
            x, edge_index, edge_attr, batch, attn = layer(x, edge_index, edge_attr, batch)
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)
        return out, attn


class EquivariantSAGPoolTanhScalar(EquivariantBinaryClassificationSAGPoolScalar):
    def __init__(
            self,
            args,
            activation="tanh",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        out_channels = args["output_dim"]
        super(EquivariantSAGPoolTanhScalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )
        self.output_network = nn.Sequential(
            nn.Linear(x_channels, out_channels, bias=False),
            act_class_mapping[activation](),
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)


class EquivariantTanhStarPoolScalar(EquivariantBinaryClassificationStarPoolScalar):
    def __init__(
            self,
            args,
            activation="tanh",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        out_channels = args["output_dim"]
        super(EquivariantTanhStarPoolScalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )
        self.output_network = nn.Sequential(
            nn.Linear(x_channels, out_channels, bias=False),
            act_class_mapping[activation](),
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)


class EquivariantTanhSAGPoolScalar(EquivariantSAGPoolTanhScalar):
    def __init__(
            self,
            args,
            activation="tanh",
            allow_prior_model=True,
    ):
        super(EquivariantTanhSAGPoolScalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )


class EquivariantTanhScalar(EquivariantBinaryClassificationScalar):
    def __init__(
            self,
            args,
            activation="tanh",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        out_channels = args["output_dim"]
        super(EquivariantTanhScalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )
        self.output_network = nn.Sequential(
            nn.Linear(x_channels, out_channels, bias=False),
            act_class_mapping[activation](),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)


class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return nn.functional.linear(x, self.weight.exp())


class EquivariantBinaryClassificationTanh2Scalar(EquivariantBinaryClassificationScalar):
    def __init__(
            self,
            args,
            activation="tanh",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        out_channels = args["output_dim"]
        super(EquivariantBinaryClassificationTanh2Scalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )
        self.output_network = nn.Sequential(
            PositiveLinear(x_channels, out_channels),
            act_class_mapping[activation](),
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)

    def reduce(self, x, edge_index, edge_attr, batch):
        return scatter(torch.square(x), batch, dim=0, reduce=self.reduce_op), None


class EquivariantRegressionScalar(OutputModel):
    def __init__(
            self,
            args,
            activation="pass",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        if args["model"] == "pass-forward":
            x_channels = args["x_in_channels"]
            if args["add_msa"]:
                x_channels += 199
        out_channels = args["output_dim"]
        reduce_op = args["reduce_op"]
        super(EquivariantRegressionScalar, self).__init__(
            allow_prior_model=allow_prior_model, reduce_op=reduce_op
        )
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(x_channels, out_channels),
            act_class(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], pos, batch):
        return x

    def post_reduce(self, x):
        return self.output_network(x)


class EquivariantRegressionSAGPoolScalar(EquivariantRegressionScalar):
    def __init__(
            self,
            args,
            activation="pass",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        super(EquivariantRegressionSAGPoolScalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )
        # apply two layers of SAGPool
        self.SAGPoolLayers = nn.ModuleList([
            SAGPool(hidden_channels=x_channels,
                    edge_channels=args["num_rbf"] + args["num_edge_attr"],
                    cutoff_lower=args["cutoff_lower"],
                    cutoff_upper=args["cutoff_upper"],
                    ratio=0.5),
            SAGPool(hidden_channels=x_channels,
                    edge_channels=args["num_rbf"] + args["num_edge_attr"],
                    cutoff_lower=args["cutoff_lower"],
                    cutoff_upper=args["cutoff_upper"],
                    ratio=0.5),
        ])

    def reduce(self, x, edge_index, edge_attr, batch):
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)
        for layer in self.SAGPoolLayers:
            x, edge_index, edge_attr, batch, perm = layer(x, edge_index, edge_attr, batch)
            out += scatter(x, batch, dim=0, reduce=self.reduce_op)
        return out, None


class EquivariantRegressionStarPoolScalar(EquivariantRegressionScalar):
    def __init__(
            self,
            args,
            activation="pass",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        super(EquivariantRegressionStarPoolScalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )
        # apply two layers of SAGPool
        # apply two layers of SAGPool
        self.StarPoolLayers = nn.ModuleList([
            StarPool(hidden_channels=x_channels,
                     edge_channels=args["num_rbf"] + args["num_edge_attr"],
                     cutoff_lower=args["cutoff_lower"],
                     cutoff_upper=args["cutoff_upper"],
                     use_lora=args["use_lora"],
                     drop_out=args["drop_out"],
                     ratio=0.5),
        ])

    def reduce(self, x, edge_index, edge_attr, batch):
        for layer in self.StarPoolLayers:
            x, edge_index, edge_attr, batch, attn = layer(x, edge_index, edge_attr, batch)
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)
        return out, attn


class EquivariantMaskPredictScalar(OutputModel):
    def __init__(
            self,
            args,
            lm_weight,
            activation="gelu",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        out_channels = args["x_channels"]
        reduce_op = args["reduce_op"]
        super(EquivariantMaskPredictScalar, self).__init__(
            allow_prior_model=allow_prior_model, reduce_op=reduce_op
        )
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(x_channels, out_channels),
            act_class(),
            nn.LayerNorm(out_channels),
        )
        self.lm_weight = lm_weight
        self.bias = nn.Parameter(torch.zeros(args["x_in_channels"]))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], pos, batch):
        x = self.output_network(x)
        x = nn.functional.linear(x, self.lm_weight) + self.bias
        return x

    def post_reduce(self, x):
        return x


class EquivariantMaskPredictLogLogitsScalar(EquivariantMaskPredictScalar):
    def __init__(
            self,
            args,
            lm_weight,
            activation="gelu",
            allow_prior_model=True,
    ):
        super(EquivariantMaskPredictLogLogitsScalar, self).__init__(
            args=args,
            lm_weight=lm_weight,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )

    def pre_reduce(self, x, v: Optional[torch.Tensor], pos, batch):
        x = self.output_network(x)
        x = nn.functional.linear(x, self.lm_weight) + self.bias
        x = torch.log_softmax(x, dim=-1)
        return x


class Scalar(OutputModel):
    def __init__(
            self,
            args,
            activation="silu",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        out_channels = args["output_dim"]
        reduce_op = args["reduce_op"]
        super(Scalar, self).__init__(
            allow_prior_model=allow_prior_model, reduce_op=reduce_op
        )
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(x_channels, x_channels // 2),
            act_class(),
            nn.Linear(x_channels // 2, out_channels),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], pos, batch):
        return self.output_network(x)


class SAGPool(MessagePassing, metaclass=ABCMeta):
    def __init__(self,
                 hidden_channels,
                 edge_channels,
                 cutoff_lower,
                 cutoff_upper,
                 ratio=0.5,
                 non_linearity=torch.tanh):
        super(SAGPool, self).__init__(aggr="mean")
        self.distance_proj = nn.Linear(edge_channels, 1)
        self.score = nn.Linear(hidden_channels, 1)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.ratio = ratio
        self.non_linearity = non_linearity
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.distance_proj.weight)
        self.distance_proj.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        W = self.distance_proj(edge_attr)
        # convolution to get node scores
        x_score = self.score(x)
        score = self.propagate(edge_index, x=x_score, W=W, size=None).squeeze()
        # perform topK pooling
        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm

    def message(self, x_j, W):
        return x_j * W


def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class StarPool_archive(MessagePassing, metaclass=ABCMeta):
    def __init__(self,
                 hidden_channels,
                 edge_channels,
                 cutoff_lower,
                 cutoff_upper,
                 ratio=0.5,
                 drop_out=0.0,
                 num_heads=32,
                 non_linearity=torch.tanh):
        super(StarPool, self).__init__(aggr="mean")
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dk_proj = nn.Linear(edge_channels, hidden_channels)
        self.layernorm_in = nn.LayerNorm(hidden_channels)
        self.layernorm_out = nn.LayerNorm(hidden_channels)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.x_head_dim = hidden_channels // num_heads
        self.node_dim = 0
        # self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.attn_activation = act_class_mapping["silu"]()
        self.drop_out = nn.Dropout(drop_out)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.dk_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)
        self.dk_proj.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr, batch=None):
        residue = x
        x = self.layernorm_in(x)
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        q = self.q_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        dk = self.dk_proj(edge_attr).reshape(-1, self.num_heads, self.x_head_dim)
        x, _ = self.propagate(
            edge_index = edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            size=None
        )
        x = x.reshape(-1, self.hidden_channels)
        x = residue + x
        # perform topK pooling
        center_nodes = torch.unique(edge_index[1])
        perm = center_nodes
        x = x[perm]
        batch = batch[perm]
        residue = residue[perm]
        x = self.layernorm_out(x)
        x = gelu(self.fc1(x))
        x = self.drop_out(x)
        x = self.fc2(x)
        x = residue + x

        return x, edge_index, edge_attr, batch, perm

    def message(self, q_i, k_j, v_j, dk):
        attn = (q_i * k_j * dk).sum(dim=-1)
        # attention activation function
        attn = self.attn_activation(attn)
        # update scalar features
        x = v_j * attn.unsqueeze(2)
        return x, attn

    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, attn = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x, attn
    
    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs

    def message_and_aggregate(self, adj_t) -> torch.Tensor:
        pass

    def edge_update(self) -> torch.Tensor:
        pass


class StarPool(MessagePassing, metaclass=ABCMeta):
    def __init__(self,
                 hidden_channels,
                 edge_channels,
                 cutoff_lower,
                 cutoff_upper,
                 ratio=0.5,
                 drop_out=0.0,
                 num_heads=32,
                 use_lora=None,
                 non_linearity=torch.tanh):
        super(StarPool, self).__init__(aggr="mean")
        if use_lora is not None:
            self.q_proj = lora.Linear(hidden_channels, hidden_channels, r=use_lora)
            self.kv_proj = lora.Linear(hidden_channels, hidden_channels, r=use_lora)
            self.dk_proj = lora.Linear(edge_channels, hidden_channels, r=use_lora)
            # self.fc1 = lora.Linear(hidden_channels, hidden_channels, r=use_lora)
            # self.fc2 = lora.Linear(hidden_channels, hidden_channels, r=use_lora)
        else:
            self.q_proj = nn.Linear(hidden_channels, hidden_channels)
            self.kv_proj = nn.Linear(hidden_channels, hidden_channels)
            self.dk_proj = nn.Linear(edge_channels, hidden_channels)
            # self.fc1 = nn.Linear(hidden_channels, hidden_channels)
            # self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.layernorm_in = nn.LayerNorm(hidden_channels)
        self.layernorm_out = nn.LayerNorm(hidden_channels)
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.x_head_dim = hidden_channels // num_heads
        self.node_dim = 0
        self.attn_activation = act_class_mapping["silu"]()
        self.drop_out = nn.Dropout(drop_out)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        # nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.dk_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.kv_proj.bias.data.fill_(0)
        # self.v_proj.bias.data.fill_(0)
        self.dk_proj.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr, batch=None):
        residue = x
        x = self.layernorm_in(x)
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        q = self.q_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        k = self.kv_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        # v = self.v_proj(x).reshape(-1, self.num_heads, self.x_head_dim)
        v = k
        dk = self.dk_proj(edge_attr).reshape(-1, self.num_heads, self.x_head_dim)
        x, attn = self.propagate(
            edge_index = edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            size=None
        )
        x = x.reshape(-1, self.hidden_channels)
        x = residue + x
        # perform topK pooling
        center_nodes = torch.unique(edge_index[1])
        perm = center_nodes
        x = x[perm]
        batch = batch[perm]
        residue = residue[perm]
        x = self.layernorm_out(x)
        x = residue + self.drop_out(x)

        return x, edge_index, edge_attr, batch, attn

    def message(self, q_i, k_j, v_j, dk):
        attn = (q_i * k_j * dk).sum(dim=-1)
        # attention activation function
        attn = self.attn_activation(attn)
        # update scalar features
        x = v_j * attn.unsqueeze(2)
        return x, attn

    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, attn = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x, attn
    
    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs

    def message_and_aggregate(self, adj_t) -> torch.Tensor:
        pass

    def edge_update(self) -> torch.Tensor:
        pass
