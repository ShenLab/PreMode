from abc import abstractmethod, ABCMeta
from typing import Optional, Tuple
import math
import torch
from torch import _dynamo
_dynamo.config.suppress_errors = True
from torch import nn
from torch_geometric.nn import MessagePassing
# from torch_geometric.nn.pool.topk_pool import topk, filter_adj # Abort SAGPool
from torch_scatter import scatter
import loralib as lora
import gpytorch
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from pyro.nn.module import to_pyro_module_
from data.utils import AA_DICT_HUMAN, ESM_TOKENS
from model.module.utils import act_class_mapping
from .attention import PAEMultiHeadAttentionSoftMaxStarGraph, MultiHeadAttentionSoftMaxStarGraph
from esm.modules import RobertaLMHead


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


class ESMScalar(OutputModel):
    # ESMOutputModel is a special output model for ESM-2
    # it has the same initial weights as the ESM-2 model
    # it outputs the log probs of all 20 amino acids
    # currently it only handles single aa change
    def __init__(self, args,
            activation="sigmoid",
            allow_prior_model=True,
            lm_head: RobertaLMHead=None):
        # have a language model head weights
        x_channels = args["x_channels"]
        out_channels = args["output_dim"]
        reduce_op = args["reduce_op"]
        super(ESMScalar, self).__init__(
            allow_prior_model=allow_prior_model, reduce_op=reduce_op
        )
        self.activation = act_class_mapping[activation]()
        # first is lm dense layer
        self.lm_dense = nn.Linear(x_channels, x_channels)
        self.lm_dense.weight.data.copy_(lm_head.dense.weight)
        self.lm_dense.bias.data.copy_(lm_head.dense.bias)
        # next is lm_weight layer
        self.lm_weight = nn.Parameter(torch.zeros(len(ESM_TOKENS), x_channels, out_channels))
        self.lm_bias = nn.Parameter(torch.zeros(len(ESM_TOKENS), out_channels))
        # self.sigmoid_bias = nn.Parameter(torch.ones(out_channels) * -4)
        # self.sigmoid_weight = nn.Parameter(torch.ones(out_channels) * -1)
        # last is the layer norm
        self.lm_layer_norm = nn.LayerNorm(x_channels)
        # copy lm_weight to self.lm_weight
        for i in range(out_channels):
            self.lm_weight[:, :, i].data.copy_(lm_head.weight)
            self.lm_bias[:, i].data.copy_(lm_head.bias)
        self.lm_layer_norm.weight.data.copy_(lm_head.layer_norm.weight)
        
    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def pre_reduce(self, x, v, pos, batch):
        return x

    def reduce(self, x, edge_index, edge_attr, batch):
        # return the center node features
        # center node index is the most common node index in the edge_index
        center_nodes = torch.unique(edge_index[1])
        x = x[center_nodes]
        return x, None

    def post_reduce(self, x, score_mask=None):
        # apply the language model head
        # first dense layer
        x = self.lm_dense(x)
        # next gelu
        x = self.gelu(x)
        # next layer norm
        x = self.lm_layer_norm(x)
        # last linear layer
        x = torch.einsum('bh,ths->bts', x, self.lm_weight) + self.lm_bias.unsqueeze(0)
        # should apply the score mask here, -1 means reference, 1 means alternative, 0 means non of interest
        if score_mask is not None:
            x = (x * score_mask.unsqueeze(-1)).sum(dim=1)
        else:
            x = x.sum(dim=1)
        # return self.activation(x * self.sigmoid_weight + self.sigmoid_bias)
        return self.activation(x)


class ESMFullGraphScalar(ESMScalar):
    # ESMOutputModel is a special output model for ESM-2
    # it has the same initial weights as the ESM-2 model
    # it outputs the log probs of all 20 amino acids
    # currently it only handles single aa change
    def __init__(self, args,
            activation="sigmoid",
            allow_prior_model=True,
            lm_head: RobertaLMHead=None):
        # have a language model head weights
        super(ESMFullGraphScalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
            lm_head=lm_head
        )
     
    def reduce(self, x, x_mask):
        # return the center node features
        # center node index is the most common node index in the edge_index
        x = (x * x_mask).sum(dim=1)
        return x, None


class EquivariantNoGraphScalar(OutputModel):
    def __init__(
            self,
            args,
            activation="sigmoid",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        out_channels = args["output_dim"]
        reduce_op = args["reduce_op"]
        super(EquivariantNoGraphScalar, self).__init__(
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


class EquivariantScalar(OutputModel):
    def __init__(
            self,
            args,
            activation="sigmoid",
            allow_prior_model=True,
            init_fn='uniform',
    ):
        x_channels = args["x_channels"]
        if args["model"] == "pass-forward":
            x_channels = args["x_in_channels"] if args["x_in_channels"] is not None else args["x_channels"]
            if args["add_msa"]:
                x_channels += 199
        out_channels = args["output_dim"]
        reduce_op = args["reduce_op"]
        super(EquivariantScalar, self).__init__(
            allow_prior_model=allow_prior_model, reduce_op=reduce_op
        )
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(x_channels, out_channels),
            act_class(),
        )
        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_fn == 'uniform':
            nn.init.xavier_uniform_(self.output_network[0].weight)
        else:
            nn.init.constant_(self.output_network[0].weight, 0)
        self.output_network[0].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], pos, batch):
        return x

    def post_reduce(self, x):
        return self.output_network(x)


class EquivariantPAEAttnScalar(EquivariantScalar):
    def __init__(
            self,
            args,
            activation="sigmoid",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        super(EquivariantAttnScalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )
        # apply two layers of StarPool
        if args["loss_fn"] == "weighted_combined_loss" or args["loss_fn"] == "combined_loss":
            use_lora = args["use_lora"]
        else:
            use_lora = None
        input_dic = {
            "x_channels": args["x_channels"],
            "x_hidden_channels": args["x_hidden_channels"],
            "vec_channels": args["vec_channels"],
            "vec_hidden_channels": args["vec_hidden_channels"],
            "share_kv": args["share_kv"],
            "edge_attr_dist_channels": args["num_rbf"],
            "edge_attr_channels": args["num_edge_attr"],
            "distance_influence": args["distance_influence"],
            "num_heads": args["num_heads"],
            "activation": act_class_mapping[args["activation"]],
            "cutoff_lower": args["cutoff_lower"],
            "cutoff_upper": args["cutoff_upper"],
            "use_lora": use_lora
        }
        self.AttnPoolLayers = nn.ModuleList([
            PAEMultiHeadAttentionSoftMaxStarGraph(**input_dic),
        ])

    def reduce(self, x, x_center_index, w_ij, f_dist_ij, f_ij, plddt, key_padding_mask):
        # x don't have to reduce to x_center_index but w_ij and f_dist_ij have to
        w_ij = w_ij[x_center_index].unsqueeze(1)
        f_dist_ij = f_dist_ij[x_center_index].unsqueeze(1)
        f_ij = f_ij[x_center_index].unsqueeze(1)
        for layer in self.AttnPoolLayers:
            x, attn = layer(x, x_center_index, w_ij, f_dist_ij, f_ij, key_padding_mask, True)
        return x, attn


class EquivariantAttnScalar(EquivariantScalar):
    def __init__(
            self,
            args,
            activation="sigmoid",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        super(EquivariantAttnScalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )
        # apply two layers of StarPool
        if args["loss_fn"] == "weighted_combined_loss" or args["loss_fn"] == "combined_loss":
            use_lora = args["use_lora"]
        else:
            use_lora = None
        input_dic = {
            "x_channels": args["x_channels"],
            "x_hidden_channels": args["x_hidden_channels"],
            "vec_channels": args["vec_channels"],
            "vec_hidden_channels": args["vec_hidden_channels"],
            "share_kv": args["share_kv"],
            "edge_attr_dist_channels": args["num_rbf"],
            "edge_attr_channels": args["num_edge_attr"],
            "distance_influence": args["distance_influence"],
            "num_heads": args["num_heads"],
            "activation": act_class_mapping[args["activation"]],
            "cutoff_lower": args["cutoff_lower"],
            "cutoff_upper": args["cutoff_upper"],
            "use_lora": use_lora
        }
        self.AttnPoolLayers = nn.ModuleList([
            MultiHeadAttentionSoftMaxStarGraph(**input_dic),
        ])

    def reduce(self, x, x_center_index, w_ij, f_dist_ij, f_ij, plddt, key_padding_mask):
        # x don't have to reduce to x_center_index but w_ij and f_dist_ij have to
        # w_ij = w_ij[x_center_index].unsqueeze(1)
        # f_dist_ij = f_dist_ij[x_center_index].unsqueeze(1)
        f_ij = f_ij[x_center_index].unsqueeze(1)
        for layer in self.AttnPoolLayers:
            x, attn = layer(x, x_center_index, None, None, f_ij, key_padding_mask, True)
        return x, attn


class EquivariantAttnOneSiteScalar(EquivariantAttnScalar):
    def __init__(
            self,
            args,
            activation="sigmoid",
            allow_prior_model=True,
    ):
        self.output_dim = args["output_dim"]
        args["output_dim"] = len(AA_DICT_HUMAN) * self.output_dim
        super(EquivariantAttnOneSiteScalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )

    def post_reduce(self, x):
        res = self.output_network(x).reshape(-1, len(AA_DICT_HUMAN), self.output_dim)
        return res


class EquivariantStarPoolScalar(EquivariantScalar):
    def __init__(
            self,
            args,
            activation="sigmoid",
            allow_prior_model=True,
            init_fn='uniform',
    ):
        x_channels = args["x_channels"]
        super(EquivariantStarPoolScalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
            init_fn=init_fn,
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


class EquivariantStarPoolOneSiteScalar(EquivariantStarPoolScalar):
    def __init__(
            self,
            args,
            activation="sigmoid",
            allow_prior_model=True,
    ):
        self.output_dim = args["output_dim"]
        args["output_dim"] = len(AA_DICT_HUMAN) * self.output_dim
        super(EquivariantStarPoolOneSiteScalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )

    def post_reduce(self, x):
        res = self.output_network(x).reshape(-1, len(AA_DICT_HUMAN), self.output_dim)
        return res


class EquivariantStarPoolMeanVarScalar(EquivariantStarPoolScalar):
    def __init__(
            self,
            args,
            activation="sigmoid",
            allow_prior_model=True,
    ):
        self.output_dim = args["output_dim"]
        # make a copy of args
        args_copy = args.copy()
        args_copy["output_dim"] = 2 * self.output_dim
        super(EquivariantStarPoolMeanVarScalar, self).__init__(
            args=args_copy,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )

    def post_reduce(self, x):
        # output mean and variance
        return self.output_network(x).reshape(-1, 2, self.output_dim)


class EquivariantStarPoolPyroScalar(EquivariantStarPoolScalar):
    def __init__(
            self,
            args,
            activation="sigmoid",
            allow_prior_model=True,
    ):
        super(EquivariantStarPoolPyroScalar, self).__init__(
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )
        to_pyro_module_(self.output_network)

    def post_reduce(self, x):
        return self.output_network(x)


# GP layer basic class
class GaussianProcessLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=64, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        # initialize variational strategy
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super(GaussianProcessLayer, self).__init__(variational_strategy, input_dims, output_dims)
        if mean_type == 'constant':
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = gpytorch.means.LinearMean(input_dims)
        # Cannot use RBFKernel here because it guarantee the diagnal values to be same
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel(num_dimensions=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        y = gpytorch.distributions.MultivariateNormal(mean, covar)
        y_new = y.to_data_independent_dist()
        y_new.lazy_covariance_matrix = gpytorch.lazy.DiagLazyTensor(y.lazy_covariance_matrix.diag())
        return y_new

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)
        # TODO: here the are_samples are set to always true, not sure if it is correct
        # return super().__call__(x, are_samples=bool(len(other_inputs)), **kwargs)
        return super().__call__(x, are_samples=True, **kwargs)
    

class EquivariantStarPoolGPScalar(EquivariantStarPoolScalar, DeepGP):
    def __init__(
            self,
            args,
            activation="sigmoid",
            grid_bounds=(-10., 10.),
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        DeepGP.__init__(self)
        EquivariantStarPoolScalar.__init__(
            self,
            args=args,
            activation=activation,
            allow_prior_model=allow_prior_model,
        )
        # change the output network to GP
        self.output_network = GaussianProcessLayer(input_dims=x_channels, output_dims=args["output_dim"], mean_type='linear')
        self.grid_bounds = grid_bounds
        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    def reset_parameters(self):
        # Do nothing
        return

    def post_reduce(self, x):
        # Scale the input to be between [-10, 10]
        x = self.scale_to_bounds(x)
        # Get the predicted latent function values
        x = self.output_network(x)
        return x


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


class EquivariantRegressionClassificationStarPoolScalar(EquivariantStarPoolScalar):
    def __init__(
            self,
            args,
            activation="pass",
            allow_prior_model=True,
    ):
        x_channels = args["x_channels"]
        super(EquivariantRegressionClassificationStarPoolScalar, self).__init__(
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
        self.output_network_1 = nn.Sequential(
            nn.Linear(x_channels, args["output_dim_1"]),
            act_class_mapping["pass"](),
        )
        self.output_network_2 = nn.Sequential(
            nn.Linear(args["output_dim_1"], args["output_dim_2"]),
            act_class_mapping["sigmoid"](),
        )

    def reduce(self, x, edge_index, edge_attr, batch):
        for layer in self.StarPoolLayers:
            x, edge_index, edge_attr, batch, attn = layer(x, edge_index, edge_attr, batch)
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)
        return out, attn
    
    def post_reduce(self, x):
        x = self.output_network_1(x)
        # concat x and self.output_network_2(x) to get the final output
        output = torch.cat((self.output_network_2(x), x), dim=-1)
        return output


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


def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


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


def build_output_model(output_model_name, args, **kwargs):
    if output_model_name == "EquivariantBinaryClassificationNoGraphScalar":
        return EquivariantNoGraphScalar(args=args, activation="sigmoid")
    elif output_model_name == "EquivariantBinaryClassificationScalar":
        return EquivariantScalar(args=args, activation="sigmoid")
    elif output_model_name == "ESMBinaryClassificationScalar":
        return ESMScalar(args=args, activation="sigmoid", **kwargs)
    elif output_model_name == "ESMFullGraphBinaryClassificationScalar":
        return ESMFullGraphScalar(args=args, activation="sigmoid", **kwargs)
    elif output_model_name == "EquivariantBinaryClassificationStarPoolScalar":
        return EquivariantStarPoolScalar(args=args, activation="sigmoid", init_fn=args["init_fn"])
    elif output_model_name == "EquivariantBinaryClassificationStarPoolMeanVarScalar":
        return EquivariantStarPoolMeanVarScalar(args=args, activation="softplus")
    elif output_model_name == "EquivariantBinaryClassificationAttnScalar":
        return EquivariantAttnScalar(args=args, activation="sigmoid")
    elif output_model_name == "EquivariantBinaryClassificationPAEAttnScalar":
        return EquivariantPAEAttnScalar(args=args, activation="sigmoid")
    elif output_model_name == "EquivariantBinaryClassificationStarPoolOneSiteScalar":
        return EquivariantStarPoolOneSiteScalar(args=args, activation="sigmoid")
    elif output_model_name == "EquivariantBinaryClassificationAttnOneSiteScalar":
        return EquivariantAttnOneSiteScalar(args=args, activation="sigmoid")
    elif output_model_name == "EquivariantBinaryClassificationStarPoolGPScalar":
        return EquivariantStarPoolGPScalar(args=args, activation="sigmoid")
    elif output_model_name == "EquivariantRegressionScalar":
        return EquivariantScalar(args=args, activation="pass")
    elif output_model_name == "ESMRegressionScalar":
        return ESMScalar(args=args, activation="pass", **kwargs)
    elif output_model_name == "ESMFullGraphRegressionScalar":
        return ESMFullGraphScalar(args=args, activation="pass", **kwargs)
    elif output_model_name == "EquivariantRegressionStarPoolScalar":
        return EquivariantStarPoolScalar(args=args, activation="pass")
    elif output_model_name == "EquivariantRegressionStarPoolMeanVarScalar":
        return EquivariantStarPoolMeanVarScalar(args=args, activation="pass")
    elif output_model_name == "EquivariantRegressionAttnScalar":
        return EquivariantAttnScalar(args=args, activation="pass")
    elif output_model_name == "EquivariantRegressionPAEAttnScalar":
        return EquivariantPAEAttnScalar(args=args, activation="pass")
    elif output_model_name == "EquivariantRegressionStarPoolOneSiteScalar":
        return EquivariantStarPoolOneSiteScalar(args=args, activation="pass")
    elif output_model_name == "EquivariantRegressionAttnOneSiteScalar":
        return EquivariantAttnOneSiteScalar(args=args, activation="pass")
    else:
        raise NotImplementedError