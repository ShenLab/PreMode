import re
import warnings
from typing import Optional, List, Tuple, Dict

import torch
from torch import _dynamo
_dynamo.config.suppress_errors = True
from torch import nn, Tensor
from model.module.representation import eqStar2PAETransformerSoftMax, eqStar2WeightedPAETransformerSoftMax, eqStar2FullGraphPAETransformerSoftMax
from model.module import output

__all__ = ["PreMode", "PreMode_Star_CON", "PreMode_DIFF", "PreMode_SSP", "PreMode_Mask_Predict", "PreMode_Single"]


def create_model(args, model_class="PreMode"):
    shared_args = dict(
        num_heads=args["num_heads"],
        x_in_channels=args["x_in_channels"],
        x_channels=args["x_channels"],
        vec_channels=args["vec_channels"],
        vec_in_channels=args["vec_in_channels"],
        x_hidden_channels=args["x_hidden_channels"],
        vec_hidden_channels=args["vec_hidden_channels"],
        num_layers=args["num_layers"],
        num_edge_attr=args["num_edge_attr"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        attn_activation=args["attn_activation"],
        neighbor_embedding=args["neighbor_embedding"],
        cutoff_lower=args["cutoff_lower"],
        cutoff_upper=args["cutoff_upper"],
        x_in_embedding_type=args["x_in_embedding_type"],
        x_use_msa=args['add_msa'] or args['zero_msa'],
        drop_out_rate=args["drop_out"],
    )

    # representation network
    if args["model"] == "equivariant-transformer":
        from model.module.representation import eqTransformer
        
        model_fn = eqTransformer
    elif args["model"] == "equivariant-transformer-star":
        from model.module.representation import eqStarTransformer
        model_fn = eqStarTransformer
    elif args["model"] == "equivariant-transformer-softmax":
        from model.module.representation import eqTransformerSoftMax
        model_fn = eqTransformerSoftMax
    elif args["model"] == "equivariant-transformer-star-softmax":
        from model.module.representation import eqStarTransformerSoftMax
        model_fn = eqStarTransformerSoftMax
    elif args["model"] == "equivariant-transformer-star2-softmax":
        from model.module.representation import eqStar2TransformerSoftMax
        model_fn = eqStar2TransformerSoftMax
        shared_args["use_lora"]=args["use_lora"]
        shared_args["share_kv"]=args["share_kv"]
    elif args["model"] == "equivariant-transformer-PAE-star2-softmax":
        model_fn = eqStar2PAETransformerSoftMax
        shared_args["use_lora"]=args["use_lora"]
        shared_args["share_kv"]=args["share_kv"]
        args["num_rbf"] = 0 # cancel the rbf in PAE model
    elif args["model"] == "equivariant-transformer-weighted-PAE-star2-softmax":
        model_fn = eqStar2WeightedPAETransformerSoftMax
        shared_args["use_lora"]=args["use_lora"]
        shared_args["share_kv"]=args["share_kv"]
        args["num_rbf"] = 0 # cancel the rbf in PAE model
    elif args["model"] == "equivariant-transformer-PAE-star2-fullgraph-softmax":
        model_fn = eqStar2FullGraphPAETransformerSoftMax
        shared_args["use_lora"]=args["use_lora"]
        shared_args["share_kv"]=args["share_kv"]
    elif args["model"] == "transformer-fullgraph-softmax":
        from model.module.representation import FullGraphPAETransformerSoftMax
        model_fn = FullGraphPAETransformerSoftMax
        shared_args["use_lora"]=args["use_lora"]
        shared_args["share_kv"]=args["share_kv"]
    elif args["model"] == "equivariant-triangular-attention-transformer":
        from model.module.representation import eqTriAttnTransformer
        model_fn = eqTriAttnTransformer
        shared_args["pariwise_state_dim"]=args["vec_hidden_channels"]
    elif args["model"] == "equivariant-triangular-star-transformer":
        from model.module.representation import eqTriStarTransformer
        model_fn = eqTriStarTransformer
    elif args["model"] == "equivariant-msa-triangular-star-transformer":
        from model.module.representation import eqMSATriStarTransformer
        model_fn = eqMSATriStarTransformer
        shared_args["ee_channels"]=args["ee_channels"]
        shared_args["triangular_update"]=args["triangular_update"]
    elif args["model"] == "equivariant-msa-triangular-star-drop-transformer":
        from model.module.representation import eqMSATriStarDropTransformer
        model_fn = eqMSATriStarDropTransformer
        shared_args["ee_channels"]=args["ee_channels"]
        shared_args["triangular_update"]=args["triangular_update"]
        shared_args["use_lora"]=args["use_lora"]
    elif args["model"] == "equivariant-msa-triangular-star-gru-transformer":
        from model.module.representation import eqMSATriStarGRUTransformer
        model_fn = eqMSATriStarGRUTransformer
        shared_args["ee_channels"]=args["ee_channels"]
        shared_args["triangular_update"]=args["triangular_update"]
    elif args["model"] == "equivariant-msa-triangular-star-drop-gru-transformer":
        from model.module.representation import eqMSATriStarDropGRUTransformer
        model_fn = eqMSATriStarDropGRUTransformer
        shared_args["ee_channels"]=args["ee_channels"]
        shared_args["triangular_update"]=args["triangular_update"]
        shared_args["use_lora"]=args["use_lora"]
    elif args["model"] == "pass-forward":
        from model.module.representation import PassForward
        model_fn = PassForward
    elif args["model"] == "lora-esm":
        from model.module.representation import LoRAESM2
        model_fn = LoRAESM2
    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')
    representation_model = model_fn(
            **shared_args,
        )
    # create output network
    if "MaskPredict" in args["output_model"]:
        output_model = getattr(output, args["output_model"])(
            args=args,
            lm_weight=representation_model.node_x_proj.weight,
        )
    elif "ESM" in args["output_model"]:
        # get lm_weight from esm2
        import esm
        esm_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        output_model = output.build_output_model(
            args["output_model"], 
            args=args,
            lm_head=esm_model.lm_head,
        )
    else:
        # for non-clinvar tasks, use non_uniform init
        if args["init_fn"] is None:
            if args["data_type"] != "ClinVar":
                args["init_fn"] = "non_uniform"
            else:
                args["init_fn"] = "uniform"
        if hasattr(output, args["output_model"]):
            output_model = getattr(output, args["output_model"])(
                args=args,
            )
        else:
            output_model = output.build_output_model(args["output_model"], args=args)

    # combine representation and output network
    model = globals()[model_class](
        representation_model,
        output_model,
        alt_projector=args["alt_projector"],
    )
    return model


def create_model_and_load(args, model_class="PreMode"):
    model = create_model(args, model_class)
    state_dict = torch.load(args["load_model"], map_location="cpu")
    # The following are for backward compatibility with models created when atomref was
    # the only supported prior.
    output_model_state_dict = {}
    representation_model_state_dict = {}
    for key in state_dict.keys():
        # delete _orig_mod
        if key.startswith("_orig_mod"):
            newkey = key.replace("_orig_mod.", "")
        else:
            newkey = key
        if newkey.startswith("output_model"):
            output_model_state_dict[newkey.replace("output_model.", "")] = state_dict[key]
        elif newkey.startswith("representation_model"):
            if newkey.startswith("representation_model.node_x_proj.weight"):
                if args["partial_load_model"]:
                    embedding_weight = state_dict[key]
                    print('only use the first 26 embedding of MaskPredict')
                    embedding_weight = embedding_weight[:26]  # exclude the embedding of mask
                    representation_model_state_dict["node_x_proj.weight"] = \
                        torch.concat((embedding_weight,
                                      torch.zeros(args["x_in_channels"] - embedding_weight.shape[0],
                                                  embedding_weight.shape[1]))).T
                    representation_model_state_dict["node_x_proj.bias"] = \
                        torch.zeros(args["x_channels"])
                else:
                    representation_model_state_dict[newkey.replace("representation_model.", "")] = state_dict[key]
            else:
                representation_model_state_dict[newkey.replace("representation_model.", "")] = state_dict[key]
    model.representation_model.load_state_dict(representation_model_state_dict, strict=False)
    if args["data_type"] == "ClinVar" \
        or args['loss_fn'] == "combined_loss" \
        or args['loss_fn'] == "weighted_combined_loss" \
        or args['use_output_head']:
        # or args['loss_fn'] == "weighted_loss":
        try:
            # check the output network module dimension
            if output_model_state_dict['output_network.0.weight'].shape[0] != args['output_dim']:
                # if output network is EquivariantAttnOneSiteScalar, we can use it
                if "OneSite" in args['output_model'] and args['use_output_head']:
                    rep_time = args['output_dim'] // output_model_state_dict['output_network.0.weight'].shape[0] 
                    # repeat the weight and bias repeat_interleave
                    output_model_state_dict['output_network.0.weight'] = output_model_state_dict['output_network.0.weight'].repeat_interleave(rep_time, 0)
                    output_model_state_dict['output_network.0.bias'] = output_model_state_dict['output_network.0.bias'].repeat_interleave(rep_time)
                else:
                    print('Warning: output network module dimension is not equal to output_dim, now changing the dimension')
                    output_network_weight = torch.concat(
                        (output_model_state_dict['output_network.0.weight'],
                        torch.zeros(args['output_dim'] - output_model_state_dict['output_network.0.weight'].shape[0], 
                                    output_model_state_dict['output_network.0.weight'].shape[1])
                        )
                    )
                    output_network_bias = torch.concat(
                        (output_model_state_dict['output_network.0.bias'],
                        torch.zeros(args['output_dim'] - output_model_state_dict['output_network.0.bias'].shape[0])
                        )
                    )
                    output_model_state_dict['output_network.0.weight'] = output_network_weight
                    output_model_state_dict['output_network.0.bias'] = output_network_bias
                model.output_model.load_state_dict(output_model_state_dict, strict=False)
            print(f"loaded the output model state dict including the output module")
        except RuntimeError:
            print(f"Warning: Didn't load output model state dict because keys didn't match.")
    else:
        print(f"Warning: Didn't load output model because task is not ClinVar")
    return model


def load_model(filepath, args=None, device="cpu", model_class="PreMode", **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    model = create_model(args, model_class=model_class)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}

    model.load_state_dict(state_dict)
    return model.to(device)


class PreMode(nn.Module):
    def __init__(
            self,
            representation_model,
            output_model,
            alt_projector=None,
    ):
        super(PreMode, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model
        if alt_projector is not None:
            # need to have a linear layer to project the concatenated vector to the same dimension as the original vector
            out_dim = representation_model.x_channels if representation_model.x_in_channels is None else representation_model.x_in_channels
            self.alt_linear = nn.Linear(alt_projector, out_dim, bias=False)
        else:
            self.alt_linear = None

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()

    def forward(
            self,
            x: Tensor,
            x_mask: Tensor,
            x_alt: Tensor,
            pos: Tensor,
            edge_index: Tensor,
            edge_index_star: Tensor = None,
            edge_attr: Tensor = None,
            edge_attr_star: Tensor = None,
            node_vec_attr: Tensor = None,
            batch: Optional[Tensor] = None,
            extra_args: Optional[Dict[str, Tensor]] = None,
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, List]:
        
        # assert x.dim() == 2
        batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device) if batch is None else batch

        # get the graph representation of origin protein first
        # if there is msa in x, split it
        if (self.representation_model.x_in_channels is not None and x.shape[1] > self.representation_model.x_in_channels):
            x_orig, _ = x[:, :self.representation_model.x_in_channels], x[:, self.representation_model.x_in_channels:]
        elif x.shape[1] > self.representation_model.x_channels:
            x_orig, _ = x[:, :self.representation_model.x_channels], x[:, self.representation_model.x_channels:]
        else:
            x_orig = x

        if self.alt_linear is not None:
            x_alt = self.alt_linear(x_alt)
        # update x to alt aa
        x = x * x_mask + x_alt * x_mask

        # run the potentially wrapped representation model
        if extra_args is not None and "y_mask" in extra_args:
            x, v, pos, edge_attr, batch, attn_weight_layers = self.representation_model(
                x=x,
                pos=pos,
                batch=batch,
                edge_index=edge_index,
                edge_index_star=edge_index_star,
                edge_attr=edge_attr,
                edge_attr_star=edge_attr_star,
                node_vec_attr=node_vec_attr,
                mask=extra_args["y_mask"].to(x.device, non_blocking=True),
                return_attn=return_attn, )
        else:
            x, v, pos, edge_attr, batch, attn_weight_layers = self.representation_model(
                x=x,
                pos=pos,
                batch=batch,
                edge_index=edge_index,
                edge_index_star=edge_index_star,
                edge_attr=edge_attr,
                edge_attr_star=edge_attr_star,
                node_vec_attr=node_vec_attr,
                return_attn=return_attn, )

        # apply the output network
        x = self.output_model.pre_reduce(x, v, pos, batch)

        # aggregate residues
        if extra_args is not None and "y_mask" in extra_args:
            x = x * extra_args["y_mask"].unsqueeze(2).to(x.device, non_blocking=True)
        
        # reduce nodes
        x, attn_out = self.output_model.reduce(x - x_orig, edge_index, edge_attr, batch)
        # x = self.output_model.reduce(x, edge_index, edge_attr, batch)
        attn_weight_layers.append(attn_out)
        # apply output model after reduction
        y = self.output_model.post_reduce(x)

        return y, x, attn_weight_layers


class PreMode_Star_CON(nn.Module):
    def __init__(
            self,
            representation_model,
            output_model,
            alt_projector=None,
    ):
        super(PreMode_Star_CON, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model
        self.alt_projector = alt_projector
        if alt_projector is not None:
            # need to have a linear layer to project the concatenated vector to the same dimension as the original vector
            out_dim = representation_model.x_channels if representation_model.x_in_channels is None else representation_model.x_in_channels
            self.alt_linear = nn.Sequential(nn.Linear(alt_projector, out_dim, bias=False), nn.SiLU())
        else:
            self.alt_linear = None
        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()

    def forward(
            self,
            x: Tensor,
            x_mask: Tensor,
            x_alt: Tensor,
            pos: Tensor,
            edge_index: Tensor,
            edge_index_star: Tensor = None,
            edge_attr: Tensor = None,
            edge_attr_star: Tensor = None,
            node_vec_attr: Tensor = None,
            batch: Optional[Tensor] = None,
            extra_args: Optional[Dict[str, Tensor]] = None,
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, List]:
        # assert x.dim() == 2
        batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device) if batch is None else batch
        # get the graph representation of origin protein first
        # if there is msa in x, split it
        if self.representation_model.x_in_channels is not None:
            if x.shape[-1] > self.representation_model.x_in_channels:
                x, msa = x[..., :self.representation_model.x_in_channels], x[..., self.representation_model.x_in_channels:]
                split = True
            else:
                split = False
        elif x.shape[-1] > self.representation_model.x_channels:
            x, msa = x[..., :self.representation_model.x_channels], x[..., self.representation_model.x_channels:]
            split = True
        else:
            split = False
        if len(x.shape) == 3 or len(x_mask.shape) == 1:
            x_mask = x_mask.unsqueeze(-1)
        else:
            x_mask = x_mask[:, 0].unsqueeze(1)
        if self.alt_linear is not None:
            x_alt = x_alt[..., :self.alt_projector]
            x_alt = self.alt_linear(x_alt)
        else:
            x_alt = x_alt[..., :x.shape[-1]]
        # update x to alt aa
        x = x * x_mask + x_alt * (~x_mask)
        # concat with msa
        if split:
            x = torch.cat((x, msa), dim=-1)

        # run the potentially wrapped representation model
        # wrap input features
        input = {"x": x, 
                 "pos": pos, 
                 "batch": batch, 
                 "edge_index": edge_index, 
                 "edge_index_star": edge_index_star, 
                 "edge_attr": edge_attr,
                 "edge_attr_star": edge_attr_star, 
                 "node_vec_attr": node_vec_attr,
                 "return_attn": return_attn}
        
        if extra_args is not None and "y_mask" in extra_args:
            input["mask"] = extra_args["y_mask"].to(x.device, non_blocking=True)
        if extra_args is not None and "x_padding_mask" in extra_args:
            input["x_padding_mask"] = extra_args["x_padding_mask"].to(x.device, non_blocking=True)    
        if isinstance(self.representation_model, eqStar2PAETransformerSoftMax) or \
            isinstance(self.representation_model, eqStar2WeightedPAETransformerSoftMax) or \
            isinstance(self.representation_model, eqStar2FullGraphPAETransformerSoftMax):
            # means we are using PAE model
            input["plddt"] = extra_args["plddt"].to(x.device, non_blocking=True) \
                if "plddt" in extra_args else None
            input["edge_confidence"] = extra_args["edge_confidence"].to(x.device, non_blocking=True) \
                if "edge_confidence" in extra_args else None
            input["edge_confidence_star"] = extra_args["edge_confidence_star"].to(x.device, non_blocking=True) \
                if "edge_confidence_star" in extra_args else None
        x, v, pos, edge_attr, batch, attn_weight_layers = self.representation_model(**input)
        # apply the output network
        x = self.output_model.pre_reduce(x, v, pos, batch)

        # aggregate residues
        if extra_args is not None and "y_mask" in extra_args:
            x = x * extra_args["y_mask"].unsqueeze(2).to(x.device, non_blocking=True)
        
        # if edge_attr is same shape as edge_index_star, it means that edge_attr is actually updated to edge_attr_star
        if len(x.shape) < 3:
            # # for nodes not connected by edges, set their x to 0
            # reduce nodes by star graph
            end_node_count = edge_index_star[1].unique(return_counts=True)
            end_nodes = end_node_count[0][end_node_count[1] > 1]
            if edge_attr is not None and edge_attr.shape[0] == edge_index_star.shape[1]:
                x, attn_out = self.output_model.reduce(x, 
                                        edge_index_star[:, torch.isin(edge_index_star[1], end_nodes)],
                                        edge_attr[torch.isin(edge_index_star[1], end_nodes), :],
                                        batch)
            else:
                # if edge_attr is not updated, use edge_attr_star
                x, attn_out = self.output_model.reduce(x,
                                            edge_index_star[:, torch.isin(edge_index_star[1], end_nodes)],
                                            edge_attr_star[torch.isin(edge_index_star[1], end_nodes), :],
                                            batch)
        else:
            x, attn_out = self.output_model.reduce(
                x, (~x_mask).squeeze(2),
                edge_attr[0], edge_attr[1], edge_attr[2], edge_attr[3], 
                input["x_padding_mask"])
            if 'score_mask' not in extra_args:
                x = x.unsqueeze(1)
        # x = self.output_model.reduce(x, edge_index, edge_attr, batch)
        attn_weight_layers.append(attn_out)
        # apply output model after reduction
        # if esm_mask is in extra_args, it means we are using esm model
        if "esm_mask" in extra_args:
            y = self.output_model.post_reduce(x, extra_args["esm_mask"].to(x.device, non_blocking=True))
        else:
            y = self.output_model.post_reduce(x)

        return y, x, attn_weight_layers


class PreMode_SSP(PreMode):
    def __init__(
            self,
            representation_model,
            output_model,
            vec_in_channels=4,
    ):
        super(PreMode_SSP, self).__init__(representation_model=representation_model,
                                          output_model=output_model,)
        self.vec_reconstruct = nn.Linear(representation_model.vec_channels, vec_in_channels, bias=False)

    def forward(
            self,
            x: Tensor,
            x_mask: Tensor,
            x_alt: Tensor,
            pos: Tensor,
            edge_index: Tensor,
            edge_index_star: Tensor = None,
            edge_attr: Tensor = None,
            edge_attr_star: Tensor = None,
            edge_vec: Tensor = None,
            edge_vec_star: Tensor = None,
            node_vec_attr: Tensor = None,
            batch: Optional[Tensor] = None,
            extra_args: Optional[Dict[str, Tensor]] = None,
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List]:

        assert x.dim() == 2 and x.dtype == torch.float
        batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device) if batch is None else batch

        # get the graph representation of origin protein first
        x_orig = x

        # update x to alt aa
        x = x * x_mask + x_alt

        # run the potentially wrapped representation model
        x, v, pos, edge_attr, batch, attn_weight_layers = self.representation_model(
            x=x,
            pos=pos,
            batch=batch,
            edge_index=edge_index,
            edge_index_star=edge_index_star,
            edge_attr=edge_attr,
            edge_attr_star=edge_attr_star,
            node_vec_attr=node_vec_attr,
            return_attn=return_attn, )

        vec = self.vec_reconstruct(v)
        # apply the output network
        x_graph: Tensor = x
        x = self.output_model.pre_reduce(x, v, pos, batch)

        # aggregate residues
        x, _ = self.output_model.reduce(x - x_orig, edge_index, edge_attr, batch)

        # apply output model after reduction
        y = self.output_model.post_reduce(x)

        return x_graph, vec, y, x, attn_weight_layers


class PreMode_DIFF(PreMode):
    def __init__(
            self,
            representation_model,
            output_model,
            alt_projector=None,
    ):
        super(PreMode_DIFF, self).__init__(representation_model=representation_model,
                                           output_model=output_model,)

    def forward(
            self,
            x: Tensor,
            x_mask: Tensor,
            x_alt: Tensor,
            pos: Tensor,
            edge_index: Tensor,
            edge_index_star: Tensor = None,
            edge_attr: Tensor = None,
            edge_attr_star: Tensor = None,
            edge_vec: Tensor = None,
            edge_vec_star: Tensor = None,
            node_vec_attr: Tensor = None,
            batch: Optional[Tensor] = None,
            extra_args: Optional[Dict[str, Tensor]] = None,
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, List]:

        # assert x.dim() == 2 and x.dtype == torch.float
        batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device) if batch is None else batch

        # get the graph representation of origin protein first
        x_orig, v, pos, _, batch, attn_weight_layers_ref = self.representation_model(
            x=x,
            pos=pos,
            batch=batch,
            edge_index=edge_index,
            edge_index_star=edge_index_star,
            edge_attr=edge_attr,
            edge_attr_star=edge_attr_star,
            node_vec_attr=node_vec_attr,
            return_attn=return_attn, )
        x_orig = self.output_model.pre_reduce(x_orig, v, pos, batch)

        # update x to alt aa
        x = x * x_mask + x_alt

        # run the potentially wrapped representation model
        x, v, pos, edge_attr, batch, attn_weight_layers_alt = self.representation_model(
            x=x,
            pos=pos,
            batch=batch,
            edge_index=edge_index,
            edge_index_star=edge_index_star,
            edge_attr=edge_attr,
            edge_attr_star=edge_attr_star,
            node_vec_attr=node_vec_attr,
            return_attn=return_attn, )

        # apply the output network
        x = self.output_model.pre_reduce(x, v, pos, batch)

        # aggregate residues
        x, _ = self.output_model.reduce(x - x_orig, edge_index, edge_attr, batch)

        # apply output model after reduction
        y = self.output_model.post_reduce(x)

        return y, x, [attn_weight_layers_ref, attn_weight_layers_alt]


class PreMode_Mask_Predict(PreMode):
    def __init__(
            self,
            representation_model,
            output_model,
            alt_projector=None,
    ):
        super(PreMode_Mask_Predict, self).__init__(representation_model=representation_model,
                                                   output_model=output_model,)

    def forward(
            self,
            x: Tensor,
            x_mask: Tensor,
            x_alt: Tensor,
            pos: Tensor,
            edge_index: Tensor,
            edge_index_star: Tensor = None,
            edge_attr: Tensor = None,
            edge_attr_star: Tensor = None,
            edge_vec: Tensor = None,
            edge_vec_star: Tensor = None,
            node_vec_attr: Tensor = None,
            batch: Optional[Tensor] = None,
            extra_args: Optional[Dict[str, Tensor]] = None,
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, List]:
        
        # assert x.dim() == 2 and x.dtype == torch.float
        batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device) if batch is None else batch

        # update x to alt aa
        x = x * x_mask + x_alt

        # get the graph representation of origin protein first
        if "y_mask" in extra_args:
            # means that it is non-graph model
            x_embed, v, pos, _, batch, attn_weight_layers_ref = self.representation_model(
                x=x,
                pos=pos,
                mask=extra_args["y_mask"].to(x.device, non_blocking=True),
                return_attn=return_attn, )
        else:
            x_embed, v, pos, _, batch, attn_weight_layers_ref = self.representation_model(
                x=x,
                pos=pos,
                batch=batch,
                edge_index=edge_index,
                edge_index_star=edge_index_star,
                edge_attr=edge_attr,
                edge_attr_star=edge_attr_star,
                node_vec_attr=node_vec_attr,
                return_attn=return_attn, )
        # pre reduce is to reduce to one hot alphabet
        y = self.output_model.pre_reduce(x_embed, v, pos, batch)

        return y, y, attn_weight_layers_ref


class PreMode_Single(PreMode):
    def __init__(
            self,
            representation_model,
            output_model,
            alt_projector=None,
    ):
        super(PreMode_Single, self).__init__(representation_model=representation_model,
                                             output_model=output_model,)

    def forward(
            self,
            x: Tensor,
            x_mask: Tensor,
            x_alt: Tensor,
            pos: Tensor,
            edge_index: Tensor,
            edge_index_star: Tensor = None,
            edge_attr: Tensor = None,
            edge_attr_star: Tensor = None,
            edge_vec: Tensor = None,
            edge_vec_star: Tensor = None,
            node_vec_attr: Tensor = None,
            batch: Optional[Tensor] = None,
            extra_args: Optional[Dict[str, Tensor]] = None,
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, List]:

        assert x.dim() == 2
        batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device) if batch is None else batch

        # get the graph representation of origin protein first
        # if there is msa in x, split it
        # if there is msa in x, split it
        if (self.representation_model.x_in_channels is not None and x.shape[1] > self.representation_model.x_in_channels):
            x, msa = x[:, :self.representation_model.x_in_channels], x[:, self.representation_model.x_in_channels:]
            split = True
        elif x.shape[1] > self.representation_model.x_channels:
            x, msa = x[:, :self.representation_model.x_channels], x[:, self.representation_model.x_channels:]
            split = True
        else:
            split = False
        x_mask = x_mask[:, 0]
        if self.alt_linear is not None:
            x_alt = x_alt[:, :self.alt_projector]
            x_alt = self.alt_linear(x_alt)
        else:
            x_alt = x_alt[:, :x.shape[1]]
        # update x to alt aa
        x = x * x_mask.unsqueeze(1) + x_alt * (~x_mask).unsqueeze(1)
        # concat with msa
        if split:
            x = torch.cat((x, msa), dim=1)

        # run the potentially wrapped representation model
        x, v, pos, edge_attr, batch, attn_weight_layers = self.representation_model(
            x=x,
            pos=pos,
            batch=batch,
            edge_index=edge_index,
            edge_index_star=edge_index_star,
            edge_attr=edge_attr,
            edge_attr_star=edge_attr_star,
            node_vec_attr=node_vec_attr,
            return_attn=return_attn, )

        # apply the output network
        x = self.output_model.pre_reduce(x, v, pos, batch)

        # aggregate residues
        x, _ = self.output_model.reduce(x, edge_index, edge_attr, batch)

        # apply output model after reduction
        y = self.output_model.post_reduce(x)

        return y, x, attn_weight_layers
