import re
import warnings
from typing import Optional, List, Tuple, Dict

import torch
from torch import nn, Tensor

from model.module import output

__all__ = ["PreMode", "PreMode_DIFF", "PreMode_SSP", "PreMode_Mask_Predict", "PreMode_Single"]


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
    # TODO: remove this
    elif args["model"] == "equivariant-transformer-star-softmax":
        from model.module.representation import eqStarTransformerSoftMax
        model_fn = eqStarTransformerSoftMax
    elif args["model"] == "equivariant-transformer-star2-softmax":
        from model.module.representation import eqStar2TransformerSoftMax
        model_fn = eqStar2TransformerSoftMax
    elif args["model"] == "equivariant-triangular-attention-transformer":
        from model.module.representation import eqTriAttnTransformer
        model_fn = eqTriAttnTransformer
        shared_args["pariwise_state_dim"]=args["vec_hidden_channels"]
    elif args["model"] == "pass-forward":
        from model.module.representation import PassForward
        model_fn = PassForward
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
    else:
        output_model = getattr(output, args["output_model"])(
            args=args,
        )

    # combine representation and output network
    model = globals()[model_class](
        representation_model,
        output_model,
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
        if key.startswith("output_model"):
            output_model_state_dict[key.replace("output_model.", "")] = state_dict[key]
        elif key.startswith("representation_model"):
            if key.startswith("representation_model.node_x_proj.weight"):
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
                    representation_model_state_dict[key.replace("representation_model.", "")] = state_dict[key]
            else:
                representation_model_state_dict[key.replace("representation_model.", "")] = state_dict[key]
    model.representation_model.load_state_dict(representation_model_state_dict)
    if args["data_type"] == "ClinVar":
        try:
            model.output_model.load_state_dict(output_model_state_dict)
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
    ):
        super(PreMode, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model

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
        x_orig = x

        # update x to alt aa
        x = x * x_mask + x_alt

        # run the potentially wrapped representation model
        if "y_mask" in extra_args:
            x, v, pos, edge_attr, batch, attn_weight_layers = self.representation_model(
                x=x,
                pos=pos,
                batch=batch,
                edge_index=edge_index,
                edge_index_star=edge_index_star,
                edge_attr=edge_attr,
                edge_attr_star=edge_attr_star,
                node_vec_attr=node_vec_attr,
                mask=extra_args["y_mask"].to(x.device),
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
                node_vec_attr=node_vec_attr, )

        # apply the output network
        x = self.output_model.pre_reduce(x, v, pos, batch)

        # aggregate residues
        if "y_mask" in extra_args:
            x = x * extra_args["y_mask"].unsqueeze(2).to(x.device)
        
        # # for nodes not connected by edges, set their x to 0
        # x_in_edge_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        # x_in_edge_mask[edge_index[0]] = True
        # x_in_edge_mask[edge_index[1]] = True
        # x = x * x_in_edge_mask.unsqueeze(1)
        # x_orig = x_orig * x_in_edge_mask.unsqueeze(1)
        
        # reduce nodes
        x = self.output_model.reduce(x - x_orig, edge_index, edge_attr, batch)
        # x = self.output_model.reduce(x, edge_index, edge_attr, batch)

        # apply output model after reduction
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
        x = self.output_model.reduce(x - x_orig, edge_index, edge_attr, batch)

        # apply output model after reduction
        y = self.output_model.post_reduce(x)

        return x_graph, vec, y, x, attn_weight_layers


class PreMode_DIFF(PreMode):
    def __init__(
            self,
            representation_model,
            output_model,
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
        x = self.output_model.reduce(x - x_orig, edge_index, edge_attr, batch)

        # apply output model after reduction
        y = self.output_model.post_reduce(x)

        return y, x, [attn_weight_layers_ref, attn_weight_layers_alt]


class PreMode_CON(PreMode):
    def __init__(
            self,
            representation_model,
            output_model,
    ):
        super(PreMode_CON, self).__init__(representation_model=representation_model,
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
            node_vec_attr: Tensor = None,
            batch: Optional[Tensor] = None,
            extra_args: Optional[Dict[str, Tensor]] = None,
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, List]:
        
        # assert x.dim() == 2
        batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device) if batch is None else batch

        # get the graph representation of origin protein first
        x_orig = x

        # update x to alt aa
        x = torch.concat((x, x_alt), dim=1)

        # run the potentially wrapped representation model
        if "y_mask" in extra_args:
            x, v, pos, edge_attr, batch, attn_weight_layers = self.representation_model(
                x=x,
                pos=pos,
                batch=batch,
                edge_index=edge_index,
                edge_index_star=edge_index_star,
                edge_attr=edge_attr,
                edge_attr_star=edge_attr_star,
                node_vec_attr=node_vec_attr,
                mask=extra_args["y_mask"].to(x.device),
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
                node_vec_attr=node_vec_attr, )

        # apply the output network
        x = self.output_model.pre_reduce(x, v, pos, batch)

        # aggregate residues
        if "y_mask" in extra_args:
            x = x * extra_args["y_mask"].unsqueeze(2).to(x.device)
        
        # # for nodes not connected by edges, set their x to 0
        # x_in_edge_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        # x_in_edge_mask[edge_index[0]] = True
        # x_in_edge_mask[edge_index[1]] = True
        # x = x * x_in_edge_mask.unsqueeze(1)
        # x_orig = x_orig * x_in_edge_mask.unsqueeze(1)
        
        # reduce nodes
        # x = self.output_model.reduce(x - x_orig, edge_index, edge_attr, batch)
        x = self.output_model.reduce(x, edge_index, edge_attr, batch)

        # apply output model after reduction
        y = self.output_model.post_reduce(x)

        return y, x, attn_weight_layers


class PreMode_Mask_Predict(PreMode):
    def __init__(
            self,
            representation_model,
            output_model,
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
                mask=extra_args["y_mask"].to(x.device),
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
            node_vec_attr: Tensor = None,
            batch: Optional[Tensor] = None,
            extra_args: Optional[Dict[str, Tensor]] = None,
            return_attn: bool = False,
    ) -> Tuple[Tensor, Tensor, List]:

        assert x.dim() == 2
        batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device) if batch is None else batch

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

        # apply the output network
        x = self.output_model.pre_reduce(x, v, pos, batch)

        # aggregate residues
        x = self.output_model.reduce(x, edge_index, edge_attr, batch)

        # apply output model after reduction
        y = self.output_model.post_reduce(x)

        return y, x, attn_weight_layers
