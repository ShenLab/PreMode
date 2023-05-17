import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn.functional import mse_loss
from model.trainer import gRESCVE_trainer


class gRESCVE_trainer_SSP(gRESCVE_trainer):
    """
    A wrapper for dataloader, summary writer, optimizer, scheduler
    """

    def __init__(self, hparams, model, stage: str = "train", dataset=None, device_id=None):
        super(gRESCVE_trainer_SSP, self).__init__(hparams, model, stage, dataset, device_id)
        self.loss_fn_ssp = mse_loss

    def step(self, batch, stage="train"):
        with torch.set_grad_enabled(stage == "train"):
            extra_args = batch.to_dict()
            # extra_args actually won't be used in the model
            for a in ('y', 'x', 'x_mask', 'x_ref', 'x_alt',
                      'pos', 'batch', 'edge_index', 'edge_attr',
                      'node_vec_attr', 'node_vec_attr_mask'):
                if a in extra_args:
                    del extra_args[a]
            node_vec_attr_forward = batch.node_vec_attr.to(self.device) * batch.node_vec_attr_mask.to(self.device)
            # mutate forward
            x_hidden, vec_hidden, y, x_embed, attn_weight_layers_forward = self.forward(
                x=batch.x.to(self.device),
                x_mask=batch.x_mask.to(self.device),
                x_alt=batch.x_alt.to(self.device),
                pos=batch.pos.to(self.device),
                batch=batch.batch.to(self.device) if "batch" in batch else None,
                edge_index=batch.edge_index.to(self.device),
                edge_index_star=batch.edge_index_star.to(self.device) if "edge_index_star" in batch else None,
                edge_attr=batch.edge_attr.to(self.device),
                edge_attr_star=batch.edge_attr_star.to(self.device) if "edge_attr_star" in batch else None,
                node_vec_attr=node_vec_attr_forward,
                extra_args=extra_args,
                return_attn=stage == "interpret",
            )
            if stage == "test":
                # if test stage, don't have to mutate back
                self.predictions['y'].append(y.detach().cpu().numpy())
            else:
                node_vec_attr_backward = vec_hidden * batch.node_vec_attr_mask.to(self.device)
                # if train or val stage, mutate back
                x_reconstruct, vec_reconstruct, _, _, _, attn_weight_layers_backward = self.forward(
                    x=x_hidden,
                    x_mask=batch.x_mask.to(self.device),
                    x_alt=batch.x_ref.to(self.device),
                    pos=batch.pos.to(self.device),
                    batch=batch.batch.to(self.device) if "batch" in batch else None,
                    edge_index=batch.edge_index.to(self.device),
                    edge_index_star=batch.edge_index_star.to(self.device) if "edge_index_star" in batch else None,
                    edge_attr=batch.edge_attr.to(self.device),
                    edge_attr_star=batch.edge_attr_star.to(self.device) if "edge_attr_star" in batch else None,
                    node_vec_attr=node_vec_attr_backward,
                    extra_args=extra_args,
                    return_attn=stage == "interpret",
                )
        loss_y, loss_x, loss_vec = 0, 0, 0
        # y loss
        if "y" in batch:
            if batch.y.ndim == 1:
                batch.y = batch.y.unsqueeze(1)
            if batch.y.shape[1] == 1:
                # y loss, remove y = -1
                print(f"Rank {self.device_id} batch {self.global_step} "
                      f"self-supervised samples: {sum(batch.y == -2).item()}")
                loss_y = (self.loss_fn(
                    y, batch.y.to(self.device), reduction='none'
                ) * (batch.y != -2).to(self.device)).mean()
            else:
                # y loss
                loss_y = self.loss_fn(y, batch.y.to(self.device))

            if self.hparams.y_weight > 0 and not stage == "interpret":
                self.losses[stage + "_y"].append(loss_y.detach().cpu())
        # x loss
        if not stage == "test" and not stage == "interpret":
            loss_x = self.loss_fn_ssp(x_reconstruct, batch.x.to(self.device))
            self.losses[stage + "_x"].append(loss_x.detach().cpu())
        # total loss
        loss = loss_y * self.hparams.y_weight + loss_x
        if not stage == "interpret":
            self.losses[stage].append(loss.detach().cpu())
        if stage == "interpret":
            return loss, y, x_embed, [attn_weight_layers_forward, attn_weight_layers_backward]
        else:
            return loss

    def validation_epoch_end(self, reset_train_loss=False):
        self.val_iterator = None
        # construct dict of logged metrics
        result_dict = {
            "epoch": int(self.current_epoch),
            "lr": self.optimizer.param_groups[0]["lr"],
            "train_loss": torch.stack(self.losses["train"]).mean().item(),
            "val_loss": torch.stack(self.losses["val"]).mean().item(),
        }

        # add test loss if available
        if len(self.losses["test"]) > 0:
            result_dict["test_loss"] = torch.stack(self.losses["test"]).mean().item()

        # if predictions are present, also log them separately
        if len(self.losses["train_y"]) > 0 and len(self.losses["train_x"]) > 0:
            result_dict["train_loss_y"] = torch.stack(
                self.losses["train_y"]
            ).mean().item()
            result_dict["train_loss_x"] = torch.stack(
                self.losses["train_x"]
            ).mean().item()
            result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean().item()
            result_dict["val_loss_x"] = torch.stack(self.losses["val_x"]).mean().item()

            if len(self.losses["test"]) > 0:
                result_dict["test_loss_y"] = torch.stack(
                    self.losses["test_y"]
                ).mean().item()
        if reset_train_loss:
            self._reset_losses_dict()
        else:
            self._reset_val_losses_dict()
        return result_dict

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
            "train_y": [],
            "train_x": [],
            "val_y": [],
            "val_x": [],
            "test_y": [],
        }

    def _reset_val_losses_dict(self):
        self.losses["val"] = []
        self.losses["val_y"] = []
        self.losses["val_x"] = []
