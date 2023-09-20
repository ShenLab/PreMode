import json
import os
import time
from os.path import join

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import loralib as lora

import data
import utils.configs
from model.module.utils import loss_fn_mapping

torch.multiprocessing.set_sharing_strategy('file_system')


class PreMode_trainer(object):
    """
    A wrapper for dataloader, summary writer, optimizer, scheduler
    """

    def __init__(self, hparams, model, stage: str = "train", dataset=None, device_id=None):
        super(PreMode_trainer, self).__init__()

        self.hparams = hparams

        # Don't load model, just store the model from input.
        self.model = model
        # save the ddp_rank to write the log
        self.device_id = device_id
        self.device = f"cuda:{device_id}" if device_id is not None else "cpu"

        # initialize dataloaders
        self.dataset = dataset
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.split_fn = self.hparams.data_split_fn
        self.setup_dataloaders(stage, self.split_fn)
        if self.train_dataloader is not None:
            self.batchs_per_epoch = len(self.train_dataloader)
        self.reset_train_dataloader_each_epoch = self.hparams.reset_train_dataloader_each_epoch
        self.reset_train_dataloader_each_epoch_seed = self.hparams.reset_train_dataloader_each_epoch_seed
        self.train_iterator = None
        self.val_iterator = None
        self.test_iterator = None

        # initialize loss function
        if self.hparams.loss_fn == "weighted_combined_loss" or "weighted_loss" in self.hparams.loss_fn:
            label_counts = self.dataset.get_label_counts() 
            if len(label_counts) == 4:
                # [lof, beni, gain, patho]
                # note that we changed to 2-dim scheme now.
                total_count_1 = label_counts.sum()
                task_weight = total_count_1 / (label_counts[0] + label_counts[2]) # patho / glof 
                total_count_2 = total_count_1 - label_counts[3] - label_counts[0] # gof + lof
                if label_counts[1] != 0:
                    weight_1 = torch.tensor([total_count_1 / label_counts[1] / 2, 
                                            total_count_1 / (total_count_1 - label_counts[1]) / 2], 
                                            dtype=torch.float32, device=self.device)
                    weight_2 = torch.tensor([total_count_2 / label_counts[0] / 2, 
                                            total_count_2 / label_counts[2] / 2], 
                                            dtype=torch.float32, device=self.device)
                else:
                    weight_1 = torch.ones(2, dtype=torch.float32, device=self.device)
                    weight_2 = torch.tensor([total_count_2 / label_counts[0] / 2, 
                                             total_count_2 / label_counts[2] / 2], 
                                             dtype=torch.float32, device=self.device)
            elif len(label_counts) == 2:
                # [beni, patho]
                task_weight = 0
                total_count_1 = label_counts.sum()
                if label_counts[0] != 0:
                    weight_1 = torch.tensor([total_count_1 / label_counts[0] / 2, 
                                             total_count_1 / label_counts[1] / 2], 
                                             dtype=torch.float32, device=self.device)
                    weight_2 = torch.zeros(2, dtype=torch.float32, device=self.device)
                else:
                    weight_1 = torch.ones(2, dtype=torch.float32, device=self.device)
                    weight_2 = torch.zeros(2, dtype=torch.float32, device=self.device)
            else:
                raise ValueError("The number of labels should be 2 or 4.")
            weight=torch.cat([weight_1, weight_2])
            print(f"set up weighted loss function with weight: {weight}")
            self.loss_fn = loss_fn_mapping[self.hparams.loss_fn](weight=weight, task_weight=task_weight)
            # Archived, as we are not using the 3-dim scheme any more.
            # print("Initialize the output module to fit the weighted loss function.")
            # with torch.no_grad():
            #     if isinstance(self.model, DDP):
            #         self.model.module.output_model.output_network[0].weight[1].copy_(self.model.module.output_model.output_network[0].weight[2])
            #     else:
            #         self.model.output_model.output_network[0].weight[1].copy_(self.model.output_model.output_network[0].weight[2])
        else:
            self.loss_fn = loss_fn_mapping[self.hparams.loss_fn]
            
        # freeze representation module if hparams.freeze_representation is True
        if self.hparams.freeze_representation:
            for param in self.model.representation_model.parameters():
                param.requires_grad = False
            # deactivate dropout
            self.model.representation_model.eval()
        if self.hparams.freeze_representation_but_attention:
            for param in self.model.representation_model.parameters():
                param.requires_grad = False
            # deactivate dropout
            self.model.representation_model.eval()
            for param in self.model.representation_model.attention_layers.parameters():
                param.requires_grad = True
        if self.hparams.freeze_representation_but_gru:
            for param in self.model.representation_model.parameters():
                param.requires_grad = False
            # deactivate dropout
            self.model.representation_model.eval()
            for layer in self.model.representation_model.attention_layers:
                assert layer.gru is not None
                for param in layer.gru.parameters():
                    param.requires_grad = True
        if self.hparams.use_lora is not None:
            self.model.eval()
            lora.mark_only_lora_as_trainable(model)
            if self.hparams.loss_fn == "weighted_combined_loss" or self.hparams.loss_fn == "combined_loss":
                self.model.output_model.output_network.requires_grad_(True)
            elif self.hparams.loss_fn == "weighted_loss":
                self.model.output_model.requires_grad_(True)
            self.use_lora = True
        else:
            self.use_lora = False


        # initialize loss collection
        self.losses = None
        self._reset_losses_dict()

        # initialize the prediction collection
        self.predictions = None
        self._reset_predictions_dict()

        # initialize global step and epoch
        self.global_step = 0
        self.current_epoch = 0

        # initialize optimizers
        self.updated = True
        self.optimizer = None
        self.scheduler = None
        self.lr_scheduler = None
        self.configure_optimizers()

        # initialize contrastive loss
        self.contrastive_loss = loss_fn_mapping[self.hparams.contrastive_loss_fn] if self.hparams.contrastive_loss_fn is not None else None

        # initialize summary writer
        self.writer = SummaryWriter(log_dir=f'{self.hparams.log_dir}/log/')

    def setup_dataloaders(self, stage: str = 'train', split_fn="_by_uniprot_id"):
        if self.dataset is None:
            self.dataset = getattr(data, self.hparams["dataset"])(
                data_file=self.hparams.data_file_train,
                data_type=self.hparams.data_type,
                radius=self.hparams.radius,
                max_neighbors=self.hparams.max_num_neighbors,
                loop=self.hparams.loop,
            )
        if stage == 'train':
            # make train/val split
            if self.hparams.val_size > 0:
                idx_train, idx_val = getattr(utils.configs, "make_splits_train_val" + split_fn)(
                    self.dataset,
                    self.hparams.train_size,
                    self.hparams.val_size,
                    self.hparams.seed,
                    self.hparams.batch_size,
                    join(self.hparams.log_dir, f"splits.{self.device_id}.npz"),
                )
                print(f"train {len(idx_train)}, val {len(idx_val)}")

                self.train_dataset = Subset(self.dataset, idx_train)
                self.idx_train = idx_train
                self.val_dataset = Subset(self.dataset, idx_val)
                self.idx_val = idx_val
            else:
                self.train_dataset = self.dataset
                self.val_dataset = None
                self.idx_train = np.arange(len(self.dataset))
                self.idx_val = None

            self.train_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=0,
                pin_memory=True,
                pin_memory_device='cpu',
                shuffle=False,
            )
            if self.val_dataset is not None:
                self.val_dataloader = DataLoader(
                    dataset=self.val_dataset,
                    batch_size=self.hparams.batch_size,
                    num_workers=0,
                    pin_memory=True,
                    pin_memory_device='cpu',
                    shuffle=False,
                )
            else:
                self.val_dataloader = None
        elif stage == 'test':
            # only prepare test dataloader
            self.test_dataset = self.dataset
            self.test_dataloader = DataLoader(
                dataset=self.test_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=0,
                pin_memory=True,
                pin_memory_device='cpu',
                shuffle=False,
            )
        elif stage == 'all':
            # make train/test/val split
            idx_train, idx_val, idx_test = getattr(utils.configs, "make_splits_train_val_test" + split_fn)(
                self.dataset,
                self.hparams.train_size,
                self.hparams.val_size,
                self.hparams.test_size,
                0,
                self.hparams.batch_size * self.hparams.num_workers,
                join(self.hparams.log_dir, "splits.npz"),
                self.hparams.splits,
            )
            print(f"train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}")

            self.train_dataset = Subset(self.dataset, idx_train)
            self.idx_train = idx_train
            self.val_dataset = Subset(self.dataset, idx_val)
            self.idx_val = idx_val
            self.test_dataset = Subset(self.dataset, idx_test)
            self.idx_test = idx_test

            self.train_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=0,
                pin_memory=True,
                pin_memory_device='cpu',
                shuffle=False,
            )
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=0,
                pin_memory=True,
                pin_memory_device='cpu',
                shuffle=False,
            )
            self.test_dataloader = DataLoader(
                dataset=self.test_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=0,
                pin_memory=True,
                pin_memory_device='cpu',
                shuffle=False,
            )
        else:
            raise ValueError(f"stage {stage} not supported")

    def configure_optimizers(self):
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=self.hparams.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            "min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=float(self.hparams.lr_min),
        )
        self.lr_scheduler = {
            "scheduler": self.scheduler,
            "monitor": getattr(self.hparams, "lr_metric", "val_loss"),
            "interval": "epoch",
            "frequency": 1,
        }

    def forward(self, x, x_mask, x_alt, pos, batch=None,
                edge_index=None, edge_attr=None,
                edge_index_star=None, edge_attr_star=None,
                node_vec_attr=None,
                extra_args=None,
                return_attn=False):
        return self.model(x=x,
                          x_mask=x_mask,
                          x_alt=x_alt,
                          pos=pos,
                          batch=batch,
                          edge_index=edge_index,
                          edge_attr=edge_attr,
                          edge_index_star=edge_index_star,
                          edge_attr_star=edge_attr_star,
                          node_vec_attr=node_vec_attr,
                          extra_args=extra_args,
                          return_attn=return_attn)

    def training_step(self):
        if self.train_iterator is None:
            raise ValueError("train_iterator is None, please call training_epoch_begin() first")
        batch = next(self.train_iterator)
        loss = self.step(batch, "train") / self.hparams.num_steps_update
        loss.backward()
        self.write_loss_log("train", loss)
        # parameters_without_grad = []
        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         parameters_without_grad.append(name)
        # print("Parameters without gradients:")
        # for param_name in parameters_without_grad:
        #     print(param_name)
        # import ipdb; ipdb.set_trace()
        self.updated = False
        self.global_step += 1  # update global step
        return loss

    def validation_step(self):
        if self.val_iterator is None:
            raise ValueError("val_iterator is None, please call validation_epoch_begin() first")
        batch = next(self.val_iterator)
        with torch.no_grad():
            loss = self.step(batch, "val")
        # self.write_loss_log("val", loss)
        return loss

    def test_step(self):
        if self.test_iterator is None:
            raise ValueError("test_iterator is None, please call test_epoch_begin() first")
        batch = next(self.test_iterator)
        with torch.no_grad():
            return self.step(batch, "test")

    def interpret_step(self, batch):
        with torch.no_grad():
            return self.step(batch, "interpret")

    def step(self, batch, stage):
        with torch.set_grad_enabled(stage == "train"):
            extra_args = batch.to_dict()
            # extra_args actually won't be used in the model
            for a in ('y', 'x', 'x_mask', 'x_alt', 'pos', 'batch',
                      'edge_index', 'edge_attr',
                      'edge_index_star', 'edge_attr_star',
                      'node_vec_attr'):
                if a in extra_args:
                    del extra_args[a]
            y, x_embed, attn_weight_layers = self.forward(
                x=batch.x.to(self.device),
                x_mask=batch.x_mask.to(self.device),
                x_alt=batch.x_alt.to(self.device),
                pos=batch.pos.to(self.device),
                batch=batch.batch.to(self.device) if "batch" in batch else None,
                edge_index=batch.edge_index.to(self.device) if batch.edge_index is not None else None,
                edge_index_star=batch.edge_index_star.to(self.device) if "edge_index_star" in batch else None,
                edge_attr=batch.edge_attr.to(self.device) if batch.edge_attr is not None else None,
                edge_attr_star=batch.edge_attr_star.to(self.device) if "edge_attr_star" in batch else None,
                node_vec_attr=batch.node_vec_attr.to(self.device),
                extra_args=extra_args,
                return_attn=stage == "interpret",
            )
            if stage == "test":
                if self.hparams.dataset.startswith("Mask"):
                    # if mask dataset, and we are testing, then we don't want to mark other locations but mask
                    self.predictions['y'].append(y[batch.x_mask == False].detach().cpu().numpy())
                else:
                    self.predictions['y'].append(y.detach().cpu().numpy())
        loss_y = 0
        
        if stage != "interpret":
            if "y" in batch:
                if batch.y.ndim == 1 and self.hparams.loss_fn != "cross_entropy":
                    batch.y = batch.y.unsqueeze(1)

                # y loss, if mask predict, only predict the non-masked locations
                if self.hparams.dataset.startswith("Mask"):
                    y = y[batch.x_mask==False]
                    batch.y = batch.y[batch.x_mask==False]
                loss_y = self.loss_fn(y, batch.y.to(self.device))
                
                if self.contrastive_loss is not None:
                    loss_cont = self.contrastive_loss(x_embed, batch.y.to(self.device))
                else:
                    loss_cont = 0

                if self.hparams.y_weight > 0 and stage != "interpret":
                    self.losses[stage + "_y"].append(loss_y.detach().cpu())

            # total loss
            loss = loss_y * self.hparams.y_weight + loss_cont
            self.losses[stage].append(loss.detach().cpu())
            return loss
        else:
            return None, y, x_embed, attn_weight_layers

    def optimizer_step(self, loss=None):
        # optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_scale * float(self.hparams.lr)
        # loss is not used in optimizer step anymore
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.updated = True        

    def scheduler_step(self, val_loss):
        self.scheduler.step(val_loss)

    def training_epoch_begin(self):
        self.train_iterator = iter(self.train_dataloader)

    def training_epoch_end(self):
        self.train_iterator = None
        self._reset_losses_dict()
        self.current_epoch += 1
        if self.reset_train_dataloader_each_epoch:
            idx_train = getattr(utils.configs, "reshuffle_train" + self.split_fn)(self.idx_train, self.hparams.batch_size,
                                                                                   self.dataset,
                                                                                   seed=self.current_epoch if self.reset_train_dataloader_each_epoch_seed else None)
            self.train_dataset = Subset(self.dataset, idx_train)
            self.train_dataloader = DataLoader(
                    dataset=self.train_dataset,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    pin_memory=True,
                    pin_memory_device='cpu',
                    shuffle=False,
                )

    def validation_epoch_begin(self):
        if self.val_dataloader is None:
            self.val_iterator = iter(self.train_dataloader)
        else:
            self.val_iterator = iter(self.val_dataloader)

    def validation_epoch_end(self, reset_train_loss=False):
        self.val_iterator = None
        # construct dict of logged metrics
        result_dict = {
            "epoch": int(self.current_epoch),
            "lr": self.optimizer.param_groups[0]["lr"],
            "train_loss": torch.stack(self.losses["train"]).mean().item() if len(self.losses["train"]) > 0 else None,
        }
        if self.val_dataset is not None:
            result_dict["val_loss"] = torch.stack(self.losses["val"]).mean().item()
            self.write_loss_log("val", torch.stack(self.losses["val"]).mean())
        else:
            # use train loss as val loss if no val dataset is present
            result_dict["val_loss"] = torch.stack(self.losses["train"]).mean().item()
            self.write_loss_log("val", torch.stack(self.losses["train"]).mean())
        # add test loss if available
        if len(self.losses["test"]) > 0:
            result_dict["test_loss"] = torch.stack(self.losses["test"]).mean().item()

        # if predictions are present, also log them separately
        if len(self.losses["train_y"]) > 0:
            result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean().item()
            if self.val_dataset is not None:
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean().item()

            if len(self.losses["test"]) > 0:
                result_dict["test_loss_y"] = torch.stack(
                    self.losses["test_y"]
                ).mean().item()
        if reset_train_loss:
            self._reset_losses_dict()
        else:
            self._reset_val_losses_dict()
        return result_dict

    def testing_epoch_begin(self):
        self.test_iterator = iter(self.test_dataloader)

    def testing_epoch_end(self):
        self.test_iterator = None
        # construct dict of logged metrics
        result_dict = {
            "epoch": int(self.current_epoch),
            "lr": self.optimizer.param_groups[0]["lr"],
            "test_loss": torch.stack(self.losses["test"]).mean().item(),
        }
        # if predictions are present, also log them separately
        if len(self.losses["test_y"]) > 0:
            if len(self.losses["test"]) > 0:
                result_dict["test_loss_y"] = torch.stack(
                    self.losses["test_y"]
                ).mean().item()
        self._reset_losses_dict()
        # prepare result data frame
        y_result = pd.DataFrame(np.concatenate(self.predictions['y'], axis=0),
                                index=self.dataset.data.index)
        y_result.columns = [f'y.{i}' for i in y_result.columns]
        result_df = pd.concat(
            [self.dataset.data,
             y_result,
             ],
            axis=1
        )
        self._reset_predictions_dict()
        return result_dict, result_df

    def write_loss_log(self, stage, loss):
        if self.device_id is None:
            scalar_name = f"loss/{stage}"
        else:
            scalar_name = f"loss/ddp_rank.{self.device_id}.{stage}"
        self.writer.add_scalar(scalar_name, loss, self.global_step)
        if stage == "train" and self.device_id == 0:
            for tag, value in self.model.named_parameters():
                    tag = tag.replace('.', '/')
                    self.writer.add_histogram('weights/'+tag, value.data.cpu().numpy(), self.global_step)
                    try:
                        self.writer.add_histogram('grads/'+tag, value.grad.data.cpu().numpy(), self.global_step)
                    except:
                        print(f"failed to add grad histogram for '{tag}' in counter: {self.global_step}")

    def write_model(self, epoch=None, step=None, save_optimizer=False, optimizer_rank=None):
        if save_optimizer:
            assert optimizer_rank is not None
        if epoch is None:
            if step is None:
                model_save_file_name = f"{self.hparams.log_dir}/model.epoch.{self.current_epoch}.step.{self.global_step}.pt"
                if save_optimizer:
                    optimizer_save_file_name = f"{self.hparams.log_dir}/optimizer.epoch.{self.current_epoch}.step.{self.global_step}.rank.{optimizer_rank}.pt"
                    scheduler_save_file_name = f"{self.hparams.log_dir}/scheduler.epoch.{self.current_epoch}.step.{self.global_step}.rank.{optimizer_rank}.pt"
            else:
                model_save_file_name = f"{self.hparams.log_dir}/model.step.{step}.pt"
                if save_optimizer:
                    optimizer_save_file_name = f"{self.hparams.log_dir}/optimizer.step.{step}.rank.{optimizer_rank}.pt"
                    scheduler_save_file_name = f"{self.hparams.log_dir}/scheduler.step.{step}.rank.{optimizer_rank}.pt"
        else:
            if step is None:
                model_save_file_name = f"{self.hparams.log_dir}/model.epoch.{epoch}.pt"
                if save_optimizer:
                    optimizer_save_file_name = f"{self.hparams.log_dir}/optimizer.epoch.{epoch}.rank.{optimizer_rank}.pt"
                    scheduler_save_file_name = f"{self.hparams.log_dir}/scheduler.epoch.{epoch}.rank.{optimizer_rank}.pt"
            else:
                model_save_file_name = f"{self.hparams.log_dir}/model.epoch.{epoch}.step.{step}.pt"
                if save_optimizer:
                    optimizer_save_file_name = f"{self.hparams.log_dir}/optimizer.epoch.{epoch}.step.{step}.rank.{optimizer_rank}.pt"
                    scheduler_save_file_name = f"{self.hparams.log_dir}/scheduler.epoch.{epoch}.step.{step}.rank.{optimizer_rank}.pt"
        if isinstance(self.model, DDP):
            if self.use_lora:
                state_dic = lora.lora_state_dict(self.model.module)
                # add output_model to state_dic
                output_model_state_dic = self.model.module.output_model.state_dict()
                for key, value in output_model_state_dic.items():
                    state_dic[f"module.output_model.{key}"] = value
                torch.save(state_dic, model_save_file_name)
            else:
                torch.save(self.model.module.state_dict(), model_save_file_name)
        else:
            if self.use_lora:
                state_dic = lora.lora_state_dict(self.model)
                # add output_model to state_dic
                output_model_state_dic = self.model.output_model.output_network.state_dict()
                for key, value in output_model_state_dic.items():
                    state_dic[f"output_model.output_network.{key}"] = value
                torch.save(state_dic, model_save_file_name)
            else:
                torch.save(self.model.state_dict(), model_save_file_name)
        if save_optimizer:
            torch.save(self.optimizer.state_dict(), optimizer_save_file_name)
            torch.save(self.scheduler.state_dict(), scheduler_save_file_name)
    
    def write_optimizer(self, epoch=None, step=None, optimizer_rank=None):
        if epoch is None:
            if step is None:
                optimizer_save_file_name = f"{self.hparams.log_dir}/optimizer.epoch.{self.current_epoch}.step.{self.global_step}.rank.{optimizer_rank}.pt"
                scheduler_save_file_name = f"{self.hparams.log_dir}/scheduler.epoch.{self.current_epoch}.step.{self.global_step}.rank.{optimizer_rank}.pt"
            else:
                optimizer_save_file_name = f"{self.hparams.log_dir}/optimizer.step.{step}.rank.{optimizer_rank}.pt"
                scheduler_save_file_name = f"{self.hparams.log_dir}/scheduler.step.{step}.rank.{optimizer_rank}.pt"
        else:
            if step is None:
                optimizer_save_file_name = f"{self.hparams.log_dir}/optimizer.epoch.{epoch}.rank.{optimizer_rank}.pt"
                scheduler_save_file_name = f"{self.hparams.log_dir}/scheduler.epoch.{epoch}.rank.{optimizer_rank}.pt"
            else:
                optimizer_save_file_name = f"{self.hparams.log_dir}/optimizer.epoch.{epoch}.step.{step}.rank.{optimizer_rank}.pt"
                scheduler_save_file_name = f"{self.hparams.log_dir}/scheduler.epoch.{epoch}.step.{step}.rank.{optimizer_rank}.pt"
        torch.save(self.optimizer.state_dict(), optimizer_save_file_name)
        torch.save(self.scheduler.state_dict(), scheduler_save_file_name)

    def load_model(self, epoch=None, step=None, update_count=False):
        if epoch is None:
            if step is None:
                _state_dict = torch.load(
                    f"{self.hparams.log_dir}/model.epoch.{self.current_epoch}.step.{self.global_step}.pt",
                    maplocation=self.device
                )
            else:
                _state_dict = torch.load(
                    f"{self.hparams.log_dir}/model.step.{step}.pt",
                    map_location=self.device
                )
                if update_count:
                    self.global_step = step
                    self.current_epoch = step // self.batchs_per_epoch
        else:
            if step is None:
                _state_dict = torch.load(
                    f"{self.hparams.log_dir}/model.epoch.{epoch}.pt",
                    map_location=self.device
                )
                if update_count:
                    self.current_epoch = epoch
                    self.global_step = epoch * self.batchs_per_epoch
            else:
                _state_dict = torch.load(
                    f"{self.hparams.log_dir}/model.epoch.{epoch}.step.{step}.pt",
                    map_location=self.device
                )
                if update_count:
                    self.current_epoch = epoch
                    self.global_step = step
        _state_dict_is_ddp = list(_state_dict.keys())[0].startswith("module.")
        if isinstance(self.model, DDP):
            if _state_dict_is_ddp:
                self.model.load_state_dict(_state_dict, strict=self.use_lora==False)
            else:
                self.model.module.load_state_dict(_state_dict, strict=self.use_lora==False)
        else:
            if _state_dict_is_ddp:
                # create new OrderedDict that does not contain `module.`
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in _state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                self.model.load_state_dict(new_state_dict, strict=self.use_lora==False)
            else:
                self.model.load_state_dict(_state_dict, strict=self.use_lora==False)

    def load_optimizer(self, epoch=None, step=None, optimizer_rank=0):
        if epoch is None:
            if step is None:
                optimizer_state_dict = torch.load(
                    f"{self.hparams.log_dir}/optimizer.epoch.{self.current_epoch}.step.{self.global_step}.rank.{optimizer_rank}.pt",
                    maplocation=self.device
                )
                scheduler_state_dict = torch.load(
                    f"{self.hparams.log_dir}/scheduler.epoch.{self.current_epoch}.step.{self.global_step}.rank.{optimizer_rank}.pt",
                    maplocation=self.device
                )
            else:
                optimizer_state_dict = torch.load(
                    f"{self.hparams.log_dir}/optimizer.step.{step}.rank.{optimizer_rank}.pt",
                    map_location=self.device
                )
                scheduler_state_dict = torch.load(
                    f"{self.hparams.log_dir}/scheduler.step.{step}.rank.{optimizer_rank}.pt",
                    map_location=self.device
                )
        else:
            if step is None:
                optimizer_state_dict = torch.load(
                    f"{self.hparams.log_dir}/optimizer.epoch.{epoch}.rank.{optimizer_rank}.pt",
                    map_location=self.device
                )
                scheduler_state_dict = torch.load(
                    f"{self.hparams.log_dir}/scheduler.epoch.{epoch}.rank.{optimizer_rank}.pt",
                    map_location=self.device
                )
            else:
                optimizer_state_dict = torch.load(
                    f"{self.hparams.log_dir}/optimizer.epoch.{epoch}.step.{step}.rank.{optimizer_rank}.pt",
                    map_location=self.device
                )
                scheduler_state_dict = torch.load(
                    f"{self.hparams.log_dir}/scheduler.epoch.{epoch}.step.{step}.rank.{optimizer_rank}.pt",
                    map_location=self.device
                )
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.scheduler.load_state_dict(scheduler_state_dict)
        
    def _reset_predictions_dict(self):
        self.predictions = {
            "y": [],
        }

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
            "train_y": [],
            "val_y": [],
            "test_y": [],
        }

    def _reset_val_losses_dict(self):
        self.losses["val"] = []
        self.losses["val_y"] = []


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '15423'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def data_distributed_parallel_gpu_archive(rank, model, hparams, datasets, trainer_fn=None):
    # set up training processes
    # Currently have bug if batch size does not match
    global result_dict
    world_size = hparams.ngpus
    epochs = hparams.num_epochs
    save_every_step = hparams.num_save_batches
    save_every_epoch = hparams.num_save_epochs
    setup(rank, world_size)
    device = f'cuda:{rank}'
    torch.cuda.set_per_process_memory_fraction(1.0, rank)
    model = model.to(device)
    
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    ddp_model.train()
    
    trainer = trainer_fn(hparams=hparams, model=ddp_model, dataset=datasets[rank], device_id=rank)
    # begin training
    with Join([trainer.model]):
        for i in range(epochs):
            epoch_start_time = time.time()
            train_finished = False
            trainer.training_epoch_begin()
            while not train_finished:
                try:
                    batch_start_time = time.time()
                    loss = trainer.training_step()
                    if trainer.global_step % hparams.num_steps_update == 0:
                        # only update every num_steps_update steps, to save memory
                        trainer.optimizer_step(loss)
                    batch_end_time = time.time()
                    print(f"Rank {rank} batch {trainer.global_step} time: {batch_end_time - batch_start_time}")
                    dist.barrier()
                    if trainer.global_step % save_every_step == 0:
                        if rank == 0:
                            trainer.write_model(step=trainer.global_step)
                        # validate every save_every_step steps
                        if trainer.val_dataset is not None:
                            val_finished = False
                            trainer.validation_epoch_begin()
                            while not val_finished:
                                try:
                                    trainer.validation_step()
                                    dist.barrier()
                                except StopIteration:
                                    val_finished = True
                        result_dict = trainer.validation_epoch_end(reset_train_loss=True)
                        print(f"Rank {rank} batch {trainer.global_step} result: {result_dict}")
                        with open(
                                f"{hparams.log_dir}/result_dict.batch.{trainer.global_step}.ddp_rank.{rank}.json", "w"
                        ) as f:
                            json.dump(result_dict, f)
                        dist.barrier()
                        all_val_loss = []
                        for k in range(world_size):
                            with open(
                                    f"{hparams.log_dir}/result_dict.batch.{trainer.global_step}.ddp_rank.{k}.json", "r"
                            ) as f:
                                if trainer.val_dataset is not None:
                                    all_val_loss.append(json.load(f)["val_loss"])
                                else:
                                    # train is val
                                    all_val_loss.append(json.load(f)["train_loss"])
                        print(f"Batch {trainer.global_step} all val loss: {np.mean(all_val_loss)}")
                        trainer.scheduler_step(np.mean(all_val_loss))
                        dist.barrier()
                except StopIteration:
                    train_finished = True
            # if remain unupdated parameters, update them
            if not trainer.updated:
                trainer.optimizer_step(loss)
            dist.barrier()
            # validate every epoch
            if trainer.val_dataset is not None:
                val_finished = False
                trainer.validation_epoch_begin()
                while not val_finished:
                    try:
                        trainer.validation_step()
                        dist.barrier()
                    except StopIteration:
                        val_finished = True
            result_dict = trainer.validation_epoch_end()
            print(f"Rank {rank} epoch {i} result: {result_dict}")
            with open(f"{hparams.log_dir}/result_dict.epoch.{i}.ddp_rank.{rank}.json", "w") as f:
                json.dump(result_dict, f)
            # take all val loss together
            dist.barrier()
            trainer.training_epoch_end()
            epoch_end_time = time.time()
            print(f"Epoch {i} time: ", epoch_end_time - epoch_start_time)
            dist.barrier()
            if rank == 0 and trainer.current_epoch % save_every_epoch == 0:
                trainer.write_model(epoch=trainer.current_epoch)
    cleanup()
    # return all_losses
    return trainer


def data_distributed_parallel_gpu(rank, model, hparams, datasets, trainer_fn=None, checkpoint_epoch=None):
    # set up training processes
    # Currently have bug if batch size does not match
    global result_dict
    world_size = hparams.ngpus
    epochs = hparams.num_epochs
    save_every_step = hparams.num_save_batches
    save_every_epoch = hparams.num_save_epochs
    setup(rank, world_size)
    device = f'cuda:{rank}'
    torch.cuda.set_per_process_memory_fraction(1.0, rank)
    model = model.to(device)
    
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    ddp_model.train()
    
    trainer = trainer_fn(hparams=hparams, model=ddp_model, dataset=datasets[rank], device_id=rank)
    print(f"number of trainable parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)}, " +
          f"percentage = {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad) / sum(p.numel() for p in trainer.model.parameters())}")
    # dry run to update optimizer and scheduler to the checkpoint epoch
    if checkpoint_epoch is not None:
        while trainer.current_epoch < checkpoint_epoch - 1:
            epoch_start_time = time.time()
            # trainer.training_epoch_begin()
            # trainer.training_epoch_end()
            trainer.current_epoch += 1
            epoch_end_time = time.time()
            print(f"Dry run load: Epoch {trainer.current_epoch} time: ", epoch_end_time - epoch_start_time)
            dist.barrier()
        # Set up training data set
        trainer.training_epoch_end()
        trainer.load_model(epoch=checkpoint_epoch, update_count=True)
        trainer.load_optimizer(epoch=checkpoint_epoch, optimizer_rank=rank)
        print(f"Finished dry run, loaded model from epoch {checkpoint_epoch}")
    else:
        print("No checkpoint epoch, start from scratch")
        checkpoint_epoch = 0
    # begin training
    with Join([trainer.model]):
        for i in range(checkpoint_epoch, epochs):
            epoch_start_time = time.time()
            train_finished = False
            trainer.training_epoch_begin()
            while not train_finished:
                try:
                    batch_start_time = time.time()
                    loss = trainer.training_step()
                    if trainer.global_step % hparams.num_steps_update == 0:
                        # only update every num_steps_update steps, to save memory
                        trainer.optimizer_step(loss)
                    batch_end_time = time.time()
                    print(f"Rank {rank} batch {trainer.global_step} time: {batch_end_time - batch_start_time}")
                    dist.barrier()
                    if trainer.global_step % save_every_step == 0:
                        if rank == 0:
                            trainer.write_model(step=trainer.global_step)
                        # validate every save_every_step steps
                        if trainer.val_dataset is not None:
                            val_finished = False
                            val_begin_time = time.time()
                            trainer.validation_epoch_begin()
                            while not val_finished:
                                try:
                                    trainer.validation_step()
                                    dist.barrier()
                                except StopIteration:
                                    val_finished = True
                            val_end_time = time.time()
                        result_dict = trainer.validation_epoch_end(reset_train_loss=True)
                        print(f"Rank {rank} batch {trainer.global_step} result: {result_dict}")
                        with open(
                                f"{hparams.log_dir}/result_dict.batch.{trainer.global_step}.ddp_rank.{rank}.json", "w"
                        ) as f:
                            json.dump(result_dict, f)
                        dist.barrier()
                        all_val_loss = []
                        for k in range(world_size):
                            with open(
                                    f"{hparams.log_dir}/result_dict.batch.{trainer.global_step}.ddp_rank.{k}.json", "r"
                            ) as f:
                                if trainer.val_dataset is not None:
                                    all_val_loss.append(json.load(f)["val_loss"])
                                else:
                                    # train is val
                                    all_val_loss.append(json.load(f)["train_loss"])
                        print(f"Batch {trainer.global_step} all val loss: {np.mean(all_val_loss)}")
                        print(f"Batch {trainer.global_step} val time: {val_end_time - val_begin_time}")
                        trainer.scheduler_step(np.mean(all_val_loss))
                        dist.barrier()
                except StopIteration:
                    train_finished = True
            # if remain unupdated parameters, update them
            if not trainer.updated:
                trainer.optimizer_step(loss)
            dist.barrier()
            # validate every epoch
            if trainer.val_dataset is not None:
                val_finished = False
                trainer.validation_epoch_begin()
                while not val_finished:
                    try:
                        trainer.validation_step()
                        dist.barrier()
                    except StopIteration:
                        val_finished = True
            result_dict = trainer.validation_epoch_end()
            print(f"Rank {rank} epoch {i} result: {result_dict}")
            with open(f"{hparams.log_dir}/result_dict.epoch.{i}.ddp_rank.{rank}.json", "w") as f:
                json.dump(result_dict, f)
            # take all val loss together
            dist.barrier()
            trainer.training_epoch_end()
            epoch_end_time = time.time()
            print(f"Epoch {i} time: ", epoch_end_time - epoch_start_time)
            dist.barrier()
            if trainer.current_epoch % save_every_epoch == 0:
                if rank == 0:
                    trainer.write_model(epoch=trainer.current_epoch, save_optimizer=True, optimizer_rank=rank)
                else:
                    trainer.write_optimizer(epoch=trainer.current_epoch, optimizer_rank=rank)
    cleanup()
    # return all_losses
    return trainer


def single_thread_gpu(rank, model, hparams, dataset, trainer_fn=None, checkpoint_epoch=None):
    # set up training processes
    # Currently have bug if batch size does not match
    global result_dict
    epochs = hparams.num_epochs
    save_every_step = hparams.num_save_batches
    save_every_epoch = hparams.num_save_epochs
    device = f'cuda:{rank}'
    torch.cuda.set_per_process_memory_fraction(1.0, rank)
    model = model.to(device)
    model.train()
    
    trainer = trainer_fn(hparams=hparams, model=model, dataset=dataset, device_id=rank)
    print(f"number of trainable parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)}, " +
          f"percentage = {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad) / sum(p.numel() for p in trainer.model.parameters())}")
    # begin training
    if checkpoint_epoch is not None:
        while trainer.current_epoch < checkpoint_epoch:
            epoch_start_time = time.time()
            trainer.training_epoch_begin()
            trainer.training_epoch_end()
            epoch_end_time = time.time()
            print(f"Dry run load: Epoch {trainer.current_epoch} time: ", epoch_end_time - epoch_start_time)
        trainer.load_model(epoch=checkpoint_epoch, update_count=True)
        trainer.load_optimizer(epoch=checkpoint_epoch, optimizer_rank=rank)
        print(f"Finished dry run, loaded model from epoch {checkpoint_epoch}")
    else:
        print("No checkpoint epoch, start from scratch")
        checkpoint_epoch = 0
    for i in range(checkpoint_epoch, epochs):
        epoch_start_time = time.time()
        train_finished = False
        trainer.training_epoch_begin()
        while not train_finished:
            try:
                batch_start_time = time.time()
                loss = trainer.training_step()
                if trainer.global_step % hparams.num_steps_update == 0:
                        # only update every num_steps_update steps, to save memory
                    trainer.optimizer_step(loss)
                batch_end_time = time.time()
                print(f"Rank {rank} batch {trainer.global_step} time: {batch_end_time - batch_start_time}")
                if trainer.global_step % save_every_step == 0:
                    trainer.write_model(step=trainer.global_step)
                    # validate every save_every_step steps
                    val_finished = False
                    val_start_time = time.time()
                    trainer.validation_epoch_begin()
                    while not val_finished:
                        try:
                            trainer.validation_step()
                        except StopIteration:
                            val_finished = True
                    result_dict = trainer.validation_epoch_end()
                    print(f"Rank {rank} batch {trainer.global_step} result: {result_dict}")
                    with open(
                            f"{hparams.log_dir}/result_dict.batch.{trainer.global_step}.ddp_rank.{rank}.json", "w"
                    ) as f:
                        json.dump(result_dict, f)
                    all_val_loss = result_dict["val_loss"]
                    print(f"Batch {trainer.global_step} all val loss: {all_val_loss}")
                    trainer.scheduler_step(all_val_loss)
                    val_end_time = time.time()
                    print(f"Rank {rank} batch {trainer.global_step} validation time: {val_end_time - val_start_time}")
            except StopIteration:
                train_finished = True
        # if remain unupdated parameters, update them
        if not trainer.updated:
            trainer.optimizer_step(loss)
        # validate every epoch
        val_finished = False
        trainer.validation_epoch_begin()
        while not val_finished:
            try:
                trainer.validation_step()
            except StopIteration:
                val_finished = True
        result_dict = trainer.validation_epoch_end()
        print(f"Rank {rank} epoch {i} result: {result_dict}")
        with open(f"{hparams.log_dir}/result_dict.epoch.{i}.ddp_rank.{rank}.json", "w") as f:
            json.dump(result_dict, f)
        trainer.training_epoch_end()
        epoch_end_time = time.time()
        print(f"Epoch {i} time: ", epoch_end_time - epoch_start_time)
        if trainer.current_epoch % save_every_epoch == 0:
            trainer.write_model(epoch=trainer.current_epoch, save_optimizer=True, optimizer_rank=rank)
    # return all_losses
    return trainer
