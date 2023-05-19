import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from os.path import join
from torch.utils.data import DataLoader, Subset
from model.trainer import PreMode_trainer

import data
import utils.configs

class PreMode_trainer_noGraph(PreMode_trainer):
    """
    A wrapper for dataloader, summary writer, optimizer, scheduler
    """

    def __init__(self, hparams, model, stage: str = "train", dataset=None, device_id=None):
        super(PreMode_trainer_noGraph, self).__init__(hparams, model, stage, dataset, device_id)

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
                    self.hparams.batch_size * self.hparams.num_workers,
                    join(self.hparams.log_dir, f"splits.{self.device_id}.npz"),
                    self.hparams.splits,
                )
                print(f"train {len(idx_train)}, val {len(idx_val)}")

                self.train_dataset = Subset(self.dataset, idx_train)
                self.idx_train = idx_train
                self.val_dataset = Subset(self.dataset, idx_val)
                self.idx_val = idx_val
            else:
                self.train_dataset = self.dataset
                self.val_dataset = None

            self.train_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                shuffle=False,
            )
            if self.val_dataset is not None:
                self.val_dataloader = DataLoader(
                    dataset=self.val_dataset,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    pin_memory=True,
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
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                shuffle=False,
            )
        elif stage == 'all':
            # make train/test/val split
            idx_train, idx_val, idx_test = getattr(utils.configs, "make_splits_train_val_test" + split_fn)(
                self.dataset,
                self.hparams.train_size,
                self.hparams.val_size,
                self.hparams.test_size,
                self.hparams.seed,
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
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                shuffle=False,
            )
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                shuffle=False,
            )
            self.test_dataloader = DataLoader(
                dataset=self.test_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                shuffle=False,
            )
        else:
            raise ValueError(f"stage {stage} not supported")

    def training_epoch_end(self):
        self.train_iterator = None
        self._reset_losses_dict()
        self.current_epoch += 1
        if self.reset_train_dataloader_each_epoch:
            idx_train = getattr(utils.configs, "reshuffle_train" + self.split_fn)(self.idx_train, self.hparams.batch_size, self.dataset)
            self.train_dataset = Subset(self.dataset, idx_train)
            self.train_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                shuffle=False,
            )

    def step(self, batch, stage):
        with torch.set_grad_enabled(stage == "train"):
            extra_args = batch.copy()
            # extra_args actually won't be used in the model
            for a in ('y', 'x', 'x_mask', 'x_alt', 'pos', 'batch',
                      'edge_index', 'edge_attr',
                      'edge_index_star', 'edge_attr_star',
                      'node_vec_attr'):
                if a in extra_args:
                    del extra_args[a]
            y, x_embed, attn_weight_layers = self.forward(
                x=batch["x"].to(self.device),
                x_mask=batch["x_mask"].to(self.device),
                x_alt=batch["x_alt"].to(self.device),
                pos=batch["pos"].to(self.device),
                extra_args=extra_args,
                return_attn=stage == "interpret",
            )
            if stage == "test":
                if self.hparams.dataset.startswith("Mask"):
                    # if mask dataset, and we are testing, then we don't want to mark other locations but mask
                    self.predictions['y'].append(y[batch["x_mask"] == False].detach().cpu().numpy())
                else:
                    self.predictions['y'].append(y.detach().cpu().numpy())
        loss_y = 0
        
        if "y" in batch:
            if batch["y"].ndim == 1 and self.hparams.loss_fn != "cross_entropy":
                batch["y"] = batch["y"].unsqueeze(1)
            if self.hparams.dataset.startswith("Mask"):
                # resize B x L x D to B*L x D
                batch["y"] = batch["y"].view(-1, batch["y"].shape[-1])
                y = y.view(-1, y.shape[-1])
                batch["y_mask"] = batch["y_mask"].view(-1)
                # only calculate loss for non-masked locations
                y = y[batch["y_mask"]]
                batch["y"] = batch["y"][batch["y_mask"]]
            # y loss
            loss_y = self.loss_fn(y, batch["y"].to(self.device))

            if self.hparams.y_weight > 0 and stage != "interpret":
                self.losses[stage + "_y"].append(loss_y.detach().cpu())

        # total loss
        loss = loss_y * self.hparams.y_weight

        if stage != "interpret":
            self.losses[stage].append(loss.detach().cpu())
        if stage == "interpret":
            return loss, y, x_embed, attn_weight_layers
        else:
            return loss
