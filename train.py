import argparse
import json
import os
import subprocess
import pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.multiprocessing as mp

import data
from model import model
from model.model import create_model, create_model_and_load
from model.trainer import (data_distributed_parallel_gpu, PreMode_trainer, single_thread_gpu)
from model.trainer_noGraph import PreMode_trainer_noGraph
from model.trainer_ssp import PreMode_trainer_SSP
from utils.configs import save_argparse, LoadFromFile
from captum.attr import IntegratedGradients
mp.set_sharing_strategy('file_system')


def get_args():
    parser = argparse.ArgumentParser(description='Training')
    # keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')
    # data set specific
    parser.add_argument('--dataset', default=None, type=str, choices=data.__all__,
                        help='Name of the dataset')
    parser.add_argument('--data-file-train', default=None, type=str,
                        help='Custom training files')
    parser.add_argument('--data-file-train-ddp-prefix', default=None, type=str,
                        help='Prefix of custom training files if use DDP')
    parser.add_argument('--data-file-test', default=None, type=str,
                        help='Custom testing files')
    parser.add_argument('--data-type', default=None, type=str,
                        help='Data type for the task')
    parser.add_argument('--loop', type=bool, default=False,
                        help='Add self loop to nodes or not')
    parser.add_argument('--max-num-neighbors', type=int, default=32,
                        help='Maximum number of neighbors to consider in the network')
    parser.add_argument('--node-embedding-type', type=str, default='esm',
                        help='Node embedding type. Choose from esm, one-hot, one-hot-idx, or aa-5dim')
    parser.add_argument('--graph-type', type=str, default='af2',
                        help='Graph type. Choose from af2 or 1d-neighbor')
    parser.add_argument('--add-plddt', type=bool, default=False,
                        help='Whether to add plddt or not')
    parser.add_argument('--add-conservation', type=bool, default=False,
                        help='Whether to add conservation or not')
    parser.add_argument('--add-dssp', type=bool, default=False,
                        help='Whether to add dssp or not')
    parser.add_argument('--add-position', type=bool, default=False,
                        help='Whether to add positional wise encoding or not')
    parser.add_argument('--add-sidechain', type=bool, default=False,
                        help='Whether to add sidechain or not')
    parser.add_argument('--use-cb', type=bool, default=False,
                        help='Whether to use CB as distance or not')
    parser.add_argument('--add-msa', type=bool, default=False,
                        help='Whether to add msa to features or not')
    parser.add_argument('--loaded-msa', type=bool, default=False,
                        help='Whether to preload msa to features or not')
    parser.add_argument('--alt-type', type=str, default='alt',
                        help='alt type in data, either alt or concat')
    parser.add_argument('--computed-graph', type=bool, default=True,
                        help='Whether to use computed graph or not')
    parser.add_argument('--neighbor-type', type=str, default='KNN',
                        help='The type of neighbor selection. Choose from KNN or radius')
    parser.add_argument('--max-len', type=int, default=2251,
                        help='Maximum length of input sequences')
    parser.add_argument('--radius', type=float, default=50,
                        help='Radius of AA to be selected')
    parser.add_argument('--data-augment', type=bool, default=False,
                        help='Whether to augument data, if so, the data will be augumented in the training process by reverse the ref and alt')
    parser.add_argument('--score-transfer', type=bool, default=False,
                        help='Whether to transfer scer, if so, the score will be transfered to 0, 3')
    
    # model specific
    parser.add_argument('--load-model', type=str, default=None,
                        help='Restart training using a model checkpoint')
    parser.add_argument('--partial-load-model', type=bool, default=False,
                        help='Partial load model, particullay from maskpredict model using a model checkpoint')
    parser.add_argument('--model-class', type=str, default=None, choices=model.__all__,
                        help='Which model to use')
    parser.add_argument('--model', type=str, default=None,
                        help='Which representation model to use')
    parser.add_argument('--triangular-update', type=bool, default=True,
                        help='Whether do triangular update')
    parser.add_argument('--alt-projector', type=int, default=None,
                        help='Alt projector size')
    parser.add_argument('--neighbor-embedding', type=bool, default=False,
                        help='If a neighbor embedding should be applied before interactions')
    parser.add_argument('--cutoff-lower', type=float, default=0.0,
                        help='Lower cutoff in model')
    parser.add_argument('--cutoff-upper', type=float, default=5.0,
                        help='Upper cutoff in model')
    parser.add_argument('--x-in-channels', type=int, default=None,
                        help='x input size, only used if different from x_channels')
    parser.add_argument('--x-in-embedding-type', type=str, default=None,
                        help='x input embedding type, only used if x-in-channels is not None')
    parser.add_argument('--x-channels', type=int, default=1280,
                        help='x embedding size')
    parser.add_argument('--x-hidden-channels', type=int, default=640,
                        help='x hidden size')
    parser.add_argument('--vec-in-channels', type=int, default=4,
                        help='vector embedding size')
    parser.add_argument('--vec-channels', type=int, default=64,
                        help='vector hidden size')
    parser.add_argument('--vec-hidden-channels', type=int, default=1280,
                        help='vector hidden size, must be equal to x_channels')
    parser.add_argument('--share-kv', type=bool, default=False,
                        help='Whether to share key and value')
    parser.add_argument('--ee-channels', type=int, default=None,
                        help='edge-edge update channel that depends on start/end node distances')
    parser.add_argument('--distance-influence', type=str, default='both',
                        help='Which distance influences to use')
    parser.add_argument('--num-heads', type=int, default=16,
                        help='number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--num-edge-attr', type=int, default=1,
                        help='number of edge attributes')
    parser.add_argument('--num-nodes', type=int, default=1,
                        help='number of nodes')
    parser.add_argument('--num-rbf', type=int, default=32,
                        help='number of radial basis functions')
    parser.add_argument('--rbf-type', type=str, default="expnorm",
                        help='type of radial basis functions')
    parser.add_argument('--trainable-rbf', type=bool, default=False,
                        help='to train rbf or not')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='number of workers')
    parser.add_argument('--output-model', type=str, default='EquivariantBinaryClassificationSAGPoolScalar',
                        help='The type of output model')
    parser.add_argument('--reduce-op', type=str, default='mean',
                        help='The type of reduce operation')
    parser.add_argument('--output-dim', type=int, default=1,
                        help='The dimension of output model')
    parser.add_argument('--activation', type=str, default='silu',
                        help='The activation function')
    parser.add_argument('--attn-activation', type=str, default='silu',
                        help='The attention activation function')
    parser.add_argument('--drop-out', type=float, default=0.1,
                        help='Drop out rate at each layer') 
    parser.add_argument('--use-lora', type=int, default=None,
                        help='Whether to use lora or not')
   
    # training specific
    parser.add_argument('--trainer-fn', type=str, default='PreMode_trainer', 
                        help='trainer function')
    parser.add_argument('--freeze-representation', type=bool, default=False, 
                        help='freeze representation module or not')
    parser.add_argument('--freeze-representation-but-attention', type=bool, default=False, 
                        help='freeze representation module but without attention, or not')
    parser.add_argument('--freeze-representation-but-gru', type=bool, default=False, 
                        help='freeze representation module but without gru, or not')
    parser.add_argument('--seed', type=int, default=0, 
                        help='random seed')
    parser.add_argument('--lr', type=float, default=1e-5, 
                        help='learning rate')
    parser.add_argument('--lr-factor', type=float, default=0.8, 
                        help='factor by which the learning rate will be reduced')
    parser.add_argument('--weight-decay', type=float, default=0.0, 
                        help='factor by which the learning rate will be decayed in AdamW, default 0.0')
    parser.add_argument('--lr-min', type=float, default=1e-6, 
                        help='minimum learning rate')
    parser.add_argument('--lr-patience', type=int, default=2, 
                        help='number of epochs with no improvement after which learning rate will be reduced')
    parser.add_argument('--num-steps-update', type=int, default=1, 
                        help='number of steps after which to update the model')
    parser.add_argument('--lr-warmup-steps', type=int, default=2000, 
                        help='number of warmup steps for learning rate')
    parser.add_argument('--batch-size', type=int, default=6, 
                        help='batch size for training')
    parser.add_argument('--ngpus', type=int, default=4, 
                        help='number of gpus to use')
    parser.add_argument('--gpu-id', type=int, default=0, 
                        help='default of gpu to use in processing the dataset')
    parser.add_argument('--num-epochs', type=int, default=10, 
                        help='number of epochs to train for')
    parser.add_argument('--loss-fn', type=str, default='binary_cross_entropy', 
                        help='loss function to use')
    parser.add_argument('--y-weight', type=float, default=1.0, 
                        help='weight of y in loss function')
    parser.add_argument('--data-split-fn', type=str, default='_by_good_batch', 
                        help='function for splitting data')
    parser.add_argument('--contrastive-loss-fn', type=str, default='cosin_contrastive_loss', 
                        help='contrastive loss function to use')
    parser.add_argument('--reset-train-dataloader-each-epoch', type=bool, default=True, 
                        help='whether to reset train dataloader each epoch')
    parser.add_argument('--reset-train-dataloader-each-epoch-seed', type=bool, default=False,
                        help='whether to set the seed of shuffle train dataloader each epoch')
    parser.add_argument('--test-size', type=int, default=None, 
                        help='size of the test set')
    parser.add_argument('--train-size', type=float, default=0.95, 
                        help='fraction of data to use for training')
    parser.add_argument('--val-size', type=float, default=0.05, 
                        help='fraction of data to use for validation')
    
    # log specific
    parser.add_argument('--num-save-epochs', type=int, default=1, 
                        help='number of epochs after which to save the model')
    parser.add_argument('--num-save-batches', type=int, default=1000, 
                        help='number of batches after which to save the model')
    parser.add_argument('--log-dir', type=str, default='/share/vault/Users/gz2294/RESCVE/CHPs.v1.ct/', 
                        help='directory for saving logs')
    
    # script specific
    parser.add_argument('--mode', type=str, default="train_and_test", 
                        help='mode of training')
    parser.add_argument('--re-test', type=bool, default=False, 
                        help='re-test the model or not')
    parser.add_argument('--test-by', type=str, default='epoch_and_batch', 
                        help='test by batch or epoch')
    parser.add_argument('--interpret-by', type=str, default=None, 
                        help='interpret by batch or epoch')
    parser.add_argument('--interpret-step', type=int, default=None, 
                        help='interpret step')
    parser.add_argument('--interpret-epoch', type=int, default=None, 
                        help='interpret epoch')
    parser.add_argument('--out-dir', type=str, default=None, 
                        help='The output directory / file for interpret mode')
    parser.add_argument('--interpret-idxes', type=str, default=None, 
                        help='The index of the data point to interpret, split by comma')
    parser.add_argument('--save-attn', type=bool, default=False, 
                        help='Whether save attention matrix for interpret mode')
    parser.add_argument('--use-ig', type=bool, default=False, 
                        help='Whether to use integrated gradient for interpret mode')
    # aggregate
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])

    return args


def main(args, continue_train=False):
    pl.seed_everything(args.seed, workers=True)

    hparams = vars(args)
    model_class = args.model_class
    # initialize model
    if args.load_model == "None" or args.load_model == "null" or args.load_model is None:
        my_model = create_model(hparams, model_class=model_class)
    else:
        my_model = create_model_and_load(hparams, model_class=model_class)

    # TODO: consider implement early stopping
    # early_stopping = EarlyStopping("val_loss", patience=args.early_stopping_patience)
    dataset_att = {"data_type": args.data_type,
                   "radius": args.radius,
                   "max_neighbors": args.max_num_neighbors,
                   "loop": args.loop,
                   "shuffle": False, 
                   "node_embedding_type": args.node_embedding_type,
                   "graph_type": args.graph_type,
                   "add_plddt": args.add_plddt,
                   "add_conservation": args.add_conservation,
                   "add_position": args.add_position,
                   "add_sidechain": args.add_sidechain,
                   "add_dssp": args.add_dssp,
                   "add_msa": args.add_msa,
                   "loaded_msa": args.loaded_msa,
                   "data_augment": args.data_augment,
                   "score_transfer": args.score_transfer,
                   "alt_type": args.alt_type,
                   "computed_graph": args.computed_graph,
                   "neighbor_type": args.neighbor_type,
                   "max_len": args.max_len,}
    if args.trainer_fn == "PreMode_trainer_noGraph":
        trainer_fn = PreMode_trainer_noGraph
        dataset_extra_args = {"padding": args.batch_size > 1}
    elif args.trainer_fn == "PreMode_trainer":
        trainer_fn = PreMode_trainer
        dataset_extra_args = {}
    elif args.trainer_fn == "PreMode_trainer_SSP":
        trainer_fn = PreMode_trainer_SSP
        dataset_extra_args = {}
    else:
        raise ValueError(f"trainer_fn {args.trainer_fn} not supported")
    if continue_train:
        for i in range(args.num_epochs):
            if os.path.exists(os.path.join(args.log_dir, f"result_dict.epoch.{i}.ddp_rank.0.json")):
                continue
            else:
                break
        if i == args.num_epochs - 1:
            print(f"model for epoch {args.num_epochs} already exists")
            return
        check_point_epoch = i
        print(f"continue training from epoch {check_point_epoch}")
    else:
        check_point_epoch = None
    if args.ngpus > 1:
        datasets = [getattr(data, args.dataset)(
            data_file=f"{args.data_file_train_ddp_prefix}.{rank}.csv",
            gpu_id=rank,
            **dataset_att,
            **dataset_extra_args,
        ) for rank in range(args.ngpus)]
        mp.spawn(data_distributed_parallel_gpu,
                    args=(my_model, args, datasets, trainer_fn, check_point_epoch),
                    nprocs=args.ngpus,
                    join=True)
    else:
        dataset = getattr(data, args.dataset)(
            data_file=args.data_file_train,
            **dataset_att,
            **dataset_extra_args,
        )
        single_thread_gpu(args.gpu_id, my_model, args, dataset, trainer_fn, check_point_epoch)


def _test(args):
    pl.seed_everything(args.seed, workers=True)

    hparams = vars(args)
    model_class = args.model_class
    # initialize model
    my_model = create_model(hparams, model_class=model_class)

    dataset_att = {"data_type": args.data_type,
                   "radius": args.radius,
                   "max_neighbors": args.max_num_neighbors,
                   "loop": args.loop,
                   "shuffle": False, 
                   "node_embedding_type": args.node_embedding_type,
                   "graph_type": args.graph_type,
                   "add_plddt": args.add_plddt,
                   "add_conservation": args.add_conservation,
                   "add_position": args.add_position,
                   "add_sidechain": args.add_sidechain,
                   "add_dssp": args.add_dssp,
                   "add_msa": args.add_msa,
                   "loaded_msa": args.loaded_msa,
                   "data_augment": args.data_augment,
                   "score_transfer": args.score_transfer,
                   "alt_type": args.alt_type,
                   "computed_graph": args.computed_graph,
                   "neighbor_type": args.neighbor_type,
                   "max_len": args.max_len,}
    if args.trainer_fn == "PreMode_trainer_noGraph":
        trainer_fn = PreMode_trainer_noGraph
        dataset_extra_args = {"padding": args.batch_size > 1}
    elif args.trainer_fn == "PreMode_trainer":
        trainer_fn = PreMode_trainer
        dataset_extra_args = {}
    elif args.trainer_fn == "PreMode_trainer_SSP":
        trainer_fn = PreMode_trainer_SSP
        dataset_extra_args = {}
    else:
        raise ValueError(f"trainer_fn {args.trainer_fn} not supported")
    dataset = getattr(data, args.dataset)(
        data_file=args.data_file_test,
        **dataset_att,
        **dataset_extra_args,
    )
    # import ipdb; ipdb.set_trace()
    my_model = my_model.to(f"cuda:{args.gpu_id}")
    my_model.eval()
    trainer = trainer_fn(hparams=args, model=my_model, stage="test",
                         dataset=dataset, device_id=args.gpu_id)
    if "epoch" in args.test_by:
        # test by epoch
        print(f'num_saved_epochs: {args.num_epochs}')
        for epoch in range(1, args.num_epochs + 1):
            if os.path.exists(os.path.join(args.log_dir, f"test_result.epoch.{epoch}.txt")) and not args.re_test:
                print(f"test result for epoch {epoch} already exists")
                continue
            if os.path.exists(os.path.join(args.log_dir, f"result_dict.epoch.{epoch-1}.ddp_rank.0.json")):
                print(f"begin test for epoch {epoch}")
                trainer.load_model(epoch=epoch)
                test_result_dict, test_result_df = _test_one_epoch(trainer)
                with open(os.path.join(args.log_dir, f"test_result.epoch.{epoch}.txt"), "w") as f:
                    f.write(str(test_result_dict))
                test_result_df.to_csv(os.path.join(args.log_dir, f"test_result.epoch.{epoch}.csv"), index=False)
            else:
                print(f"model for epoch {epoch} not exist")

    if "batch" in args.test_by:
        # test by batch steps
        import numpy as np
        train_data_size = subprocess.check_output(f'wc -l {args.data_file_train}', shell=True)
        train_data_size = int(str(train_data_size).split(' ')[0][2:]) - 1
        num_saved_batches = int(np.floor(np.ceil(np.ceil(train_data_size * args.train_size)
                                                / args.ngpus / args.batch_size)
                                        * args.num_epochs / args.num_save_batches) + 1)
        print(f'num_saved_batches: {num_saved_batches}')
        for step in range(args.num_save_batches,
                          num_saved_batches * args.num_save_batches,
                          args.num_save_batches):
            if os.path.exists(os.path.join(args.log_dir, f"test_result.step.{step}.txt")) and not args.re_test:
                print(f"test result for step {step} already exists")
                continue
            if os.path.exists(os.path.join(args.log_dir, f"result_dict.batch.{step}.ddp_rank.0.json")):
                print(f"begin test for step {step}")
                trainer.load_model(step=step)
                test_result_dict, test_result_df = _test_one_epoch(trainer)
                with open(os.path.join(args.log_dir, f"test_result.step.{step}.txt"), "w") as f:
                    f.write(str(test_result_dict))
                test_result_df.to_csv(os.path.join(args.log_dir, f"test_result.step.{step}.csv"), index=False)
            else:
                print(f"model for step {step} not exists")
                continue


def _test_one_epoch(trainer):
    trainer.testing_epoch_begin()
    while True:
        try:
            trainer.test_step()
        except StopIteration:
            break
    test_result_dict, test_result_df = trainer.testing_epoch_end()
    return test_result_dict, test_result_df


def ig_forward(x, trainer, batch, out_idx=0):
    # integrated gradient forward
    # x: (batch_size, num_nodes, x_channels)
    extra_args = batch.to_dict()
    # extra_args actually won't be used in the model
    for a in ('y', 'x', 'x_mask', 'x_alt', 'pos', 'batch',
                'edge_index', 'edge_attr',
                'edge_index_star', 'edge_attr_star',
                'node_vec_attr'):
        if a in extra_args:
            del extra_args[a]
    out, _, _ = trainer.forward(
        x.to(trainer.device), 
        x_mask=batch.x_mask.to(trainer.device),
        x_alt=batch.x_alt.to(trainer.device),
        pos=batch.pos.to(trainer.device),
        batch=batch.batch.to(trainer.device) if "batch" in batch else None,
        edge_index=batch.edge_index.to(trainer.device) if batch.edge_index is not None else None,
        edge_index_star=batch.edge_index_star.to(trainer.device) if "edge_index_star" in batch else None,
        edge_attr=batch.edge_attr.to(trainer.device) if batch.edge_attr is not None else None,
        edge_attr_star=batch.edge_attr_star.to(trainer.device) if "edge_attr_star" in batch else None,
        node_vec_attr=batch.node_vec_attr.to(trainer.device),
        extra_args=extra_args,
        return_attn=False,)
    # out is one-dim tensor
    # out = out.squeeze()
    return out[:, [out_idx, out_idx]]


def interpret(args, idxs=None, epoch=None, step=None):
    # interpret a dataset by attention, only for the data point of idxs in the dataset
    pl.seed_everything(args.seed, workers=True)

    hparams = vars(args)
    model_class = args.model_class
    # initialize model
    if args.load_model == "None" or args.load_model == "null" or args.load_model is None:
        my_model = create_model(hparams, model_class=model_class)
    else:
        my_model = create_model_and_load(hparams, model_class=model_class)

    dataset_att = {"data_type": args.data_type,
                   "radius": args.radius,
                   "max_neighbors": args.max_num_neighbors,
                   "loop": args.loop,
                   "shuffle": False, 
                   "node_embedding_type": args.node_embedding_type,
                   "graph_type": args.graph_type,
                   "add_plddt": args.add_plddt,
                   "add_conservation": args.add_conservation,
                   "add_position": args.add_position,
                   "add_sidechain": args.add_sidechain,
                   "add_dssp": args.add_dssp,
                   "add_msa": args.add_msa,
                   "loaded_msa": args.loaded_msa,
                   "data_augment": args.data_augment,
                   "score_transfer": args.score_transfer,
                   "alt_type": args.alt_type,
                   "computed_graph": args.computed_graph,
                   "neighbor_type": args.neighbor_type,
                   "max_len": args.max_len,}
    if args.trainer_fn == "PreMode_trainer_noGraph":
        trainer_fn = PreMode_trainer_noGraph
        dataset_extra_args = {"padding": args.batch_size > 1}
    elif args.trainer_fn == "PreMode_trainer":
        trainer_fn = PreMode_trainer
        dataset_extra_args = {}
    elif args.trainer_fn == "PreMode_trainer_SSP":
        trainer_fn = PreMode_trainer_SSP
        dataset_extra_args = {}
    else:
        raise ValueError(f"trainer_fn {args.trainer_fn} not supported")
    dataset = getattr(data, args.dataset)(
        data_file=args.data_file_test,
        **dataset_att,
        **dataset_extra_args,
    )
    my_model = my_model.to(f"cuda:{args.gpu_id}")
    my_model.eval()
    trainer = trainer_fn(hparams=args, model=my_model, 
                         stage="test", dataset=dataset, device_id=args.gpu_id)
    
    if epoch is not None:
        trainer.load_model(epoch=epoch)
    elif step is not None:
        trainer.load_model(step=step)
    else:
        if args.interpret_by is None:
            train_data_size = subprocess.check_output(f'wc -l {args.data_file_train}', shell=True)
            train_data_size = int(str(train_data_size).split(' ')[0][2:]) - 1
            num_saved_batches = int(np.floor(np.ceil(np.ceil(train_data_size * args.train_size)
                                                    / args.ngpus / args.batch_size)
                                            * args.num_epochs / args.num_save_batches) + 1)
            if num_saved_batches > args.num_epochs:
                args.interpret_by = "batch"
            else:
                args.interpret_by = "epoch"
        if args.interpret_by == "epoch":
            # find the min val loss epoch
            val_losses = []
            for epoch in range(args.num_epochs):
                val_loss = []
                for rank in range(args.ngpus):
                    with open(f"{args.log_dir}/result_dict.epoch.{epoch}.ddp_rank.{rank}.json", "r") as f:
                        result_dict = json.load(f)
                        val_loss.append(result_dict["val_loss"])
                val_losses.append(np.mean(val_loss))
            min_val_loss_epoch = np.argmin(val_losses) + 1
            trainer.load_model(epoch=min_val_loss_epoch)
        elif args.interpret_by == "batch":
            # find the min val loss batch
            val_losses = []
            train_data_size = subprocess.check_output(f'wc -l {args.data_file_train}', shell=True)
            train_data_size = int(str(train_data_size).split(' ')[0][2:]) - 1
            num_saved_batches = int(np.floor(np.ceil(np.ceil(train_data_size * args.train_size)
                                                    / args.ngpus / args.batch_size)
                                            * args.num_epochs / args.num_save_batches) + 1)
            print(f'num_saved_batches: {num_saved_batches}')
            steps = list(range(args.num_save_batches,
                               num_saved_batches * args.num_save_batches,
                               args.num_save_batches))
            for step in steps:
                val_loss = []
                for rank in range(args.ngpus):
                    if (os.path.exists(f"{args.log_dir}/result_dict.batch.{step}.ddp_rank.{rank}.json")):
                        with open(f"{args.log_dir}/result_dict.batch.{step}.ddp_rank.{rank}.json", "r") as f:
                            result_dict = json.load(f)
                            val_loss.append(result_dict["val_loss"])
                val_losses.append(np.mean(val_loss))
            min_val_loss_batch = steps[np.argmin(np.array(val_losses)[~np.isnan(val_losses)])]
            print(f"min_val_loss_batch: {min_val_loss_batch}")
            trainer.load_model(step=min_val_loss_batch)
        elif args.interpret_by == "both":
            # find the min val loss epoch
            val_losses = []
            for epoch in range(args.num_epochs):
                val_loss = []
                for rank in range(args.ngpus):
                    with open(f"{args.log_dir}/result_dict.epoch.{epoch}.ddp_rank.{rank}.json", "r") as f:
                        result_dict = json.load(f)
                        val_loss.append(result_dict["val_loss"])
                val_losses.append(np.mean(val_loss))
            min_val_loss_epoch = np.argmin(val_losses) + 1
            min_epoch_loss = np.min(val_losses)
            # find the min val loss batch
            val_losses = []
            train_data_size = subprocess.check_output(f'wc -l {args.data_file_train}', shell=True)
            train_data_size = int(str(train_data_size).split(' ')[0][2:]) - 1
            num_saved_batches = int(np.floor(np.ceil(np.ceil(train_data_size * args.train_size)
                                                    / args.ngpus / args.batch_size)
                                            * args.num_epochs / args.num_save_batches) + 1)
            print(f'num_saved_batches: {num_saved_batches}')
            steps = list(range(args.num_save_batches,
                               num_saved_batches * args.num_save_batches,
                               args.num_save_batches))
            for step in steps:
                val_loss = []
                for rank in range(args.ngpus):
                    if (os.path.exists(f"{args.log_dir}/result_dict.batch.{step}.ddp_rank.{rank}.json")):
                        with open(f"{args.log_dir}/result_dict.batch.{step}.ddp_rank.{rank}.json", "r") as f:
                            result_dict = json.load(f)
                            val_loss.append(result_dict["val_loss"])
                val_losses.append(np.mean(val_loss))
            min_val_loss_batch = steps[np.argmin(np.array(val_losses)[~np.isnan(val_losses)])]
            min_batch_loss = np.min(val_losses)
            if min_epoch_loss < min_batch_loss:
                print(f"min_val_loss_epoch: {min_val_loss_epoch}")
                trainer.load_model(epoch=min_val_loss_epoch)
            else:
                print(f"min_val_loss_batch: {min_val_loss_batch}")
                trainer.load_model(step=min_val_loss_batch)
    if idxs is None:
        idxs = list(range(len(dataset)))
        out_index = dataset.data.index
    else:
        out_index = dataset.data.index[idxs]
    # interpret one sequence at a time
    x_embeds = []
    ys = []
    if args.use_ig:
        ig = IntegratedGradients(ig_forward)
    for idx in idxs:
        batch = dataset.get(idx)
        _, y, x_embed, attn_weight_layers = trainer.interpret_step(batch)
        if args.use_ig:
            x = batch.x.requires_grad_(True)
            x_gradients = []
            for i in range(batch.y.shape[1]):
                x_gradient = ig.attribute(
                    x, target=batch.y[:, [i]].int(),
                    additional_forward_args=(trainer, batch, i),
                    internal_batch_size=1,
                    )
                x_gradients.append(x_gradient.detach().cpu().numpy())
        else:
            x_gradients = None
        x_embeds.append(x_embed.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        # print(f"loss for data point idx {idx}: {loss}")
        # print(f"logits for data point idx {idx}: {y}")
        # print(f"save attention weights for data point idx {idx}")
        if args.save_attn:
            if args.out_dir is None:
                save_file_name = f"{args.log_dir}/attn_weights.{idx}.pkl"
            else:
                if not args.out_dir.endswith(".csv"):
                    save_file_name = f"{args.out_dir}/attn_weights.{idx}.pkl"
                else:
                    save_file_name = f"{args.out_dir.replace('.csv', '')}.{idx}.pkl"
            with open(save_file_name, "wb") as f:
                pickle.dump(([a.detach().cpu().numpy() for a in attn_weight_layers], 
                                batch.edge_index_star.detach().cpu().numpy(),
                                dataset.data.iloc[idx], 
                                (dataset.mutations[idx].seq_start_orig, dataset.mutations[idx].seq_end_orig,
                                 dataset.mutations[idx].seq_start, dataset.mutations[idx].seq_end), 
                                y.detach().cpu().numpy(),
                                x_gradients), f)
    x_embeds = np.concatenate(x_embeds, axis=0)
    ys = pd.DataFrame(np.concatenate(ys, axis=0), index=out_index)
    if ys.shape[1] == 1:
        ys.columns = ["logits"]
    else:
        ys.columns=[f'logits.{i}' for i in range(ys.shape[1])]
    x_embed_df = pd.concat([dataset.data.loc[out_index],
                            ys,
                            pd.DataFrame(x_embeds, index=out_index)],
                           axis=1)
    if args.out_dir is None:
        args.out_dir = args.log_dir
    if not os.path.exists(args.out_dir) and not args.out_dir.endswith(".csv"):
        os.makedirs(args.out_dir)
    if args.out_dir.endswith(".csv"):
        x_embed_df.to_csv(args.out_dir, index=False)
    else:
        x_embed_df.to_csv(f"{args.out_dir}/x_embeds.csv", index=False)


def interpret_datapoint(idx):
    with open(f'/share/vault/Users/gz2294/RESCVE/CHPs.ContactsOnly.1280Dim.Star.SoftMax.pLDDT/attn_weights.{idx}.pkl',
              'rb') as f:
        tmp = pickle.load(f)
    attn_weight_layers, batch, datapoint, point_mutation, loss, y = tmp
    attn_weight_1 = attn_weight_layers[0].detach().cpu().numpy()
    tmp = np.split(attn_weight_1, [2, attn_weight_1.shape[1] + 1], axis=1)
    edge_idx_1 = tmp[0]
    attn_weight_1 = tmp[1]
    attn_weight_2 = attn_weight_layers[1].detach().cpu().numpy()
    tmp = np.split(attn_weight_2, [2, attn_weight_2.shape[1] + 1], axis=1)
    edge_idx_2 = tmp[0]
    attn_weight_2 = tmp[1]
    # find the highest attention in each attention head
    max_layer_1 = np.zeros((0, attn_weight_1.shape[1]))
    min_layer_1 = np.zeros((0, attn_weight_1.shape[1]))
    max_layer_2 = np.zeros((0, attn_weight_2.shape[1]))
    min_layer_2 = np.zeros((0, attn_weight_2.shape[1]))
    target_node_layer_1 = np.unique(edge_idx_1[:, 1])
    target_node_layer_2 = np.unique(edge_idx_2[:, 1])
    for target_node in target_node_layer_1:
        target_node_attn = attn_weight_1[edge_idx_1[:, 1] == target_node]
        max_start_node = [edge_idx_1[np.argmax(target_node_attn[:, i]), 0] + point_mutation.seq_start
                          for i in range(target_node_attn.shape[1])]
        min_start_node = [edge_idx_1[np.argmin(target_node_attn[:, i]), 0] + point_mutation.seq_start
                          for i in range(target_node_attn.shape[1])]
        max_layer_1 = np.vstack((max_layer_1, np.array(max_start_node)))
        min_layer_1 = np.vstack((min_layer_1, np.array(min_start_node)))
    for target_node in target_node_layer_2:
        target_node_attn = attn_weight_2[edge_idx_2[:, 1] == target_node]
        max_start_node = [edge_idx_2[np.argmax(target_node_attn[:, i]), 0] + point_mutation.seq_start
                          for i in range(target_node_attn.shape[1])]
        min_start_node = [edge_idx_2[np.argmin(target_node_attn[:, i]), 0] + point_mutation.seq_start
                          for i in range(target_node_attn.shape[1])]
        max_layer_2 = np.vstack((max_layer_2, np.array(max_start_node)))
        min_layer_2 = np.vstack((min_layer_2, np.array(min_start_node)))
    max_layer_1 = pd.DataFrame(max_layer_1, columns=[f"attn_head_{i}" for i in range(attn_weight_1.shape[1])],
                               index=target_node_layer_1 + point_mutation.seq_start)
    min_layer_1 = pd.DataFrame(min_layer_1, columns=[f"attn_head_{i}" for i in range(attn_weight_1.shape[1])],
                               index=target_node_layer_1 + point_mutation.seq_start)
    max_layer_2 = pd.DataFrame(max_layer_2, columns=[f"attn_head_{i}" for i in range(attn_weight_2.shape[1])],
                               index=target_node_layer_2 + point_mutation.seq_start)
    min_layer_2 = pd.DataFrame(min_layer_2, columns=[f"attn_head_{i}" for i in range(attn_weight_2.shape[1])],
                               index=target_node_layer_2 + point_mutation.seq_start)
    return max_layer_1, min_layer_1, max_layer_2, min_layer_2, batch, datapoint, loss, y


def test_scalar_invariance(args):
    import torch
    hparams = vars(args)
    torch.manual_seed(1234)
    rotate = torch.tensor(
        [
            [0.9886788, -0.1102370, 0.1017945],
            [0.1363630, 0.9431761, -0.3030248],
            [-0.0626055, 0.3134752, 0.9475304],
        ]
    )
    dataset_att = {"data_type": args.data_type,
                   "radius": args.radius,
                   "max_neighbors": args.max_num_neighbors,
                   "loop": args.loop,
                   "shuffle": False, 
                   "node_embedding_type": args.node_embedding_type,
                   "graph_type": args.graph_type,
                   "add_plddt": args.add_plddt,
                   "add_dssp": args.add_dssp,
                   "add_conservation": args.add_conservation,
                   "add_position": args.add_position,
                   "computed_graph": args.computed_graph,
                   "neighbor_type": args.neighbor_type,
                   "max_len": args.max_len,}
    datasets = getattr(data, args.dataset)(
            data_file=f"{args.data_file_train}",
            gpu_id=0,
            **dataset_att,
        )
    data_point = datasets.get(0)
    model = create_model(hparams, model_class=args.model_class)
    # get data points
    pos = torch.randn(data_point.x.shape[0], 3)
    node_vec_attr = torch.randn(data_point.x.shape[0], 3, 4)

    y = model(data_point.x, data_point.x_mask, data_point.x_alt, pos, 
              data_point.edge_index, data_point.edge_index_star, 
              data_point.edge_attr, data_point.edge_attr_star, 
              node_vec_attr, data_point.batch, )[0]
    y_rot = model(data_point.x, data_point.x_mask, data_point.x_alt, pos @ rotate, 
                  data_point.edge_index, data_point.edge_index_star, 
                  data_point.edge_attr, data_point.edge_attr_star, 
                  (node_vec_attr.permute(0, 2, 1) @ rotate).permute(0, 2, 1), data_point.batch)[0]
    torch.testing.assert_allclose(y, y_rot)


if __name__ == "__main__":
    # main_pl()
    _args = get_args()
    if _args.mode == "train":
        main(_args)
    elif _args.mode == "continue_train":
        main(_args, continue_train=True)
    elif _args.mode == "test":
        _test(_args)
    elif _args.mode == "train_and_test":
        main(_args)
        _args.re_test = True
        _test(_args)
    elif _args.mode == "interpret":
        if _args.interpret_idxes is not None:
            _args.interpret_idxes = [int(i) for i in _args.interpret_idxes.split(",")]
        interpret(_args, idxs=_args.interpret_idxes, step=_args.interpret_step, epoch=_args.interpret_epoch)
    elif _args.mode == "test_equivariancy":
        test_scalar_invariance(_args)
