import argparse
import json
import os
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
    parser.add_argument('--add-position', type=bool, default=False,
                        help='Whether to add positional wise encoding or not')
    parser.add_argument('--computed-graph', type=bool, default=True,
                        help='Whether to use computed graph or not')
    parser.add_argument('--neighbor-type', type=str, default='KNN',
                        help='The type of neighbor selection. Choose from KNN or radius')
    parser.add_argument('--max-len', type=int, default=2251,
                        help='Maximum length of input sequences')
    parser.add_argument('--radius', type=float, default=50,
                        help='Radius of AA to be selected')

    # model specific
    parser.add_argument('--load-model', type=str, default=None,
                        help='Restart training using a model checkpoint')
    parser.add_argument('--partial-load-model', type=bool, default=False,
                        help='Partial load model, particullay from maskpredict model using a model checkpoint')
    parser.add_argument('--model-class', type=str, default=None, choices=model.__all__,
                        help='Which model to use')
    parser.add_argument('--model', type=str, default=None,
                        help='Which representation model to use')
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
   
    # training specific
    parser.add_argument('--trainer-fn', type=str, default='PreMode_trainer', 
                        help='trainer function')
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
    parser.add_argument('--out-dir', type=str, default=None, 
                        help='The output directory / file for interpret mode')
    
    # aggregate
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])

    return args


def main(args):
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
    if args.ngpus > 1:
        datasets = [getattr(data, args.dataset)(
            data_file=f"{args.data_file_train_ddp_prefix}.{rank}.csv",
            gpu_id=rank,
            **dataset_att,
            **dataset_extra_args,
        ) for rank in range(args.ngpus)]
        mp.spawn(data_distributed_parallel_gpu,
                    args=(my_model, args, datasets, trainer_fn),
                    nprocs=args.ngpus,
                    join=True)
    else:
        dataset = getattr(data, args.dataset)(
            data_file=args.data_file_train,
            **dataset_att,
            **dataset_extra_args,
        )
        single_thread_gpu(args.gpu_id, my_model, args, dataset, trainer_fn)


# TODO: to be implemented
def main_continue(args):
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
    if args.ngpus > 1:
        datasets = [getattr(data, args.dataset)(
            data_file=f"{args.data_file_train_ddp_prefix}.{rank}.csv",
            gpu_id=rank,
            **dataset_att,
            **dataset_extra_args,
        ) for rank in range(args.ngpus)]
        mp.spawn(data_distributed_parallel_gpu,
                    args=(my_model, args, datasets, trainer_fn),
                    nprocs=args.ngpus,
                    join=True)
    else:
        dataset = getattr(data, args.dataset)(
            data_file=args.data_file_train,
            **dataset_att,
            **dataset_extra_args,
        )
        single_thread_gpu(args.gpu_id, my_model, args, dataset, trainer_fn)


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
                   "gpu_id": args.gpu_id,
                   "shuffle": False, 
                   "node_embedding_type": args.node_embedding_type,
                   "graph_type": args.graph_type,
                   "add_plddt": args.add_plddt,
                   "add_conservation": args.add_conservation,
                   "add_position": args.add_position,
                   "computed_graph": args.computed_graph,
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
            print(f"begin test for epoch {epoch}")
            trainer.load_model(epoch=epoch)
            test_result_dict, test_result_df = _test_one_epoch(trainer)
            with open(os.path.join(args.log_dir, f"test_result.epoch.{epoch}.txt"), "w") as f:
                f.write(str(test_result_dict))
            test_result_df.to_csv(os.path.join(args.log_dir, f"test_result.epoch.{epoch}.csv"), index=False)

    if "batch" in args.test_by:
        # test by batch steps
        import numpy as np
        train_dataset = pd.read_csv(args.data_file_train)
        num_saved_batches = int(np.floor(np.ceil(np.ceil(len(train_dataset) * args.train_size)
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
                   "gpu_id": args.gpu_id,
                   "shuffle": False, 
                   "node_embedding_type": args.node_embedding_type,
                   "graph_type": args.graph_type,
                   "add_plddt": args.add_plddt,
                   "add_conservation": args.add_conservation,
                   "add_position": args.add_position,
                   "computed_graph": args.computed_graph,
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

    if idxs is None:
        idxs = list(range(len(dataset)))
    # interpret one sequence at a time
    x_embeds = []
    ys = []
    for idx in idxs:
        batch = dataset.get(idx)
        loss, y, x_embed, attn_weight_layers = trainer.interpret_step(batch)
        x_embeds.append(x_embed.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        print(f"loss for data point idx {idx}: {loss}")
        print(f"logits for data point idx {idx}: {y}")
        print(f"save attention weights for data point idx {idx}")
        with open(f"{args.log_dir}/attn_weights.{idx}.pkl", "wb") as f:
            pickle.dump((attn_weight_layers, batch, dataset.data.iloc[idx], dataset.mutations[idx], loss, y), f)
    x_embeds = np.concatenate(x_embeds, axis=0)
    ys = np.concatenate(ys, axis=0)
    x_embed_df = pd.concat([dataset.data,
                            pd.DataFrame(ys, index=dataset.data.index, columns=["logits"]),
                            pd.DataFrame(x_embeds, index=dataset.data.index)],
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


if __name__ == "__main__":
    # main_pl()
    _args = get_args()
    if _args.mode == "train":
        main(_args)
    elif _args.mode == "continue_train":
        main_continue(_args)
    elif _args.mode == "test":
        _test(_args)
    elif _args.mode == "train_and_test":
        main(_args)
        _args.re_test = True
        _test(_args)
    elif _args.mode == "interpret":
        interpret(_args)
