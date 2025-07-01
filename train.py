import argparse
import json
import os
import subprocess
import pickle

import numpy as np
import pandas as pd
import random
import torch
import torch.multiprocessing as mp
from types import SimpleNamespace as sn
import data
from model import model
from model.model import create_model, create_model_and_load
from model.trainer import data_distributed_parallel_gpu, PreMode_trainer, single_thread_gpu, ray_tune, single_thread_gpu_4_fold, multiple_thread_gpu_4_fold
from utils.configs import save_argparse, LoadFromFile
from captum.attr import IntegratedGradients
from functools import partial


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
    parser.add_argument('--convert-to-onesite', type=bool, default=False,
                        help='Convert the data to one site-date or not, only works for FullGraph dataset')
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
    parser.add_argument('--scale-plddt', type=bool, default=False,
                        help='Whether to scale plddt or not')
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
    parser.add_argument('--zero-msa', type=bool, default=False,
                        help='Whether to make msa zero')
    parser.add_argument('--add-msa-contacts', type=bool, default=True,
                        help='Whether to add msa contacts to features or not')
    parser.add_argument('--add-confidence', type=bool, default=False,
                        help='Whether to add af2 predicted confidence or not')
    parser.add_argument('--add-ptm', type=bool, default=False,
                        help='Whether to add post translational modification information or not')
    parser.add_argument('--add-af2-single', type=bool, default=False,
                        help='Whether to add alphafold single representation or not')
    parser.add_argument('--add-af2-pairwise', type=bool, default=False,
                        help='Whether to add alphafold pairwise representation or not')
    parser.add_argument('--loaded-af2-single', type=bool, default=False,
                        help='Whether to load af2 single representation or not')
    parser.add_argument('--loaded-af2-pairwise', type=bool, default=False,
                        help='Whether to load af2 pairwise representation or not')
    parser.add_argument('--loaded-confidence', type=bool, default=False,
                        help='Whether to load af2 predicted confidence or not')
    parser.add_argument('--loaded-msa', type=bool, default=False,
                        help='Whether to preload msa to features or not')
    parser.add_argument('--loaded-esm', type=bool, default=False,
                        help='Whether to preload esm to features or not')
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
    parser.add_argument('--use-lmdb', type=bool, default=False,
                        help='Whether to use preloaded lmdb')
    
    # model specific
    parser.add_argument('--load-model', type=str, default=None,
                        help='Restart training using a model checkpoint')
    parser.add_argument('--partial-load-model', type=bool, default=False,
                        help='Partial load model, particullay from maskpredict model using a model checkpoint')
    parser.add_argument('--use-output-head', type=bool, default=False,
                        help='Use output head or not')
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
    parser.add_argument('--output-dim-1', type=int, default=1,
                        help='The first dimension of output model, only used in regression-classification')
    parser.add_argument('--output-dim-2', type=int, default=1,
                        help='The second dimension of output model, only used in regression-classification')
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
    parser.add_argument('--seed-with-pl', type=bool, default=False, 
                        help='Initialize with pytorch lightning seed')
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
    parser.add_argument('--hp-tune', type=bool, default=False, 
                        help='Whether use hyperparameter tuning or not')
    parser.add_argument('--adaptive-rounds', type=int, default=6, 
                        help='active learning rounds')
    parser.add_argument('--init-fn', type=str, default=None, 
                        help='Initialization function for output model')
    
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
    parser.add_argument('--use-jacob', type=bool, default=False, 
                        help='Whether to use jacobian at the output reduce layer for interpret mode')
    # aggregate
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    if "train" in args.mode:
        save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])

    return args


def main(args, continue_train=False, four_fold=False):
    if args.seed_with_pl:
        import pytorch_lightning as pl
        pl.seed_everything(args.seed)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

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
                   "scale_plddt": args.scale_plddt,
                   "add_conservation": args.add_conservation,
                   "add_position": args.add_position,
                   "add_sidechain": args.add_sidechain,
                   "add_dssp": args.add_dssp,
                   "add_msa": args.add_msa,
                   "add_confidence": args.add_confidence,
                   "add_msa_contacts": args.add_msa_contacts,
                   "add_ptm": args.add_ptm,
                   "add_af2_single": args.add_af2_single,
                   "add_af2_pairwise": args.add_af2_pairwise,
                   "loaded_af2_single": args.loaded_af2_single,
                   "loaded_af2_pairwise": args.loaded_af2_pairwise,
                   "loaded_msa": args.loaded_msa,
                   "loaded_esm": args.loaded_esm,
                   "loaded_confidence": args.loaded_confidence,
                   "data_augment": args.data_augment,
                   "score_transfer": args.score_transfer,
                   "alt_type": args.alt_type,
                   "computed_graph": args.computed_graph,
                   "neighbor_type": args.neighbor_type,
                   "max_len": args.max_len,
                   "use_lmdb": args.use_lmdb,}
    if "Onesite" in args.dataset:
        dataset_att['convert_to_onesite'] = args.convert_to_onesite
    if args.trainer_fn == "PreMode_trainer":
        trainer_fn = PreMode_trainer
        dataset_extra_args = {}
    else:
        raise ValueError(f"trainer_fn {args.trainer_fn} not supported")
    if continue_train:
        for i in range(args.num_epochs):
            if os.path.exists(os.path.join(args.log_dir, f"result_dict.epoch.{i}.ddp_rank.0.json")) and os.path.exists(os.path.join(args.log_dir, f"model.epoch.{i+1}.pt")):
                continue
            else:
                break
        if i == args.num_epochs:
            print(f"model for epoch {args.num_epochs} already exists")
            return
        if i == 0:
            check_point_epoch = None
        else:
            check_point_epoch = i
        print(f"continue training from epoch {check_point_epoch}")
    else:
        check_point_epoch = None
    if args.ngpus > 1:
        # assert four_fold is False, "fold 4 is not supported in distributed training"
        if four_fold:
            dataset = getattr(data, args.dataset)(
                data_file=args.data_file_train,
                **dataset_att,
                **dataset_extra_args,
            )
            multiple_thread_gpu_4_fold(args.gpu_id, my_model, args, dataset, trainer_fn, check_point_epoch)
        else:
            mp.spawn(data_distributed_parallel_gpu,
                        args=(my_model, args, dataset_att, dataset_extra_args, trainer_fn, check_point_epoch),
                        nprocs=args.ngpus,
                        join=True)
    else:
        dataset = getattr(data, args.dataset)(
            data_file=args.data_file_train,
            **dataset_att,
            **dataset_extra_args,
        )
        if four_fold:
            single_thread_gpu_4_fold(args.gpu_id, my_model, args, dataset, trainer_fn, check_point_epoch)
        else:
            single_thread_gpu(args.gpu_id, my_model, args, dataset, trainer_fn, check_point_epoch)


def adaptive_main(args):
    if args.seed_with_pl:        
        import pytorch_lightning as pl
        pl.seed_everything(args.seed)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

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
                   "scale_plddt": args.scale_plddt,
                   "add_conservation": args.add_conservation,
                   "add_position": args.add_position,
                   "add_sidechain": args.add_sidechain,
                   "add_dssp": args.add_dssp,
                   "add_msa": args.add_msa,
                   "add_confidence": args.add_confidence,
                   "add_msa_contacts": args.add_msa_contacts,
                   "add_ptm": args.add_ptm,
                   "add_af2_single": args.add_af2_single,
                   "add_af2_pairwise": args.add_af2_pairwise,
                   "loaded_af2_single": args.loaded_af2_single,
                   "loaded_af2_pairwise": args.loaded_af2_pairwise,
                   "loaded_msa": args.loaded_msa,
                   "loaded_esm": args.loaded_esm,
                   "loaded_confidence": args.loaded_confidence,
                   "data_augment": args.data_augment,
                   "score_transfer": args.score_transfer,
                   "alt_type": args.alt_type,
                   "computed_graph": args.computed_graph,
                   "neighbor_type": args.neighbor_type,
                   "max_len": args.max_len,}
    if "Onesite" in args.dataset:
        dataset_att['convert_to_onesite'] = args.convert_to_onesite
    if args.trainer_fn == "PreMode_trainer":
        trainer_fn = PreMode_trainer
        dataset_extra_args = {}
    else:
        raise ValueError(f"trainer_fn {args.trainer_fn} not supported")
    # read in the data_file
    try:
        data_file = pd.read_csv(args.data_file_train, index_col=0, low_memory=False)
        data_file_test = pd.read_csv(args.data_file_test, index_col=0, low_memory=False)
    except UnicodeDecodeError:
        data_file = pd.read_csv(args.data_file_train, index_col=0, encoding='ISO-8859-1')
        data_file_test = pd.read_csv(args.data_file_test, index_col=0, encoding='ISO-8859-1')
    # if split fn is by_anno, start from beginning
    if args.data_split_fn == "_by_anno":
        # pick the data file 'split' column with 'train' or 'val' value
        data_file_train = data_file[data_file['split'].isin(['train', 'val'])]
        size_each_round = data_file_train['split'].value_counts()['train']
        data_file_candidate_train = data_file[~data_file['split'].isin(['train', 'val'])]
    else:
        # randomly select 10% of the data as train set
        data_file_train = data_file.sample(frac=0.1, random_state=args.seed)
        size_each_round = data_file_train.shape[0]
        data_file_candidate_train = data_file.drop(data_file_train.index)
    # save the train and test data file
    # make sure the log_dir exists
    base_log_dir = args.log_dir
    os.makedirs(base_log_dir, exist_ok=True)
    
    if args.data_split_fn == "_by_anno":
        # val_size_each_round = sum(data_file_train['split'] == 'val')
        val_size_each_round = 0
    else:
        val_size_each_round = int(size_each_round * args.val_size)
    for i in range(args.adaptive_rounds):
        # we only do 6 rounds of adaptive learning, suppose the init amount is 10%
        # for each round, we train the model on the train set, and test on the test set
        # then we select the top 10% of the test set as the new train set
        # and the rest as the new test set
        # set the log_dir for each round
        if os.path.exists(os.path.join(base_log_dir, f"candidate.training.round.{i}.csv")):
            print(f"round {i} already exists, skip")
            x_embed_df = pd.read_csv(os.path.join(base_log_dir, f"candidate.training.round.{i}.csv"), index_col=0)
            # read the results from the previous round
            data_file_train = pd.read_csv(os.path.join(base_log_dir, f"data_file_train.round.{i}.csv"), index_col=0)
            data_file_candidate_train = pd.read_csv(os.path.join(base_log_dir, f"data_file_candidate.round.{i}.csv"), index_col=0)
        else:
            args.log_dir = os.path.join(base_log_dir, f"round_{i}")
            os.makedirs(args.log_dir, exist_ok=True)
            data_file_train.to_csv(os.path.join(base_log_dir, f"data_file_train.round.{i}.csv"))
            data_file_candidate_train.to_csv(os.path.join(base_log_dir, f"data_file_candidate.round.{i}.csv"))
            dataset = getattr(data, args.dataset)(
                data_file=data_file_train,
                **dataset_att,
                **dataset_extra_args,
            )
            # check if model check point exists
            for e in range(args.num_epochs):
                if os.path.exists(os.path.join(args.log_dir, f"result_dict.epoch.{e}.ddp_rank.0.json")) and os.path.exists(os.path.join(args.log_dir, f"model.epoch.{e+1}.pt")):
                    continue
                else:
                    break
            if e == args.num_epochs - 1:
                print(f"model for epoch {args.num_epochs} already exists")
            else:
                if e == 0:
                    check_point_epoch = None
                else:
                    check_point_epoch = e
                print(f"continue training from epoch {check_point_epoch}")
                single_thread_gpu(args.gpu_id, my_model, args, dataset, trainer_fn, check_point_epoch)
            # test the model on the test set first
            dataset = getattr(data, args.dataset)(
                data_file=data_file_test,
                **dataset_att,
                **dataset_extra_args,
            )
            # add interpret mode here, add some args
            args.interpret_by = "both"
            x_embed_df = interpret_core(args, dataset, idxs=None, epoch=None, step=None)
            x_embed_df.to_csv(os.path.join(base_log_dir, f"testing.round.{i}.csv"))
            # test the model on the candidate train set
            # test the model on the test set first
            dataset = getattr(data, args.dataset)(
                data_file=data_file_candidate_train,
                **dataset_att,
                **dataset_extra_args,
            )
            x_embed_df = interpret_core(args, dataset, idxs=None, epoch=None, step=None)
            x_embed_df.to_csv(os.path.join(base_log_dir, f"candidate.training.round.{i}.csv"))
        # for columns starts with logits_var, we first calculate the rank percentiles
        x_embed_df = x_embed_df.apply(lambda x: x.rank(pct=True) if isinstance(x.name, str) and x.name.startswith("logits_var") else x)
        # then we calculate the row-wise mean of the rank percentiles
        x_embed_df_rank = x_embed_df.loc[:, x_embed_df.columns.str.startswith("logits_var")].mean(axis=1)
        # then we select the top `size_each_round` of the test set as the new train set
        # and the rest as the new test set
        # if the data_split_fn is by_anno, we need to make sure the new train set "split" column is "train"
        to_pick_threshold = 1 - size_each_round / data_file_candidate_train.shape[0]
        if args.data_split_fn == "_by_anno":
            data_file_train_new = data_file_candidate_train[x_embed_df_rank > to_pick_threshold]
            data_file_train_new['split'] = 'train'
            # randomly select certain amount of data from the candidate train set as new validation set
            # data_file_train_new['split'][np.random.choice(data_file_train_new.index, val_size_each_round)] = 'val'
            data_file_candidate_train = data_file_candidate_train[x_embed_df_rank <= to_pick_threshold]
        else:
            data_file_train_new = data_file_candidate_train[x_embed_df_rank > to_pick_threshold]
            data_file_candidate_train = data_file_candidate_train[x_embed_df_rank <= to_pick_threshold]
        # we need to drop columns in data_file_train_new that are not in data_file_train
        data_file_train_new = data_file_train_new[data_file_train.columns]
        # then we concat the new train set to the old train set
        data_file_train = pd.concat([data_file_train, data_file_train_new], axis=0)


def hp_tune(args):
    # hyperparameter tuning, too expensive thus only do for some genes
    # import ray has to be here, otherwise will cause error for other functions
    import ray
    from ray import tune
    from ray.tune.analysis import ExperimentAnalysis
    from ray.tune.schedulers import ASHAScheduler
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # transfer args to dict
    hparams = vars(args)
    # early_stopping = EarlyStopping("val_loss", patience=args.early_stopping_patience)
    dataset_att = {"data_type": args.data_type,
                   "radius": args.radius,
                   "max_neighbors": args.max_num_neighbors,
                   "loop": args.loop,
                   "shuffle": False, 
                   "node_embedding_type": args.node_embedding_type,
                   "graph_type": args.graph_type,
                   "add_plddt": args.add_plddt,
                   "scale_plddt": args.scale_plddt,
                   "add_conservation": args.add_conservation,
                   "add_position": args.add_position,
                   "add_sidechain": args.add_sidechain,
                   "add_dssp": args.add_dssp,
                   "add_msa": args.add_msa,
                   "add_confidence": args.add_confidence,
                   "add_msa_contacts": args.add_msa_contacts,
                   "add_ptm": args.add_ptm,
                   "add_af2_single": args.add_af2_single,
                   "add_af2_pairwise": args.add_af2_pairwise,
                   "loaded_af2_single": args.loaded_af2_single,
                   "loaded_af2_pairwise": args.loaded_af2_pairwise,
                   "loaded_msa": args.loaded_msa,
                   "loaded_esm": args.loaded_esm,
                   "loaded_confidence": args.loaded_confidence,
                   "data_augment": args.data_augment,
                   "score_transfer": args.score_transfer,
                   "alt_type": args.alt_type,
                   "computed_graph": args.computed_graph,
                   "neighbor_type": args.neighbor_type,
                   "max_len": args.max_len,}
    dataset = getattr(data, args.dataset)(
            data_file=args.data_file_train,
            **dataset_att,
        )
    # transform args to dict, which is already done, named hparams
    # set up ray tune configs
    config = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "lr_min": tune.loguniform(1e-8, 1e-5),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "drop_out": tune.uniform(0.0, 0.9),
        "num_save_batches": tune.choice([50, 100, 200, 400]),
    }
    # add the rest of hparams to config
    for k, v in hparams.items():
        if k not in config:
            config[k] = v
    # add a param in config to indicate the trainer fn whether to use tune or not
    config["hp_tune"] = True
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=args.num_epochs,
        grace_period=3,
        reduction_factor=2,
    )
    ray.init(num_cpus=20, num_gpus=torch.cuda.device_count())
    result: ExperimentAnalysis = tune.run(
        partial(ray_tune, dataset=dataset), 
        resources_per_trial={"cpu": 4, "gpu": 1},
        config=config,
        num_samples=25,
        scheduler=scheduler,
        verbose=1,
        local_dir=args.log_dir,
        checkpoint_at_end=True,
    )
    best_trial = result.get_best_trial("loss", "min", "all")
    best_trial_id = best_trial.trial_id
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial id: {best_trial_id}")
    # create symbolic link from the best trial to the log dir
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    # initialize model
    if args.load_model == "None" or args.load_model == "null" or args.load_model is None:
        my_model = create_model(best_trial.config, model_class=args.model_class)
    else:
        my_model = create_model_and_load(best_trial.config, model_class=args.model_class)
    if args.dataset.startswith("FullGraph"):
        my_model = torch.compile(my_model.to(f"cuda:{args.gpu_id}"))
    else:
        my_model = my_model.to(f"cuda:{args.gpu_id}")
    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()
    my_model.load_state_dict(best_checkpoint_data["net_state_dict"])
    torch.save(my_model.state_dict(), f'{args.log_dir}/model.hp_tune.pt')


def _test(args):
    if args.seed_with_pl:
        import pytorch_lightning as pl
        pl.seed_everything(args.seed)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

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
                   "scale_plddt": args.scale_plddt,
                   "add_conservation": args.add_conservation,
                   "add_position": args.add_position,
                   "add_sidechain": args.add_sidechain,
                   "add_dssp": args.add_dssp,
                   "add_msa": args.add_msa,
                   "add_confidence": args.add_confidence,
                   "add_msa_contacts": args.add_msa_contacts,
                   "add_ptm": args.add_ptm,
                   "add_af2_single": args.add_af2_single,
                   "add_af2_pairwise": args.add_af2_pairwise,
                   "loaded_af2_single": args.loaded_af2_single,
                   "loaded_af2_pairwise": args.loaded_af2_pairwise,
                   "loaded_msa": args.loaded_msa,
                   "loaded_esm": args.loaded_esm,
                   "loaded_confidence": args.loaded_confidence,
                   "data_augment": args.data_augment,
                   "score_transfer": args.score_transfer,
                   "alt_type": args.alt_type,
                   "computed_graph": args.computed_graph,
                   "neighbor_type": args.neighbor_type,
                   "max_len": args.max_len,}
    if args.trainer_fn == "PreMode_trainer":
        trainer_fn = PreMode_trainer
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
    # clean up the data sets
    trainer.dataset.clean_up()


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


def interpret(args, idxs=None, epoch=None, step=None, dryrun=False, four_fold=False):
    # interpret a dataset by attention, only for the data point of idxs in the dataset
    dataset_att = {"data_type": args.data_type,
                   "radius": args.radius,
                   "max_neighbors": args.max_num_neighbors,
                   "loop": args.loop,
                   "shuffle": False, 
                   "node_embedding_type": args.node_embedding_type,
                   "graph_type": args.graph_type,
                   "add_plddt": args.add_plddt,
                   "scale_plddt": args.scale_plddt,
                   "add_conservation": args.add_conservation,
                   "add_position": args.add_position,
                   "add_sidechain": args.add_sidechain,
                   "add_dssp": args.add_dssp,
                   "add_msa": args.add_msa,
                   "add_confidence": args.add_confidence,
                   "add_msa_contacts": args.add_msa_contacts,
                   "add_ptm": args.add_ptm,
                   "add_af2_single": args.add_af2_single,
                   "add_af2_pairwise": args.add_af2_pairwise,
                   "loaded_af2_single": args.loaded_af2_single,
                   "loaded_af2_pairwise": args.loaded_af2_pairwise,
                   "loaded_msa": args.loaded_msa,
                   "loaded_esm": args.loaded_esm,
                   "loaded_confidence": args.loaded_confidence,
                   "data_augment": args.data_augment,
                   "score_transfer": args.score_transfer,
                   "alt_type": args.alt_type,
                   "computed_graph": args.computed_graph,
                   "neighbor_type": args.neighbor_type,
                   "max_len": args.max_len,}
    if args.trainer_fn == "PreMode_trainer_noGraph":
        dataset_extra_args = {"padding": args.batch_size > 1}
    elif args.trainer_fn == "PreMode_trainer":
        dataset_extra_args = {}
    elif args.trainer_fn == "PreMode_trainer_SSP":
        dataset_extra_args = {}
    else:
        raise ValueError(f"trainer_fn {args.trainer_fn} not supported")
    if dryrun:
        dataset = None
    else:
        dataset = getattr(data, args.dataset)(
            data_file=args.data_file_test,
            **dataset_att,
            **dataset_extra_args,
        )
    # apply 4 fold cross validation
    if four_fold:
        main_log_dir = args.log_dir
        if idxs is None:
            idxs = list(range(len(dataset)))
            out_index = dataset.data.index
        else:
            out_index = dataset.data.index[idxs]
        x_embed_df = dataset.data.loc[out_index]
        for FOLD in range(4):
            # change args log_dir to the fold log_dir
            args.log_dir = os.path.join(main_log_dir, f"FOLD.{FOLD}/")
            _, ys, min_loss = interpret_core(args, dataset, idxs=idxs, epoch=epoch, step=step, dryrun=dryrun, four_fold=True)
            # change ys columns to original column name + '.FOLD.{FOLD}'
            ys.columns = [f"{c}.FOLD.{FOLD}" for c in ys.columns]
            x_embed_df = pd.concat([x_embed_df, ys], axis=1)
            x_embed_df[f"min_loss.FOLD.{FOLD}"] = min_loss
    else:
        x_embed_df = interpret_core(args, dataset, idxs=idxs, epoch=epoch, step=step, dryrun=dryrun)
        if dryrun:
            return
    if args.out_dir is None:
        args.out_dir = args.log_dir
    if not os.path.exists(args.out_dir) and not args.out_dir.endswith(".csv"):
        os.makedirs(args.out_dir)
    if args.out_dir.endswith(".csv"):
        x_embed_df.to_csv(args.out_dir, index=False)
    else:
        x_embed_df.to_csv(f"{args.out_dir}/x_embeds.csv", index=False)


def interpret_core(args, dataset, idxs=None, epoch=None, step=None, dryrun=False, four_fold=False):
    if args.seed_with_pl:
        import pytorch_lightning as pl
        pl.seed_everything(args.seed)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    hparams = vars(args)
    model_class = args.model_class
    # initialize model
    if args.load_model == "None" or args.load_model == "null" or args.load_model is None:
        my_model = create_model(hparams, model_class=model_class)
    else:
        my_model = create_model_and_load(hparams, model_class=model_class)
    if args.trainer_fn == "PreMode_trainer":
        trainer_fn = PreMode_trainer
    else:
        raise ValueError(f"trainer_fn {args.trainer_fn} not supported")
    if args.dataset.startswith("FullGraph") and not args.model.startswith("lora"):
        my_model = torch.compile(my_model)
    if args.hp_tune:
        my_model.load_state_dict(torch.load(f'{args.log_dir}/model.hp_tune.pt', map_location=torch.device("cpu")))
    my_model.eval()
    if dryrun:
        trainer = None
    else:
        trainer = trainer_fn(hparams=args, model=my_model, 
                            stage="test", dataset=dataset, device_id=args.gpu_id)
    if not args.hp_tune:
        # only load model if not hp_tune
        if epoch is not None:
            # if epoch is -1, load the last epoch
            if epoch == -1:
                epoch = args.num_epochs
            trainer.load_model(epoch=epoch)
            min_loss = None
        elif step is not None:
            trainer.load_model(step=step)
            min_loss = None
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
                min_loss = np.min(np.array(val_losses)[~np.isnan(val_losses)])
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
                min_loss = np.min(np.array(val_losses)[~np.isnan(val_losses)])
            elif args.interpret_by == "both":
                # find the min val loss epoch
                val_losses = []
                for epoch in range(args.num_epochs):
                    val_loss = []
                    for rank in range(args.ngpus):
                        if os.path.exists(f"{args.log_dir}/result_dict.epoch.{epoch}.ddp_rank.{rank}.json"):
                            with open(f"{args.log_dir}/result_dict.epoch.{epoch}.ddp_rank.{rank}.json", "r") as f:
                                result_dict = json.load(f)
                                val_loss.append(result_dict["val_loss"])
                        else:
                            val_loss.append(np.nan)
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
                        else:
                            val_loss.append(np.nan)
                    val_losses.append(np.mean(val_loss))
                if len(np.array(val_losses)[~np.isnan(val_losses)]) > 0:
                    # remove nan values steps
                    steps = np.array(steps)[~np.isnan(val_losses)]
                    # remove nan values val_losses
                    val_losses = np.array(val_losses)[~np.isnan(val_losses)]
                    min_val_loss_batch = steps[np.argmin(val_losses)]
                    min_batch_loss = np.min(val_losses)
                    min_loss = min(min_epoch_loss, min_batch_loss)
                else:
                    min_loss = min_epoch_loss
                if len(np.array(val_losses)[~np.isnan(val_losses)]) == 0 or min_epoch_loss < min_batch_loss:
                    print(f"min_val_loss_epoch: {min_val_loss_epoch}, loss: {min_epoch_loss}")
                    if dryrun:
                        print("dryrun mode, skip interpret")
                        return
                    trainer.load_model(epoch=min_val_loss_epoch)
                else:
                    print(f"min_val_loss_batch: {min_val_loss_batch}, loss: {min_batch_loss}")
                    if dryrun:
                        print("dryrun mode, skip interpret")
                        return
                    trainer.load_model(step=min_val_loss_batch)
            elif args.interpret_by == "both_train_val":
                # find the min val loss epoch
                val_losses = []
                for epoch in range(args.num_epochs):
                    val_loss = []
                    for rank in range(args.ngpus):
                        if os.path.exists(f"{args.log_dir}/result_dict.epoch.{epoch}.ddp_rank.{rank}.json"):
                            try:
                                with open(f"{args.log_dir}/result_dict.epoch.{epoch}.ddp_rank.{rank}.json", "r") as f:
                                    result_dict = json.load(f)
                                    val_loss.append(result_dict["val_loss"] + result_dict["train_loss"])
                            except:
                                print(f"error in file {args.log_dir}/result_dict.epoch.{epoch}.ddp_rank.{rank}.json")
                        else:
                            val_loss.append(np.nan)
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
                                val_loss.append(result_dict["val_loss"] + result_dict["train_loss"])
                        else:
                            val_loss.append(np.nan)
                    val_losses.append(np.mean(val_loss))
                if len(np.array(val_losses)[~np.isnan(val_losses)]) > 0:
                    # remove nan values steps
                    steps = np.array(steps)[~np.isnan(val_losses)]
                    # remove nan values val_losses
                    val_losses = np.array(val_losses)[~np.isnan(val_losses)]
                    min_val_loss_batch = steps[np.argmin(val_losses)]
                    min_batch_loss = np.min(val_losses)
                    min_loss = min(min_epoch_loss, min_batch_loss)
                else:
                    min_loss = min_epoch_loss
                if len(np.array(val_losses)[~np.isnan(val_losses)]) == 0 or min_epoch_loss < min_batch_loss:
                    print(f"min_val_loss_epoch: {min_val_loss_epoch}, loss: {min_epoch_loss}")
                    if dryrun:
                        print("dryrun mode, skip interpret")
                        return
                    trainer.load_model(epoch=min_val_loss_epoch)
                else:
                    print(f"min_val_loss_batch: {min_val_loss_batch}, loss: {min_batch_loss}")
                    if dryrun:
                        print("dryrun mode, skip interpret")
                        return
                    trainer.load_model(step=min_val_loss_batch)
    if idxs is None:
        idxs = list(range(len(dataset)))
        out_index = dataset.data.index
    else:
        out_index = dataset.data.index[idxs]
    # interpret one sequence at a time
    x_embeds = []
    x_jacobs = []
    ys = []
    y_covars = []
    if args.use_ig:
        ig = IntegratedGradients(ig_forward)
    # set up jacobian function for torch input
    # x is second last layer output, y is target 
    if args.use_jacob:
        def jacobian_fn(x, tgt=None):
            y = trainer.model.output_model.post_reduce(x)
            if tgt is None:
                tgt = torch.ones_like(y).to(trainer.device)
            res = trainer.loss_fn(y, tgt, reduce=False, reduction="none")
            return res
    for idx in idxs:
        batch = dataset.__getitem__(idx)
        if args.dataset.startswith("FullGraph"):
            # we need to expand the batch dim 1
            for k in batch:
                batch[k] = batch[k].unsqueeze(0)
        y_covar, y, x_embed, attn_weight_layers = trainer.interpret_step(batch)
        if args.loss_fn == "GP_loss":
            y_covars.append(y_covar.detach().cpu().numpy())
        elif args.loss_fn == "weighted_loss_betabinomial":
            alpha = y[:, 0]
            beta = y[:, 1]
            y_mean = alpha / (alpha + beta)
            y_covar = y_mean * (1 - y_mean) / (alpha + beta + 1)
            y_covars.append(y_covar.detach().cpu().numpy())
            y = y_mean
        elif args.loss_fn == "gaussian_loss":
            y_mean = y[:, 0]
            y_covar = y[:, 1]
            y_covars.append(y_covar.detach().cpu().numpy())
            y = y_mean
        if isinstance(batch, dict):
            batch = sn(**batch)
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
        if args.use_jacob:
            x_jacob = torch.autograd.functional.jacobian(
                jacobian_fn, 
                (x_embed.to(trainer.device, non_blocking=True))
            )
            if hasattr(batch, "score_mask"):
                x_jacobs.append(x_jacob[0, np.nonzero(batch.score_mask.numpy())[1][0], :, 0, 0].detach().cpu().numpy())
            else:
                x_jacobs.append(x_jacob[0, 0].detach().cpu().numpy())
        # select the corresponding alt output for the data point
        if "Onesite" in args.dataset:
            # reshape y to (batch_size, -1)
            y = y.reshape(y.shape[0], -1)
        while len(x_embed.shape) > 2:
            x_embed = x_embed.squeeze(0)
        while (len(y.shape) > 2):
            y = y.squeeze(-1)
        # if args.dataset.startswith("FullGraph") and not args.model.startswith("lora"):
        #     x_embed = x_embed[0]
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
    x_embeds = pd.DataFrame(x_embeds, index=out_index)
    x_embeds.columns = [f'X.{i}' for i in range(x_embeds.shape[1])]
    # assign column names
    if args.use_jacob:
        x_jacobs = np.concatenate(x_jacobs, axis=0)
        x_jacobs_df = pd.DataFrame(x_jacobs, index=out_index)
        x_jacobs_df.columns = [f'jacob.{i}' for i in range(x_jacobs_df.shape[1])]
    ys = pd.DataFrame(np.concatenate(ys, axis=0), index=out_index)
    if args.loss_fn == "GP_loss" or args.loss_fn == "weighted_loss_betabinomial" or args.loss_fn == "gaussian_loss":
        y_covars = pd.DataFrame(np.concatenate(y_covars, axis=0), index=out_index)
        if y_covars.shape[1] == 1:
            y_covars.columns = ["logits_var"]
        else:
            y_covars.columns=[f'logits_var.{i}' for i in range(y_covars.shape[1])]
    if ys.shape[1] == 1:
        ys.columns = ["logits"]
    else:
        ys.columns=[f'logits.{i}' for i in range(ys.shape[1])]
    x_embed_df = pd.concat([dataset.data.loc[out_index], ys, x_embeds], axis=1)
    if args.loss_fn == "GP_loss" or args.loss_fn == "weighted_loss_betabinomial" or args.loss_fn == "gaussian_loss":
        x_embed_df = pd.concat([x_embed_df, y_covars], axis=1)
    if args.use_jacob:
        x_embed_df = pd.concat([x_embed_df, x_jacobs_df], axis=1)
    # clean up the data sets
    trainer.dataset.clean_up()
    # add min_loss to the x_embed_df
    x_embed_df["min_loss"] = min_loss
    if four_fold:
        return x_embed_df, ys, min_loss
    else:
        return x_embed_df


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
                   "scale_plddt": args.scale_plddt,
                   "add_conservation": args.add_conservation,
                   "add_position": args.add_position,
                   "add_sidechain": args.add_sidechain,
                   "add_dssp": args.add_dssp,
                   "add_msa": args.add_msa,
                   "add_confidence": args.add_confidence,
                   "add_msa_contacts": args.add_msa_contacts,
                   "add_ptm": args.add_ptm,
                   "loaded_msa": args.loaded_msa,
                   "loaded_esm": args.loaded_esm,
                   "loaded_confidence": args.loaded_confidence,
                   "data_augment": args.data_augment,
                   "score_transfer": args.score_transfer,
                   "alt_type": args.alt_type,
                   "computed_graph": args.computed_graph,
                   "neighbor_type": args.neighbor_type,
                   "max_len": args.max_len,}
    datasets = getattr(data, args.dataset)(
            data_file=f"{args.data_file_train}",
            gpu_id=0,
            **dataset_att,
        )
    dataloader = torch.utils.data.DataLoader(
        datasets,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    data_point = next(iter(dataloader))
    data_point["edge_index"] = None
    data_point["edge_index_star"] = None
    data_point["edge_attr_star"] = None
    data_point["batch"] = None
    hparams["drop_out"] = 0
    model = create_model(hparams, model_class=args.model_class)
    # get data points
    pos = torch.randn(data_point["x"].shape[0], data_point["x"].shape[1], 3)
    node_vec_attr = torch.randn(data_point["x"].shape[0], data_point["x"].shape[1], 3, 35)
    y = model(data_point["x"], data_point["x_mask"], data_point["x_alt"], pos, 
              data_point["edge_index"], data_point["edge_index_star"], 
              data_point["edge_attr"], data_point["edge_attr_star"], 
              node_vec_attr, data_point["batch"], data_point, return_attn=True)[0]
    y_rot = model(data_point["x"], data_point["x_mask"], data_point["x_alt"], pos @ rotate, 
                  data_point["edge_index"], data_point["edge_index_star"], 
                  data_point["edge_attr"], data_point["edge_attr_star"], 
                  (node_vec_attr.permute(0, 1, 3, 2) @ rotate).permute(0, 1, 3, 2), data_point["batch"], data_point,
                  return_attn=True)[0]
    torch.testing.assert_allclose(y, y_rot)


def test_scalar_invariance_2(args):
    import torch
    from torch_geometric.loader import DataLoader
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
                   "scale_plddt": args.scale_plddt,
                   "add_conservation": args.add_conservation,
                   "add_position": args.add_position,
                   "add_sidechain": args.add_sidechain,
                   "add_dssp": args.add_dssp,
                   "add_msa": args.add_msa,
                   "add_confidence": args.add_confidence,
                   "add_msa_contacts": args.add_msa_contacts,
                   "add_ptm": args.add_ptm,
                   "loaded_msa": args.loaded_msa,
                   "loaded_esm": args.loaded_esm,
                   "loaded_confidence": args.loaded_confidence,
                   "data_augment": args.data_augment,
                   "score_transfer": args.score_transfer,
                   "alt_type": args.alt_type,
                   "computed_graph": args.computed_graph,
                   "neighbor_type": args.neighbor_type,
                   "max_len": args.max_len,}
    datasets = getattr(data, args.dataset)(
            data_file=f"{args.data_file_train}",
            gpu_id=0,
            **dataset_att,
        )
    dataloader = DataLoader(
        datasets,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    data_point = next(iter(dataloader))
    model = create_model(hparams, model_class=args.model_class)
    # get data points
    pos = torch.randn(data_point.x.shape[0], 3)
    node_vec_attr = torch.randn(data_point.x.shape[0], 3, 35)
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
    elif _args.mode == "train_4_fold":
        main(_args, continue_train=False, four_fold=True)
    elif _args.mode == "adaptive_train":
        adaptive_main(_args)
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
    elif _args.mode == "interpret_dry":
        if _args.interpret_idxes is not None:
            _args.interpret_idxes = [int(i) for i in _args.interpret_idxes.split(",")]
        interpret(_args, idxs=_args.interpret_idxes, step=_args.interpret_step, epoch=_args.interpret_epoch, dryrun=True)
    elif _args.mode == "interpret_4_fold":
        if _args.interpret_idxes is not None:
            _args.interpret_idxes = [int(i) for i in _args.interpret_idxes.split(",")]
        interpret(_args, idxs=_args.interpret_idxes, step=_args.interpret_step, epoch=_args.interpret_epoch, dryrun=False, four_fold=True)
    elif _args.mode == "test_equivariancy":
        test_scalar_invariance(_args)
    elif _args.mode == "test_equivariancy_2":
        test_scalar_invariance_2(_args)
    elif _args.mode == "hp_tune":
        hp_tune(_args)    
    else:
        raise ValueError(f"mode {_args.mode} not supported")
