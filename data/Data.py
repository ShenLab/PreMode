from typing import Literal
import warnings
import time
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Dataset, Data
from torch_geometric.data.data import BaseData
from torch_geometric.utils import remove_isolated_nodes
from itertools import cycle
from multiprocessing import Pool
from multiprocessing import get_context
from typing import Any, List
import data.utils as utils
import h5py
import lmdb
import pickle
from datetime import datetime
import os
NUM_THREADS = 42

# Main Abstract Class, define a Mutation Dataset, compatible with PyTorch Geometric
class GraphMutationDataset(Dataset):
    """
    MutationDataSet dataset, input a file of mutations, output a star graph and KNN graph
    Can be either single mutation or multiple mutations.

    Args:
        data_file (string or pd.DataFrame): Path or pd.DataFrame for a csv file for a list of mutations
        data_type (string): Type of this data, 'ClinVar', 'DMS', etc
    """

    def __init__(self, data_file, data_type: str,
                 radius: float = None, max_neighbors: int = None,
                 loop: bool = False, shuffle: bool = False, gpu_id: int = None,
                 node_embedding_type: Literal['esm', 'one-hot-idx', 'one-hot', 'aa-5dim', 'esm1b'] = 'esm',
                 graph_type: Literal['af2', '1d-neighbor'] = 'af2',
                 add_plddt: bool = False, 
                 scale_plddt: bool = False,
                 add_conservation: bool = False,
                 add_position: bool = False,
                 add_sidechain: bool = False,
                 local_coord_transform: bool = False,
                 use_cb: bool = False,
                 add_msa_contacts: bool = True,
                 add_dssp: bool = False,
                 add_msa: bool = False,
                 add_confidence: bool = False,
                 loaded_confidence: bool = False,
                 loaded_esm: bool = False,
                 add_ptm: bool = False,
                 data_augment: bool = False,
                 score_transfer: bool = False,
                 alt_type: Literal['alt', 'concat', 'diff', 'zero', 'orig'] = 'alt',
                 computed_graph: bool = True,
                 loaded_msa: bool = False,
                 neighbor_type: Literal['KNN', 'radius', 'radius-KNN'] = 'KNN',
                 max_len = 2251,
                 add_af2_single: bool = False,
                 add_af2_pairwise: bool = False,
                 loaded_af2_single: bool = False,
                 loaded_af2_pairwise: bool = False,
                 use_lmdb: bool = False,
                 ):
        super(GraphMutationDataset, self).__init__()
        if isinstance(data_file, pd.DataFrame):
            self.data = data_file
            self.data_file = 'pd.DataFrame'
        elif isinstance(data_file, str):
            try:
                self.data = pd.read_csv(data_file, index_col=0, low_memory=False)
            except UnicodeDecodeError:
                self.data = pd.read_csv(data_file, index_col=0, encoding='ISO-8859-1')
            self.data_file = data_file
        else:
            raise ValueError("data_path must be a string or a pandas.DataFrame")
        self.data_type = data_type
        self._y_columns = self.data.columns[self.data.columns.str.startswith('score')]
        self._y_mask_columns = self.data.columns[self.data.columns.str.startswith('confidence.score')]
        self.node_embedding_type = node_embedding_type
        self.graph_type = graph_type
        self.neighbor_type = neighbor_type
        self.add_plddt = add_plddt
        self.scale_plddt = scale_plddt
        self.add_conservation = add_conservation
        self.add_position = add_position
        self.use_cb = use_cb
        self.add_sidechain = add_sidechain
        self.add_msa_contacts = add_msa_contacts
        self.add_dssp = add_dssp
        self.add_msa = add_msa
        self.add_af2_single = add_af2_single
        self.add_af2_pairwise = add_af2_pairwise
        self.loaded_af2_single = loaded_af2_single
        self.loaded_af2_pairwise = loaded_af2_pairwise
        self.add_confidence = add_confidence
        self.loaded_confidence = loaded_confidence
        self.add_ptm = add_ptm
        self.loaded_msa = loaded_msa
        self.loaded_esm = loaded_esm
        self.alt_type = alt_type
        self.max_len = max_len
        self.loop = loop
        self.data_augment = data_augment
        # initialize some dicts
        self.af2_file_dict = None
        self.af2_coord_dict = None
        self.af2_plddt_dict = None
        self.af2_confidence_dict = None
        self.af2_dssp_dict = None
        self.af2_graph_dict = None
        self.esm_file_dict = None
        self.esm_dict = None
        self.msa_file_dict = None
        self.msa_dict = None
        self._check_embedding_files()
        if score_transfer:
            # only do score_transfer when score is 0 or 1
            if set(self.data['score'].unique()) <= {0, 1}:
                self.data['score'] = self.data['score'] * 3
            else:
                warnings.warn("score_transfer is only applied when score is 0 or 1")
        if data_augment and set(self.data['score'].unique()) > {0, 1}:
            # reverse ref and alt and score, only when we do gof/lof
            reverse_data = self.data.copy()
            # reverse only for score == 1 and score == 0
            reverse_data = reverse_data.loc[(reverse_data['score'] == 1) | (reverse_data['score'] == 0), :]
            reverse_data['ref'] = self.data['alt']
            reverse_data['alt'] = self.data['ref']
            reverse_data['score'] = -reverse_data['score']
            self.data = pd.concat([self.data, reverse_data], ignore_index=True)
        self._set_mutations()
        self.computed_graph = computed_graph
        self._load_af2_features(radius=radius, max_neighbors=max_neighbors, loop=loop, gpu_id=gpu_id)
        if (self.add_msa or self.add_conservation) and self.loaded_msa:
            self._load_msa_features()
        if self.loaded_esm:
            self._load_esm_features()
        if self.loaded_af2_pairwise or self.loaded_af2_single:
            self._load_af2_reps()
        self._set_node_embeddings()
        self._set_edge_embeddings()
        self.unmatched_msa = 0
        # shuffle the data
        if shuffle:
            np.random.seed(0)
            shuffle_index = np.random.permutation(len(self.mutations))
            self.data = self.data.iloc[shuffle_index].reset_index(drop=True)
            self.mutations = list(map(self.mutations.__getitem__, shuffle_index))
        if self.add_ptm:
            self.ptm_ref = pd.read_csv('./data.files/ptm.small.csv', index_col=0)
        self.get_method = 'default'
        # if your machine has sufficient memory, you can uncomment the following line
        # self.load_all_to_memory()

    def _check_embedding_files(self):
        print(f"read in {len(self.data)} mutations from {self.data_file}")
        # scan uniprot files and transcript files to check if they exist
        unique_data = self.data.drop_duplicates(subset=['uniprotID'])
        print(f"found {len(unique_data)} unique wt sequences")
        # only check embeddings if we are using esm
        if self.node_embedding_type == 'esm':
            with Pool(NUM_THREADS) as p:
                embedding_exist = p.starmap(utils.get_embedding_from_esm2, zip(unique_data['uniprotID'], cycle([True])))
            # msa_exist = p.starmap(get_attn_from_msa, zip(unique_data['ENST'], unique_data['wt.orig'], cycle([True])))
            # TODO: check MSA again, consider using raw MSA only
            to_drop = unique_data['wt.orig'].loc[~np.array(embedding_exist, dtype=bool)]
            print(f"drop {np.sum(self.data['wt.orig'].isin(to_drop))} mutations that do not have embedding or msa")
            self.data = self.data[~self.data['wt.orig'].isin(to_drop)]
        else:
            print(f"skip checking embedding files for {self.node_embedding_type}")

    def _set_mutations(self):
        if 'af2_file' not in self.data.columns:
            self.data['af2_file'] = pd.NA
        with Pool(NUM_THREADS) as p:
            point_mutations = p.starmap(utils.get_mutations, zip(self.data['uniprotID'],
                                                                 self.data['ENST'] if 'ENST' in self.data.columns else cycle([None]),
                                                                 self.data['wt.orig'],
                                                                 self.data['sequence.len.orig'],
                                                                 self.data['pos.orig'],
                                                                 self.data['ref'],
                                                                 self.data['alt'], 
                                                                 cycle([self.max_len]),
                                                                 self.data['af2_file'] if 'af2_file' in self.data.columns else cycle([None]),))
        # drop the data that does not have coordinates if we are using af2
        print(f"drop {np.sum(~np.array(point_mutations, dtype=bool))} mutations that don't have coordinates")
        self.data = self.data.loc[np.array(point_mutations, dtype=bool)]
        self.mutations = list(filter(bool, point_mutations))
        print(f'Finished loading {len(self.mutations)} mutations')

    def _load_af2_features(self, radius, max_neighbors, loop, gpu_id):
        self.af2_file_dict, mutation_idx = np.unique([mutation.af2_file for mutation in self.mutations],
                                                    return_inverse=True)
        _ = list(map(lambda x, y: x.set_af2_seq_index(y), self.mutations, mutation_idx))
        with Pool(NUM_THREADS) as p:
            self.af2_coord_dict = p.starmap(utils.get_coords_from_af2, zip(self.af2_file_dict, cycle([self.add_sidechain])))
            print(f'Finished loading {len(self.af2_coord_dict)} af2 coords')
            self.af2_plddt_dict = p.starmap(utils.get_plddt_from_af2, zip(self.af2_file_dict)) if self.add_plddt else None
            print(f'Finished loading plddt')
            self.af2_confidence_dict = p.starmap(utils.get_confidence_from_af2file, zip(self.af2_file_dict, self.af2_plddt_dict)) if self.add_plddt and self.add_confidence and self.loaded_confidence else None
            print(f'Finished loading confidence')
            self.af2_dssp_dict = p.starmap(utils.get_dssp_from_af2, zip(self.af2_file_dict)) if self.add_dssp else None
            print(f'Finished loading dssp')
        if self.computed_graph:
            if self.graph_type == 'af2':
                if self.neighbor_type == 'KNN':
                    self.af2_graph_dict = list(map(utils.get_knn_graphs_from_af2, self.af2_coord_dict,
                                            cycle([radius]), cycle([max_neighbors]), cycle([loop]), cycle([gpu_id])))
                    print(f'Finished constructing {len(self.af2_graph_dict)} af2 graphs')
                else:
                    # if radius graph, don't compute until needed
                    self.computed_graph = False
                    print(f'Do not construct graphs from af2 files to save RAM')
            elif self.graph_type == '1d-neighbor':
                self.af2_graph_dict = list(map(utils.get_graphs_from_neighbor, self.af2_coord_dict,
                                        cycle([max_neighbors]), cycle([loop])))
                print(f'Finished constructing {len(self.af2_graph_dict)} af2 graphs')
        else:
            print(f'Do not construct graphs from af2 files to save RAM')
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.loop = loop
        self.gpu_id = gpu_id
    
    def _load_esm_features(self):
        self.esm_file_dict, mutation_idx = np.unique([mutation.ESM_prefix for mutation in self.mutations],
                                                    return_inverse=True)
        _ = list(map(lambda x, y: x.set_esm_seq_index(y), self.mutations, mutation_idx))
        with Pool(NUM_THREADS) as p:
            self.esm_dict = p.starmap(utils.get_esm_dict_from_uniprot, zip(self.esm_file_dict))
            print(f'Finished loading {len(self.esm_file_dict)} esm embeddings')

    def _load_af2_reps(self):
        self.af2_rep_file_prefix_dict, mutation_idx = np.unique([mutation.af2_rep_file_prefix for mutation in self.mutations],
                                                                return_inverse=True)
        _ = list(map(lambda x, y: x.set_af2_rep_index(y), self.mutations, mutation_idx))
        with Pool(NUM_THREADS) as p:
            if self.add_af2_single and self.loaded_af2_single:
                self.af2_single_dict = p.starmap(utils.get_af2_single_rep_dict_from_prefix, zip(self.af2_rep_file_prefix_dict))
                print(f'Finished loading {len(self.af2_rep_file_prefix_dict)} alphafold2 single representations')
            # because the pairwise representation is too large to fit in RAM, we have to select a subset of them
        if self.add_af2_pairwise and self.loaded_af2_pairwise:
            raise ValueError("Not implemented in this version")
            
    def _load_msa_features(self):
        self.msa_file_dict, mutation_idx = np.unique([mutation.uniprot_id for mutation in self.mutations],
                                                     return_inverse=True)
        _ = list(map(lambda x, y: x.set_msa_seq_index(y), self.mutations, mutation_idx))
        with Pool(NUM_THREADS) as p:
            # msa_dict: msa_seq, conservation, msa
            self.msa_dict = p.starmap(utils.get_msa_dict_from_transcript, zip(self.msa_file_dict))
        print(f'Finished loading {len(self.msa_dict)} msa seqs')
        
    def _set_node_embeddings(self):
        pass

    def _set_edge_embeddings(self):
        pass
    
    def get_mask(self, mutation: utils.Mutation):
        return mutation.pos - 1, mutation

    def get_graph_and_mask(self, mutation: utils.Mutation):
        # get the ordinary graph
        coords: np.ndarray = self.af2_coord_dict[mutation.af2_seq_index]  # N, C, O, CA, CB
        if self.computed_graph:
            edge_index = self.af2_graph_dict[mutation.af2_seq_index]  # 2, E
        else:
            if self.graph_type == 'af2':
                if self.neighbor_type == 'KNN':
                    edge_index = utils.get_knn_graphs_from_af2(coords, self.radius, self.max_neighbors, self.loop, self.gpu_id)
                elif self.neighbor_type == 'radius':
                    edge_index = utils.get_radius_graphs_from_af2(coords, self.radius, self.loop, self.gpu_id)
                    # delete nodes that are not connected with variant node.
                    connected_nodes = edge_index[:, np.isin(edge_index[0], mutation.pos - 1)].flatten()
                    edge_index = edge_index[:, np.isin(edge_index[0], connected_nodes) | np.isin(edge_index[1], connected_nodes)]
                else:
                    edge_index = utils.get_radius_knn_graphs_from_af2(coords, mutation.pos - 1, self.radius, self.max_neighbors, self.loop)
            elif self.graph_type == '1d-neighbor':
                edge_index = utils.get_graphs_from_neighbor(coords, self.max_neighbors, self.loop)
        # remember we could have cropped sequence
        if mutation.crop:
            coords = coords[mutation.seq_start - 1:mutation.seq_end, :]
            edge_index = edge_index[:, (edge_index[0, :] >= mutation.seq_start - 1) &
                                       (edge_index[1, :] >= mutation.seq_start - 1) &
                                       (edge_index[0, :] < mutation.seq_end) &
                                       (edge_index[1, :] < mutation.seq_end)]
            edge_index[0, :] -= mutation.seq_start - 1
            edge_index[1, :] -= mutation.seq_start - 1
        # get the mask
        mask_idx, mutation = self.get_mask(mutation)
        # star graph of other positions to variant sites and reverse
        edge_matrix_star = np.zeros((coords.shape[0], coords.shape[0]))
        edge_matrix_star[:, mask_idx] = 1
        edge_matrix_star[mask_idx, :] = 1
        edge_index_star = np.array(np.where(edge_matrix_star == 1))
        # if radius graph, only keep the edges of nodes in the edge_index
        if self.neighbor_type == 'radius' or self.neighbor_type == 'KNN':
            edge_index_star = edge_index_star[:, np.isin(edge_index_star[0], edge_index.flatten()) &
                                                np.isin(edge_index_star[1], edge_index.flatten())]
        elif self.neighbor_type == 'radius-KNN':
            edge_index_star = edge_index_star[:, np.isin(edge_index_star[0], np.concatenate((edge_index.flatten(), mask_idx))) &
                                                np.isin(edge_index_star[1], np.concatenate((edge_index.flatten(), mask_idx)))]
        # cancel self loop
        if not self.loop:
            edge_index_star = edge_index_star[:, edge_index_star[0] != edge_index_star[1]]
        if self.add_msa_contacts:
            coevo_strength = utils.get_contacts_from_msa(mutation, False)
            if isinstance(coevo_strength, int):
                coevo_strength = np.zeros([mutation.seq_end - mutation.seq_start + 1,
                                        mutation.seq_end - mutation.seq_start + 1, 1])
        else:
            coevo_strength = np.zeros([mutation.seq_end - mutation.seq_start + 1,
                                        mutation.seq_end - mutation.seq_start + 1, 0])
        start = time.time()
        if self.add_af2_pairwise:
            if self.loaded_af2_pairwise:
                # we don't use the self.af2_pair_dict anymore because it won't fit in RAM
                # we load from lmdb
                byteflow = self.af2_pairwise_txn.get(u'{}'.format(mutation.af2_rep_file_prefix.split('/')[-1]).encode('ascii'))
                pairwise_rep = pickle.loads(byteflow)
                if pairwise_rep is None:
                    pairwise_rep = utils.get_af2_pairwise_rep_dict_from_prefix(mutation.af2_rep_file_prefix)
            else:
                pairwise_rep = utils.get_af2_pairwise_rep_dict_from_prefix(mutation.af2_rep_file_prefix)
            # crop the pairwise_rep, if necessary
            if mutation.af2_rep_file_prefix.find('-F') == -1:
                pairwise_rep = pairwise_rep[mutation.seq_start_orig - 1: mutation.seq_end_orig,
                                            mutation.seq_start_orig - 1: mutation.seq_end_orig]
            if mutation.crop:
                pairwise_rep = pairwise_rep[mutation.seq_start - 1: mutation.seq_end,
                                            mutation.seq_start - 1: mutation.seq_end]
            coevo_strength = np.concatenate([coevo_strength, pairwise_rep], axis=2)
        end = time.time()
        print(f'Finished loading pairwise in {end - start:.2f} seconds')
        edge_attr = coevo_strength[edge_index[0], edge_index[1], :]
        edge_attr_star = coevo_strength[edge_index_star[0], edge_index_star[1], :]
        # if add positional embedding, add it here
        if self.add_position:
            # add a sin positional embedding that reflects the relative position of the residue
            edge_attr = np.concatenate(
                (edge_attr, np.sin(np.pi / 2 * (edge_index[1] - edge_index[0]) / self.max_len).reshape(-1, 1)), 
                axis=1)
            edge_attr_star = np.concatenate(
                (edge_attr_star, np.sin(np.pi / 2 * (edge_index_star[1] - edge_index_star[0]) / self.max_len).reshape(-1, 1)),
                axis=1)
        return coords, edge_index, edge_index_star, edge_attr, edge_attr_star, mask_idx, mutation

    def get_one_mutation(self, idx):
        mutation: utils.Mutation = self.mutations[idx]
        # get the graph
        coords, edge_index, edge_index_star, edge_attr, edge_attr_star, mask_idx, mutation = self.get_graph_and_mask(mutation)
        # get embeddings
        if self.node_embedding_type == 'esm':
            if self.loaded_esm:
                embed_data = utils.get_embedding_from_esm2(self.esm_dict[mutation.esm_seq_index], False,
                                                           mutation.seq_start, mutation.seq_end)
            else:
                embed_data = utils.get_embedding_from_esm2(mutation.ESM_prefix, False,
                                                           mutation.seq_start, mutation.seq_end)
            to_alt = np.concatenate([utils.ESM_AA_EMBEDDING_DICT[alt_aa].reshape(1, -1) for alt_aa in mutation.alt_aa])
            to_ref = np.concatenate([utils.ESM_AA_EMBEDDING_DICT[ref_aa].reshape(1, -1) for ref_aa in mutation.ref_aa])
        elif self.node_embedding_type == 'one-hot-idx':
            assert not self.add_conservation and not self.add_plddt
            embed_logits, embed_data, one_hot_mat = utils.get_embedding_from_onehot_nonzero(mutation.seq, return_idx=True, return_onehot_mat=True)
            to_alt = np.concatenate([np.array(utils.AA_DICT.index(alt_aa)).reshape(1, -1) for alt_aa in mutation.alt_aa])
            to_ref = np.concatenate([np.array(utils.AA_DICT.index(ref_aa)).reshape(1, -1) for ref_aa in mutation.ref_aa])
        elif self.node_embedding_type == 'one-hot':
            embed_data, one_hot_mat = utils.get_embedding_from_onehot(mutation.seq, return_idx=False, return_onehot_mat=True)
            to_alt = np.concatenate([np.eye(len(utils.AA_DICT))[utils.AA_DICT.index(alt_aa)].reshape(1, -1) for alt_aa in mutation.alt_aa])
            to_ref = np.concatenate([np.eye(len(utils.AA_DICT))[utils.AA_DICT.index(ref_aa)].reshape(1, -1) for ref_aa in mutation.ref_aa])
        elif self.node_embedding_type == 'aa-5dim':
            embed_data = utils.get_embedding_from_5dim(mutation.seq)
            to_alt = np.concatenate([np.array(utils.AA_5DIM_EMBED[alt_aa]).reshape(1, -1) for alt_aa in mutation.alt_aa])
            to_ref = np.concatenate([np.array(utils.AA_5DIM_EMBED[ref_aa]).reshape(1, -1) for ref_aa in mutation.ref_aa])
        elif self.node_embedding_type == 'esm1b':
            embed_data = utils.get_embedding_from_esm1b(mutation.ESM_prefix, False,
                                                        mutation.seq_start, mutation.seq_end)
            to_alt = np.concatenate([utils.ESM1b_AA_EMBEDDING_DICT[alt_aa].reshape(1, -1) for alt_aa in mutation.alt_aa])
            to_ref = np.concatenate([utils.ESM1b_AA_EMBEDDING_DICT[ref_aa].reshape(1, -1) for ref_aa in mutation.ref_aa])
        if self.alt_type == "zero":
            to_alt = np.zeros_like(to_alt)[[0]]
        # add conservation, if needed
        if self.loaded_msa and (self.add_msa or self.add_conservation):
            msa_seq = self.msa_dict[mutation.msa_seq_index][0]
            conservation_data = self.msa_dict[mutation.msa_seq_index][1]
            msa_data = self.msa_dict[mutation.msa_seq_index][2]
        else:
            if self.add_conservation or self.add_msa:
                msa_seq, conservation_data, msa_data = utils.get_msa_dict_from_transcript(mutation.uniprot_id)
        if self.add_conservation:
            if conservation_data.shape[0] == 0:
                conservation_data = np.zeros((embed_data.shape[0], 20))
            else:
                msa_seq_check = msa_seq[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                conservation_data = conservation_data[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                if mutation.crop:
                    msa_seq_check = msa_seq_check[mutation.seq_start - 1: mutation.seq_end]
                    conservation_data = conservation_data[mutation.seq_start - 1: mutation.seq_end]
                if msa_seq_check != mutation.seq:
                    # warnings.warn(f'MSA file: {mutation.transcript_id} does not match mutation sequence')
                    self.unmatched_msa += 1
                    print(f'Unmatched MSA: {self.unmatched_msa}')
                    conservation_data = np.zeros((embed_data.shape[0], 20))
            embed_data = np.concatenate([embed_data, conservation_data], axis=1)
            to_alt = np.concatenate([to_alt, conservation_data[mask_idx]], axis=1)
            if self.alt_type == 'diff':
                to_ref = np.concatenate([to_ref, conservation_data[mask_idx]], axis=1)
        # add pLDDT, if needed
        if self.add_plddt:
            # get plddt
            plddt_data = self.af2_plddt_dict[mutation.af2_seq_index]  # N, C, O, CA, CB
            if mutation.crop:
                plddt_data = plddt_data[mutation.seq_start - 1: mutation.seq_end]
            if self.add_confidence:
                confidence_data = plddt_data / 100
            if plddt_data.shape[0] != embed_data.shape[0]:
                warnings.warn(f'pLDDT {plddt_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                f'pLDDT file: {mutation.af2_file}, '
                                f'ESM prefix: {mutation.ESM_prefix}')
                plddt_data = np.ones_like(embed_data[:, 0]) * 50
                if self.add_confidence:
                    # assign 0.5 confidence to all points
                    confidence_data = np.ones_like(embed_data[:, 0]) / 2
            if self.scale_plddt:
                plddt_data = plddt_data / 100
            embed_data = np.concatenate([embed_data, plddt_data[:, None]], axis=1)
            to_alt = np.concatenate([to_alt, plddt_data[mask_idx, None]], axis=1)
            if self.alt_type == 'diff':
                to_ref = np.concatenate([to_ref, plddt_data[mask_idx]], axis=1)
        # add dssp, if needed
        if self.add_dssp:
            # get dssp
            dssp_data = self.af2_dssp_dict[mutation.af2_seq_index]
            if mutation.crop:
                dssp_data = dssp_data[mutation.seq_start - 1: mutation.seq_end]
            if dssp_data.shape[0] != embed_data.shape[0]:
                warnings.warn(f'DSSP {dssp_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                f'DSSP file: {mutation.af2_file}, '
                                f'ESM prefix: {mutation.ESM_prefix}')
                dssp_data = np.zeros_like(embed_data[:, 0])
            # if dssp_data size axis is 1, add a dimension
            if len(dssp_data.shape) == 1:
                dssp_data = dssp_data[:, None]
            embed_data = np.concatenate([embed_data, dssp_data], axis=1)
            to_alt = np.concatenate([to_alt, dssp_data[mask_idx]], axis=1)
            if self.alt_type == 'diff':
                to_ref = np.concatenate([to_ref, dssp_data[mask_idx]], axis=1)
        if self.add_ptm:
            # ptm used to behind msa, moved it here
            ptm_data = utils.get_ptm_from_mutation(mutation, self.ptm_ref)
            embed_data = np.concatenate([embed_data, ptm_data], axis=1)
            to_alt = np.concatenate([to_alt, ptm_data[mask_idx]], axis=1)
            if self.alt_type == 'diff':
                to_ref = np.concatenate([to_ref, ptm_data[mask_idx]], axis=1)
        if self.add_af2_single:
            if self.loaded_af2_single:
                single_rep = self.af2_single_dict[mutation.af2_rep_index]
            else:
                single_rep = utils.get_af2_single_rep_dict_from_prefix(mutation.af2_rep_file_prefix)
            # crop the pairwise_rep, if necessary
            if mutation.af2_rep_file_prefix.find('-F') == -1:
                single_rep = single_rep[mutation.seq_start_orig - 1: mutation.seq_end_orig]
            if mutation.crop:
                single_rep = single_rep[mutation.seq_start - 1: mutation.seq_end]
            embed_data = np.concatenate([embed_data, single_rep], axis=1)
            to_alt = np.concatenate([to_alt, single_rep[mask_idx]], axis=1)
            if self.alt_type == 'diff':
                to_ref = np.concatenate([to_ref, single_rep[mask_idx]], axis=1)
        if self.add_msa:
            # msa must be the last feature
            if msa_data.shape[0] == 0:
                msa_data = np.zeros((embed_data.shape[0], 199))
            else:
                msa_seq_check = msa_seq[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                msa_data = msa_data[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                if mutation.crop:
                    msa_seq_check = msa_seq_check[mutation.seq_start - 1: mutation.seq_end]
                    msa_data = msa_data[mutation.seq_start - 1: mutation.seq_end]
                if msa_seq_check != mutation.seq:
                    print(f'Unmatched MSA: {self.unmatched_msa}')
                    msa_data = np.zeros((embed_data.shape[0], 199))
            embed_data = np.concatenate([embed_data, msa_data], axis=1)
            if self.alt_type == 'alt' or self.alt_type == 'zero':
                to_alt = np.concatenate([to_alt, msa_data[mask_idx]], axis=1)
            if self.alt_type == 'diff':
                to_ref = np.concatenate([to_ref, msa_data[mask_idx]], axis=1)
        # replace the embedding with the mutation, note pos is 1-based
        # but we don't modify the embedding matrix, instead we return a mask matrix
        embed_data_mask = np.ones_like(embed_data)
        embed_data_mask[mask_idx] = 0
        if self.alt_type == 'alt' or self.alt_type == 'zero':
            alt_embed_data = np.zeros_like(embed_data)
            alt_embed_data[mask_idx] = to_alt
        elif self.alt_type == 'concat':
            alt_embed_data = np.zeros((embed_data.shape[0], to_alt.shape[1] + to_ref.shape[1]))
            alt_embed_data[mask_idx] = np.concatenate([to_alt, to_ref], axis=1)
        elif self.alt_type == 'diff':
            alt_embed_data = np.zeros_like(embed_data)
            alt_embed_data[mask_idx] = to_alt
            embed_data[mask_idx] = to_ref
        elif self.alt_type == 'orig':
            # do nothing
            alt_embed_data = embed_data
        else:
            raise ValueError(f'alt_type {self.alt_type} not supported')
        # prepare node vector features
        # get CA_coords
        CA_coord = coords[:, 3]
        CB_coord = coords[:, 4]
        # add CB_coord for GLY
        CB_coord[np.isnan(CB_coord)] = CA_coord[np.isnan(CB_coord)]
        if self.graph_type == '1d-neighbor':
            CA_coord[:, 0] = np.arange(coords.shape[0])
            CB_coord[:, 0] = np.arange(coords.shape[0])
            coords = np.zeros_like(coords)
        CA_CB = coords[:, [4]] - coords[:, [3]]  # Note that glycine does not have CB
        CA_CB[np.isnan(CA_CB)] = 0
        # Change the CA_CB of the mutated residue to 0
        # but we don't modify the CA_CB matrix, instead we return a mask matrix
        CA_C = coords[:, [1]] - coords[:, [3]]
        CA_O = coords[:, [2]] - coords[:, [3]]
        CA_N = coords[:, [0]] - coords[:, [3]]
        nodes_vector = np.transpose(np.concatenate([CA_CB, CA_C, CA_O, CA_N], axis=1), (0, 2, 1))
        if self.add_sidechain:
            # get sidechain coords
            sidechain_nodes_vector = coords[:, 5:] - coords[:, [3]]
            sidechain_nodes_vector[np.isnan(sidechain_nodes_vector)] = 0
            sidechain_nodes_vector = np.transpose(sidechain_nodes_vector, (0, 2, 1))
            nodes_vector = np.concatenate([nodes_vector, sidechain_nodes_vector], axis=2)
        # prepare graph
        features = dict(
            embed_logits=embed_logits if self.node_embedding_type == 'one-hot-idx' else None,
            one_hot_mat=one_hot_mat if self.node_embedding_type.startswith('one-hot') else None,
            mask_idx=mask_idx,
            embed_data=embed_data,
            embed_data_mask=embed_data_mask,
            alt_embed_data=alt_embed_data,
            coords=coords,
            CA_coord=CA_coord,
            CB_coord=CB_coord,
            edge_index=edge_index,
            edge_index_star=edge_index_star,
            edge_attr=edge_attr,
            edge_attr_star=edge_attr_star,
            nodes_vector=nodes_vector,
        )
        if self.add_confidence:
            # add position wise confidence
            if self.add_plddt:
                features['plddt'] = confidence_data
                if self.loaded_confidence:
                    pae = self.af2_confidence_dict[mutation.af2_seq_index]
                else:
                    pae = utils.get_confidence_from_af2file(mutation.af2_file, self.af2_plddt_dict[mutation.af2_seq_index])
                if mutation.crop:
                    pae = pae[mutation.seq_start - 1: mutation.seq_end, mutation.seq_start - 1: mutation.seq_end]
            else:
                # get plddt
                plddt_data = utils.get_plddt_from_af2(mutation.af2_file)
                pae = utils.get_confidence_from_af2file(mutation.af2_file, plddt_data)
                if mutation.crop:
                    confidence_data = plddt_data[mutation.seq_start - 1: mutation.seq_end] / 100
                    pae = pae[mutation.seq_start - 1: mutation.seq_end, mutation.seq_start - 1: mutation.seq_end]
                if confidence_data.shape[0] != embed_data.shape[0]:
                    warnings.warn(f'pLDDT {confidence_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                    f'pLDDT file: {mutation.af2_file}, '
                                    f'ESM prefix: {mutation.ESM_prefix}')
                    confidence_data = np.ones_like(embed_data[:, 0]) * 0.8
                features['plddt'] = confidence_data
            # add pairwise confidence
            features['edge_confidence'] = pae[edge_index[0], edge_index[1]]
            features['edge_confidence_star'] = pae[edge_index_star[0], edge_index_star[1]]
        return features

    def get(self, idx):
        features_np = self.get_one_mutation(idx)
        if self.node_embedding_type == 'one-hot-idx':
            x = torch.from_numpy(features_np['embed_data']).to(torch.long)
        else:
            x = torch.from_numpy(features_np['embed_data']).to(torch.float32)
        features = dict(
            x=x,
            x_mask=torch.from_numpy(features_np['embed_data_mask']).to(torch.bool),
            x_alt=torch.from_numpy(features_np['alt_embed_data']).to(torch.float32),
            pos=torch.from_numpy(features_np['CA_coord']).to(torch.float32) if not self.use_cb else torch.from_numpy(features_np['CB_coord']).to(torch.float32),
            edge_index=torch.from_numpy(features_np['edge_index']).to(torch.long),
            edge_index_star=torch.from_numpy(features_np['edge_index_star']).to(torch.long),
            edge_attr=torch.from_numpy(features_np['edge_attr']).to(torch.float32),
            edge_attr_star=torch.from_numpy(features_np['edge_attr_star']).to(torch.float32),
            node_vec_attr=torch.from_numpy(features_np['nodes_vector']).to(torch.float32),
            y=torch.tensor([self.data[self._y_columns].iloc[int(idx)]]).to(torch.float32),
        )
        if self.add_confidence:
            features['plddt'] = torch.from_numpy(features_np['plddt']).to(torch.float32)
            features['edge_confidence'] = torch.from_numpy(features_np['edge_confidence']).to(torch.float32)
            features['edge_confidence_star'] = torch.from_numpy(features_np['edge_confidence_star']).to(torch.float32)
        if self.neighbor_type == 'radius' or self.neighbor_type == 'radius-KNN':
            # first concat edge_index and edge_index_star
            concat_edge_index = torch.cat((features["edge_index"], features["edge_index_star"]), dim=1)
            concat_edge_attr = torch.cat((features["edge_attr"], features["edge_attr_star"]), dim=0)
            # then remove isolated nodes
            concat_edge_index, concat_edge_attr, mask = \
                remove_isolated_nodes(concat_edge_index, concat_edge_attr, x.shape[0])
            # then split edge_index and edge_attr
            features["edge_index"] = concat_edge_index[:, :features["edge_index"].shape[1]]
            features["edge_index_star"] = concat_edge_index[:, features["edge_index"].shape[1]:]
            features["edge_attr"] = concat_edge_attr[:features["edge_attr"].shape[0]]
            features["edge_attr_star"] = concat_edge_attr[features["edge_attr"].shape[0]:]
        else:
            features["edge_index"], features["edge_attr"], mask = \
                remove_isolated_nodes(features["edge_index"], features["edge_attr"], x.shape[0])
            features["edge_index_star"], features["edge_attr_star"], mask = \
                remove_isolated_nodes(features["edge_index_star"], features["edge_attr_star"], x.shape[0])
        features["x"] = features["x"][mask]
        features["x_mask"] = features["x_mask"][mask]
        features["x_alt"] = features["x_alt"][mask]
        features["pos"] = features["pos"][mask]
        features["node_vec_attr"] = features["node_vec_attr"][mask]
        if len(self._y_mask_columns) > 0:
            features['score_mask'] = torch.tensor([self.data[self._y_mask_columns].iloc[int(idx)]]).to(torch.float)
        return Data(**features)

    def get_from_hdf5(self, idx):
        if not hasattr(self, 'hdf5_keys') or self.hdf5_file is None:
            raise ValueError('hdf5 file is not set')
        else:
            features = {}
            with h5py.File(self.hdf5_file, 'r') as f:
                for key in self.hdf5_keys:
                    features[key] = torch.tensor(f[f'{self.hdf5_idx_map[idx]}/{key}'])
            return Data(**features)
    
    def open_lmdb(self):
         self.env = lmdb.open(self.lmdb_path, subdir=False,
                              readonly=True, lock=False,
                              readahead=False, meminit=False)
         self.txn = self.env.begin(write=False, buffers=True)
    
    def get_from_lmdb(self, idx):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        byteflow = self.txn.get(u'{}'.format(self.lmdb_idx_map[idx]).encode('ascii'))
        unpacked = pickle.loads(byteflow)
        return unpacked
    
    def __getitem__(self, idx):
        # record time
        start = time.time()
        if self.get_method == 'default':
            data = self.get(idx)
            print(f'default Finished loading {idx} in {time.time() - start:.2f} seconds')
        elif self.get_method == 'hdf5':
            data = self.get_from_hdf5(idx)
            print(f'hdf5 Finished loading {idx} in {time.time() - start:.2f} seconds')
        elif self.get_method == 'lmdb':
            data = self.get_from_lmdb(idx)
            print(f'lmdb Finished loading {idx} in {time.time() - start:.2f} seconds')
        elif self.get_method == 'memory':
            data = self.parsed_data[idx]
            print(f'memory Finished loading {idx} in {time.time() - start:.2f} seconds')
        return data

    def __len__(self):
        return len(self.mutations)

    def len(self) -> int:
        return len(self.mutations)
    
    def subset(self, idxs):
        self.data = self.data.iloc[idxs].reset_index(drop=True)
        self.mutations = list(map(self.mutations.__getitem__, idxs))
        # get unique af2 graphs
        subset_af2_file_dict, mutation_idx = np.unique([mutation.af2_file for mutation in self.mutations],
                                                       return_inverse=True)
        # find the index of the af2 file in the subset
        if hasattr(self, 'af2_file_dict') and self.af2_file_dict is not None:
            af2_file_idx = np.array([np.where(self.af2_file_dict==i)[0][0] for i in subset_af2_file_dict])
            self.af2_file_dict = subset_af2_file_dict
            # get the subset of af2 graphs
            self.af2_coord_dict = list(map(self.af2_coord_dict.__getitem__, af2_file_idx)) if self.af2_coord_dict is not None else None
            self.af2_plddt_dict = list(map(self.af2_plddt_dict.__getitem__, af2_file_idx)) if self.af2_plddt_dict is not None else None
            self.af2_confidence_dict = list(map(self.af2_confidence_dict.__getitem__, af2_file_idx)) if self.af2_confidence_dict is not None else None
            self.af2_dssp_dict = list(map(self.af2_dssp_dict.__getitem__, af2_file_idx)) if self.af2_dssp_dict is not None else None
            self.af2_graph_dict = list(map(self.af2_graph_dict.__getitem__, af2_file_idx)) if self.af2_graph_dict is not None else None
            # reset the af2_seq_index
            _ = list(map(lambda x, y: x.set_af2_seq_index(y), self.mutations, mutation_idx))
        # get unique esm files
        if hasattr(self, 'esm_file_dict') and self.esm_file_dict is not None:
            subset_esm_file_dict, mutation_idx = np.unique([mutation.ESM_prefix for mutation in self.mutations],
                                                        return_inverse=True)
            # find the index of the esm file in the subset
            esm_file_idx = np.array([np.where(self.esm_file_dict==i)[0][0] for i in subset_esm_file_dict])
            self.esm_file_dict = subset_esm_file_dict
            # get the subset of esm embeddings
            self.esm_dict = list(map(self.esm_dict.__getitem__, esm_file_idx)) if self.esm_dict is not None else None
            # reset the esm_seq_index
            _ = list(map(lambda x, y: x.set_esm_seq_index(y), self.mutations, mutation_idx))
        # get unique msa files
        if hasattr(self, 'msa_file_dict') and self.msa_file_dict is not None:
            subset_msa_file_dict, mutation_idx = np.unique([mutation.uniprot_id for mutation in self.mutations],
                                                           return_inverse=True)
            # find the index of the msa file in the subset
            msa_file_idx = np.array([np.where(self.msa_file_dict==i)[0][0] for i in subset_msa_file_dict])
            self.msa_file_dict = subset_msa_file_dict
            # get the subset of msa embeddings
            self.msa_dict = list(map(self.msa_dict.__getitem__, msa_file_idx)) if self.msa_dict is not None else None
            # reset the msa_seq_index
            _ = list(map(lambda x, y: x.set_msa_seq_index(y), self.mutations, mutation_idx))
        # subset hdf5 idx map, if exists
        if hasattr(self, 'hdf5_idx_map') and self.hdf5_idx_map is not None:
            self.hdf5_idx_map = self.hdf5_idx_map[idxs]
        # subset lmdb idx map, if exists
        if hasattr(self, 'lmdb_idx_map') and self.lmdb_idx_map is not None:
            self.lmdb_idx_map = self.lmdb_idx_map[idxs]
        if hasattr(self, 'parsed_data') and self.parsed_data is not None:
            self.parsed_data = list(map(self.parsed_data.__getitem__, idxs))
        return self

    def shuffle(self, idxs):
        # for shuffle, we only need to shuffle self.mutations and self.data
        self.data = self.data.iloc[idxs].reset_index(drop=True)
        self.mutations = list(map(self.mutations.__getitem__, idxs))
        # shuffle hdf5 idx map, if exists
        if self.hdf5_idx_map is not None:
            self.hdf5_idx_map = self.hdf5_idx_map[idxs]
        # shuffle lmdb idx map, if exists
        if self.lmdb_idx_map is not None:
            self.lmdb_idx_map = self.lmdb_idx_map[idxs]

    def get_label_counts(self) -> np.ndarray:
        if self.data.columns.isin(['score']).any():
            if (-1 in self.data['score'].values):
                lof = (self.data['score']==-1).sum()
                benign = (self.data['score']==0).sum()
                gof = (self.data['score']==1).sum()
                patho = (self.data['score']==3).sum()
                if lof != 0 and gof != 0:
                    return np.array([lof, benign, gof, patho])
                else:
                    return np.array([benign, patho])
            else:
                benign = (self.data['score']==0).sum()
                patho = (self.data['score']==1).sum()
                return np.array([benign, patho])
        else:
            return np.array([0, 0])
    
    # create a hdf5 file for the dataset, for faster loading        
    def create_hdf5(self):
        hdf5_file = self.data_file.replace('.csv', f'.{datetime.now()}.hdf5')
        self.hdf5_file = hdf5_file
        self.get_method = 'hdf5'
        self.hdf5_keys = None
        # create a mapping from mutation index to hdf5 index, in case of subset or shuffle
        self.hdf5_idx_map = np.arange(len(self))
        with h5py.File(hdf5_file, 'w') as f:
            for i in range(len(self)):
                features = self.get(i)
                # store feature keys into self
                if self.hdf5_keys is None:
                    self.hdf5_keys = list(features.keys())
                for key in features.keys():
                    f.create_dataset(f'{i}/{key}', data=features[key])
        return
    
    # create a lmdb file for the dataset, for faster loading
    def create_lmdb(self, write_frequency=1000):
        lmdb_path = self.data_file.replace('.csv', f'.{datetime.now()}.lmdb')
        map_size = 5e12 # 5TB
        db = lmdb.open(lmdb_path, subdir=False, map_size=map_size, readonly=False, meminit=False, map_async=True)
        print(f"Begin loading {len(self)} points into lmdb")
        txn = db.begin(write=True)
        for idx in range(len(self)):
            d = self.get(idx)
            txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps(d))
            print(f'Finished loading {idx}')
            if (idx + 1) % write_frequency == 0:
                txn.commit()
                txn = db.begin(write=True)
        txn.commit()
        print(f"Finished loading {len(self)} points into lmdb")
        self.lmdb_path = lmdb_path
        self.lmdb_idx_map = np.arange(len(self))
        self.get_method = 'lmdb'
        print("Flushing database ...")
        db.sync()
        db.close()
        return

    def load_all_to_memory(self):
        # load all data into memory
        self.get_method = 'memory'
        self.parsed_data = []
        ctime = time.time()
        tmp_data = []
        app = tmp_data.append
        for i in range(len(self)):
            app(self.get(i))
            if (i+1) % 200 == 0:
                print(f'rank {self.gpu_id} Finished loading {i+1} points in {time.time() - ctime:.2f} seconds')
                ctime = time.time()
                self.parsed_data.extend(tmp_data)
                tmp_data = []
                app = tmp_data.append
                print(f'rank {self.gpu_id} Extended {i+1} points in {time.time() - ctime:.2f} seconds')
        self.parsed_data.extend(tmp_data)
        # safe to delete all 'dict' data
        if hasattr(self, 'af2_file_dict'):
            del self.af2_file_dict
        if hasattr(self, 'af2_coord_dict'):
            del self.af2_coord_dict
        if hasattr(self, 'af2_plddt_dict'):
            del self.af2_plddt_dict
        if hasattr(self, 'af2_confidence_dict'):
            del self.af2_confidence_dict
        if hasattr(self, 'af2_dssp_dict'):
            del self.af2_dssp_dict
        if hasattr(self, 'af2_graph_dict'):
            del self.af2_graph_dict
        if hasattr(self, 'esm_file_dict'):
            del self.esm_file_dict
        if hasattr(self, 'esm_dict'):
            del self.esm_dict
        if hasattr(self, 'msa_file_dict'):
            del self.msa_file_dict
        if hasattr(self, 'msa_dict'):
            del self.msa_dict
        if hasattr(self, 'af2_single_dict'):
            del self.af2_single_dict
        if hasattr(self, 'af2_pairwise_dict'):
            del self.af2_pairwise_dict
        return

    # clean up hdf5 and lmdb files
    def clean_up(self):
        if hasattr(self, 'hdf5_file') and self.hdf5_file is not None and os.path.exists(self.hdf5_file):
            os.remove(self.hdf5_file)
        if hasattr(self, 'lmdb_path') and self.lmdb_path is not None and os.path.exists(self.lmdb_path):
            os.remove(self.lmdb_path)
        if hasattr(self, 'af2_pair_dict_lmdb_path') and self.af2_pair_dict_lmdb_path is not None:
            for lmdb_path in self.af2_pair_dict_lmdb_path:
                if os.path.exists(lmdb_path):
                    os.remove(lmdb_path)
        # close lmdb env, if exists
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
        if hasattr(self, 'af2_pairwise_env') and self.af2_pairwise_env is not None:
            self.af2_pairwise_env.close()
        return


class FullGraphMutationDataset(TorchDataset):
    """
    MutationDataSet dataset, input a file of mutations, output a star graph and KNN graph
    Can be either single mutation or multiple mutations.

    Args:
        data_file (string or pd.DataFrame): Path or pd.DataFrame for a csv file for a list of mutations
        data_type (string): Type of this data, 'ClinVar', 'DMS', etc
    """

    def __init__(self, data_file, data_type: str,
                 radius: float = None, max_neighbors: int = None,
                 loop: bool = False, shuffle: bool = False, gpu_id: int = None,
                 node_embedding_type: Literal['esm', 'one-hot-idx', 'one-hot', 'aa-5dim', 'esm1b'] = 'esm',
                 graph_type: Literal['af2', '1d-neighbor'] = 'af2',
                 add_plddt: bool = False, 
                 scale_plddt: bool = False,
                 add_conservation: bool = False,
                 add_position: bool = False,
                 add_sidechain: bool = False,
                 local_coord_transform: bool = False,
                 use_cb: bool = False,
                 add_msa_contacts: bool = True,
                 add_dssp: bool = False,
                 add_msa: bool = False,
                 add_confidence: bool = False,
                 loaded_confidence: bool = False,
                 loaded_esm: bool = False,
                 add_ptm: bool = False,
                 data_augment: bool = False,
                 score_transfer: bool = False,
                 alt_type: Literal['alt', 'concat', 'diff'] = 'alt',
                 computed_graph: bool = False,
                 loaded_msa: bool = False,
                 neighbor_type: Literal['KNN', 'radius', 'radius-KNN'] = 'KNN',
                 max_len = 2251,
                 convert_to_onesite: bool = False,
                 add_af2_single: bool = False,
                 add_af2_pairwise: bool = False,
                 loaded_af2_single: bool = False,
                 loaded_af2_pairwise: bool = False,
                 use_lmdb: bool = False
                 ):
        super(FullGraphMutationDataset, self).__init__()
        if isinstance(data_file, pd.DataFrame):
            self.data = data_file
            self.data_file = 'pd.DataFrame'
        elif isinstance(data_file, str):
            try:
                self.data = pd.read_csv(data_file, index_col=0, low_memory=False)
            except UnicodeDecodeError:
                self.data = pd.read_csv(data_file, index_col=0, encoding='ISO-8859-1')
            self.data_file = data_file
        else:
            raise ValueError("data_path must be a string or a pandas.DataFrame")
        if convert_to_onesite:
            self.data = utils.convert_to_onesite(self.data)
        self.data_type = data_type
        self._y_columns = self.data.columns[self.data.columns.str.startswith('score')]
        self.node_embedding_type = node_embedding_type
        self.graph_type = graph_type
        self.neighbor_type = neighbor_type
        self.add_plddt = add_plddt
        self.scale_plddt = scale_plddt
        self.add_conservation = add_conservation
        self.add_position = add_position
        self.use_cb = use_cb
        self.add_sidechain = add_sidechain
        self.add_msa_contacts = add_msa_contacts
        self.add_dssp = add_dssp
        self.add_msa = add_msa
        self.add_confidence = add_confidence
        self.add_af2_single = add_af2_single
        self.add_af2_pairwise = add_af2_pairwise
        self.loaded_af2_single = loaded_af2_single
        self.loaded_af2_pairwise = loaded_af2_pairwise
        self.loaded_confidence = loaded_confidence
        self.add_ptm = add_ptm
        self.loaded_msa = loaded_msa
        self.loaded_esm = loaded_esm
        self.alt_type = alt_type
        self.max_len = max_len
        self.loop = loop
        self.data_augment = data_augment
        # initialize some dicts
        self.af2_file_dict = None
        self.af2_coord_dict = None
        self.af2_plddt_dict = None
        self.af2_confidence_dict = None
        self.af2_dssp_dict = None
        self.af2_graph_dict = None
        self.esm_file_dict = None
        self.esm_dict = None
        self.msa_file_dict = None
        self.msa_dict = None
        self._check_embedding_files()
        if score_transfer:
            # only do score_transfer when score is 0 or 1
            if set(self.data['score'].unique()) <= {0, 1}:
                self.data['score'] = self.data['score'] * 3
            else:
                warnings.warn("score_transfer is only applied when score is 0 or 1")
        if data_augment and set(self.data['score'].unique()) > {0, 1}:
            # reverse ref and alt and score, only when we do gof/lof
            reverse_data = self.data.copy()
            # reverse only for score == 1 and score == 0
            reverse_data = reverse_data.loc[(reverse_data['score'] == 1) | (reverse_data['score'] == 0), :]
            reverse_data['ref'] = self.data['alt']
            reverse_data['alt'] = self.data['ref']
            reverse_data['score'] = -reverse_data['score']
            self.data = pd.concat([self.data, reverse_data], ignore_index=True)
        self._set_mutations()
        self.computed_graph = computed_graph # do not need to compute graph as we will use full graph
        self._load_af2_features(radius=radius, max_neighbors=max_neighbors, loop=loop, gpu_id=gpu_id)
        if (self.add_msa or self.add_conservation) and self.loaded_msa:
            self._load_msa_features()
        if self.loaded_esm:
            self._load_esm_features()
        if self.loaded_af2_pairwise or self.loaded_af2_single:
            self._load_af2_reps()
        self._set_node_embeddings()
        self._set_edge_embeddings()
        self.unmatched_msa = 0
        # TODO: consider load language model embeddings to RAM
        # shuffle the data
        if shuffle:
            np.random.seed(0)
            shuffle_index = np.random.permutation(len(self.mutations))
            self.data = self.data.iloc[shuffle_index].reset_index(drop=True)
            self.mutations = list(map(self.mutations.__getitem__, shuffle_index))
        if self.add_ptm:
            self.ptm_ref = pd.read_csv('./data.files/ptm.small.csv', index_col=0)
        self.get_method = 'default'
        if use_lmdb:
            self.get_method = 'lmdb'
            self.lmdb_path = data_file.replace('.csv', '.lmdb')
            self.lmdb_idx_map = np.arange(len(self))

    def _check_embedding_files(self):
        print(f"read in {len(self.data)} mutations from {self.data_file}")
        # scan uniprot files and transcript files to check if they exist
        unique_data = self.data.drop_duplicates(subset=['uniprotID'])
        print(f"found {len(unique_data)} unique wt sequences")
        # only check embeddings if we are using esm
        if self.node_embedding_type == 'esm':
            with Pool(NUM_THREADS) as p:
                embedding_exist = p.starmap(utils.get_embedding_from_esm2, zip(unique_data['uniprotID'], cycle([True])))
            # msa_exist = p.starmap(get_attn_from_msa, zip(unique_data['ENST'], unique_data['wt.orig'], cycle([True])))
            # TODO: check MSA again, consider using raw MSA only
            to_drop = unique_data['wt.orig'].loc[~np.array(embedding_exist, dtype=bool)]
            print(f"drop {np.sum(self.data['wt.orig'].isin(to_drop))} mutations that do not have embedding or msa")
            self.data = self.data[~self.data['wt.orig'].isin(to_drop)]
        else:
            print(f"skip checking embedding files for {self.node_embedding_type}")

    def _set_mutations(self):
        if 'af2_file' not in self.data.columns:
            self.data['af2_file'] = pd.NA
        with Pool(NUM_THREADS) as p:
            point_mutations = p.starmap(utils.get_mutations, zip(self.data['uniprotID'],
                                                                 self.data['ENST'] if 'ENST' in self.data.columns else cycle([None]),
                                                                 self.data['wt.orig'],
                                                                 self.data['sequence.len.orig'],
                                                                 self.data['pos.orig'],
                                                                 self.data['ref'],
                                                                 self.data['alt'], 
                                                                 cycle([self.max_len]),
                                                                 self.data['af2_file'],))
        # drop the data that does not have coordinates if we are using af2
        # if self.graph_type == 'af2':
        print(f"drop {np.sum(~np.array(point_mutations, dtype=bool))} mutations that don't have coordinates")
        self.data = self.data.loc[np.array(point_mutations, dtype=bool)]
        self.mutations = list(filter(bool, point_mutations))
        print(f'Finished loading {len(self.mutations)} mutations')

    def _load_af2_features(self, radius, max_neighbors, loop, gpu_id):
        self.af2_file_dict, mutation_idx = np.unique([mutation.af2_file for mutation in self.mutations],
                                                    return_inverse=True)
        _ = list(map(lambda x, y: x.set_af2_seq_index(y), self.mutations, mutation_idx))
        with Pool(NUM_THREADS) as p:
            self.af2_coord_dict = p.starmap(utils.get_coords_from_af2, zip(self.af2_file_dict, cycle([self.add_sidechain])))
            print(f'Finished loading {len(self.af2_coord_dict)} af2 coords')
            self.af2_plddt_dict = p.starmap(utils.get_plddt_from_af2, zip(self.af2_file_dict)) if self.add_plddt else None
            print(f'Finished loading plddt')
            self.af2_confidence_dict = p.starmap(utils.get_confidence_from_af2file, zip(self.af2_file_dict, self.af2_plddt_dict)) if self.add_plddt and self.add_confidence and self.loaded_confidence else None
            print(f'Finished loading confidence')
            self.af2_dssp_dict = p.starmap(utils.get_dssp_from_af2, zip(self.af2_file_dict)) if self.add_dssp else None
            print(f'Finished loading dssp')
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.loop = loop
        self.gpu_id = gpu_id
    
    def _load_esm_features(self):
        self.esm_file_dict, mutation_idx = np.unique([mutation.ESM_prefix for mutation in self.mutations],
                                                    return_inverse=True)
        _ = list(map(lambda x, y: x.set_esm_seq_index(y), self.mutations, mutation_idx))
        with Pool(NUM_THREADS) as p:
            self.esm_dict = p.starmap(utils.get_esm_dict_from_uniprot, zip(self.esm_file_dict))
            print(f'Finished loading {len(self.esm_file_dict)} esm embeddings')

    def _load_af2_reps(self):
        self.af2_rep_file_prefix_dict, mutation_idx = np.unique([mutation.af2_rep_file_prefix for mutation in self.mutations],
                                                                return_inverse=True)
        _ = list(map(lambda x, y: x.set_af2_rep_index(y), self.mutations, mutation_idx))
        with Pool(NUM_THREADS) as p:
            if self.add_af2_single and self.loaded_af2_single:
                self.af2_single_dict = p.starmap(utils.get_af2_single_rep_dict_from_prefix, zip(self.af2_rep_file_prefix_dict))
            print(f'Finished loading {len(self.af2_rep_file_prefix_dict)} alphafold2 single representations')
            # because the pairwise representation is too large to fit in RAM, we have to select a subset of them
        if self.add_af2_pairwise and self.loaded_af2_pairwise:
            raise ValueError("Not implemented in this version")
        
    def _load_msa_features(self):
        self.msa_file_dict, mutation_idx = np.unique([mutation.uniprot_id for mutation in self.mutations],
                                                     return_inverse=True)
        _ = list(map(lambda x, y: x.set_msa_seq_index(y), self.mutations, mutation_idx))
        with get_context('spawn').Pool(NUM_THREADS) as p:
            # msa_dict: msa_seq, conservation, msa
            self.msa_dict = p.starmap(utils.get_msa_dict_from_transcript, zip(self.msa_file_dict))
        print(f'Finished loading {len(self.msa_dict)} msa seqs')
        
    def _set_node_embeddings(self):
        pass

    def _set_edge_embeddings(self):
        pass
    
    def get_mask(self, mutation: utils.Mutation):
        return mutation.pos - 1, mutation

    def get_graph_and_mask(self, mutation: utils.Mutation):
        # get the ordinary graph
        coords: np.ndarray = self.af2_coord_dict[mutation.af2_seq_index]  # N, C, O, CA, CB
        # remember we could have cropped sequence
        if mutation.crop:
            coords = coords[mutation.seq_start - 1:mutation.seq_end, :]
        # get the mask
        mask_idx, mutation = self.get_mask(mutation)
        # prepare edge features
        if self.add_msa_contacts:
            coevo_strength = utils.get_contacts_from_msa(mutation, False)
            if isinstance(coevo_strength, int):
                coevo_strength = np.zeros([mutation.seq_end - mutation.seq_start + 1,
                                        mutation.seq_end - mutation.seq_start + 1, 1])
        else:
            coevo_strength = np.zeros([mutation.seq_end - mutation.seq_start + 1,
                                        mutation.seq_end - mutation.seq_start + 1, 0])
        start = time.time()
        if self.add_af2_pairwise:
            if self.loaded_af2_pairwise:
                # we don't use the self.af2_pair_dict anymore because it won't fit in RAM
                # pairwise_rep = self.af2_pair_dict[mutation.af2_rep_index]
                # we load from lmdb
                byteflow = self.af2_pairwise_txn.get(u'{}'.format(mutation.af2_rep_file_prefix.split('/')[-1]).encode('ascii'))
                pairwise_rep = pickle.loads(byteflow)
                if pairwise_rep is None:
                    pairwise_rep = utils.get_af2_pairwise_rep_dict_from_prefix(mutation.af2_rep_file_prefix)
                # instead we load from lmdb
                # if not hasattr(self, 'af2_pairwise_txn'):
                #     # open all lmdb in self.af2_pair_dict_lmdb_path
                #     self.af2_pairwise_env = []
                #     self.af2_pairwise_txn = []
                #     for lmdb_path in self.af2_pair_dict_lmdb_path:
                #         af2_pairwise_env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
                #         self.af2_pairwise_txn.append(af2_pairwise_env.begin(write=False, buffers=True))
                #         self.af2_pairwise_env.append(af2_pairwise_env)
                # byteflow = self.af2_pairwise_txn[mutation.af2_rep_index // 20].get(u'{}'.format(mutation.af2_rep_index).encode('ascii'))
                # pairwise_rep = pickle.loads(byteflow)
            else:
                pairwise_rep = utils.get_af2_pairwise_rep_dict_from_prefix(mutation.af2_rep_file_prefix)
            # crop the pairwise_rep, if necessary
            if mutation.af2_rep_file_prefix.find('-F') == -1:
                pairwise_rep = pairwise_rep[mutation.seq_start_orig - 1: mutation.seq_end_orig,
                                            mutation.seq_start_orig - 1: mutation.seq_end_orig]
            if mutation.crop:
                pairwise_rep = pairwise_rep[mutation.seq_start - 1: mutation.seq_end,
                                            mutation.seq_start - 1: mutation.seq_end]
            coevo_strength = np.concatenate([coevo_strength, pairwise_rep], axis=2)
        end = time.time()
        print(f'Finished loading pairwise in {end - start:.2f} seconds')
        edge_attr = coevo_strength # N, N, 1
        # if add positional embedding, add it here
        if self.add_position:
            # add a sin positional embedding that reflects the relative position of the residue
            edge_position = np.arange(coords.shape[0])[:, None] - np.arange(coords.shape[0])[None, :]
            edge_attr = np.concatenate(
                (edge_attr, np.sin(np.pi / 2 * edge_position / self.max_len)[:, :, None]), 
                axis=2)
        return coords, None, None, edge_attr, None, mask_idx, mutation

    def get_one_mutation(self, idx):
        mutation: utils.Mutation = self.mutations[idx]
        # get the graph
        coords, _, _, edge_attr, _, mask_idx, mutation = self.get_graph_and_mask(mutation)
        # get embeddings
        if self.node_embedding_type == 'esm':
            if self.loaded_esm:
                # esm embeddings have <start> token, so starts at 1
                embed_data = self.esm_dict[mutation.esm_seq_index][mutation.seq_start:mutation.seq_end + 1]
            else:
                embed_data = utils.get_embedding_from_esm2(mutation.ESM_prefix, False,
                                                           mutation.seq_start, mutation.seq_end)
            to_alt = np.concatenate([utils.ESM_AA_EMBEDDING_DICT[alt_aa].reshape(1, -1) for alt_aa in mutation.alt_aa])
            to_ref = np.concatenate([utils.ESM_AA_EMBEDDING_DICT[ref_aa].reshape(1, -1) for ref_aa in mutation.ref_aa])
        elif self.node_embedding_type == 'one-hot-idx':
            assert not self.add_conservation and not self.add_plddt
            embed_logits, embed_data, one_hot_mat = utils.get_embedding_from_onehot_nonzero(mutation.seq, return_idx=True, return_onehot_mat=True)
            to_alt = np.concatenate([np.array(utils.AA_DICT.index(alt_aa)).reshape(1, -1) for alt_aa in mutation.alt_aa])
            to_ref = np.concatenate([np.array(utils.AA_DICT.index(ref_aa)).reshape(1, -1) for ref_aa in mutation.ref_aa])
        elif self.node_embedding_type == 'one-hot':
            embed_data, one_hot_mat = utils.get_embedding_from_onehot(mutation.seq, return_idx=False, return_onehot_mat=True)
            to_alt = np.concatenate([np.eye(len(utils.AA_DICT))[utils.AA_DICT.index(alt_aa)].reshape(1, -1) for alt_aa in mutation.alt_aa])
            to_ref = np.concatenate([np.eye(len(utils.AA_DICT))[utils.AA_DICT.index(ref_aa)].reshape(1, -1) for ref_aa in mutation.ref_aa])
        elif self.node_embedding_type == 'aa-5dim':
            embed_data = utils.get_embedding_from_5dim(mutation.seq)
            to_alt = np.concatenate([np.array(utils.AA_5DIM_EMBED[alt_aa]).reshape(1, -1) for alt_aa in mutation.alt_aa])
            to_ref = np.concatenate([np.array(utils.AA_5DIM_EMBED[ref_aa]).reshape(1, -1) for ref_aa in mutation.ref_aa])
        elif self.node_embedding_type == 'esm1b':
            embed_data = utils.get_embedding_from_esm1b(mutation.ESM_prefix, False,
                                                        mutation.seq_start, mutation.seq_end)
            to_alt = np.concatenate([utils.ESM1b_AA_EMBEDDING_DICT[alt_aa].reshape(1, -1) for alt_aa in mutation.alt_aa])
            to_ref = np.concatenate([utils.ESM1b_AA_EMBEDDING_DICT[ref_aa].reshape(1, -1) for ref_aa in mutation.ref_aa])
        # add conservation, if needed
        if self.loaded_msa and (self.add_msa or self.add_conservation):
            msa_seq = self.msa_dict[mutation.msa_seq_index][0]
            conservation_data = self.msa_dict[mutation.msa_seq_index][1]
            msa_data = self.msa_dict[mutation.msa_seq_index][2]
        else:
            if self.add_conservation or self.add_msa:
                msa_seq, conservation_data, msa_data = utils.get_msa_dict_from_transcript(mutation.uniprot_id)
        if self.add_conservation:
            if conservation_data.shape[0] == 0:
                conservation_data = np.zeros((embed_data.shape[0], 20))
            else:
                msa_seq_check = msa_seq[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                conservation_data = conservation_data[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                if mutation.crop:
                    msa_seq_check = msa_seq_check[mutation.seq_start - 1: mutation.seq_end]
                    conservation_data = conservation_data[mutation.seq_start - 1: mutation.seq_end]
                if msa_seq_check != mutation.seq:
                    # warnings.warn(f'MSA file: {mutation.transcript_id} does not match mutation sequence')
                    self.unmatched_msa += 1
                    print(f'Unmatched MSA: {self.unmatched_msa}')
                    conservation_data = np.zeros((embed_data.shape[0], 20))
            embed_data = np.concatenate([embed_data, conservation_data], axis=1)
            to_alt = np.concatenate([to_alt, conservation_data[mask_idx]], axis=1)
            if self.alt_type == 'diff':
                to_ref = np.concatenate([to_ref, conservation_data[mask_idx]], axis=1)
        # add pLDDT, if needed
        if self.add_plddt:
            # get plddt
            plddt_data = self.af2_plddt_dict[mutation.af2_seq_index]  # N, C, O, CA, CB
            if mutation.crop:
                plddt_data = plddt_data[mutation.seq_start - 1: mutation.seq_end]
            if self.add_confidence:
                confidence_data = plddt_data / 100
            if plddt_data.shape[0] != embed_data.shape[0]:
                warnings.warn(f'pLDDT {plddt_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                f'pLDDT file: {mutation.af2_file}, '
                                f'ESM prefix: {mutation.ESM_prefix}')
                plddt_data = np.ones_like(embed_data[:, 0]) * 50
                if self.add_confidence:
                    # assign 0.5 confidence to all points
                    confidence_data = np.ones_like(embed_data[:, 0]) / 2
            if self.scale_plddt:
                plddt_data = plddt_data / 100
            embed_data = np.concatenate([embed_data, plddt_data[:, None]], axis=1)
            to_alt = np.concatenate([to_alt, plddt_data[mask_idx, None]], axis=1)
            if self.alt_type == 'diff':
                to_ref = np.concatenate([to_ref, plddt_data[mask_idx]], axis=1)
        # add dssp, if needed
        if self.add_dssp:
            # get dssp
            dssp_data = self.af2_dssp_dict[mutation.af2_seq_index]
            if mutation.crop:
                dssp_data = dssp_data[mutation.seq_start - 1: mutation.seq_end]
            if dssp_data.shape[0] != embed_data.shape[0]:
                warnings.warn(f'DSSP {dssp_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                f'DSSP file: {mutation.af2_file}, '
                                f'ESM prefix: {mutation.ESM_prefix}')
                dssp_data = np.zeros_like(embed_data[:, 0])
            # if dssp_data size axis is 1, add a dimension
            if len(dssp_data.shape) == 1:
                dssp_data = dssp_data[:, None]
            embed_data = np.concatenate([embed_data, dssp_data], axis=1)
            to_alt = np.concatenate([to_alt, dssp_data[mask_idx]], axis=1)
            if self.alt_type == 'diff':
                to_ref = np.concatenate([to_ref, dssp_data[mask_idx]], axis=1)
        if self.add_ptm:
            # ptm used to behind msa, moved it here
            ptm_data = utils.get_ptm_from_mutation(mutation, self.ptm_ref)
            embed_data = np.concatenate([embed_data, ptm_data], axis=1)
            to_alt = np.concatenate([to_alt, ptm_data[mask_idx]], axis=1)
            if self.alt_type == 'diff':
                to_ref = np.concatenate([to_ref, ptm_data[mask_idx]], axis=1)
        if self.add_af2_single:
            if self.loaded_af2_single:
                single_rep = self.af2_single_dict[mutation.af2_rep_index]
            else:
                single_rep = utils.get_af2_single_rep_dict_from_prefix(mutation.af2_rep_file_prefix)
            # crop the pairwise_rep, if necessary
            if mutation.af2_rep_file_prefix.find('-F') == -1:
                single_rep = single_rep[mutation.seq_start_orig - 1: mutation.seq_end_orig]
            if mutation.crop:
                single_rep = single_rep[mutation.seq_start - 1: mutation.seq_end]
            embed_data = np.concatenate([embed_data, single_rep], axis=1)
            to_alt = np.concatenate([to_alt, single_rep[mask_idx]], axis=1)
            if self.alt_type == 'diff':
                to_ref = np.concatenate([to_ref, single_rep[mask_idx]], axis=1)
        if self.add_msa:
            if msa_data.shape[0] == 0:
                msa_data = np.zeros((embed_data.shape[0], 199))
            else:
                msa_seq_check = msa_seq[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                msa_data = msa_data[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                if mutation.crop:
                    msa_seq_check = msa_seq_check[mutation.seq_start - 1: mutation.seq_end]
                    msa_data = msa_data[mutation.seq_start - 1: mutation.seq_end]
                if msa_seq_check != mutation.seq:
                    # warnings.warn(f'MSA file: {mutation.transcript_id} does not match mutation sequence')
                    msa_data = np.zeros((embed_data.shape[0], 199))
            embed_data = np.concatenate([embed_data, msa_data], axis=1)
            if self.alt_type == 'alt':
                to_alt = np.concatenate([to_alt, msa_data[mask_idx]], axis=1)
            if self.alt_type == 'diff':
                to_ref = np.concatenate([to_ref, msa_data[mask_idx]], axis=1)
        # replace the embedding with the mutation, note pos is 1-based
        # but we don't modify the embedding matrix, instead we return a mask matrix
        embed_data_mask = np.ones_like(embed_data)
        embed_data_mask[mask_idx] = 0
        if self.alt_type == 'alt':
            alt_embed_data = np.zeros_like(embed_data)
            alt_embed_data[mask_idx] = to_alt
        elif self.alt_type == 'concat':
            alt_embed_data = np.zeros((embed_data.shape[0], to_alt.shape[1] + to_ref.shape[1]))
            alt_embed_data[mask_idx] = np.concatenate([to_alt, to_ref], axis=1)
        elif self.alt_type == 'diff':
            alt_embed_data = np.zeros_like(embed_data)
            alt_embed_data[mask_idx] = to_alt
            embed_data[mask_idx] = to_ref
        else:
            raise ValueError(f'alt_type {self.alt_type} not supported')
        # prepare node vector features
        # get CA_coords
        CA_coord = coords[:, 3]
        CB_coord = coords[:, 4]
        # add CB_coord for GLY
        CB_coord[np.isnan(CB_coord)] = CA_coord[np.isnan(CB_coord)]
        if self.graph_type == '1d-neighbor':
            CA_coord[:, 0] = np.arange(coords.shape[0])
            CB_coord[:, 0] = np.arange(coords.shape[0])
            coords = np.zeros_like(coords)
        CA_CB = coords[:, [4]] - coords[:, [3]]  # Note that glycine does not have CB
        CA_CB[np.isnan(CA_CB)] = 0
        # Change the CA_CB of the mutated residue to 0
        # but we don't modify the CA_CB matrix, instead we return a mask matrix
        CA_C = coords[:, [1]] - coords[:, [3]]
        CA_O = coords[:, [2]] - coords[:, [3]]
        CA_N = coords[:, [0]] - coords[:, [3]]
        nodes_vector = np.transpose(np.concatenate([CA_CB, CA_C, CA_O, CA_N], axis=1), (0, 2, 1))
        if self.add_sidechain:
            # get sidechain coords
            sidechain_nodes_vector = coords[:, 5:] - coords[:, [3]]
            sidechain_nodes_vector[np.isnan(sidechain_nodes_vector)] = 0
            sidechain_nodes_vector = np.transpose(sidechain_nodes_vector, (0, 2, 1))
            nodes_vector = np.concatenate([nodes_vector, sidechain_nodes_vector], axis=2)
        # prepare graph
        features = dict(
            embed_logits=embed_logits if self.node_embedding_type == 'one-hot-idx' else None,
            one_hot_mat=one_hot_mat if self.node_embedding_type.startswith('one-hot') else None,
            mask_idx=mask_idx,
            embed_data=embed_data,
            embed_data_mask=embed_data_mask,
            alt_embed_data=alt_embed_data,
            coords=coords,
            CA_coord=CA_coord,
            CB_coord=CB_coord,
            edge_index=None,
            edge_index_star=None,
            edge_attr=edge_attr,
            edge_attr_star=None,
            nodes_vector=nodes_vector,
        )
        if self.add_confidence:
            # add position wise confidence
            if self.add_plddt:
                features['plddt'] = confidence_data
                if self.loaded_confidence:
                    pae = self.af2_confidence_dict[mutation.af2_seq_index]
                else:
                    pae = utils.get_confidence_from_af2file(mutation.af2_file, self.af2_plddt_dict[mutation.af2_seq_index])
                if mutation.crop:
                    pae = pae[mutation.seq_start - 1: mutation.seq_end, mutation.seq_start - 1: mutation.seq_end]
            else:
                # get plddt
                plddt_data = utils.get_plddt_from_af2(mutation.af2_file)
                pae = utils.get_confidence_from_af2file(mutation.af2_file, plddt_data)
                if mutation.crop:
                    confidence_data = plddt_data[mutation.seq_start - 1: mutation.seq_end] / 100
                    pae = pae[mutation.seq_start - 1: mutation.seq_end, mutation.seq_start - 1: mutation.seq_end]
                if confidence_data.shape[0] != embed_data.shape[0]:
                    warnings.warn(f'pLDDT {confidence_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                    f'pLDDT file: {mutation.af2_file}, '
                                    f'ESM prefix: {mutation.ESM_prefix}')
                    confidence_data = np.ones_like(embed_data[:, 0]) * 0.8
                features['plddt'] = confidence_data
            # add pairwise confidence
            features['edge_confidence'] = pae
        return features

    def get(self, idx):
        start_time=time.time()
        features_np = self.get_one_mutation(idx)
        if self.node_embedding_type == 'one-hot-idx':
            x = torch.from_numpy(features_np['embed_data']).to(torch.long)
        else:
            x = torch.from_numpy(features_np['embed_data']).to(torch.float32)
        # padding x to the max length
        x_padding_mask = torch.zeros(self.max_len, dtype=torch.bool)
        pos=torch.from_numpy(features_np['CB_coord']).to(torch.float32) if self.use_cb else torch.from_numpy(features_np['CA_coord']).to(torch.float32)
        node_vec_attr=torch.from_numpy(features_np['nodes_vector']).to(torch.float32)
        edge_attr=torch.from_numpy(features_np['edge_attr']).to(torch.float32)
        x_mask=torch.from_numpy(features_np['embed_data_mask'][:, 0]).to(torch.bool)
        x_alt=torch.from_numpy(features_np['alt_embed_data']).to(torch.float32)
        if self.add_confidence:
            plddt=torch.from_numpy(features_np['plddt']).to(torch.float32)
            edge_confidence=torch.from_numpy(features_np['edge_confidence']).to(torch.float32)
        if x.shape[0] < self.max_len:
            x_padding_mask[x.shape[0]:] = True
            x = torch.nn.functional.pad(x, (0, 0, 0, self.max_len - x.shape[0]))
            pos = torch.nn.functional.pad(pos, (0, 0, 0, self.max_len - pos.shape[0]))
            node_vec_attr = torch.nn.functional.pad(node_vec_attr, (0, 0, 0, 0, 0, self.max_len - node_vec_attr.shape[0]))
            edge_attr = torch.nn.functional.pad(edge_attr, (0, 0, 0, self.max_len - edge_attr.shape[0], 0, self.max_len - edge_attr.shape[0]))
            x_alt = torch.nn.functional.pad(x_alt, (0, 0, 0, self.max_len - x_alt.shape[0]))
            x_mask = torch.nn.functional.pad(x_mask, (0, self.max_len - x_mask.shape[0]), 'constant', True)
            if self.add_confidence:
                edge_confidence = torch.nn.functional.pad(edge_confidence, (0, self.max_len - edge_confidence.shape[0], 0, self.max_len - edge_confidence.shape[0]))
                plddt = torch.nn.functional.pad(plddt, (0, self.max_len - plddt.shape[0]))
        features = dict(
            x=x,
            x_padding_mask=x_padding_mask,
            x_mask=x_mask,
            x_alt=x_alt,
            pos=pos,
            edge_attr=edge_attr,
            node_vec_attr=node_vec_attr,
            y=torch.tensor([self.data[self._y_columns].iloc[int(idx)]]).to(torch.float32).unsqueeze(0),
        )
        if self.add_confidence:
            features['plddt'] = plddt
            features['edge_confidence'] = edge_confidence
        print(f'Finished loading {idx}th mutation in {time.time() - start_time} seconds')
        return features

    def get_from_hdf5(self, idx):
        if not hasattr(self, 'hdf5_keys') or self.hdf5_file is None:
            raise ValueError('hdf5 file is not set')
        else:
            features = {}
            with h5py.File(self.hdf5_file, 'r') as f:
                for key in self.hdf5_keys:
                    features[key] = torch.tensor(f[f'{self.hdf5_idx_map[idx]}/{key}'])
            return Data(**features)
    
    def open_lmdb(self):
         self.env = lmdb.open(self.lmdb_path, subdir=False,
                              readonly=True, lock=False,
                              readahead=False, meminit=False)
         self.txn = self.env.begin(write=False, buffers=True)
    
    def get_from_lmdb(self, idx):
        if not hasattr(self, 'txn') or self.txn is None:
            self.open_lmdb()
        byteflow = self.txn.get(u'{}'.format(self.lmdb_idx_map[idx]).encode('ascii'))
        if byteflow is None:
            return self.get(idx)
        else:
            unpacked = pickle.loads(byteflow)
            return unpacked
    
    def __getitem__(self, idx):
        # record time
        start = time.time()
        if self.get_method == 'default':
            data = self.get(idx)
            print(f'default Finished loading {idx} in {time.time() - start:.2f} seconds')
        elif self.get_method == 'hdf5':
            data = self.get_from_hdf5(idx)
            print(f'hdf5 Finished loading {idx} in {time.time() - start:.2f} seconds')
        elif self.get_method == 'lmdb':
            data = self.get_from_lmdb(idx)
            print(f'lmdb Finished loading {idx} in {time.time() - start:.2f} seconds')
        return data

    def __len__(self):
        return len(self.mutations)

    def len(self) -> int:
        return len(self.mutations)
    
    def subset(self, idxs):
        self.data = self.data.iloc[idxs].reset_index(drop=True)
        self.mutations = list(map(self.mutations.__getitem__, idxs))
        # get unique af2 graphs
        subset_af2_file_dict, mutation_idx = np.unique([mutation.af2_file for mutation in self.mutations],
                                                       return_inverse=True)
        # find the index of the af2 file in the subset
        if self.af2_file_dict is not None:
            af2_file_idx = np.array([np.where(self.af2_file_dict==i)[0][0] for i in subset_af2_file_dict])
            self.af2_file_dict = subset_af2_file_dict
            # get the subset of af2 graphs
            self.af2_coord_dict = list(map(self.af2_coord_dict.__getitem__, af2_file_idx)) if self.af2_coord_dict is not None else None
            self.af2_plddt_dict = list(map(self.af2_plddt_dict.__getitem__, af2_file_idx)) if self.af2_plddt_dict is not None else None
            self.af2_confidence_dict = list(map(self.af2_confidence_dict.__getitem__, af2_file_idx)) if self.af2_confidence_dict is not None else None
            self.af2_dssp_dict = list(map(self.af2_dssp_dict.__getitem__, af2_file_idx)) if self.af2_dssp_dict is not None else None
            self.af2_graph_dict = list(map(self.af2_graph_dict.__getitem__, af2_file_idx)) if self.af2_graph_dict is not None else None
            # reset the af2_seq_index
            _ = list(map(lambda x, y: x.set_af2_seq_index(y), self.mutations, mutation_idx))
        # get unique esm files
        if self.esm_file_dict is not None:
            subset_esm_file_dict, mutation_idx = np.unique([mutation.ESM_prefix for mutation in self.mutations],
                                                        return_inverse=True)
            # find the index of the esm file in the subset
            esm_file_idx = np.array([np.where(self.esm_file_dict==i)[0][0] for i in subset_esm_file_dict])
            self.esm_file_dict = subset_esm_file_dict
            # get the subset of esm embeddings
            self.esm_dict = list(map(self.esm_dict.__getitem__, esm_file_idx)) if self.esm_dict is not None else None
            # reset the esm_seq_index
            _ = list(map(lambda x, y: x.set_esm_seq_index(y), self.mutations, mutation_idx))
        # get unique msa files
        if self.msa_file_dict is not None:
            subset_msa_file_dict, mutation_idx = np.unique([mutation.uniprot_id for mutation in self.mutations],
                                                           return_inverse=True)
            # find the index of the msa file in the subset
            msa_file_idx = np.array([np.where(self.msa_file_dict==i)[0][0] for i in subset_msa_file_dict])
            self.msa_file_dict = subset_msa_file_dict
            # get the subset of msa embeddings
            self.msa_dict = list(map(self.msa_dict.__getitem__, msa_file_idx)) if self.msa_dict is not None else None
            # reset the msa_seq_index
            _ = list(map(lambda x, y: x.set_msa_seq_index(y), self.mutations, mutation_idx))
        return self

    def shuffle(self, idxs):
        # for shuffle, we only need to shuffle self.mutations and self.data
        self.data = self.data.iloc[idxs].reset_index(drop=True)
        self.mutations = list(map(self.mutations.__getitem__, idxs))

    def get_label_counts(self) -> np.ndarray:
        if self.data.columns.isin(['score']).any():
            if (-1 in self.data['score'].values):
                lof = (self.data['score']==-1).sum()
                benign = (self.data['score']==0).sum()
                gof = (self.data['score']==1).sum()
                patho = (self.data['score']==3).sum()
                if lof != 0 and gof != 0:
                    return np.array([lof, benign, gof, patho])
                else:
                    return np.array([benign, patho])
            else:
                benign = (self.data['score']==0).sum()
                patho = (self.data['score']==1).sum()
                return np.array([benign, patho])
        else:
            return np.array([0, 0])

    # create a hdf5 file for the dataset, for faster loading        
    def create_hdf5(self):
        hdf5_file = self.data_file.replace('.csv', '.hdf5')
        self.hdf5_file = hdf5_file
        self.get_method = 'hdf5'
        self.hdf5_keys = None
        # create a mapping from mutation index to hdf5 index, in case of subset or shuffle
        self.hdf5_idx_map = np.arange(len(self))
        with h5py.File(hdf5_file, 'w') as f:
            for i in range(len(self)):
                features = self.get(i)
                # store feature keys into self
                if self.hdf5_keys is None:
                    self.hdf5_keys = list(features.keys())
                for key in features.keys():
                    f.create_dataset(f'{i}/{key}', data=features[key])
        return
    
    # create a lmdb file for the dataset, for faster loading
    def create_lmdb(self, write_frequency=1000):
        lmdb_path = self.data_file.replace('.csv', f'.{datetime.now()}.lmdb')
        map_size = 5e12 # 5TB
        db = lmdb.open(lmdb_path, subdir=False, map_size=map_size, readonly=False, meminit=False, map_async=True)
        print(f"Begin loading {len(self)} points into lmdb")
        txn = db.begin(write=True)
        for idx in range(len(self)):
            d = self.get(idx)
            txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps(d))
            print(f'Finished loading {idx}')
            if (idx + 1) % write_frequency == 0:
                txn.commit()
                txn = db.begin(write=True)
        txn.commit()
        print(f"Finished loading {len(self)} points into lmdb")
        self.lmdb_path = lmdb_path
        self.lmdb_idx_map = np.arange(len(self))
        self.get_method = 'lmdb'
        print("Flushing database ...")
        db.sync()
        db.close()
        return

    # clean up hdf5 and lmdb files
    def clean_up(self):
        if hasattr(self, 'hdf5_file') and self.hdf5_file is not None and os.path.exists(self.hdf5_file):
            os.remove(self.hdf5_file)
        if hasattr(self, 'lmdb_path') and self.lmdb_path is not None and os.path.exists(self.lmdb_path):
            os.remove(self.lmdb_path)
        return


class MutationDataset(GraphMutationDataset):
    """
    MutationDataSet dataset, input a file of mutations, output without graph.
    Can be either single mutation or multiple mutations.

    Args:
        data_file (string or pd.DataFrame): Path or pd.DataFrame for a csv file for a list of mutations
        data_type (string): Type of this data, 'ClinVar', 'DMS', etc
    """

    def __init__(self, data_file, data_type: str,
                 radius: float = None, max_neighbors: int = 50,
                 loop: bool = False, shuffle: bool = False, gpu_id: int = None,
                 node_embedding_type: Literal['esm', 'one-hot-idx', 'one-hot', 'aa-5dim'] = 'esm',
                 graph_type: Literal['af2', '1d-neighbor'] = 'af2',
                 precomputed_graph: bool = False,
                 add_plddt: bool = False, 
                 add_conservation: bool = False,
                 add_position: bool = False,
                 add_msa_contacts: bool = True,
                 max_len: int = 700,
                 padding: bool = False,
                 ):
        self.padding = padding
        super(MutationDataset, self).__init__(data_file, data_type, radius, max_neighbors, loop, shuffle, gpu_id,
                                              node_embedding_type, graph_type, precomputed_graph, add_plddt, add_conservation,
                                              add_position, add_msa_contacts,
                                              max_len)

    def __getitem__(self, idx):
        features_np = self.get_one_mutation(idx)
        orig_len = features_np['embed_data'].shape[0]
        if self.padding and orig_len < self.max_len:
            features_np['embed_data'] = np.pad(features_np['embed_data'], ((0, self.max_len - orig_len), (0, 0)), 'constant')
            features_np['coords'] = np.pad(features_np['coords'], ((0, self.max_len - orig_len), (0, 0), (0, 0)), 'constant')
            features_np['alt_embed_data'] = np.pad(features_np['alt_embed_data'], ((0, self.max_len - orig_len), (0, 0)), 'constant')
            features_np['embed_data_mask'] = np.pad(features_np['embed_data_mask'], ((0, self.max_len - orig_len), (0, 0)), 'constant')
            y_mask = np.concatenate((np.ones(orig_len), np.zeros(self.max_len - orig_len)))
        else:
            y_mask = np.ones(orig_len)
        # prepare data
        if self.node_embedding_type == 'one-hot-idx':
            x = torch.from_numpy(features_np['embed_data']).to(torch.long)
        else:
            x = torch.from_numpy(features_np['embed_data']).to(torch.float32)
        features = dict(
            x=x,
            x_mask=torch.from_numpy(features_np['embed_data_mask']).to(torch.bool),
            x_alt=torch.from_numpy(features_np['alt_embed_data']).to(torch.float32),
            pos=torch.from_numpy(features_np['coords']).to(torch.float32),
            edge_index=torch.tensor([torch.nan]),
            edge_index_star=torch.tensor([torch.nan]),
            edge_attr=torch.tensor([torch.nan]),
            edge_attr_star=torch.tensor([torch.nan]),
            node_vec_attr=torch.tensor([torch.nan]),
            y=torch.tensor(self.data[self._y_columns].iloc[int(idx)]).to(torch.float32),
            y_mask=torch.from_numpy(y_mask).to(torch.bool), # padding mask
        )
        return features 
    
    def get(self, idx):
        return self.__getitem__(idx)


class GraphMaskPredictMutationDataset(GraphMutationDataset):
    """
    MutationDataSet dataset, input a file of mutations, output without graph.
    Can be either single mutation or multiple mutations.

    Args:
        data_file (string or pd.DataFrame): Path or pd.DataFrame for a csv file for a list of mutations
        data_type (string): Type of this data, 'ClinVar', 'DMS', etc
    """

    def __init__(self, data_file, data_type: str,
                 radius: float = None, max_neighbors: int = 50,
                 loop: bool = False, shuffle: bool = False, gpu_id: int = None,
                 node_embedding_type: Literal['one-hot-idx', 'one-hot'] = 'one-hot-idx',
                 graph_type: Literal['af2', '1d-neighbor'] = 'af2',
                 add_plddt: bool = False, 
                 add_conservation: bool = False,
                 add_position: bool = False,
                 add_msa_contacts: bool = True,
                 computed_graph: bool = True,
                 neighbor_type: Literal['KNN', 'radius'] = 'KNN',
                 max_len: int = 700,
                 mask_percentage: float = 0.15,
                 ):
        self.mask_percentage = mask_percentage
        super(GraphMaskPredictMutationDataset, self).__init__(
            data_file, data_type, radius, max_neighbors, loop, shuffle, gpu_id,
            node_embedding_type, graph_type, add_plddt, add_conservation,
            add_position, add_msa_contacts, computed_graph, neighbor_type,
            max_len)
    
    def get_mask(self, mutation: utils.Mutation):
        # randomly mask self.mask_percentage of the residues
        seq_len = mutation.seq_end - mutation.seq_start + 1
        if not pd.isna(mutation.alt_aa):
            # add the point mutation to random mask
            points_to_mask = int(seq_len * self.mask_percentage)
            if points_to_mask > 1:
                mask_idx = np.random.choice(seq_len, int(seq_len * 0.15) - 1, replace=False)
                mask_idx = np.append(mask_idx, mutation.pos - 1)
            else:
                mask_idx = np.array([mutation.pos - 1])
        else:
            mask_idx = np.random.choice(seq_len, int(seq_len * 0.15), replace=False)
        mutation.ref_aa = np.array(list(mutation.seq))[mask_idx]
        mutation.alt_aa = np.array(['<mask>'] * (len(mask_idx)))
        return mask_idx, mutation

    def get(self, idx):
        features_np = self.get_one_mutation(idx)
        embed_logits = features_np['embed_logits']
        one_hot_mat = features_np['one_hot_mat']
        mutation: utils.Mutation = self.mutaions[idx]
        # change embed logits to mask
        if not pd.isna(mutation.alt_aa):
            embed_logits[mutation.pos - 1] = (one_hot_mat[utils.AA_DICT.index(mutation.ref_aa)]
                                              + one_hot_mat[utils.AA_DICT.index(mutation.alt_aa)]) / 2
        # prepare data
        if self.node_embedding_type == 'one-hot-idx':
            x = torch.from_numpy(features_np['embed_data']).to(torch.long)
        else:
            x = torch.from_numpy(features_np['embed_data']).to(torch.float32)
        features = dict(
            x=x,
            x_mask=torch.from_numpy(features_np['embed_data_mask']).to(torch.bool),
            x_alt=torch.from_numpy(features_np['alt_embed_data']).to(torch.float32),
            pos=torch.from_numpy(features_np['CA_coord']).to(torch.float32),
            edge_index=torch.from_numpy(features_np['edge_index']).to(torch.long),
            edge_index_star=torch.from_numpy(features_np['edge_index_star']).to(torch.long),
            edge_attr=torch.from_numpy(features_np['edge_attr']).to(torch.float32),
            edge_attr_star=torch.from_numpy(features_np['edge_attr_star']).to(torch.float32),
            node_vec_attr=torch.from_numpy(features_np['nodes_vector']).to(torch.float32),
            y=torch.from_numpy(embed_logits).to(torch.float32),
        )
        features["edge_index"], features["edge_attr"], mask = \
            remove_isolated_nodes(features["edge_index"], features["edge_attr"], x.shape[0])
        features["edge_index_star"], features["edge_attr_star"], mask = \
            remove_isolated_nodes(features["edge_index_star"], features["edge_attr_star"], x.shape[0])
        features["x"] = features["x"][mask]
        features["x_mask"] = features["x_mask"][mask]
        features["x_alt"] = features["x_alt"][mask]
        features["pos"] = features["pos"][mask]
        features["node_vec_attr"] = features["node_vec_attr"][mask]
        return Data(**features)


class MaskPredictMutationDataset(GraphMaskPredictMutationDataset):
    """
    MutationDataSet dataset, input a file of mutations, output without graph.
    Can be either single mutation or multiple mutations.

    Args:
        data_file (string or pd.DataFrame): Path or pd.DataFrame for a csv file for a list of mutations
        data_type (string): Type of this data, 'ClinVar', 'DMS', etc
    """

    def __init__(self, data_file, data_type: str,
                 radius: float = None, max_neighbors: int = 50,
                 loop: bool = False, shuffle: bool = False, gpu_id: int = None,
                 node_embedding_type: Literal['one-hot-idx', 'one-hot'] = 'one-hot-idx',
                 graph_type: Literal['af2', '1d-neighbor'] = 'af2',
                 precomputed_graph: bool = False,
                 add_plddt: bool = False, 
                 add_conservation: bool = False,
                 add_position: bool = False,
                 add_msa_contacts: bool = True,
                 max_len: int = 700,
                 padding: bool = False,
                 mask_percentage: float = 0.15,
                 ):
        self.padding = padding
        super(MaskPredictMutationDataset, self).__init__(
            data_file, data_type, radius, max_neighbors, loop, shuffle, gpu_id,
            node_embedding_type, graph_type, precomputed_graph, add_plddt, add_conservation,
            add_position, add_msa_contacts,
            max_len, mask_percentage)
    
    def get_mask(self, mutation: utils.Mutation):
        # randomly mask self.mask_percentage of the residues
        seq_len = mutation.seq_end - mutation.seq_start + 1
        if not pd.isna(mutation.alt_aa):
            # add the point mutation to random mask
            points_to_mask = int(seq_len * self.mask_percentage)
            if points_to_mask > 1:
                mask_idx = np.random.choice(seq_len, int(seq_len * 0.15) - 1, replace=False)
                mask_idx = np.append(mask_idx, mutation.pos - 1)
            else:
                mask_idx = np.array([mutation.pos - 1])
        else:
            mask_idx = np.random.choice(seq_len, int(seq_len * 0.15), replace=False)
        mutation.ref_aa = np.array(list(mutation.seq))[mask_idx]
        mutation.alt_aa = np.array(['<mask>'] * (len(mask_idx)))
        return mask_idx, mutation

    def __getitem__(self, idx):
        features_np = self.get_one_mutation(idx)
        embed_logits = features_np['embed_logits']
        one_hot_mat = features_np['one_hot_mat']
        mutation: utils.Mutation = self.mutaions[idx]
        # change embed logits to mask
        if not pd.isna(mutation.alt_aa):
            embed_logits[mutation.pos - 1] = (one_hot_mat[utils.AA_DICT.index(mutation.ref_aa)]
                                              + one_hot_mat[utils.AA_DICT.index(mutation.alt_aa)]) / 2
        # padding if necessary
        orig_len = features_np['embed_data'].shape[0]
        if self.padding and orig_len < self.max_len:
            features_np['embed_data'] = np.pad(features_np['embed_data'], ((0, self.max_len - orig_len), (0, 0)), 'constant')
            embed_logits = np.pad(embed_logits, ((0, self.max_len - orig_len), (0, 0)), 'constant')
            features_np['coords'] = np.pad(features_np['coords'], ((0, self.max_len - orig_len), (0, 0), (0, 0)), 'constant')
            features_np['alt_embed_data'] = np.pad(features_np['alt_embed_data'], ((0, self.max_len - orig_len), (0, 0)), 'constant')
            features_np['embed_data_mask'] = np.pad(features_np['embed_data_mask'], ((0, self.max_len - orig_len), (0, 0)), 'constant')
            y_mask = np.concatenate((np.ones(orig_len), np.zeros(self.max_len - orig_len)))
        else:
            y_mask = np.ones(orig_len)
        # prepare data
        if self.node_embedding_type == 'one-hot-idx':
            x = torch.from_numpy(features_np['embed_data']).to(torch.long)
        else:
            x = torch.from_numpy(features_np['embed_data']).to(torch.float32)
        features = dict(
            x=x,
            x_mask=torch.from_numpy(features_np['embed_data_mask']).to(torch.bool),
            x_alt=torch.from_numpy(features_np['alt_embed_data']).to(torch.float32),
            pos=torch.from_numpy(features_np['CA_coord']).to(torch.float32),
            edge_index=torch.tensor([torch.nan]),
            edge_index_star=torch.tensor([torch.nan]),
            edge_attr=torch.tensor([torch.nan]),
            edge_attr_star=torch.tensor([torch.nan]),
            node_vec_attr=torch.tensor([torch.nan]),
            y=torch.from_numpy(embed_logits).to(torch.float32),
            y_mask=torch.from_numpy(y_mask).to(torch.bool), # padding mask
        )
        return features


class GraphMultiOnesiteMutationDataset(GraphMutationDataset):
    def __init__(self, data_file, data_type: str,
                 radius: float = None, max_neighbors: int = None,
                 loop: bool = False, shuffle: bool = False, gpu_id: int = None,
                 node_embedding_type: Literal['esm', 'one-hot-idx', 'one-hot', 'aa-5dim', 'esm1b'] = 'esm',
                 graph_type: Literal['af2', '1d-neighbor'] = 'af2',
                 add_plddt: bool = False, 
                 scale_plddt: bool = False,
                 add_conservation: bool = False,
                 add_position: bool = False,
                 add_sidechain: bool = False,
                 local_coord_transform: bool = False,
                 use_cb: bool = False,
                 add_msa_contacts: bool = True,
                 add_dssp: bool = False,
                 add_msa: bool = False,
                 add_confidence: bool = False,
                 loaded_confidence: bool = False,
                 loaded_esm: bool = False,
                 add_ptm: bool = False,
                 data_augment: bool = False,
                 score_transfer: bool = False,
                 alt_type: Literal['alt', 'concat', 'diff'] = 'alt',
                 computed_graph: bool = True,
                 loaded_msa: bool = False,
                 neighbor_type: Literal['KNN', 'radius', 'radius-KNN'] = 'KNN',
                 max_len = 2251,
                 convert_to_onesite: bool = False,
                 add_af2_single: bool = False,
                 add_af2_pairwise: bool = False,
                 loaded_af2_single: bool = False,
                 loaded_af2_pairwise: bool = False,
                 ):
        super(GraphMultiOnesiteMutationDataset, self).__init__(
            data_file, data_type, radius, max_neighbors, loop, shuffle, gpu_id,
            node_embedding_type, graph_type, add_plddt, scale_plddt,
            add_conservation, add_position, add_sidechain,
            local_coord_transform, use_cb, add_msa_contacts, add_dssp,
            add_msa, add_confidence, loaded_confidence, loaded_esm,
            add_ptm, data_augment, score_transfer, alt_type,
            computed_graph, loaded_msa, neighbor_type, max_len)
        self._y_mask_columns = self.data.columns[self.data.columns.str.startswith('confidence.score')]

    def get_one_mutation(self, idx):
        mutation: utils.Mutation = self.mutations[idx]
        # get the graph
        coords, edge_index, edge_index_star, edge_attr, edge_attr_star, mask_idx, mutation = self.get_graph_and_mask(mutation)
        # get embeddings
        if self.node_embedding_type == 'esm':
            if self.loaded_esm:
                # esm embeddings have <start> token, so starts at 1
                embed_data = self.esm_dict[mutation.esm_seq_index][mutation.seq_start:mutation.seq_end + 1]
            else:
                embed_data = utils.get_embedding_from_esm2(mutation.ESM_prefix, False,
                                                           mutation.seq_start, mutation.seq_end)
        elif self.node_embedding_type == 'one-hot-idx':
            assert not self.add_conservation and not self.add_plddt
            embed_logits, embed_data, one_hot_mat = utils.get_embedding_from_onehot_nonzero(mutation.seq, return_idx=True, return_onehot_mat=True)
        elif self.node_embedding_type == 'one-hot':
            embed_data, one_hot_mat = utils.get_embedding_from_onehot(mutation.seq, return_idx=False, return_onehot_mat=True)
        elif self.node_embedding_type == 'aa-5dim':
            embed_data = utils.get_embedding_from_5dim(mutation.seq)
        elif self.node_embedding_type == 'esm1b':
            embed_data = utils.get_embedding_from_esm1b(mutation.ESM_prefix, False,
                                                        mutation.seq_start, mutation.seq_end)
        # add conservation, if needed
        if self.loaded_msa and (self.add_msa or self.add_conservation):
            msa_seq = self.msa_dict[mutation.msa_seq_index][0]
            conservation_data = self.msa_dict[mutation.msa_seq_index][1]
            msa_data = self.msa_dict[mutation.msa_seq_index][2]
        else:
            if self.add_conservation or self.add_msa:
                msa_seq, conservation_data, msa_data = utils.get_msa_dict_from_transcript(mutation.uniprot_id)
        if self.add_conservation:
            if conservation_data.shape[0] == 0:
                conservation_data = np.zeros((embed_data.shape[0], 20))
            else:
                msa_seq_check = msa_seq[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                conservation_data = conservation_data[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                if mutation.crop:
                    msa_seq_check = msa_seq_check[mutation.seq_start - 1: mutation.seq_end]
                    conservation_data = conservation_data[mutation.seq_start - 1: mutation.seq_end]
                if msa_seq_check != mutation.seq:
                    # warnings.warn(f'MSA file: {mutation.transcript_id} does not match mutation sequence')
                    self.unmatched_msa += 1
                    print(f'Unmatched MSA: {self.unmatched_msa}')
                    conservation_data = np.zeros((embed_data.shape[0], 20))
            embed_data = np.concatenate([embed_data, conservation_data], axis=1)
        # add pLDDT, if needed
        if self.add_plddt:
            # get plddt
            plddt_data = self.af2_plddt_dict[mutation.af2_seq_index]  # N, C, O, CA, CB
            if mutation.crop:
                plddt_data = plddt_data[mutation.seq_start - 1: mutation.seq_end]
            if self.add_confidence:
                confidence_data = plddt_data / 100
            if plddt_data.shape[0] != embed_data.shape[0]:
                warnings.warn(f'pLDDT {plddt_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                f'pLDDT file: {mutation.af2_file}, '
                                f'ESM prefix: {mutation.ESM_prefix}')
                plddt_data = np.ones_like(embed_data[:, 0]) * 50
                if self.add_confidence:
                    # assign 0.5 confidence to all points
                    confidence_data = np.ones_like(embed_data[:, 0]) / 2
            if self.scale_plddt:
                plddt_data = plddt_data / 100
            embed_data = np.concatenate([embed_data, plddt_data[:, None]], axis=1)
        # add dssp, if needed
        if self.add_dssp:
            # get dssp
            dssp_data = self.af2_dssp_dict[mutation.af2_seq_index]
            if mutation.crop:
                dssp_data = dssp_data[mutation.seq_start - 1: mutation.seq_end]
            if dssp_data.shape[0] != embed_data.shape[0]:
                warnings.warn(f'DSSP {dssp_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                f'DSSP file: {mutation.af2_file}, '
                                f'ESM prefix: {mutation.ESM_prefix}')
                dssp_data = np.zeros_like(embed_data[:, 0])
            # if dssp_data size axis is 1, add a dimension
            if len(dssp_data.shape) == 1:
                dssp_data = dssp_data[:, None]
            embed_data = np.concatenate([embed_data, dssp_data], axis=1)
        if self.add_msa:
            if msa_data.shape[0] == 0:
                msa_data = np.zeros((embed_data.shape[0], 199))
            else:
                msa_seq_check = msa_seq[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                msa_data = msa_data[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                if mutation.crop:
                    msa_seq_check = msa_seq_check[mutation.seq_start - 1: mutation.seq_end]
                    msa_data = msa_data[mutation.seq_start - 1: mutation.seq_end]
                if msa_seq_check != mutation.seq:
                    warnings.warn(f'MSA file: {mutation.transcript_id} does not match mutation sequence')
                    msa_data = np.zeros((embed_data.shape[0], 199))
            embed_data = np.concatenate([embed_data, msa_data], axis=1)
        if self.add_ptm:
            ptm_data = utils.get_ptm_from_mutation(mutation, self.ptm_ref)
            embed_data = np.concatenate([embed_data, ptm_data], axis=1)
        # replace the embedding with the mutation, note pos is 1-based
        # but we don't modify the embedding matrix, instead we return a mask matrix
        embed_data_mask = np.ones_like(embed_data)
        embed_data_mask[mask_idx] = 0
        # prepare node vector features
        # get CA_coords
        CA_coord = coords[:, 3]
        CB_coord = coords[:, 4]
        # add CB_coord for GLY
        CB_coord[np.isnan(CB_coord)] = CA_coord[np.isnan(CB_coord)]
        if self.graph_type == '1d-neighbor':
            CA_coord[:, 0] = np.arange(coords.shape[0])
            CB_coord[:, 0] = np.arange(coords.shape[0])
            coords = np.zeros_like(coords)
        CA_CB = coords[:, [4]] - coords[:, [3]]  # Note that glycine does not have CB
        CA_CB[np.isnan(CA_CB)] = 0
        # Change the CA_CB of the mutated residue to 0
        # but we don't modify the CA_CB matrix, instead we return a mask matrix
        CA_C = coords[:, [1]] - coords[:, [3]]
        CA_O = coords[:, [2]] - coords[:, [3]]
        CA_N = coords[:, [0]] - coords[:, [3]]
        nodes_vector = np.transpose(np.concatenate([CA_CB, CA_C, CA_O, CA_N], axis=1), (0, 2, 1))
        # if self.add_sidechain:
        # get sidechain coords
        sidechain_nodes_vector = coords[:, 5:] - coords[:, [3]]
        sidechain_nodes_vector[np.isnan(sidechain_nodes_vector)] = 0
        sidechain_nodes_vector = np.transpose(sidechain_nodes_vector, (0, 2, 1))
        nodes_vector = np.concatenate([nodes_vector, sidechain_nodes_vector], axis=2)
        # prepare graph
        features = dict(
            embed_logits=embed_logits if self.node_embedding_type == 'one-hot-idx' else None,
            one_hot_mat=one_hot_mat if self.node_embedding_type.startswith('one-hot') else None,
            mask_idx=mask_idx,
            embed_data=embed_data,
            embed_data_mask=embed_data_mask,
            alt_embed_data=None,
            coords=coords,
            CA_coord=CA_coord,
            CB_coord=CB_coord,
            edge_index=edge_index,
            edge_index_star=edge_index_star,
            edge_attr=edge_attr,
            edge_attr_star=edge_attr_star,
            nodes_vector=nodes_vector,
        )
        if self.add_confidence:
            # add position wise confidence
            if self.add_plddt:
                features['plddt'] = confidence_data
                if self.loaded_confidence:
                    pae = self.af2_confidence_dict[mutation.af2_seq_index]
                else:
                    pae = utils.get_confidence_from_af2file(mutation.af2_file, self.af2_plddt_dict[mutation.af2_seq_index])
                if mutation.crop:
                    pae = pae[mutation.seq_start - 1: mutation.seq_end, mutation.seq_start - 1: mutation.seq_end]
            else:
                # get plddt
                plddt_data = utils.get_plddt_from_af2(mutation.af2_file)
                pae = utils.get_confidence_from_af2file(mutation.af2_file, plddt_data)
                if mutation.crop:
                    confidence_data = plddt_data[mutation.seq_start - 1: mutation.seq_end] / 100
                    pae = pae[mutation.seq_start - 1: mutation.seq_end, mutation.seq_start - 1: mutation.seq_end]
                if confidence_data.shape[0] != embed_data.shape[0]:
                    warnings.warn(f'pLDDT {confidence_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                    f'pLDDT file: {mutation.af2_file}, '
                                    f'ESM prefix: {mutation.ESM_prefix}')
                    confidence_data = np.ones_like(embed_data[:, 0]) * 0.8
                features['plddt'] = confidence_data
            # add pairwise confidence
            features['edge_confidence'] = pae[edge_index[0], edge_index[1]]
            features['edge_confidence_star'] = pae[edge_index_star[0], edge_index_star[1]]
        return features

    def get(self, idx):
        features_np = self.get_one_mutation(idx)
        if self.node_embedding_type == 'one-hot-idx':
            x = torch.from_numpy(features_np['embed_data']).to(torch.long)
        else:
            x = torch.from_numpy(features_np['embed_data']).to(torch.float32)
        features = dict(
            x=x,
            x_mask=torch.from_numpy(features_np['embed_data_mask']).to(torch.bool),
            x_alt=torch.zeros_like(x),
            pos=torch.from_numpy(features_np['CA_coord']).to(torch.float32) if not self.use_cb else torch.from_numpy(features_np['CB_coord']).to(torch.float32),
            edge_index=torch.from_numpy(features_np['edge_index']).to(torch.long),
            edge_index_star=torch.from_numpy(features_np['edge_index_star']).to(torch.long),
            edge_attr=torch.from_numpy(features_np['edge_attr']).to(torch.float32),
            edge_attr_star=torch.from_numpy(features_np['edge_attr_star']).to(torch.float32),
            node_vec_attr=torch.from_numpy(features_np['nodes_vector']).to(torch.float32),
        )
        if self.add_confidence:
            features['plddt'] = torch.from_numpy(features_np['plddt']).to(torch.float32)
            features['edge_confidence'] = torch.from_numpy(features_np['edge_confidence']).to(torch.float32)
            features['edge_confidence_star'] = torch.from_numpy(features_np['edge_confidence_star']).to(torch.float32)
        if self.neighbor_type == 'radius' or self.neighbor_type == 'radius-KNN':
            # first concat edge_index and edge_index_star
            concat_edge_index = torch.cat((features["edge_index"], features["edge_index_star"]), dim=1)
            concat_edge_attr = torch.cat((features["edge_attr"], features["edge_attr_star"]), dim=0)
            # then remove isolated nodes
            concat_edge_index, concat_edge_attr, mask = \
                remove_isolated_nodes(concat_edge_index, concat_edge_attr, x.shape[0])
            # then split edge_index and edge_attr
            features["edge_index"] = concat_edge_index[:, :features["edge_index"].shape[1]]
            features["edge_index_star"] = concat_edge_index[:, features["edge_index"].shape[1]:]
            features["edge_attr"] = concat_edge_attr[:features["edge_attr"].shape[0]]
            features["edge_attr_star"] = concat_edge_attr[features["edge_attr"].shape[0]:]
        else:
            features["edge_index"], features["edge_attr"], mask = \
                remove_isolated_nodes(features["edge_index"], features["edge_attr"], x.shape[0])
            features["edge_index_star"], features["edge_attr_star"], mask = \
                remove_isolated_nodes(features["edge_index_star"], features["edge_attr_star"], x.shape[0])
        features["x"] = features["x"][mask]
        features["x_mask"] = features["x_mask"][mask]
        features["x_alt"] = features["x_alt"][mask]
        features["pos"] = features["pos"][mask]
        features["node_vec_attr"] = features["node_vec_attr"][mask]
        # need to process y, which is separated by comma and float
        y_scores = self.data[self._y_columns].iloc[int(idx)]
        # if mask exists, we need to mask the y_scores
        if len(self._y_mask_columns) > 0:
            y_masks = self.data[self._y_mask_columns].iloc[int(idx)]
        else:
            # create fake y_masks that are all None
            y_masks = [None] * len(y_scores)
        # we need a y that is 1 x 20 x n that depends on the length of y_scores
        y = torch.zeros([1, len(utils.AA_DICT_HUMAN), len(y_scores)]).to(torch.float32)
        y_mask = torch.zeros_like(y)
        # y_score might be multi-dimensional
        # need another y_mask that is 1 x 20 x n, to tell which location is target
        for i in range(len(y_scores)):
            y_scores_i = np.array(y_scores[i].split(';')).astype(np.float32) if isinstance(y_scores[i], str) else np.array([y_scores[i]]).astype(np.float32)
            if y_masks[i] is not None:
                y_masks_i = np.array(y_masks[i].split(';')).astype(np.float32) if isinstance(y_masks[i], str) else np.array([y_masks[i]]).astype(np.float32)
            else:
                y_masks_i = np.ones_like(y_scores_i)
            # match the values in y based on AA_DICT
            alt_aa_idxs = [utils.AA_DICT_HUMAN.index(aa) if aa != 'X' else 19 for aa in self.mutations[idx].alt_aa]
            y[0, alt_aa_idxs, i] = torch.from_numpy(y_scores_i)
            y_mask[0, alt_aa_idxs, i] = torch.from_numpy(y_masks_i)
        features["y"] = y.to(torch.float32)
        features["score_mask"] = y_mask.to(torch.float32)
        return Data(**features)


class GraphESMMutationDataset(GraphMutationDataset):
    def __init__(self, data_file, data_type: str,
                 radius: float = None, max_neighbors: int = None,
                 loop: bool = False, shuffle: bool = False, gpu_id: int = None,
                 node_embedding_type: Literal['esm', 'one-hot-idx', 'one-hot', 'aa-5dim', 'esm1b'] = 'esm',
                 graph_type: Literal['af2', '1d-neighbor'] = 'af2',
                 add_plddt: bool = False, 
                 scale_plddt: bool = False,
                 add_conservation: bool = False,
                 add_position: bool = False,
                 add_sidechain: bool = False,
                 local_coord_transform: bool = False,
                 use_cb: bool = False,
                 add_msa_contacts: bool = True,
                 add_dssp: bool = False,
                 add_msa: bool = False,
                 add_confidence: bool = False,
                 loaded_confidence: bool = False,
                 loaded_esm: bool = False,
                 add_ptm: bool = False,
                 data_augment: bool = False,
                 score_transfer: bool = False,
                 alt_type: Literal['alt', 'concat', 'diff', 'orig'] = 'orig',
                 computed_graph: bool = True,
                 loaded_msa: bool = False,
                 neighbor_type: Literal['KNN', 'radius', 'radius-KNN'] = 'KNN',
                 max_len = 2251,
                 convert_to_onesite: bool = False,
                 add_af2_single: bool = False,
                 add_af2_pairwise: bool = False,
                 loaded_af2_single: bool = False,
                 loaded_af2_pairwise: bool = False,
                 ):
        super(GraphESMMutationDataset, self).__init__(
            data_file, data_type, radius, max_neighbors, loop, shuffle, gpu_id,
            node_embedding_type, graph_type, add_plddt, scale_plddt,
            add_conservation, add_position, add_sidechain,
            local_coord_transform, use_cb, add_msa_contacts, add_dssp,
            add_msa, add_confidence, loaded_confidence, loaded_esm,
            add_ptm, data_augment, score_transfer, alt_type,
            computed_graph, loaded_msa, neighbor_type, max_len)
        self._y_mask_columns = self.data.columns[self.data.columns.str.startswith('confidence.score')]

    def get_one_mutation(self, idx):
        mutation: utils.Mutation = self.mutations[idx]
        # get the graph
        coords, edge_index, edge_index_star, edge_attr, edge_attr_star, mask_idx, mutation = self.get_graph_and_mask(mutation)
        # get embeddings
        if self.node_embedding_type == 'esm':
            if self.loaded_esm:
                # esm embeddings have <start> token, so starts at 1
                embed_data = self.esm_dict[mutation.esm_seq_index][mutation.seq_start:mutation.seq_end + 1]
            else:
                embed_data = utils.get_embedding_from_esm2(mutation.ESM_prefix, False,
                                                           mutation.seq_start, mutation.seq_end)
        elif self.node_embedding_type == 'one-hot-idx':
            assert not self.add_conservation and not self.add_plddt
            embed_logits, embed_data, one_hot_mat = utils.get_embedding_from_onehot_nonzero(mutation.seq, return_idx=True, return_onehot_mat=True)
        elif self.node_embedding_type == 'one-hot':
            embed_data, one_hot_mat = utils.get_embedding_from_onehot(mutation.seq, return_idx=False, return_onehot_mat=True)
        elif self.node_embedding_type == 'aa-5dim':
            embed_data = utils.get_embedding_from_5dim(mutation.seq)
        elif self.node_embedding_type == 'esm1b':
            embed_data = utils.get_embedding_from_esm1b(mutation.ESM_prefix, False,
                                                        mutation.seq_start, mutation.seq_end)
        # add conservation, if needed
        if self.loaded_msa and (self.add_msa or self.add_conservation):
            msa_seq = self.msa_dict[mutation.msa_seq_index][0]
            conservation_data = self.msa_dict[mutation.msa_seq_index][1]
            msa_data = self.msa_dict[mutation.msa_seq_index][2]
        else:
            if self.add_conservation or self.add_msa:
                msa_seq, conservation_data, msa_data = utils.get_msa_dict_from_transcript(mutation.uniprot_id)
        if self.add_conservation:
            if conservation_data.shape[0] == 0:
                conservation_data = np.zeros((embed_data.shape[0], 20))
            else:
                msa_seq_check = msa_seq[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                conservation_data = conservation_data[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                if mutation.crop:
                    msa_seq_check = msa_seq_check[mutation.seq_start - 1: mutation.seq_end]
                    conservation_data = conservation_data[mutation.seq_start - 1: mutation.seq_end]
                if msa_seq_check != mutation.seq:
                    # warnings.warn(f'MSA file: {mutation.transcript_id} does not match mutation sequence')
                    self.unmatched_msa += 1
                    print(f'Unmatched MSA: {self.unmatched_msa}')
                    conservation_data = np.zeros((embed_data.shape[0], 20))
            embed_data = np.concatenate([embed_data, conservation_data], axis=1)
        # add pLDDT, if needed
        if self.add_plddt:
            # get plddt
            plddt_data = self.af2_plddt_dict[mutation.af2_seq_index]  # N, C, O, CA, CB
            if mutation.crop:
                plddt_data = plddt_data[mutation.seq_start - 1: mutation.seq_end]
            if self.add_confidence:
                confidence_data = plddt_data / 100
            if plddt_data.shape[0] != embed_data.shape[0]:
                warnings.warn(f'pLDDT {plddt_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                f'pLDDT file: {mutation.af2_file}, '
                                f'ESM prefix: {mutation.ESM_prefix}')
                plddt_data = np.ones_like(embed_data[:, 0]) * 50
                if self.add_confidence:
                    # assign 0.5 confidence to all points
                    confidence_data = np.ones_like(embed_data[:, 0]) / 2
            if self.scale_plddt:
                plddt_data = plddt_data / 100
            embed_data = np.concatenate([embed_data, plddt_data[:, None]], axis=1)
        # add dssp, if needed
        if self.add_dssp:
            # get dssp
            dssp_data = self.af2_dssp_dict[mutation.af2_seq_index]
            if mutation.crop:
                dssp_data = dssp_data[mutation.seq_start - 1: mutation.seq_end]
            if dssp_data.shape[0] != embed_data.shape[0]:
                warnings.warn(f'DSSP {dssp_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                f'DSSP file: {mutation.af2_file}, '
                                f'ESM prefix: {mutation.ESM_prefix}')
                dssp_data = np.zeros_like(embed_data[:, 0])
            # if dssp_data size axis is 1, add a dimension
            if len(dssp_data.shape) == 1:
                dssp_data = dssp_data[:, None]
            embed_data = np.concatenate([embed_data, dssp_data], axis=1)
        if self.add_msa:
            if msa_data.shape[0] == 0:
                msa_data = np.zeros((embed_data.shape[0], 199))
            else:
                msa_seq_check = msa_seq[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                msa_data = msa_data[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                if mutation.crop:
                    msa_seq_check = msa_seq_check[mutation.seq_start - 1: mutation.seq_end]
                    msa_data = msa_data[mutation.seq_start - 1: mutation.seq_end]
                if msa_seq_check != mutation.seq:
                    warnings.warn(f'MSA file: {mutation.transcript_id} does not match mutation sequence')
                    msa_data = np.zeros((embed_data.shape[0], 199))
            embed_data = np.concatenate([embed_data, msa_data], axis=1)
        if self.add_ptm:
            ptm_data = utils.get_ptm_from_mutation(mutation, self.ptm_ref)
            embed_data = np.concatenate([embed_data, ptm_data], axis=1)
        # replace the embedding with the mutation, note pos is 1-based
        # but we don't modify the embedding matrix, instead we return a mask matrix
        embed_data_mask = np.ones_like(embed_data)
        embed_data_mask[mask_idx] = 0
        # prepare node vector features
        # get CA_coords
        CA_coord = coords[:, 3]
        CB_coord = coords[:, 4]
        # add CB_coord for GLY
        CB_coord[np.isnan(CB_coord)] = CA_coord[np.isnan(CB_coord)]
        if self.graph_type == '1d-neighbor':
            CA_coord[:, 0] = np.arange(coords.shape[0])
            CB_coord[:, 0] = np.arange(coords.shape[0])
            coords = np.zeros_like(coords)
        CA_CB = coords[:, [4]] - coords[:, [3]]  # Note that glycine does not have CB
        CA_CB[np.isnan(CA_CB)] = 0
        # Change the CA_CB of the mutated residue to 0
        # but we don't modify the CA_CB matrix, instead we return a mask matrix
        CA_C = coords[:, [1]] - coords[:, [3]]
        CA_O = coords[:, [2]] - coords[:, [3]]
        CA_N = coords[:, [0]] - coords[:, [3]]
        nodes_vector = np.transpose(np.concatenate([CA_CB, CA_C, CA_O, CA_N], axis=1), (0, 2, 1))
        # if self.add_sidechain:
        # get sidechain coords
        sidechain_nodes_vector = coords[:, 5:] - coords[:, [3]]
        sidechain_nodes_vector[np.isnan(sidechain_nodes_vector)] = 0
        sidechain_nodes_vector = np.transpose(sidechain_nodes_vector, (0, 2, 1))
        nodes_vector = np.concatenate([nodes_vector, sidechain_nodes_vector], axis=2)
        # prepare graph
        features = dict(
            embed_logits=embed_logits if self.node_embedding_type == 'one-hot-idx' else None,
            one_hot_mat=one_hot_mat if self.node_embedding_type.startswith('one-hot') else None,
            mask_idx=mask_idx,
            embed_data=embed_data,
            embed_data_mask=embed_data_mask,
            alt_embed_data=None,
            coords=coords,
            CA_coord=CA_coord,
            CB_coord=CB_coord,
            edge_index=edge_index,
            edge_index_star=edge_index_star,
            edge_attr=edge_attr,
            edge_attr_star=edge_attr_star,
            nodes_vector=nodes_vector,
        )
        if self.add_confidence:
            # add position wise confidence
            if self.add_plddt:
                features['plddt'] = confidence_data
                if self.loaded_confidence:
                    pae = self.af2_confidence_dict[mutation.af2_seq_index]
                else:
                    pae = utils.get_confidence_from_af2file(mutation.af2_file, self.af2_plddt_dict[mutation.af2_seq_index])
                if mutation.crop:
                    pae = pae[mutation.seq_start - 1: mutation.seq_end, mutation.seq_start - 1: mutation.seq_end]
            else:
                # get plddt
                plddt_data = utils.get_plddt_from_af2(mutation.af2_file)
                pae = utils.get_confidence_from_af2file(mutation.af2_file, plddt_data)
                if mutation.crop:
                    confidence_data = plddt_data[mutation.seq_start - 1: mutation.seq_end] / 100
                    pae = pae[mutation.seq_start - 1: mutation.seq_end, mutation.seq_start - 1: mutation.seq_end]
                if confidence_data.shape[0] != embed_data.shape[0]:
                    warnings.warn(f'pLDDT {confidence_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                    f'pLDDT file: {mutation.af2_file}, '
                                    f'ESM prefix: {mutation.ESM_prefix}')
                    confidence_data = np.ones_like(embed_data[:, 0]) * 0.8
                features['plddt'] = confidence_data
            # add pairwise confidence
            features['edge_confidence'] = pae[edge_index[0], edge_index[1]]
            features['edge_confidence_star'] = pae[edge_index_star[0], edge_index_star[1]]
        return features

    def get(self, idx):
        features_np = self.get_one_mutation(idx)
        if self.node_embedding_type == 'one-hot-idx':
            x = torch.from_numpy(features_np['embed_data']).to(torch.long)
        else:
            x = torch.from_numpy(features_np['embed_data']).to(torch.float32)
        features = dict(
            x=x,
            x_mask=torch.from_numpy(features_np['embed_data_mask']).to(torch.bool),
            x_alt=x.clone(),
            pos=torch.from_numpy(features_np['CA_coord']).to(torch.float32) if not self.use_cb else torch.from_numpy(features_np['CB_coord']).to(torch.float32),
            edge_index=torch.from_numpy(features_np['edge_index']).to(torch.long),
            edge_index_star=torch.from_numpy(features_np['edge_index_star']).to(torch.long),
            edge_attr=torch.from_numpy(features_np['edge_attr']).to(torch.float32),
            edge_attr_star=torch.from_numpy(features_np['edge_attr_star']).to(torch.float32),
            node_vec_attr=torch.from_numpy(features_np['nodes_vector']).to(torch.float32),
        )
        if self.add_confidence:
            features['plddt'] = torch.from_numpy(features_np['plddt']).to(torch.float32)
            features['edge_confidence'] = torch.from_numpy(features_np['edge_confidence']).to(torch.float32)
            features['edge_confidence_star'] = torch.from_numpy(features_np['edge_confidence_star']).to(torch.float32)
        if self.neighbor_type == 'radius' or self.neighbor_type == 'radius-KNN':
            # first concat edge_index and edge_index_star
            concat_edge_index = torch.cat((features["edge_index"], features["edge_index_star"]), dim=1)
            concat_edge_attr = torch.cat((features["edge_attr"], features["edge_attr_star"]), dim=0)
            # then remove isolated nodes
            concat_edge_index, concat_edge_attr, mask = \
                remove_isolated_nodes(concat_edge_index, concat_edge_attr, x.shape[0])
            # then split edge_index and edge_attr
            features["edge_index"] = concat_edge_index[:, :features["edge_index"].shape[1]]
            features["edge_index_star"] = concat_edge_index[:, features["edge_index"].shape[1]:]
            features["edge_attr"] = concat_edge_attr[:features["edge_attr"].shape[0]]
            features["edge_attr_star"] = concat_edge_attr[features["edge_attr"].shape[0]:]
        else:
            features["edge_index"], features["edge_attr"], mask = \
                remove_isolated_nodes(features["edge_index"], features["edge_attr"], x.shape[0])
            features["edge_index_star"], features["edge_attr_star"], mask = \
                remove_isolated_nodes(features["edge_index_star"], features["edge_attr_star"], x.shape[0])
        features["x"] = features["x"][mask]
        features["x_mask"] = features["x_mask"][mask]
        features["x_alt"] = features["x_alt"][mask]
        features["pos"] = features["pos"][mask]
        features["node_vec_attr"] = features["node_vec_attr"][mask]
        # we need a y_mask that is 1 x ESM x n that depends on the length of y_scores
        y_mask = torch.zeros([1, len(utils.ESM_TOKENS)]).to(torch.float32)
        # y_score might be multi-dimensional
        # need another y_mask that is 1 x 20 x n, to tell which location is target
        # match the aa and ref based on ESM_TOKENS
        alt_aa_idxs = [utils.ESM_TOKENS.index(aa) for aa in self.mutations[idx].alt_aa]
        ref_aa_idxs = [utils.ESM_TOKENS.index(aa) for aa in self.mutations[idx].ref_aa]
        y_mask[0, alt_aa_idxs] = 1
        y_mask[0, ref_aa_idxs] = -1
        features["y"] = torch.tensor([self.data[self._y_columns].iloc[int(idx)]]).to(torch.float32)
        features["esm_mask"] = y_mask.to(torch.float32)
        return Data(**features)


class FullGraphESMMutationDataset(FullGraphMutationDataset):
    def __init__(self, data_file, data_type: str,
                 radius: float = None, max_neighbors: int = None,
                 loop: bool = False, shuffle: bool = False, gpu_id: int = None,
                 node_embedding_type: Literal['esm', 'one-hot-idx', 'one-hot', 'aa-5dim', 'esm1b'] = 'esm',
                 graph_type: Literal['af2', '1d-neighbor'] = 'af2',
                 add_plddt: bool = False, 
                 scale_plddt: bool = False,
                 add_conservation: bool = False,
                 add_position: bool = False,
                 add_sidechain: bool = False,
                 local_coord_transform: bool = False,
                 use_cb: bool = False,
                 add_msa_contacts: bool = True,
                 add_dssp: bool = False,
                 add_msa: bool = False,
                 add_confidence: bool = False,
                 loaded_confidence: bool = False,
                 loaded_esm: bool = False,
                 add_ptm: bool = False,
                 data_augment: bool = False,
                 score_transfer: bool = False,
                 alt_type: Literal['alt', 'concat', 'diff'] = 'alt',
                 computed_graph: bool = True,
                 loaded_msa: bool = False,
                 neighbor_type: Literal['KNN', 'radius', 'radius-KNN'] = 'KNN',
                 max_len = 2251,
                 convert_to_onesite: bool = False,
                 add_af2_single: bool = False,
                 add_af2_pairwise: bool = False,
                 loaded_af2_single: bool = False,
                 loaded_af2_pairwise: bool = False,
                 ):
        super(FullGraphESMMutationDataset, self).__init__(
            data_file, data_type, radius, max_neighbors, loop, shuffle, gpu_id,
            node_embedding_type, graph_type, add_plddt, scale_plddt,
            add_conservation, add_position, add_sidechain,
            local_coord_transform, use_cb, add_msa_contacts, add_dssp,
            add_msa, add_confidence, loaded_confidence, loaded_esm,
            add_ptm, data_augment, score_transfer, alt_type,
            computed_graph, loaded_msa, neighbor_type, max_len, convert_to_onesite)
        self._y_mask_columns = self.data.columns[self.data.columns.str.startswith('confidence.score')]

    def get_graph_and_mask(self, mutation: utils.Mutation):
        # get the ordinary graph
        coords: np.ndarray = self.af2_coord_dict[mutation.af2_seq_index]  # N, C, O, CA, CB
        # remember we could have cropped sequence
        if mutation.crop:
            coords = coords[mutation.seq_start - 1:mutation.seq_end, :]
        # get the mask
        mask_idx, mutation = self.get_mask(mutation)
        # prepare edge features
        # if self.add_msa_contacts:
        #     coevo_strength = utils.get_contacts_from_msa(mutation, False)
        #     if isinstance(coevo_strength, int):
        #         coevo_strength = np.zeros([mutation.seq_end - mutation.seq_start + 1,
        #                                 mutation.seq_end - mutation.seq_start + 1, 1])
        # else:
        coevo_strength = np.zeros([mutation.seq_end - mutation.seq_start + 1,
                                        mutation.seq_end - mutation.seq_start + 1, 0])
        edge_attr = coevo_strength # N, N, 1
        # if add positional embedding, add it here
        # if self.add_position:
            # add a sin positional embedding that reflects the relative position of the residue
        edge_position = np.arange(coords.shape[0])[:, None] - np.arange(coords.shape[0])[None, :]
        edge_attr = np.concatenate(
            (edge_attr, np.sin(np.pi / 2 * edge_position / self.max_len)[:, :, None]), 
            axis=2)
        return coords, None, None, edge_attr, None, mask_idx, mutation

    def get_one_mutation(self, idx):
        mutation: utils.Mutation = self.mutations[idx]
        # get the graph
        coords, _, _, edge_attr, _, mask_idx, mutation = self.get_graph_and_mask(mutation)
        # get embeddings
        # embed data should be N x 20
        embed_logits, embed_data, one_hot_mat = utils.get_embedding_from_esm_onehot(mutation.seq, return_idx=True, return_onehot_mat=True)
        # mask_idx should plus 1 as we add <start> token
        mask_idx += 1
        # add conservation, if needed
        if self.loaded_msa and (self.add_msa or self.add_conservation):
            msa_seq = self.msa_dict[mutation.msa_seq_index][0]
            conservation_data = self.msa_dict[mutation.msa_seq_index][1]
            msa_data = self.msa_dict[mutation.msa_seq_index][2]
        else:
            if self.add_conservation or self.add_msa:
                msa_seq, conservation_data, msa_data = utils.get_msa_dict_from_transcript(mutation.uniprot_id)
        if self.add_conservation:
            if conservation_data.shape[0] == 0:
                conservation_data = np.zeros((embed_data.shape[0], 20))
            else:
                msa_seq_check = msa_seq[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                conservation_data = conservation_data[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                if mutation.crop:
                    msa_seq_check = msa_seq_check[mutation.seq_start - 1: mutation.seq_end]
                    conservation_data = conservation_data[mutation.seq_start - 1: mutation.seq_end]
                if msa_seq_check != mutation.seq:
                    # warnings.warn(f'MSA file: {mutation.transcript_id} does not match mutation sequence')
                    self.unmatched_msa += 1
                    print(f'Unmatched MSA: {self.unmatched_msa}')
                    conservation_data = np.zeros((embed_data.shape[0], 20))
            embed_data = np.concatenate([embed_data, conservation_data], axis=1)
        # add pLDDT, if needed
        if self.add_plddt:
            # get plddt
            plddt_data = self.af2_plddt_dict[mutation.af2_seq_index]  # N, C, O, CA, CB
            if mutation.crop:
                plddt_data = plddt_data[mutation.seq_start - 1: mutation.seq_end]
            if self.add_confidence:
                confidence_data = plddt_data / 100
            if plddt_data.shape[0] != embed_data.shape[0]:
                warnings.warn(f'pLDDT {plddt_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                f'pLDDT file: {mutation.af2_file}, '
                                f'ESM prefix: {mutation.ESM_prefix}')
                plddt_data = np.ones_like(embed_data[:, 0]) * 50
                if self.add_confidence:
                    # assign 0.5 confidence to all points
                    confidence_data = np.ones_like(embed_data[:, 0]) / 2
            if self.scale_plddt:
                plddt_data = plddt_data / 100
            embed_data = np.concatenate([embed_data, plddt_data[:, None]], axis=1)
        # add dssp, if needed
        if self.add_dssp:
            # get dssp
            dssp_data = self.af2_dssp_dict[mutation.af2_seq_index]
            if mutation.crop:
                dssp_data = dssp_data[mutation.seq_start - 1: mutation.seq_end]
            if dssp_data.shape[0] != embed_data.shape[0]:
                warnings.warn(f'DSSP {dssp_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                f'DSSP file: {mutation.af2_file}, '
                                f'ESM prefix: {mutation.ESM_prefix}')
                dssp_data = np.zeros_like(embed_data[:, 0])
            # if dssp_data size axis is 1, add a dimension
            if len(dssp_data.shape) == 1:
                dssp_data = dssp_data[:, None]
            embed_data = np.concatenate([embed_data, dssp_data], axis=1)
        if self.add_msa:
            if msa_data.shape[0] == 0:
                msa_data = np.zeros((embed_data.shape[0], 199))
            else:
                msa_seq_check = msa_seq[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                msa_data = msa_data[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                if mutation.crop:
                    msa_seq_check = msa_seq_check[mutation.seq_start - 1: mutation.seq_end]
                    msa_data = msa_data[mutation.seq_start - 1: mutation.seq_end]
                if msa_seq_check != mutation.seq:
                    warnings.warn(f'MSA file: {mutation.transcript_id} does not match mutation sequence')
                    msa_data = np.zeros((embed_data.shape[0], 199))
            embed_data = np.concatenate([embed_data, msa_data], axis=1)
        if self.add_ptm:
            ptm_data = utils.get_ptm_from_mutation(mutation, self.ptm_ref)
            embed_data = np.concatenate([embed_data, ptm_data], axis=1)
        # replace the embedding with the mutation, note pos is 1-based
        # but we don't modify the embedding matrix, instead we return a mask matrix
        embed_data_mask = np.ones_like(embed_data)
        embed_data_mask[mask_idx] = 0
        # prepare node vector features
        # get CA_coords
        CA_coord = coords[:, 3]
        CB_coord = coords[:, 4]
        # add CB_coord for GLY
        CB_coord[np.isnan(CB_coord)] = CA_coord[np.isnan(CB_coord)]
        if self.graph_type == '1d-neighbor':
            CA_coord[:, 0] = np.arange(coords.shape[0])
            CB_coord[:, 0] = np.arange(coords.shape[0])
            coords = np.zeros_like(coords)
        CA_CB = coords[:, [4]] - coords[:, [3]]  # Note that glycine does not have CB
        CA_CB[np.isnan(CA_CB)] = 0
        # Change the CA_CB of the mutated residue to 0
        # but we don't modify the CA_CB matrix, instead we return a mask matrix
        CA_C = coords[:, [1]] - coords[:, [3]]
        CA_O = coords[:, [2]] - coords[:, [3]]
        CA_N = coords[:, [0]] - coords[:, [3]]
        nodes_vector = np.transpose(np.concatenate([CA_CB, CA_C, CA_O, CA_N], axis=1), (0, 2, 1))
        # if self.add_sidechain:
        # get sidechain coords
        sidechain_nodes_vector = coords[:, 5:] - coords[:, [3]]
        sidechain_nodes_vector[np.isnan(sidechain_nodes_vector)] = 0
        sidechain_nodes_vector = np.transpose(sidechain_nodes_vector, (0, 2, 1))
        nodes_vector = np.concatenate([nodes_vector, sidechain_nodes_vector], axis=2)
        # prepare graph
        features = dict(
            embed_logits=None,
            one_hot_mat=None,
            mask_idx=mask_idx,
            embed_data=embed_data,
            embed_data_mask=embed_data_mask,
            alt_embed_data=None,
            coords=coords,
            CA_coord=CA_coord,
            CB_coord=CB_coord,
            edge_index=None,
            edge_index_star=None,
            edge_attr=edge_attr,
            edge_attr_star=None,
            nodes_vector=nodes_vector,
        )
        if self.add_confidence:
            # add position wise confidence
            if self.add_plddt:
                features['plddt'] = confidence_data
                if self.loaded_confidence:
                    pae = self.af2_confidence_dict[mutation.af2_seq_index]
                else:
                    pae = utils.get_confidence_from_af2file(mutation.af2_file, self.af2_plddt_dict[mutation.af2_seq_index])
                if mutation.crop:
                    pae = pae[mutation.seq_start - 1: mutation.seq_end, mutation.seq_start - 1: mutation.seq_end]
            else:
                # get plddt
                plddt_data = utils.get_plddt_from_af2(mutation.af2_file)
                pae = utils.get_confidence_from_af2file(mutation.af2_file, plddt_data)
                if mutation.crop:
                    confidence_data = plddt_data[mutation.seq_start - 1: mutation.seq_end] / 100
                    pae = pae[mutation.seq_start - 1: mutation.seq_end, mutation.seq_start - 1: mutation.seq_end]
                if confidence_data.shape[0] != embed_data.shape[0]:
                    warnings.warn(f'pLDDT {confidence_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                  f'pLDDT file: {mutation.af2_file}, '
                                  f'ESM prefix: {mutation.ESM_prefix}')
                    confidence_data = np.ones_like(embed_data[:, 0]) * 0.8
                features['plddt'] = confidence_data
            # add pairwise confidence
            features['edge_confidence'] = pae
        return features

    def get(self, idx):
        start = time.time()
        features_np = self.get_one_mutation(idx)
        if self.node_embedding_type == 'one-hot-idx':
            x = torch.from_numpy(features_np['embed_data']).to(torch.long)
        else:
            x = torch.from_numpy(features_np['embed_data']).to(torch.long)
        # padding x to the max length
        # x_padding_mask = torch.zeros(self.max_len, dtype=torch.bool)
        pos=torch.from_numpy(features_np['CB_coord']).to(torch.float32) if self.use_cb else torch.from_numpy(features_np['CA_coord']).to(torch.float32)
        node_vec_attr=torch.from_numpy(features_np['nodes_vector']).to(torch.float32)
        edge_attr=torch.from_numpy(features_np['edge_attr']).to(torch.float32)
        x_mask=torch.from_numpy(features_np['embed_data_mask']).to(torch.bool)
        if self.add_confidence:
            plddt=torch.from_numpy(features_np['plddt']).to(torch.float32)
        if x.shape[0] < self.max_len + 2:
            # x_padding_mask[x.shape[0]:] = True
            x = torch.nn.functional.pad(x, (0, self.max_len + 2 - x.shape[0]), 'constant', utils.ESM_TOKENS.index('<pad>'))
            # pos = torch.nn.functional.pad(pos, (0, 0, 0, self.max_len + 2 - pos.shape[0]))
            # node_vec_attr = torch.nn.functional.pad(node_vec_attr, (0, 0, 0, 0, 0, self.max_len + 2 - node_vec_attr.shape[0]))
            # edge_attr = torch.nn.functional.pad(edge_attr, (0, 0, 0, self.max_len + 2 - edge_attr.shape[0], 0, self.max_len + 2 - edge_attr.shape[0]))
            x_mask = torch.nn.functional.pad(x_mask, (0, self.max_len + 2 - x_mask.shape[0]), 'constant', True)
            # if self.add_confidence:
            #     edge_confidence = torch.nn.functional.pad(edge_confidence, (0, self.max_len + 2 - edge_confidence.shape[0], 0, self.max_len + 2 - edge_confidence.shape[0]))
            #     plddt = torch.nn.functional.pad(plddt, (0, self.max_len + 2 - plddt.shape[0]))
        # we need a y that is 1 x 20 x n that depends on the length of y_scores
        y_mask = torch.zeros([len(utils.ESM_TOKENS)]).to(torch.float32)
        # y_score might be multi-dimensional
        # need another y_mask that is 1 x 20 x n, to tell which location is target
        # match the aa and ref based on ESM_TOKENS
        alt_aa_idxs = [utils.ESM_TOKENS.index(aa) for aa in self.mutations[idx].alt_aa]
        ref_aa_idxs = [utils.ESM_TOKENS.index(aa) for aa in self.mutations[idx].ref_aa]
        y_mask[alt_aa_idxs] = 1
        y_mask[ref_aa_idxs] = -1
        features = dict(
            x=x,
            # x_padding_mask=x_padding_mask,
            x_mask=x_mask,
            x_alt=torch.ones_like(x) * utils.ESM_TOKENS.index('<mask>'),
            # pos=pos,
            # edge_attr=edge_attr,
            # node_vec_attr=None,
            y=torch.tensor([self.data[self._y_columns].iloc[int(idx)]]).to(torch.float32).squeeze(1),
            esm_mask=y_mask.to(torch.float32),
        )
        end = time.time()
        print(f'get time: {end - start}')
        return features


class FullGraphMultiOnesiteMutationDataset(FullGraphMutationDataset):
    def __init__(self, data_file, data_type: str,
                 radius: float = None, max_neighbors: int = None,
                 loop: bool = False, shuffle: bool = False, gpu_id: int = None,
                 node_embedding_type: Literal['esm', 'one-hot-idx', 'one-hot', 'aa-5dim', 'esm1b'] = 'esm',
                 graph_type: Literal['af2', '1d-neighbor'] = 'af2',
                 add_plddt: bool = False, 
                 scale_plddt: bool = False,
                 add_conservation: bool = False,
                 add_position: bool = False,
                 add_sidechain: bool = False,
                 local_coord_transform: bool = False,
                 use_cb: bool = False,
                 add_msa_contacts: bool = True,
                 add_dssp: bool = False,
                 add_msa: bool = False,
                 add_confidence: bool = False,
                 loaded_confidence: bool = False,
                 loaded_esm: bool = False,
                 add_ptm: bool = False,
                 data_augment: bool = False,
                 score_transfer: bool = False,
                 alt_type: Literal['alt', 'concat', 'diff'] = 'alt',
                 computed_graph: bool = True,
                 loaded_msa: bool = False,
                 neighbor_type: Literal['KNN', 'radius', 'radius-KNN'] = 'KNN',
                 max_len = 2251,
                 convert_to_onesite: bool = False,
                 add_af2_single: bool = False,
                 add_af2_pairwise: bool = False,
                 loaded_af2_single: bool = False,
                 loaded_af2_pairwise: bool = False,
                 ):
        super(FullGraphMultiOnesiteMutationDataset, self).__init__(
            data_file, data_type, radius, max_neighbors, loop, shuffle, gpu_id,
            node_embedding_type, graph_type, add_plddt, scale_plddt,
            add_conservation, add_position, add_sidechain,
            local_coord_transform, use_cb, add_msa_contacts, add_dssp,
            add_msa, add_confidence, loaded_confidence, loaded_esm,
            add_ptm, data_augment, score_transfer, alt_type,
            computed_graph, loaded_msa, neighbor_type, max_len, convert_to_onesite)
        self._y_mask_columns = self.data.columns[self.data.columns.str.startswith('confidence.score')]

    def get_graph_and_mask(self, mutation: utils.Mutation):
        # get the ordinary graph
        coords: np.ndarray = self.af2_coord_dict[mutation.af2_seq_index]  # N, C, O, CA, CB
        # remember we could have cropped sequence
        if mutation.crop:
            coords = coords[mutation.seq_start - 1:mutation.seq_end, :]
        # get the mask
        mask_idx, mutation = self.get_mask(mutation)
        # prepare edge features
        # if self.add_msa_contacts:
        #     coevo_strength = utils.get_contacts_from_msa(mutation, False)
        #     if isinstance(coevo_strength, int):
        #         coevo_strength = np.zeros([mutation.seq_end - mutation.seq_start + 1,
        #                                 mutation.seq_end - mutation.seq_start + 1, 1])
        # else:
        coevo_strength = np.zeros([mutation.seq_end - mutation.seq_start + 1,
                                        mutation.seq_end - mutation.seq_start + 1, 0])
        edge_attr = coevo_strength # N, N, 1
        # if add positional embedding, add it here
        # if self.add_position:
            # add a sin positional embedding that reflects the relative position of the residue
        edge_position = np.arange(coords.shape[0])[:, None] - np.arange(coords.shape[0])[None, :]
        edge_attr = np.concatenate(
            (edge_attr, np.sin(np.pi / 2 * edge_position / self.max_len)[:, :, None]), 
            axis=2)
        return coords, None, None, edge_attr, None, mask_idx, mutation

    def get_one_mutation(self, idx):
        mutation: utils.Mutation = self.mutations[idx]
        # get the graph
        coords, _, _, edge_attr, _, mask_idx, mutation = self.get_graph_and_mask(mutation)
        # get embeddings
        if self.node_embedding_type == 'esm':
            if self.loaded_esm:
                # esm embeddings have <start> token, so starts at 1
                embed_data = self.esm_dict[mutation.esm_seq_index][mutation.seq_start:mutation.seq_end + 1]
            else:
                embed_data = utils.get_embedding_from_esm2(mutation.ESM_prefix, False,
                                                           mutation.seq_start, mutation.seq_end)
        elif self.node_embedding_type == 'one-hot-idx':
            assert not self.add_conservation and not self.add_plddt
            embed_logits, embed_data, one_hot_mat = utils.get_embedding_from_onehot_nonzero(mutation.seq, return_idx=True, return_onehot_mat=True)
        elif self.node_embedding_type == 'one-hot':
            embed_data, one_hot_mat = utils.get_embedding_from_onehot(mutation.seq, return_idx=False, return_onehot_mat=True)
        elif self.node_embedding_type == 'aa-5dim':
            embed_data = utils.get_embedding_from_5dim(mutation.seq)
        elif self.node_embedding_type == 'esm1b':
            embed_data = utils.get_embedding_from_esm1b(mutation.ESM_prefix, False,
                                                        mutation.seq_start, mutation.seq_end)
        # add conservation, if needed
        if self.loaded_msa and (self.add_msa or self.add_conservation):
            msa_seq = self.msa_dict[mutation.msa_seq_index][0]
            conservation_data = self.msa_dict[mutation.msa_seq_index][1]
            msa_data = self.msa_dict[mutation.msa_seq_index][2]
        else:
            if self.add_conservation or self.add_msa:
                msa_seq, conservation_data, msa_data = utils.get_msa_dict_from_transcript(mutation.uniprot_id)
        if self.add_conservation:
            if conservation_data.shape[0] == 0:
                conservation_data = np.zeros((embed_data.shape[0], 20))
            else:
                msa_seq_check = msa_seq[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                conservation_data = conservation_data[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                if mutation.crop:
                    msa_seq_check = msa_seq_check[mutation.seq_start - 1: mutation.seq_end]
                    conservation_data = conservation_data[mutation.seq_start - 1: mutation.seq_end]
                if msa_seq_check != mutation.seq:
                    # warnings.warn(f'MSA file: {mutation.transcript_id} does not match mutation sequence')
                    self.unmatched_msa += 1
                    print(f'Unmatched MSA: {self.unmatched_msa}')
                    conservation_data = np.zeros((embed_data.shape[0], 20))
            embed_data = np.concatenate([embed_data, conservation_data], axis=1)
        # add pLDDT, if needed
        if self.add_plddt:
            # get plddt
            plddt_data = self.af2_plddt_dict[mutation.af2_seq_index]  # N, C, O, CA, CB
            if mutation.crop:
                plddt_data = plddt_data[mutation.seq_start - 1: mutation.seq_end]
            if self.add_confidence:
                confidence_data = plddt_data / 100
            if plddt_data.shape[0] != embed_data.shape[0]:
                warnings.warn(f'pLDDT {plddt_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                f'pLDDT file: {mutation.af2_file}, '
                                f'ESM prefix: {mutation.ESM_prefix}')
                plddt_data = np.ones_like(embed_data[:, 0]) * 50
                if self.add_confidence:
                    # assign 0.5 confidence to all points
                    confidence_data = np.ones_like(embed_data[:, 0]) / 2
            if self.scale_plddt:
                plddt_data = plddt_data / 100
            embed_data = np.concatenate([embed_data, plddt_data[:, None]], axis=1)
        # add dssp, if needed
        if self.add_dssp:
            # get dssp
            dssp_data = self.af2_dssp_dict[mutation.af2_seq_index]
            if mutation.crop:
                dssp_data = dssp_data[mutation.seq_start - 1: mutation.seq_end]
            if dssp_data.shape[0] != embed_data.shape[0]:
                warnings.warn(f'DSSP {dssp_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                f'DSSP file: {mutation.af2_file}, '
                                f'ESM prefix: {mutation.ESM_prefix}')
                dssp_data = np.zeros_like(embed_data[:, 0])
            # if dssp_data size axis is 1, add a dimension
            if len(dssp_data.shape) == 1:
                dssp_data = dssp_data[:, None]
            embed_data = np.concatenate([embed_data, dssp_data], axis=1)
        if self.add_msa:
            if msa_data.shape[0] == 0:
                msa_data = np.zeros((embed_data.shape[0], 199))
            else:
                msa_seq_check = msa_seq[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                msa_data = msa_data[mutation.seq_start_orig - 1: mutation.seq_end_orig]
                if mutation.crop:
                    msa_seq_check = msa_seq_check[mutation.seq_start - 1: mutation.seq_end]
                    msa_data = msa_data[mutation.seq_start - 1: mutation.seq_end]
                if msa_seq_check != mutation.seq:
                    warnings.warn(f'MSA file: {mutation.transcript_id} does not match mutation sequence')
                    msa_data = np.zeros((embed_data.shape[0], 199))
            embed_data = np.concatenate([embed_data, msa_data], axis=1)
        if self.add_ptm:
            ptm_data = utils.get_ptm_from_mutation(mutation, self.ptm_ref)
            embed_data = np.concatenate([embed_data, ptm_data], axis=1)
        # replace the embedding with the mutation, note pos is 1-based
        # but we don't modify the embedding matrix, instead we return a mask matrix
        embed_data_mask = np.ones_like(embed_data)
        embed_data_mask[mask_idx] = 0
        # prepare node vector features
        # get CA_coords
        CA_coord = coords[:, 3]
        CB_coord = coords[:, 4]
        # add CB_coord for GLY
        CB_coord[np.isnan(CB_coord)] = CA_coord[np.isnan(CB_coord)]
        if self.graph_type == '1d-neighbor':
            CA_coord[:, 0] = np.arange(coords.shape[0])
            CB_coord[:, 0] = np.arange(coords.shape[0])
            coords = np.zeros_like(coords)
        CA_CB = coords[:, [4]] - coords[:, [3]]  # Note that glycine does not have CB
        CA_CB[np.isnan(CA_CB)] = 0
        # Change the CA_CB of the mutated residue to 0
        # but we don't modify the CA_CB matrix, instead we return a mask matrix
        CA_C = coords[:, [1]] - coords[:, [3]]
        CA_O = coords[:, [2]] - coords[:, [3]]
        CA_N = coords[:, [0]] - coords[:, [3]]
        nodes_vector = np.transpose(np.concatenate([CA_CB, CA_C, CA_O, CA_N], axis=1), (0, 2, 1))
        # if self.add_sidechain:
        # get sidechain coords
        sidechain_nodes_vector = coords[:, 5:] - coords[:, [3]]
        sidechain_nodes_vector[np.isnan(sidechain_nodes_vector)] = 0
        sidechain_nodes_vector = np.transpose(sidechain_nodes_vector, (0, 2, 1))
        nodes_vector = np.concatenate([nodes_vector, sidechain_nodes_vector], axis=2)
        # prepare graph
        features = dict(
            embed_logits=None,
            one_hot_mat=None,
            mask_idx=mask_idx,
            embed_data=embed_data,
            embed_data_mask=embed_data_mask,
            alt_embed_data=None,
            coords=coords,
            CA_coord=CA_coord,
            CB_coord=CB_coord,
            edge_index=None,
            edge_index_star=None,
            edge_attr=edge_attr,
            edge_attr_star=None,
            nodes_vector=nodes_vector,
        )
        if self.add_confidence:
            # add position wise confidence
            if self.add_plddt:
                features['plddt'] = confidence_data
                if self.loaded_confidence:
                    pae = self.af2_confidence_dict[mutation.af2_seq_index]
                else:
                    pae = utils.get_confidence_from_af2file(mutation.af2_file, self.af2_plddt_dict[mutation.af2_seq_index])
                if mutation.crop:
                    pae = pae[mutation.seq_start - 1: mutation.seq_end, mutation.seq_start - 1: mutation.seq_end]
            else:
                # get plddt
                plddt_data = utils.get_plddt_from_af2(mutation.af2_file)
                pae = utils.get_confidence_from_af2file(mutation.af2_file, plddt_data)
                if mutation.crop:
                    confidence_data = plddt_data[mutation.seq_start - 1: mutation.seq_end] / 100
                    pae = pae[mutation.seq_start - 1: mutation.seq_end, mutation.seq_start - 1: mutation.seq_end]
                if confidence_data.shape[0] != embed_data.shape[0]:
                    warnings.warn(f'pLDDT {confidence_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                  f'pLDDT file: {mutation.af2_file}, '
                                  f'ESM prefix: {mutation.ESM_prefix}')
                    confidence_data = np.ones_like(embed_data[:, 0]) * 0.8
                features['plddt'] = confidence_data
            # add pairwise confidence
            features['edge_confidence'] = pae
        return features

    def get(self, idx):
        features_np = self.get_one_mutation(idx)
        if self.node_embedding_type == 'one-hot-idx':
            x = torch.from_numpy(features_np['embed_data']).to(torch.long)
        else:
            x = torch.from_numpy(features_np['embed_data']).to(torch.float32)
        # padding x to the max length
        x_padding_mask = torch.zeros(self.max_len, dtype=torch.bool)
        pos=torch.from_numpy(features_np['CB_coord']).to(torch.float32) if self.use_cb else torch.from_numpy(features_np['CA_coord']).to(torch.float32)
        node_vec_attr=torch.from_numpy(features_np['nodes_vector']).to(torch.float32)
        edge_attr=torch.from_numpy(features_np['edge_attr']).to(torch.float32)
        x_mask=torch.from_numpy(features_np['embed_data_mask'][:, 0]).to(torch.bool)
        if self.add_confidence:
            plddt=torch.from_numpy(features_np['plddt']).to(torch.float32)
        edge_confidence=torch.from_numpy(features_np['edge_confidence']).to(torch.float32)
        if x.shape[0] < self.max_len:
            x_padding_mask[x.shape[0]:] = True
            x = torch.nn.functional.pad(x, (0, 0, 0, self.max_len - x.shape[0]))
            pos = torch.nn.functional.pad(pos, (0, 0, 0, self.max_len - pos.shape[0]))
            node_vec_attr = torch.nn.functional.pad(node_vec_attr, (0, 0, 0, 0, 0, self.max_len - node_vec_attr.shape[0]))
            edge_attr = torch.nn.functional.pad(edge_attr, (0, 0, 0, self.max_len - edge_attr.shape[0], 0, self.max_len - edge_attr.shape[0]))
            x_mask = torch.nn.functional.pad(x_mask, (0, self.max_len - x_mask.shape[0]), 'constant', True)
            if self.add_confidence:
                edge_confidence = torch.nn.functional.pad(edge_confidence, (0, self.max_len - edge_confidence.shape[0], 0, self.max_len - edge_confidence.shape[0]))
                plddt = torch.nn.functional.pad(plddt, (0, self.max_len - plddt.shape[0]))
        # need to process y, which is separated by comma and float
        y_scores = self.data[self._y_columns].iloc[int(idx)]
        # if mask exists, we need to mask the y_scores
        if len(self._y_mask_columns) > 0:
            y_masks = self.data[self._y_mask_columns].iloc[int(idx)]
        else:
            # create fake y_masks that are all None
            y_masks = [None] * len(y_scores)
        # we need a y that is 1 x 20 x n that depends on the length of y_scores
        y = torch.zeros([len(utils.AA_DICT_HUMAN), len(y_scores)]).to(torch.float32)
        y_mask = torch.zeros_like(y)
        # y_score might be multi-dimensional
        # need another y_mask that is 1 x 20 x n, to tell which location is target
        for i in range(len(y_scores)):
            y_scores_i = np.array(y_scores[i].split(';')).astype(np.float32) if isinstance(y_scores[i], str) else np.array([y_scores[i]]).astype(np.float32)
            if y_masks[i] is not None:
                y_masks_i = np.array(y_masks[i].split(';')).astype(np.float32) if isinstance(y_masks[i], str) else np.array([y_masks[i]]).astype(np.float32)
            else:
                y_masks_i = np.ones_like(y_scores_i)
            # match the values in y based on AA_DICT
            alt_aa_idxs = [utils.AA_DICT_HUMAN.index(aa) for aa in self.mutations[idx].alt_aa]
            y[alt_aa_idxs, i] = torch.from_numpy(y_scores_i)
            y_mask[alt_aa_idxs, i] = torch.from_numpy(y_masks_i)
        # don't need x_alt, but just make it zeros and same size as x
        features = dict(
            x=x,
            x_padding_mask=x_padding_mask,
            x_mask=x_mask,
            x_alt=torch.zeros_like(x),
            pos=pos,
            edge_attr=edge_attr,
            node_vec_attr=node_vec_attr,
            y=y.to(torch.float32),
            score_mask=y_mask.to(torch.float32),
        )
        if self.add_confidence:
            features['plddt'] = plddt
            features['edge_confidence'] = edge_confidence
        return features


# Not used in this version
def collate(
    data_list: List[BaseData],
    increment: bool = True,
    add_batch: bool = True,
) -> BaseData:
    # Collates a list of `data` objects into a single object of type `cls`.
    # `collate` can handle both homogeneous and heterogeneous data objects by
    # individually collating all their stores.
    # In addition, `collate` can handle nested data structures such as
    # dictionaries and lists.

    if not isinstance(data_list, (list, tuple)):
        # Materialize `data_list` to keep the `_parent` weakref alive.
        data_list = list(data_list)

    if cls != data_list[0].__class__:
        out = cls(_base_cls=data_list[0].__class__)  # Dynamic inheritance.
    else:
        out = cls()

    # Create empty stores:
    out.stores_as(data_list[0])

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])

    # Group all storage objects of every data object in the `data_list` by key,
    # i.e. `key_to_store_list = { key: [store_1, store_2, ...], ... }`:
    key_to_stores = defaultdict(list)
    for data in data_list:
        for store in data.stores:
            key_to_stores[store._key].append(store)

    # With this, we iterate over each list of storage objects and recursively
    # collate all its attributes into a unified representation:

    # We maintain two additional dictionaries:
    # * `slice_dict` stores a compressed index representation of each attribute
    #    and is needed to re-construct individual elements from mini-batches.
    # * `inc_dict` stores how individual elements need to be incremented, e.g.,
    #   `edge_index` is incremented by the cumulated sum of previous elements.
    #   We also need to make use of `inc_dict` when re-constructuing individual
    #   elements as attributes that got incremented need to be decremented
    #   while separating to obtain original values.
    device = None
    slice_dict, inc_dict = defaultdict(dict), defaultdict(dict)
    for out_store in out.stores:
        key = out_store._key
        stores = key_to_stores[key]
        for attr in stores[0].keys():

            if attr in exclude_keys:  # Do not include top-level attribute.
                continue

            values = [store[attr] for store in stores]

            # The `num_nodes` attribute needs special treatment, as we need to
            # sum their values up instead of merging them to a list:
            if attr == 'num_nodes':
                out_store._num_nodes = values
                out_store.num_nodes = sum(values)
                continue

            # Skip batching of `ptr` vectors for now:
            if attr == 'ptr':
                continue

            # Collate attributes into a unified representation:
            value, slices, incs = _collate(attr, values, data_list, stores,
                                           increment)

            if isinstance(value, Tensor) and value.is_cuda:
                device = value.device

            out_store[attr] = value
            if key is not None:
                slice_dict[key][attr] = slices
                inc_dict[key][attr] = incs
            else:
                slice_dict[attr] = slices
                inc_dict[attr] = incs

            # Add an additional batch vector for the given attribute:
            if attr in follow_batch:
                batch, ptr = _batch_and_ptr(slices, device)
                out_store[f'{attr}_batch'] = batch
                out_store[f'{attr}_ptr'] = ptr

        # In case the storage holds node, we add a top-level batch vector it:
        if (add_batch and isinstance(stores[0], NodeStorage)
                and stores[0].can_infer_num_nodes):
            repeats = [store.num_nodes for store in stores]
            out_store.batch = repeat_interleave(repeats, device=device)
            out_store.ptr = cumsum(torch.tensor(repeats, device=device))

    return out


def my_collate_fn(data_list: List[Any]) -> Any:
    batch = collate(
            data_list=data_list,
            increment=True,
            add_batch=True,
        )

    batch._num_graphs = len(data_list)
    return batch
