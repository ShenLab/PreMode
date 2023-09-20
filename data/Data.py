from typing import Literal
import warnings
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import remove_isolated_nodes
from itertools import cycle
from multiprocessing import Pool

import data.utils as utils
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
                 add_conservation: bool = False,
                 add_position: bool = False,
                 add_sidechain: bool = False,
                 local_coord_transform: bool = False,
                 use_cb: bool = False,
                 add_msa_contacts: bool = True,
                 add_dssp: bool = False,
                 add_msa: bool = False,
                 data_augment: bool = False,
                 score_transfer: bool = False,
                 alt_type: Literal['alt', 'concat', 'diff'] = 'alt',
                 computed_graph: bool = True,
                 loaded_msa: bool = False,
                 neighbor_type: Literal['KNN', 'radius'] = 'KNN',
                 max_len = 2251,
                 ):
        super(GraphMutationDataset, self).__init__()
        if isinstance(data_file, pd.DataFrame):
            self.data = data_file
            self.data_file = 'pd.DataFrame'
        elif isinstance(data_file, str):
            try:
                self.data = pd.read_csv(data_file, index_col=0, low_memory=False)
            except:
                self.data = pd.read_csv(data_file, index_col=0)
            self.data_file = data_file
        else:
            raise ValueError("data_path must be a string or a pandas.DataFrame")
        self.data_type = data_type
        self._y_columns = self.data.columns[self.data.columns.str.startswith('score')]
        self.node_embedding_type = node_embedding_type
        self.graph_type = graph_type
        self.neighbor_type = neighbor_type
        self.add_plddt = add_plddt
        self.add_conservation = add_conservation
        self.add_position = add_position
        self.use_cb = use_cb
        self.add_sidechain = add_sidechain
        self.add_msa_contacts = add_msa_contacts
        self.add_dssp = add_dssp
        self.add_msa = add_msa
        self.loaded_msa = loaded_msa
        self.alt_type = alt_type
        self.max_len = max_len
        self.loop = loop
        self.data_augment = data_augment
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
        for i in range(len(self.data)):
            res = utils.get_mutations(self.data['uniprotID'].iloc[i],
                                self.data['ENST'].iloc[i],
                                self.data['wt.orig'].iloc[i],
                                self.data['sequence.len.orig'].iloc[i],
                                self.data['pos.orig'].iloc[i],
                                self.data['ref'].iloc[i],
                                self.data['alt'].iloc[i],
                                self.max_len,
                                self.data['af2_file'].iloc[i])
        with Pool(NUM_THREADS) as p:
            point_mutations = p.starmap(utils.get_mutations, zip(self.data['uniprotID'],
                                                                 self.data['ENST'],
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
            self.af2_plddt_dict = p.starmap(utils.get_plddt_from_af2, zip(self.af2_file_dict)) if self.add_plddt else None
            self.af2_dssp_dict = p.starmap(utils.get_dssp_from_af2, zip(self.af2_file_dict)) if self.add_dssp else None
        print(f'Finished loading {len(self.af2_coord_dict)} af2 coords')
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
                    edge_index = utils.get_knn_graphs_from_af2(coords, self.radius, self.max_neighbors, self.loop)
                else:
                    edge_index = utils.get_radius_graphs_from_af2(coords, self.radius, self.loop)
                    # delete nodes that are not connected with variant node.
                    connected_nodes = edge_index[:, np.isin(edge_index[0], mutation.pos - 1)].flatten()
                    edge_index = edge_index[:, np.isin(edge_index[0], connected_nodes) | np.isin(edge_index[1], connected_nodes)]
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
        # cancel self loop
        if not self.loop:
            edge_index_star = edge_index_star[:, edge_index_star[0] != edge_index_star[1]]
        # cancel nodes that not in edge_index
        edge_index_star = edge_index_star[:, np.isin(edge_index_star[0], edge_index.flatten()) &
                                          np.isin(edge_index_star[1], edge_index.flatten())]
        # prepare edge features
        # coevo_strength = self.msa_contacts_dict[self.msa_contacts_idx[idx]]
        if self.add_msa_contacts:
            coevo_strength = utils.get_contacts_from_msa(mutation, False)
            if isinstance(coevo_strength, int):
                coevo_strength = np.zeros([mutation.seq_end - mutation.seq_start + 1,
                                        mutation.seq_end - mutation.seq_start + 1, 1])
        else:
            coevo_strength = np.zeros([mutation.seq_end - mutation.seq_start + 1,
                                        mutation.seq_end - mutation.seq_start + 1, 0])
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
        if self.loaded_msa:
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
            if plddt_data.shape[0] != embed_data.shape[0]:
                warnings.warn(f'pLDDT {plddt_data.shape[0]} does not match embedding {embed_data.shape[0]}, '
                                f'pLDDT file: {mutation.af2_file}, '
                                f'ESM prefix: {mutation.ESM_prefix}')
                plddt_data = np.zeros_like(embed_data[:, 0])
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
            embed_data = np.concatenate([embed_data, dssp_data], axis=1)
            to_alt = np.concatenate([to_alt, dssp_data[mask_idx]], axis=1)
            if self.alt_type == 'diff':
                to_ref = np.concatenate([to_ref, dssp_data[mask_idx]], axis=1)
        if self.add_msa:
            if msa_data.shape[0] == 0:
                msa_data = np.zeros((embed_data.shape[0], 199))
            else:
                msa_seq_check = msa_seq_check[mutation.seq_start_orig - 1: mutation.seq_end_orig]
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
            edge_index=edge_index,
            edge_index_star=edge_index_star,
            edge_attr=edge_attr,
            edge_attr_star=edge_attr_star,
            nodes_vector=nodes_vector,
        )
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
        # if self.alt_type == 'concat':
        #     features['edge_index'] = None
        #     features['edge_attr'] = None
        # else:
        #     features["edge_index"], features["edge_attr"], mask = \
        #         remove_isolated_nodes(features["edge_index"], features["edge_attr"], x.shape[0])
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

    def __getitem__(self, idx):
        return self.get(idx)

    def __len__(self):
        return len(self.mutations)

    def len(self) -> int:
        return len(self.mutations)
    
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

