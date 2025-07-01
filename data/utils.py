import gzip
import json
import os
import pickle
from abc import abstractmethod
from os.path import exists
from typing import List
import string
import random
import biotite.structure
import numpy as np
import pandas as pd
import socket
import torch
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from biopandas.pdb import PandasPdb
from biotite.sequence import ProteinSequence
from biotite.structure import get_chains
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from torch_cluster import radius_graph, knn_graph

# Path to the AF2 data
AF2_DATA_PATH = './data.files/af2.files/'
# unused in this version
AF2_REP_DATA_PATH = "NA"
# Path to the AF2 data
ESM_MODEL_SIZE = '650M'
ESM_DATA_PATH = f'./data.files/esm.files/'
# Path to the ESM2 data
MSA_DATA_PATH_ARCHIVE = './data.files/gMVP.MSA/'
MSA_DATA_PATH = './data.files/MSA/'
# unused in this version
PAE_DATA_PATH = 'NA'
# Path to the ESM_MSA data
# TODO: update the path
MSA_ATTN_DATA_PATH = './data.files/esm.MSA/'
NUM_THREADS = 42
# TEMP dir for DSSP
TMPDIR = './tmp/'
DSSP_PATH = "/share/vault/Users/gz2294/miniconda3/envs/RESCVE/bin/mkdssp"
# prepare esm2 embeddings
with open(f'./utils/LANGUAGE_MODEL.{ESM_MODEL_SIZE}.pkl', 'rb') as f:
    LANGUAGE_MODEL = pickle.load(f)
with open(f'./utils/ALPHABET_CONVERTER.{ESM_MODEL_SIZE}.pkl', 'rb') as f:
    ALPHABET_CONVERTER = pickle.load(f)
with open(f'./utils/ESM_AA_EMBEDDING_DICT.{ESM_MODEL_SIZE}.pkl', 'rb') as f:
    ESM_AA_EMBEDDING_DICT = pickle.load(f)
with open(f'./utils/ESM_AA_EMBEDDING_DICT.esm1b.pkl', 'rb') as f:
    ESM1b_AA_EMBEDDING_DICT = pickle.load(f)
# prepare 5dim embeddings
with open(f'./utils/AA_5_DIM_EMBED.pkl', 'rb') as f:
    AA_5DIM_EMBED = pickle.load(f)
# ESM tokens
ESM_TOKENS = ['<cls>', '<pad>', '<eos>', '<unk>',
              'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
              'X', 'B', 'U', 'Z', 'O', '.', '-',
              '<null_1>', '<mask>']
# amino acid dictionary, no padding token because it is not used (batch size is 1 as limited by GPU memory)
AA_DICT = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
           'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
           'X', 'B', 'U', 'Z', 'O', '<mask>']
AA_DICT_HUMAN = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
                 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']
DSSP_DICT = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-', 'P']
PTM_DICT = {'ac': 0, 'ga': 1, 'gl': 2, 'm1': 3, 'm2': 4, 'm3': 5, 'me': 6, 'p': 7, 'sm': 8, 'ub': 9}

class Mutation:
    """
    A mutation object that stores the information of a mutation.
    Can specify max_len of sequence to crop the sequence.
    Can specify af2_file to ignore the input sequence and use the AF2 sequence instead.
    """
    def __init__(self, uniprot_id, transcript_id, seq_orig, seq_orig_len, pos_orig, ref_aa, alt_aa, max_len=2251, af2_file=None):
        # initialize attributes
        self.seq = None
        self.seq_start = None
        self.seq_end = None
        self.seq_start_orig = None
        self.seq_end_orig = None
        self.pos = None
        self.uniprot_id = None
        self.af2_file = None
        self.af2_rep_file_prefix = None
        self.af2_seq_index = None
        self.msa_seq_index = None
        self.esm_seq_index = None
        self.af2_rep_index = None
        self.ref_aa = None
        self.alt_aa = None
        self.ESM_prefix = None
        self.crop = False
        self.seq_len = None
        self.seq_len_orig = None
        self.max_len = max_len
        self.half_max_len = max_len // 2
        self.set_af2_fragment_idx(seq_orig, seq_orig_len, uniprot_id, pos_orig, af2_file)
        self.transcript_id = transcript_id
        self.set_ref_alt_aa(ref_aa, alt_aa)
        self.init_af2_file_idx()
        self.crop_fn()
        
    def set_af2_fragment_idx(self, seq_orig, seq_orig_len, uniprot_id, pos_orig, af2_file):
        self.seq_len_orig = seq_orig_len
        if isinstance(pos_orig, str):
            pos_orig = np.array([int(i) for i in pos_orig.split(';')])
        else:
            pos_orig = np.array([int(pos_orig)])
        if af2_file is None or pd.isna(af2_file):
            if uniprot_id.find('-F') != -1:
                idx = int(uniprot_id.split('-F')[-1])
                uniprot_id = uniprot_id.split('-F')[0]
                seq_start = 1
                seq_end = seq_orig_len
                self.seq_start_orig = seq_start
                self.seq_end_orig = seq_end
                seq = seq_orig
                pos = pos_orig
                self.ESM_prefix = f'{uniprot_id}-F{idx}'
                seq_len = 1400
                self.af2_rep_file_prefix = f'{AF2_REP_DATA_PATH}/{uniprot_id}-F{idx}/{uniprot_id}-F{idx}'
            else:
                self.ESM_prefix = f'{uniprot_id}'
                if seq_orig_len > 2700:
                    idx = min(max(1, pos_orig[0] // 200 - 2), seq_orig_len // 200 - 5)
                    seq_start = (idx - 1) * 200 + 1
                    seq_end = min((idx + 6) * 200, seq_orig_len)
                    self.seq_start_orig = seq_start
                    self.seq_end_orig = seq_end
                    seq = seq_orig[seq_start - 1:seq_end]
                    pos = pos_orig - seq_start + 1
                    seq_len = seq_end - seq_start + 1
                    seq_start = 1
                    seq_end = seq_len
                else:
                    idx = 1
                    seq_start = 1
                    seq_end = seq_orig_len
                    self.seq_start_orig = seq_start
                    self.seq_end_orig = seq_end
                    seq_len = seq_orig_len
                    seq = seq_orig
                    pos = pos_orig
                if uniprot_id == "Q8WZ42": # This protein is TTN, which is too long
                    self.ESM_prefix = f'{uniprot_id}-F{idx}'
                if seq_orig_len >= 7000:
                    self.af2_rep_file_prefix = f'{AF2_REP_DATA_PATH}/{uniprot_id}-F{idx}/{uniprot_id}-F{idx}'
                else:
                    self.af2_rep_file_prefix = f'{AF2_REP_DATA_PATH}/{uniprot_id}/{uniprot_id}'
            self.seq = seq
            self.seq_start = seq_start
            self.seq_end = seq_end
            self.seq_len = seq_len
            self.pos = pos
            self.uniprot_id = uniprot_id
            self.af2_file = f'{AF2_DATA_PATH}/AF-{uniprot_id}-F{idx}-model_v4.pdb.gz'
        else:
            self.af2_file = af2_file
            self.ESM_prefix = uniprot_id
            self.seq = seq_orig
            self.seq_start = 1
            self.seq_end = seq_orig_len
            self.seq_start_orig = self.seq_start
            self.seq_end_orig = self.seq_end
            self.seq_len = seq_orig_len
            self.pos = pos_orig
            self.uniprot_id = uniprot_id

    def set_ref_alt_aa(self, ref_aa, alt_aa):
        # ref aa and alt aa are strings
        if ";" in ref_aa or ";" in alt_aa:
            # multiple mutations
            self.ref_aa = np.array(ref_aa.split(';'))
            self.alt_aa = np.array(alt_aa.split(';'))
        else:
            # single mutation
            self.ref_aa = np.array([ref_aa])
            self.alt_aa = np.array([alt_aa])

    def init_af2_file_idx(self):
        if not exists(self.af2_file):
            print(f'Warning: {self.uniprot_id} AF2 file not found: {self.af2_file}')
            self.af2_file = None
        # else:
        #     af2_seq = AF2_SEQ_DICT[self.af2_file]['seq']
        #     if af2_seq != self.seq and not self.crop:
        #         # if not match and not due to crop, then the seq is not in the AF2 file
        #         print(f'Warning: {self.uniprot_id} seq not match AF2 seq: {self.seq} vs {af2_seq}')
        #         self.af2_file = None
        self.af2_seq_index = None  # Use index to avoid loading the same seq multiple times

    def crop_fn(self):
        seq_len = self.seq_len
        pos = self.pos
        seq_start = self.seq_start
        seq_end = self.seq_end
        seq = self.seq
        # remove sequence longer than max_len
        if seq_len >= self.max_len:
            if pos[0] <= self.half_max_len:
                seq_start = 1
                seq_end = self.max_len
                seq = seq[:self.max_len]
                pos = pos
                seq_len = self.max_len
            elif seq_len - pos[0] <= self.max_len - self.half_max_len:
                seq_start = seq_len - self.max_len + 1
                seq_end = seq_len
                seq = seq[seq_start - 1:]
                pos = pos - seq_start + 1
                seq_len = self.max_len
            else:
                seq_start = pos[0] - self.half_max_len
                seq_end = pos[0] + self.max_len - self.half_max_len - 1
                seq = seq[seq_start - 1:seq_end]
                pos = pos - seq_start + 1
                seq_len = self.max_len
            self.crop = True
            self.seq = seq
            self.seq_start = seq_start
            self.seq_end = seq_end
            self.seq_len = seq_len
            self.pos = pos

    def set_af2_seq_index(self, idx):
        self.af2_seq_index = idx

    def set_msa_seq_index(self, idx):
        self.msa_seq_index = idx

    def set_esm_seq_index(self, idx):
        self.esm_seq_index = idx
    
    def set_af2_rep_index(self, idx):
        self.af2_rep_index = idx


class RandomPointMutation(Mutation):
    def __init__(self, uniprot_id, transcript_id, seq_orig, seq_orig_len, max_len=2251):
        pos_orig = np.random.randint(1, seq_orig_len + 1)
        ref_aa = seq_orig[pos_orig - 1]
        alt_aa = np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"))
        super().__init__(uniprot_id, transcript_id, seq_orig, seq_orig_len, pos_orig, ref_aa, alt_aa, max_len)


class MaskPredictPointMutation(Mutation):
    # a class that support mask and predict as well as point mutation
    def __init__(self, uniprot_id, transcript_id, seq_orig, seq_orig_len, pos_orig, ref_aa, alt_aa, max_len=2251, af2_file=None):
        if pos_orig is None or pos_orig == 0:
            pos_orig = np.random.randint(1, seq_orig_len + 1)
        self.ESM_prefix = None
        self.max_len = max_len
        self.half_max_len = max_len // 2
        super().__init__(uniprot_id, transcript_id, seq_orig, seq_orig_len, pos_orig, ref_aa, alt_aa, max_len=max_len, af2_file=af2_file)
        # don't need ESM prefix

    def init_af2_file_idx(self):
        # don't check whether seq match AF2 seq
        if not exists(self.af2_file):
            print(f'Warning: {self.uniprot_id} AF2 file not found: {self.af2_file}')
            self.af2_file = None
        self.af2_seq_index = None  # Use index to avoid loading the same seq multiple times


def convert_to_onesite(dataset: pd.DataFrame):
    # first get unique uniprotID and pos.orig
    if 'ref_aa' not in dataset.columns:
        dataset['ref_aa'] = dataset['ref']
    if 'alt_aa' not in dataset.columns:
        dataset['alt_aa'] = dataset['alt']
    dataset_onesite = dataset.copy(deep=True)
    dataset_onesite = dataset_onesite.drop_duplicates(subset=['uniprotID', 'pos.orig'])
    # then for each unique uniprotID and pos.orig, get all ref and alt aa, as well as their scores
    # if exists the confidence of score, then use it, otherwise use 1
    # get score and confidence.score columns
    score_cols = [col for col in dataset.columns if col.startswith('score')]
    confidence_cols = [col for col in dataset.columns if col.startswith('confidence.score')]
    # if confidence_cols is empty, then use 1
    if len(confidence_cols) == 0:
        confidence_cols = [f'confidence.score.{i}' for i in range(len(score_cols))]
        for col in confidence_cols:
            dataset[col] = 1
            dataset_onesite[col] = 1
    for i in dataset_onesite.index:
        subdataset = dataset[(dataset['uniprotID'] == dataset_onesite.loc[i, 'uniprotID']) & (dataset['pos.orig'] == dataset_onesite.loc[i, 'pos.orig'])]
        dataset_onesite.loc[i, 'ref_aa'] = ';'.join(subdataset['ref_aa'].values)
        dataset_onesite.loc[i, 'alt_aa'] = ';'.join(subdataset['alt_aa'].values)
        # if score_cols and confidence_cols are not empty, then concatenate them
        if len(score_cols) > 0:
            for col in score_cols:
                dataset_onesite.loc[i, col] = ';'.join(subdataset[col].values.astype('str'))
        if len(confidence_cols) > 0:
            for col in confidence_cols:
                dataset_onesite.loc[i, col] = ';'.join(subdataset[col].values.astype('str'))
    return dataset_onesite


def load_structure(fpath, chain=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith('cif'):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith('cif.gz'):
        with gzip.open(fpath, 'rt') as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith('pdb'):
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    elif fpath.endswith('pdb.gz'):
        with gzip.open(fpath, 'rt') as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    else:
        raise ValueError("Invalid file extension")
    # bbmask = filter_backbone(structure)
    # structure = structure[bbmask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain]
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure


def extract_coords_from_structure(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple coords
            - coords is an L x 5 x 3 array for N, C, O, CA, CB coordinates
    """
    coords = get_atom_coords_residue_wise(["N", "C", "O", "CA", "CB"], structure)
    return coords


def extract_sidechain_from_structure(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple coords
            - coords is an L x 31 x 3 array for side chain coordinates
    """
    coords = get_atom_coords_residue_wise(['CD', 'CD1', 'CD2', 'CE', 'CE1',
                                           'CE2', 'CE3', 'CG', 'CG1', 'CG2', 
                                           'CH2', 'CZ', 'CZ2', 'CZ3', 'ND1',
                                           'ND2', 'NE', 'NE1', 'NE2', 'NH1',
                                           'NH2', 'NZ', 'OD1', 'OD2', 'OE1',
                                           'OE2', 'OG', 'OG1', 'OH', 'SD', 
                                           'SG'],
                                           structure)
    return coords


def extract_residues_from_structure(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    residue_identities = get_residues(structure)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return seq


def get_atom_coords_residue_wise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Example for atoms argument: ["N", "O", "CA", "C", "CB"]
    """

    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        filter_sum = filters.sum(0)
        if not np.all(filter_sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[filter_sum == 0] = float("nan")
        return coords

    return biotite.structure.apply_residue_wise(struct, struct, filterfn)


def get_mutations(uniprot_id, transcript_id, seq, seq_orig_len,
                  pos_orig, ref_aa, alt_aa, max_len=1400, af2_file=None):
    mutation = Mutation(uniprot_id, transcript_id, seq, seq_orig_len, pos_orig, ref_aa, alt_aa, max_len, af2_file)
    if mutation.af2_file is None:
        print(
            f"No AF2 file found for this mutation "+
            f"{mutation.uniprot_id}:{mutation.ref_aa}:{mutation.pos}:{mutation.alt_aa}. Skipping..."
            )
        return False
    else:
        return mutation


def get_random_point_mutations(uniprot_id, transcript_id, seq, seq_orig_len,
                               pos_orig, ref_aa, alt_aa, score):
    if score == -1:
        point_mutation = RandomPointMutation(uniprot_id, transcript_id, seq, seq_orig_len)
    else:
        point_mutation = Mutation(uniprot_id, transcript_id, seq, seq_orig_len, pos_orig, ref_aa, alt_aa)
    if point_mutation.af2_file is None:
        return False
    else:
        return point_mutation


def get_mask_predict_point_mutations(uniprot_id, transcript_id, seq, seq_orig_len,
                                     pos_orig, ref_aa, alt_aa, max_len=2251, af2_file=None):
    point_mutation = MaskPredictPointMutation(uniprot_id, transcript_id, seq, seq_orig_len, pos_orig, ref_aa, alt_aa, max_len, af2_file)
    # print("finished loading point mutation")
    if point_mutation.af2_file is None:
        print(
            f"No AF2 file found for this mutation "+
            f"{point_mutation.uniprot_id}:{point_mutation.ref_aa}:{point_mutation.pos}:{point_mutation.alt_aa}. Skipping..."
            )
        return False
    else:
        return point_mutation


def get_coords_from_af2(af2_file, add_sidechain=False):
    pdb_path = af2_file
    structure = load_structure(pdb_path)
    af2_coords = extract_coords_from_structure(structure)
    if add_sidechain:
        af2_coords_sidechain = extract_sidechain_from_structure(structure)
        af2_coords = np.concatenate([af2_coords, af2_coords_sidechain], axis=1)
    return af2_coords


def get_plddt_from_af2(af2_file):
    pdb_file = PandasPdb().read_pdb(af2_file)
    pdb_file = pdb_file.df['ATOM'].drop_duplicates(subset=['residue_number'])
    plddt = pdb_file['b_factor'].values
    return plddt


def get_dssp_from_af2(af2_file):
    p = PDBParser()
    with gzip.open(af2_file, 'rt') as f:
        structure = p.get_structure('', f)
    model = structure[0]
    # try:
    #     dssp = DSSP(model, af2_file, file_type="PDB", dssp="/usr/bin/dssp")
    # except Exception or UserWarning:
    random.seed(hash(af2_file))
    tmpfile = TMPDIR+ ''.join(random.choices(string.ascii_letters, k=5)) + '.pdb'
    with open(tmpfile, 'w') as f:
        f.write(gzip.open(af2_file, 'rt').read())
    dssp = DSSP(model, tmpfile, file_type="PDB", dssp=DSSP_PATH)
    os.remove(tmpfile)
    # keys in dssp: index, aa, secondary struc, rsa, phi, psi, N-H-->O, O-->H-N, N-H-->O, O-->H-N
    dssp = pd.DataFrame(dssp)
    sec_struc = np.eye(len(DSSP_DICT), dtype=np.float32)[[DSSP_DICT.index(i) for i in dssp.iloc[:, 2].values]]
    return np.concatenate([sec_struc, 
                           dssp.iloc[:, 3].values[:, None], 
                           dssp.iloc[:, 4].values[:, None] / 180 * np.pi, 
                           dssp.iloc[:, 5].values[:, None] / 180 * np.pi], axis=1)


def get_ptm_from_mutation(mutation: Mutation, ptm_ref):
    # for each af2 file, match the PTM anno to it
    # get uniprotID
    uniprotID = mutation.uniprot_id
    ptm_ref = ptm_ref[ptm_ref['uniprotID'] == uniprotID]
    seq = mutation.seq
    # get fragment start and end
    ptm_ref['pos'] = ptm_ref['pos'] - mutation.seq_start_orig - mutation.seq_start + 1
    ptm_ref = ptm_ref[ptm_ref['pos'] >= 0]
    ptm_ref = ptm_ref[ptm_ref['pos'] < mutation.seq_len]
    ptm_mat = np.zeros([mutation.seq_len, len(PTM_DICT)])
    for i in ptm_ref.index:
        if ptm_ref['ref'].loc[i] == seq[ptm_ref['pos'].loc[i]]:
            ptm_mat[ptm_ref['pos'].loc[i], PTM_DICT[ptm_ref['type'].loc[i]]] = 1
    return ptm_mat


def get_knn_graphs_from_af2(af2_coords, radius=None, max_neighbors=None, loop=False, gpu_id=None):
    CA_coord = af2_coords[:, 3]
    if radius is None:
        edge_index = np.indices((af2_coords.shape[0], af2_coords.shape[0])).reshape(2, -1)
        # cancel self-edges
        if not loop:
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    else:
        if max_neighbors is None:
            max_neighbors = af2_coords.shape[0] + 1
        with torch.no_grad():
            CA_coord = torch.from_numpy(CA_coord)
            edge_index = knn_graph(
                x=CA_coord.to(f'cuda:{gpu_id}') if gpu_id is not None and torch.cuda.is_available() else CA_coord,
                # r=radius,
                loop=loop,
                # max_num_neighbors=max_neighbors,
                k=max_neighbors,
                num_workers=NUM_THREADS,
            ).detach().cpu().numpy()
            del CA_coord
    return edge_index


def get_radius_graphs_from_af2(af2_coords, radius, loop=False, gpu_id=None):
    CA_coord = af2_coords[:, 3]
    max_neighbors = af2_coords.shape[0] + 1
    with torch.no_grad():
        CA_coord = torch.from_numpy(CA_coord)
        edge_index = radius_graph(
            x=CA_coord.to(f'cuda:{gpu_id}') if gpu_id is not None and torch.cuda.is_available() else CA_coord,
            r=radius,
            loop=loop,
            max_num_neighbors=max_neighbors,
            num_workers=NUM_THREADS,
        ).detach().cpu().numpy()
        del CA_coord
    return edge_index


def get_radius_knn_graphs_from_af2(af2_coords, center_nodes, radius, max_neighbors, loop=False, gpu_id=None):
    # first get radius graph at the center nodes, then get knn graph for other nodes
    CA_coord = af2_coords[:, 3]
    with torch.no_grad():
        CA_coord = torch.from_numpy(CA_coord)
        edge_index = radius_graph(
            x=CA_coord.to(f'cuda:{gpu_id}') if gpu_id is not None and torch.cuda.is_available() else CA_coord,
            r=radius,
            loop=loop,
            max_num_neighbors=af2_coords.shape[0] + 1,
            num_workers=NUM_THREADS,
        ).detach().cpu().numpy()
        # filter edge_index so that only center nodes are kept
        edge_index_radius = edge_index[:, np.isin(edge_index[0], center_nodes)]
        # next get knn graph for other nodes
        edge_index = knn_graph(
                x=CA_coord.to(f'cuda:{gpu_id}') if gpu_id is not None and torch.cuda.is_available() else CA_coord,
                loop=loop,
                k=max_neighbors,
                num_workers=NUM_THREADS,
            ).detach().cpu().numpy()
        del CA_coord
        # only keep nodes that are in the radius graph
        edge_index = edge_index[:, np.isin(edge_index[0], edge_index_radius.flatten()) & np.isin(edge_index[1], edge_index_radius.flatten())]
    return edge_index


def get_graphs_from_neighbor(af2_coords, max_neighbors=None, loop=False):
    nodes = af2_coords.shape[0]
    if max_neighbors is None:
        # full graph
        max_neighbors = nodes + 1
    edge_graph = np.ones((nodes, nodes))
    # fill upper triangle with 0
    edge_graph *= np.tri(nodes, k=int(np.floor(max_neighbors / 2))) \
                  * np.tri(nodes, k=int(np.floor(max_neighbors / 2))).T
    edge_index = np.array(np.where(edge_graph == 1))
    if not loop:
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return edge_index


def get_embedding_from_esm2(protein, check_mode=True, seq_start=None, seq_end=None):
    if isinstance(protein, str):
        file_path = f"{ESM_DATA_PATH}/{protein}.representations.layer.48.npy"
        if os.path.exists(file_path):
            if check_mode:
                return True
            wt_orig = np.load(file_path)
            # TODO: I am removing the <BOS> and <EOS> tokens, not sure if this is correct
            batch_tokens = wt_orig[max(0, seq_start):
                                   min(wt_orig.shape[0] - 1, seq_end + 1)]
        else:
            if check_mode:
                return False
            batch_tokens = np.zeros([seq_end - seq_start + 1, 5120 if ESM_MODEL_SIZE == "15B" else 1280])
    elif isinstance(protein, np.ndarray):
        batch_tokens = protein[max(0, seq_start):
                               min(protein.shape[0] - 1, seq_end + 1)]
    else:
        raise ValueError("protein must be either a string of uniprotID or a numpy array")
    return batch_tokens


def get_esm_dict_from_uniprot(uniprotID):
    file_path = f"{ESM_DATA_PATH}/{uniprotID}.representations.layer.48.npy"
    wt_orig = np.load(file_path)
    return wt_orig


def get_af2_single_rep_dict_from_prefix(uniprotID_prefix, filter=False):
    # sometimes colabfold will padding the results, we need to remove the padding
    file_path = f"{uniprotID_prefix}_single_repr_rank_001_alphafold2_ptm_model_1_seed_000.npy"
    wt_orig = np.load(file_path)
    # padding_length = 0
    # last_i = 1
    # while np.all(wt_orig[-last_i-1] == wt_orig[-last_i]):
    #     # remove the last line if it is the same as the second last line
    #     last_i -= 1
    #     padding_length += 1
    # if padding_length > 0:
    #     wt_orig = wt_orig[:-(padding_length+1)]
    return wt_orig


def get_af2_pairwise_rep_dict_from_prefix(uniprotID_prefix):
    file_path = f"{uniprotID_prefix}_pair_repr_rank_001_alphafold2_ptm_model_1_seed_000.npy"
    wt_orig = np.load(file_path)
    # padding_length = 0
    # last_i = 1
    # while np.all(wt_orig[-last_i-1] == wt_orig[-last_i]):
    #     # remove the last line if it is the same as the second last line
    #     last_i -= 1
    #     padding_length += 1
    return wt_orig


def get_embedding_from_esm1b(protein, check_mode=True, seq_start=None, seq_end=None):
    if isinstance(protein, str):
        file_path = f"/share/vault/Users/gz2294/Data/DMS/ClinVar.HGMD.PrimateAI.syn/esm1b.embedding.uniprotIDs/{protein}.representations.layer.48.npy"
        if os.path.exists(file_path):
            if check_mode:
                return True
            wt_orig = np.load(file_path)
            # TODO: I am removing the <BOS> and <EOS> tokens, not sure if this is correct
            batch_tokens = wt_orig[max(0, seq_start):
                                   min(wt_orig.shape[0] - 1, seq_end + 1)]
        else:
            if check_mode:
                return False
            batch_tokens = np.zeros([seq_end - seq_start + 1, 5120 if ESM_MODEL_SIZE == "15B" else 1280])
    elif isinstance(protein, np.ndarray):
        batch_tokens = protein[max(0, seq_start):
                               min(protein.shape[0] - 1, seq_end + 1)]
    else:
        raise ValueError("protein must be either a string of uniprotID or a numpy array")
    return batch_tokens


def get_embedding_from_onehot(seq, seq_start=None, seq_end=None, return_idx=False, aa_dict=None, return_onehot_mat=False):
    if aa_dict is None:
        idx = [AA_DICT.index(aa) for aa in seq]
        protein = np.eye(len(AA_DICT))[idx]
        one_hot_mat = np.eye(len(AA_DICT))
    else:
        idx = [aa_dict.index(aa) for aa in seq]
        protein = np.eye(len(aa_dict))[idx]
        one_hot_mat = np.eye(len(aa_dict))
    if seq_start is not None and seq_end is not None:
        batch_tokens = protein[max(0, seq_start - 1): min(protein.shape[0], seq_end)]
    else:
        batch_tokens = protein
    if return_idx:
        if return_onehot_mat:
            return batch_tokens, np.array(idx), one_hot_mat
        else:
            return batch_tokens, np.array(idx)
    else:
        if return_onehot_mat:
            return batch_tokens, one_hot_mat
        else:
            return batch_tokens


def get_embedding_from_esm_onehot(seq, seq_start=None, seq_end=None, return_idx=False, aa_dict=None, return_onehot_mat=False):
    if aa_dict is None:
        idx = [ESM_TOKENS.index('<cls>')] + [ESM_TOKENS.index(aa) for aa in seq] + [ESM_TOKENS.index('<eos>')]
        # directly return idxs but not one-hot matrix
        protein = np.array(idx)
    else:
        idx = [aa_dict.index(aa) for aa in seq]
        protein = np.array(idx)
    if seq_start is not None and seq_end is not None:
        batch_tokens = protein[max(0, seq_start - 1): min(protein.shape[0], seq_end)]
    else:
        batch_tokens = protein
    if return_idx:
        if return_onehot_mat:
            return batch_tokens, np.array(idx), None
        else:
            return batch_tokens, np.array(idx)
    else:
        if return_onehot_mat:
            return batch_tokens, None
        else:
            return batch_tokens


def get_embedding_from_5dim(seq, seq_start=None, seq_end=None):
    protein = np.array([AA_5DIM_EMBED[aa] for aa in seq])
    if seq_start is not None and seq_end is not None:
        batch_tokens = protein[max(0, seq_start - 1): min(protein.shape[0], seq_end)]
    else:
        batch_tokens = protein
    return batch_tokens


def get_embedding_from_onehot_nonzero(seq, seq_start=None, seq_end=None, return_idx=False, 
                                      aa_dict=None, min_prob=0.001, return_onehot_mat=False):
    if aa_dict is None:
        aa_dict = AA_DICT
    one_hot_mat = np.eye(len(aa_dict))
    n_special_tok = 0
    for special_tok in ['<mask>', '<pad>']:
        if special_tok in aa_dict:
            one_hot_mat[aa_dict.index(special_tok), :] = -1
            one_hot_mat[:, aa_dict.index(special_tok)] = -1
            one_hot_mat[aa_dict.index(special_tok), aa_dict.index(special_tok)] = 2
            n_special_tok += 1
    one_hot_mat[one_hot_mat == 0] = min_prob
    one_hot_mat[one_hot_mat == 1] = 1 - min_prob * (len(aa_dict) - n_special_tok)
    one_hot_mat[one_hot_mat == -1] = 0
    one_hot_mat[one_hot_mat == 2] = 1
    idx = [aa_dict.index(aa) for aa in seq]
    protein = one_hot_mat[idx]
    if seq_start is not None and seq_end is not None:
        batch_tokens = protein[max(0, seq_start - 1): min(protein.shape[0], seq_end)]
    else:
        batch_tokens = protein
    if return_idx:
        if return_onehot_mat:
            return batch_tokens, np.array(idx), one_hot_mat
        else:
            return batch_tokens, np.array(idx)
    else:
        if return_onehot_mat:
            return batch_tokens, one_hot_mat
        else:
            return batch_tokens

# TODO: conservation should only from 1:21, not 1:41
def get_conservation_from_msa(mutation: Mutation, check_mode=False):
    transcript = mutation.transcript_id
    seq = mutation.seq
    seq_start = mutation.seq_start_orig
    seq_end = mutation.seq_end_orig
    if seq_start is None:
        seq_start = 1
    if seq_end is None:
        seq_end = len(seq)
    msa_alphabet = np.array(list('ACDEFGHIKLMNPQRSTVWYU'))
    if not os.path.exists(f'{MSA_DATA_PATH}/{transcript}.pickle'):
        matched_line = False
    else:
        with open(os.path.join(MSA_DATA_PATH, transcript + '.pickle'), 'rb') as file:
            msa_mat = pickle.load(file)
        msa_seq = ''.join(msa_alphabet[msa_mat[seq_start - 1:seq_end, 0].astype(int)])
        if mutation.crop:
            msa_seq = msa_seq[mutation.seq_start -1:mutation.seq_end]
        matched_line = msa_seq == seq
    if matched_line:
        if check_mode:
            return True
        # 1:20 is conservation from hhblits, 21:41 is conservation from compara
        conservation = msa_mat[seq_start - 1:seq_end, 1:41]
    else:
        if check_mode:
            return False
        conservation = np.zeros([seq_end - seq_start + 1, 40])
    if mutation.crop:
        conservation = conservation[mutation.seq_start -1:mutation.seq_end]
    return conservation


def get_msa_dict_from_transcript_archive(transcript):
    msa_alphabet = np.array(list('ACDEFGHIKLMNPQRSTVWYU'))
    if pd.isna(transcript) or not os.path.exists(f'{MSA_DATA_PATH}/{transcript}.pickle'):
        msa_seq = ''
        conservation = np.zeros([0, 20])
        msa = np.zeros([0, 200])
    else:
        with open(os.path.join(MSA_DATA_PATH, transcript + '.pickle'), 'rb') as file:
            msa_mat = pickle.load(file)
        msa_seq = ''.join(msa_alphabet[msa_mat[:, 0].astype(int)])
        conservation = msa_mat[:, 1:21]
        msa = msa_mat[:, 21:221]
    return msa_seq, conservation, msa


def get_msa_dict_from_transcript(uniprotID):
    msa_alphabet = np.array(list('ACDEFGHIKLMNPQRSTVWYU'))
    if pd.isna(uniprotID) or not os.path.exists(f'{MSA_DATA_PATH}/{uniprotID}_MSA.npy'):
        msa_seq = ''
        conservation = np.zeros([0, 20])
        msa = np.zeros([0, 199])
    else:
        msa_mat = np.load(f'{MSA_DATA_PATH}/{uniprotID}_MSA.npy')
        msa_seq = ''.join(msa_alphabet[msa_mat[:, 0].astype(int)])
        conservation = np.eye(21)[msa_mat.astype(int)].mean(axis=1)[:, :20]
        msa = msa_mat
    return msa_seq, conservation, msa


def get_confidence_from_af2file(af2file, pLDDT):
    uniprotID = af2file.split('/')[-1].split('.')[0].split('-model')[0]
    if pd.isna(uniprotID) or not os.path.exists(f'{PAE_DATA_PATH}/{uniprotID[3:6]}/{uniprotID}-predicted_aligned_error_v4.json.gz'):
        # if PAE does not exist, use pLDDT
        # pae = (pLDDT[None, :] + pLDDT[:, None]) / 200 if not pLDDT is None else None
        pae = (200 - pLDDT[None, :] - pLDDT[:, None]) / 4 if not pLDDT is None else None
    else:
        with gzip.open(f'{PAE_DATA_PATH}/{uniprotID[3:6]}/{uniprotID}-predicted_aligned_error_v4.json.gz', 'rt') as f:
            pae = json.load(f)
        # pae = np.exp(-0.08*np.array(pae[0]['predicted_aligned_error']))
        pae = np.array(pae[0]['predicted_aligned_error'])
    return pae


def get_msa(mutation: Mutation, check_mode=False):
    transcript = mutation.transcript_id
    seq = mutation.seq
    seq_start = mutation.seq_start_orig
    seq_end = mutation.seq_end_orig
    if seq_start is None:
        seq_start = 1
    if seq_end is None:
        seq_end = len(seq)
    msa_alphabet = np.array(list('ACDEFGHIKLMNPQRSTVWYU'))
    if not os.path.exists(f'{MSA_DATA_PATH}/{transcript}.pickle'):
        matched_line = False
    else:
        with open(os.path.join(MSA_DATA_PATH, transcript + '.pickle'), 'rb') as file:
            msa_mat = pickle.load(file)
        msa_seq = ''.join(msa_alphabet[msa_mat[seq_start - 1:seq_end, 0].astype(int)])
        if mutation.crop:
            msa_seq = msa_seq[mutation.seq_start -1:mutation.seq_end]
        matched_line = msa_seq == seq
    if matched_line:
        if check_mode:
            return True
        # 1:20 is conservation from hhblits, 1:21 is conservation from compara
        msa = msa_mat[seq_start - 1:seq_end, 21:221]
    else:
        if check_mode:
            return False
        msa = np.zeros([seq_end - seq_start + 1, 200])
    if mutation.crop:
        msa = msa[mutation.seq_start -1:mutation.seq_end]
    return msa


def get_logits_from_esm2(protein, check_mode=True, seq_start=None, seq_end=None):
    if isinstance(protein, str):
        file_path = f"{ESM_DATA_PATH}/{protein}.logits.npy"
        if os.path.exists(file_path):
            if check_mode:
                return True
            wt_orig = np.load(file_path)
            # TODO: I am removing the <BOS> and <EOS> tokens, not sure if this is correct
            batch_tokens = wt_orig[max(0, seq_start):
                                   min(wt_orig.shape[0] - 1, seq_end + 1)]
        else:
            if check_mode:
                return False
            batch_tokens = np.zeros([seq_end - seq_start + 1, 32])
    elif isinstance(protein, np.ndarray):
        batch_tokens = protein[max(0, seq_start):
                               min(protein.shape[0] - 1, seq_end + 1)]
    else:
        raise ValueError("protein must be either a string of uniprotID or a numpy array")
    return batch_tokens


def get_attn_from_msa(transcript, seq, check_mode=False, seq_start=None, seq_end=None):
    NUM_LAYERS = 6
    msa_alphabet = np.array(list('ACDEFGHIKLMNPQRSTVWYU'))
    if isinstance(transcript, str):
        if pd.isna(transcript) \
                or not os.path.exists(f'{MSA_DATA_PATH}/{transcript}.pickle') \
                or not os.path.exists(f'{MSA_ATTN_DATA_PATH}/{transcript}.row_attentions.pt'):
            matched_line = False
        else:
            with open(os.path.join(MSA_DATA_PATH, transcript + '.pickle'), 'rb') as file:
                msa_mat = pickle.load(file)
            if seq_start is None:
                seq_start = 1
            if seq_end is None:
                seq_end = len(seq)
            msa_seq = ''.join(msa_alphabet[msa_mat[seq_start - 1:seq_end, 0].astype(int)])
            matched_line = msa_seq == seq
        if matched_line:
            if check_mode:
                return True
            msa_row_attns = torch.load(
                os.path.join(MSA_ATTN_DATA_PATH, transcript + '.row_attentions.pt')).detach().numpy()
            msa_contacts = torch.load(os.path.join(MSA_ATTN_DATA_PATH, transcript + '.contacts.pt')).detach().numpy()
            # R file parse seq_start starting from 1, so we need to minus 1
            # only use last 6 attn layers
            msa_row_attns = msa_row_attns[:, (12 - NUM_LAYERS):, :, seq_start - 1:seq_end, seq_start - 1:seq_end]
            msa_contacts = msa_contacts[:, seq_start - 1:seq_end, seq_start - 1:seq_end]
            msa_pairwise = np.concatenate([msa_row_attns.reshape(-1, msa_row_attns.shape[-2], msa_row_attns.shape[-1]),
                                           msa_contacts], axis=0).transpose((1, 2, 0))
        else:
            if check_mode:
                return False
            msa_pairwise = np.zeros([seq_end - seq_start + 1, seq_end - seq_start + 1, NUM_LAYERS * 12 + 1])
    elif isinstance(transcript, tuple):
        msa_row_attns = transcript[0]
        msa_contacts = transcript[1]
        if msa_row_attns is not None and msa_contacts is not None:
            msa_row_attns = msa_row_attns[:, (12 - NUM_LAYERS):, :, seq_start - 1:seq_end, seq_start - 1:seq_end]
            msa_contacts = msa_contacts[:, seq_start - 1:seq_end, seq_start - 1:seq_end]
            msa_pairwise = np.concatenate([msa_row_attns.reshape(-1, msa_row_attns.shape[-2], msa_row_attns.shape[-1]),
                                           msa_contacts], axis=0).transpose((1, 2, 0))
        else:
            msa_pairwise = np.zeros([seq_end - seq_start + 1, seq_end - seq_start + 1, NUM_LAYERS * 12 + 1])
    else:
        raise ValueError("transcript must be either a string of transcriptID"
                         " or a tuple of msa_row_attns and msa_contacts")
    return msa_pairwise


def get_contacts_from_msa(mutation: Mutation, check_mode=False):
    transcript = mutation.transcript_id
    seq = mutation.seq
    seq_start = mutation.seq_start
    seq_end = mutation.seq_end
    msa_alphabet = np.array(list('ACDEFGHIKLMNPQRSTVWYU'))
    if pd.isna(transcript) \
            or not os.path.exists(f'{MSA_DATA_PATH_ARCHIVE}/{transcript}.pickle') \
            or not os.path.exists(f'{MSA_ATTN_DATA_PATH}/{transcript}.contacts.pt'):
        matched_line = False
    else:
        with open(os.path.join(MSA_DATA_PATH_ARCHIVE, transcript + '.pickle'), 'rb') as file:
            msa_mat = pickle.load(file)
        if seq_start is None:
            seq_start = 1
        if seq_end is None:
            seq_end = len(seq)
        msa_seq = ''.join(msa_alphabet[msa_mat[seq_start - 1:seq_end, 0].astype(int)])
        matched_line = msa_seq == seq
    if matched_line:
        if check_mode:
            return True
        msa_contacts = torch.load(os.path.join(MSA_ATTN_DATA_PATH, transcript + '.contacts.pt')).detach().numpy()
        # R file parse seq_start starting from 1, so we need to minus 1
        msa_contacts = msa_contacts[:, seq_start - 1:seq_end, seq_start - 1:seq_end]
        msa_pairwise = msa_contacts.transpose((1, 2, 0))
    else:
        # no esm_msa file, try esm2 predicted contacts instead
        if not os.path.exists(f'{ESM_DATA_PATH}/{mutation.ESM_prefix}.contacts.npy'):
            if check_mode:
                return False
            msa_pairwise = np.zeros([seq_end - seq_start + 1, seq_end - seq_start + 1, 1])
        else:
            if check_mode:
                return True
            msa_pairwise = np.load(f'{ESM_DATA_PATH}/{mutation.ESM_prefix}.contacts.npy')
            msa_pairwise = np.expand_dims(msa_pairwise[seq_start - 1:seq_end, seq_start - 1:seq_end], axis=2)
    return msa_pairwise

# unused
def get_contacts_from_msa_by_identifier(identifier):
    str_split = identifier.split(":")
    transcript = str_split[0]
    seq = str_split[1]
    seq_start = int(str_split[2])
    seq_end = int(str_split[3])
    check_mode = False
    return get_contacts_from_msa(transcript, seq, check_mode, seq_start, seq_end)

# unused
def load_embedding_from_esm2(protein):
    file_path = f"{ESM_DATA_PATH}/{protein}.representations.layer.48.npy"
    assert os.path.exists(file_path)
    return np.load(file_path)

# unused
def load_logits_from_esm2(protein):
    file_path = f"{ESM_DATA_PATH}/{protein}.logits.npy"
    assert os.path.exists(file_path)
    return np.load(file_path)

# unused
def load_attn_from_msa(transcript):
    if os.path.exists(os.path.join(MSA_ATTN_DATA_PATH, transcript + '.row_attentions.pt')) and \
            os.path.exists(os.path.join(MSA_ATTN_DATA_PATH, transcript + '.contacts.pt')):
        msa_row_attns = torch.load(os.path.join(MSA_ATTN_DATA_PATH, transcript + '.row_attentions.pt')).detach().numpy()
        msa_contacts = torch.load(os.path.join(MSA_ATTN_DATA_PATH, transcript + '.contacts.pt')).detach().numpy()
        return msa_row_attns, msa_contacts
    else:
        return None, None


def _test_load():
    test_file = pd.read_csv('/share/terra/Users/gz2294/ld1/Data/DMS/ClinVar.HGMD.PrimateAI.syn/training.csv',
                            index_col=0)
    # idx = np.where(test_file.uniprotID == 'Q8WZ42')[0][0]
    idx = np.where(test_file['sequence.len.orig'] == 4753)[0][0]
    point_mutation = get_mutations(test_file['uniprotID'].iloc[idx],
                                    test_file['ENST'].iloc[idx],
                                    test_file['wt.orig'].iloc[idx],
                                    test_file['sequence.len.orig'].iloc[idx],
                                    test_file['pos.orig'].iloc[idx],
                                    test_file['ref'].iloc[idx],
                                    test_file['alt'].iloc[idx])
    coords = get_coords_from_af2(point_mutation.af2_file)

    CA_coord = coords[:, 3]
    embed_data = get_embedding_from_esm2(point_mutation.uniprot_id, False,
                                         point_mutation.seq_start, point_mutation.seq_end)
    # prepare edge features
    coev_strength = get_attn_from_msa(point_mutation.transcript_id, point_mutation.seq, False,
                                      point_mutation.seq_start, point_mutation.seq_end)
    edge_index = np.indices((coords.shape[0], coords.shape[0])).reshape(2, -1)
    # cancel self-edges
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    edge_attr = coev_strength[edge_index[0], edge_index[1], :]
    # prepare node vector features
    CA_CB = coords[:, [4]] - coords[:, [3]]
    CA_C = coords[:, [1]] - coords[:, [3]]
    CA_O = coords[:, [2]] - coords[:, [3]]
    CA_N = coords[:, [0]] - coords[:, [3]]
    nodes_vector = np.concatenate([CA_CB, CA_C, CA_O, CA_N], axis=1)
    # prepare graph
    features = dict(
        pos=torch.from_numpy(CA_coord), x=torch.from_numpy(embed_data),
        edge_index=torch.from_numpy(edge_index), edge_attr=torch.from_numpy(edge_attr).to(torch.float),
        node_vec_attr=torch.from_numpy(nodes_vector).transpose(1, 2)
    )
    from torch_geometric.data import Data

    map_data = Data(**features)
    return map_data


if __name__ == '__main__':
    print(_test_load())
