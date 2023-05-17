import torch
import torch.nn as nn
import esm
import pandas as pd
import numpy as np
import h5py
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import os
import pickle
from multiprocessing import Pool
from typing import List, Tuple, Sequence, Optional, Dict


class VrtReprAgentBatchConverter(object):
    """
    Converts a batch of sequences to a batch of tokens
    """

    def __init__(self, alphabet, max_len=1001):
        self.alphabet = alphabet
        self.max_len = max_len

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        max_len = max(self.max_len, max(len(seq_encoded) for seq_encoded in seq_encoded_list))
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
                zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
            i,
            int(self.alphabet.prepend_bos): len(seq_encoded)
                                            + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        return labels, strs, tokens


class VrtReprAgentDataSet(Dataset):
    """
    Dataset for the VRT representation agent.
    """

    def __init__(self, data_path, batch_converter, batch_size, batch_number=None, shuffle=True, num_workers=32):
        """
        @param    data_path: the directory containing the data
        @param    batch_size: the batch size
        @param    batch_number: optional, the number of batches to load
        @param    shuffle: whether to shuffle the data
        @param    num_workers: the number of workers to use for data loading
        """
        if isinstance(data_path, pd.DataFrame):
            self.data = data_path
        elif isinstance(data_path, str):
            self.data = pd.read_csv(data_path, index_col=0)
        else:
            raise ValueError("data_path must be a string or a pandas.DataFrame")
        if shuffle:
            self.data = self.data.sample(frac=1, random_state=0)
        if batch_size is None:
            assert batch_number is not None, "batch_size and batch_number cannot both be None"
            self.batch_size = int(np.ceil(self.data.shape[0] / batch_number))
            self.batch_number = batch_number
        else:
            self.batch_size = batch_size
            self.batch_number = int(np.ceil(self.data.shape[0] / self.batch_size))
        self.batch_converter = batch_converter
        self.num_workers = num_workers
        # prepare data
        data_wt = tuple(zip(self.data['VarID'].astype('str'), self.data['wt'].astype('str')))
        data_vr = self.data['sequence'].astype('str').tolist()
        for i in range(len(data_vr)):
            data_vr[i] = data_vr[i][:self.data.pos.iloc[i] - 1] + '<mask>' + data_vr[i][self.data.pos.iloc[i]:]
        data_vr = tuple(zip(self.data['VarID'].astype('str'), data_vr))
        _, _, batch_tokens_wt = self.batch_converter(data_wt)
        _, _, batch_tokens_vr = self.batch_converter(data_vr)
        labels = torch.tensor(self.data['pos'].to_numpy(), dtype=torch.long)
        # note we use pos-1 as the variant position is 1-indexed
        pos = torch.tensor(self.data['pos'].to_numpy() - 1, dtype=torch.long)
        self.batch_tokens_wt = batch_tokens_wt
        self.batch_tokens_vr = batch_tokens_vr
        self.labels = labels
        self.pos = pos

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.batch_tokens_wt[idx], self.batch_tokens_vr[idx], \
               self.pos[idx], \
               self.labels[idx]

    def count_labels(self):
        return self.data['score'].value_counts().sort_index().values

    def get_max_index(self):
        return int(np.ceil(len(self) / self.batch_size))

    def __iter__(self):
        return VrtReprAgentDataIterator(self)


class VrtReprAgentDataIterator:
    """
    Iterable class that returns batches of data.
    """

    def __init__(self, data_set: VrtReprAgentDataSet):
        """
        @param    data_set: the data set to iterate over
        """
        self._dataset = data_set
        self._index = 0
        self._max_index = int(np.ceil(len(self._dataset) / self._dataset.batch_size))
        self._batch_size = self._dataset.batch_size

    def __next__(self):
        """
        Returns the next batch of data.
        """
        if self._index < self._max_index:
            batch_idx = np.arange(self._index * self._batch_size,
                                  min((self._index + 1) * self._batch_size, len(self._dataset)))
            self._index += 1
            return self._dataset.batch_tokens_vr[batch_idx], \
                   self._dataset.pos[batch_idx], \
                   self._dataset.labels[batch_idx]
        # End of Iteration
        raise StopIteration


def inference_one_file(filename, model, batch_converter):
    dataset = VrtReprAgentDataSet(filename,
                                  batch_converter, batch_size=130, batch_number=None,
                                  shuffle=False, num_workers=32)
    results_vr_df = pd.DataFrame()
    results_wt_df = pd.DataFrame()
    with torch.no_grad():
        model.eval()
        for batch_tokens_wt, batch_tokens_vr, pos, _ in DataLoader(dataset, shuffle=False, batch_size=dataset.batch_size):
            batch_tokens_wt = batch_tokens_wt.to(device)
            batch_tokens_vr = batch_tokens_vr.to(device)
            pos = pos.to(device)
            results_vr = model(batch_tokens_vr, repr_layers=[], return_contacts=False)
            logits_vr = results_vr["logits"][range(pos.shape[0]), pos + 1, :]
            results_vr_df = pd.concat([results_vr_df, pd.DataFrame(logits_vr.cpu().numpy())], axis=0)
            results_wt = model(batch_tokens_wt, repr_layers=[], return_contacts=False)
            logits_wt = results_wt["logits"][range(pos.shape[0]), pos + 1, :]
            results_wt_df = pd.concat([results_wt_df, pd.DataFrame(logits_wt.cpu().numpy())], axis=0)

    return results_vr_df, results_wt_df


def inference_mask_one_file(filename, model, batch_converter):
    dataset = VrtReprAgentDataSet(filename,
                                  batch_converter, batch_size=130, batch_number=None,
                                  shuffle=False, num_workers=32)
    results_vr_df = pd.DataFrame()
    tokens_vr_df = pd.DataFrame()
    # results_wt_df = pd.DataFrame()
    with torch.no_grad():
        model.eval()
        for _, batch_tokens_vr, pos, _ in DataLoader(dataset, shuffle=False, batch_size=dataset.batch_size):
            # batch_tokens_wt = batch_tokens_wt.to(device)
            batch_tokens_vr = batch_tokens_vr.to(device)
            pos = pos.to(device)
            results_vr = model(batch_tokens_vr, repr_layers=[33], return_contacts=False)
            logits_vr = results_vr["logits"][range(pos.shape[0]), pos + 1, :]
            tokens_vr = results_vr["representations"][33][range(pos.shape[0]), pos + 1, :]
            results_vr_df = pd.concat([results_vr_df, pd.DataFrame(logits_vr.cpu().numpy())], axis=0)
            tokens_vr_df = pd.concat([tokens_vr_df, pd.DataFrame(tokens_vr.cpu().numpy())], axis=0)

    return results_vr_df, tokens_vr_df


if __name__ == '__main__':
    device = 'cuda:0'
    esm_name = 'esm2'
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.to(device)
    batch_converter = VrtReprAgentBatchConverter(alphabet)
    for dataset in ["testing", "training"]:
        for gene in ["PF00028", "PF00047", "PF00069", "PF00454", "PF00520", "PF06512", "PF07679", "PF07714", "PF11933", "IonChannel"]:
            os.makedirs(f'/share/terra/Users/gz2294/RESCVE.final/analysis/{esm_name}.inference/{gene}/', exist_ok=True)
            filename = f'/share/terra/Users/gz2294/ld1/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8/{gene}/{dataset}.csv'
            results_vr_df, tokens_vr_df = inference_mask_one_file(filename, model, batch_converter)
            results_vr_df.to_csv(
                f'/share/terra/Users/gz2294/RESCVE.final/analysis/'
                f'{esm_name}.inference/{gene}/{dataset}.logits.csv')
            tokens_vr_df.to_csv(
                f'/share/terra/Users/gz2294/RESCVE.final/analysis/'
                f'{esm_name}.inference/{gene}/{dataset}.tokens.csv')

    

