import argparse
import numpy as np
import torch
import yaml


class LoadFromFile(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for key in config.keys():
                if key not in namespace:
                    raise ValueError(f"Unknown argument in config file: {key}")
            if (
                    "load_model" in config
                    and namespace.load_model is not None
                    and config["load_model"] != namespace.load_model
            ):
                Warning(
                    f"The load model argument was specified as a command line argument "
                    f"({namespace.load_model}) and in the config file ({config['load_model']}). "
                    f"Ignoring 'load_model' from the config file and loading {namespace.load_model}."
                )
                del config["load_model"]
            namespace.__dict__.update(config)
        else:
            raise ValueError("Configuration file must end with yaml or yml")


# TODO: for all functions, shuffle in a way that guarantee the same batch does not contain same protein.
def train_val_test_split(dset_len, train_size, val_size, test_size, seed, order=None):
    assert (train_size is None) + (val_size is None) + (
            test_size is None
    ) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
        isinstance(test_size, float),
    )

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size
    test_size = round(dset_len * test_size) if is_float[2] else test_size

    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size

    if train_size + val_size + test_size > dset_len:
        if is_float[2]:
            test_size -= 1
        elif is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0 and test_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) or "
        f"testing ({test_size}) splits ended up with a negative size."
    )

    total = train_size + val_size + test_size
    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
    if total < dset_len:
        Warning(f"{dset_len - total} samples were excluded from the dataset")

    idxs = np.arange(dset_len, dtype=int)
    if order is None:
        idxs = np.random.default_rng(seed).permutation(idxs)

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size: train_size + val_size]
    idx_test = idxs[train_size + val_size: total]

    if order is not None:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
        idx_test = [order[i] for i in idx_test]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def make_splits_train_val_test(
        dataset_len,
        train_size,
        val_size,
        test_size,
        seed,
        filename=None,
        splits=None,
        order=None,
):
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len, train_size, val_size, test_size, seed, order
        )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
        torch.from_numpy(idx_test),
    )


def train_val_split(dset_len, train_size, val_size, seed, order=None):
    assert (train_size is None) + (
            val_size is None) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
    )
    # dset_len: int = len(dset)

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size

    if train_size is None:
        train_size = dset_len - val_size
    elif val_size is None:
        val_size = dset_len - train_size

    if train_size + val_size > dset_len:
        if is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0, (
        f"One of training ({train_size}), validation ({val_size})"
        f" splits ended up with a negative size."
    )

    total = train_size + val_size
    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
    if total < dset_len:
        Warning(f"{dset_len - total} samples were excluded from the dataset")

    idxs = np.arange(dset_len, dtype=int)
    if order is None:
        idxs = np.random.default_rng(seed).permutation(idxs)

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size: total]

    if order is not None:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]

    return np.array(idx_train), np.array(idx_val)


def make_splits_train_val(
        dset,
        train_size,
        val_size,
        seed,
        batch_size=48,  # unused
        filename=None,
        splits=None,
        order=None,
):
    dset_len = len(dset)
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
    else:
        idx_train, idx_val = train_val_split(dset_len, train_size, val_size, seed, order)

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
    )


def make_splits_train_val_by_anno(
        dset,
        train_size,
        val_size,
        seed,
        batch_size=48,  # unused
        filename=None,
        splits=None,
        order=None,
):
    dset_len = len(dset)
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
    else:
        idx_train = np.where(dset.data['split'] == 'train')[0]
        idx_val = np.where(dset.data['split'] == 'val')[0]

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
    )


def train_val_split_by_uniprot_id(dset, train_size, val_size, seed, batch_size=48, order=None):
    assert (train_size is None) + (
            val_size is None) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
    )
    dset_len: int = len(dset)

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size

    if train_size is None:
        train_size = dset_len - val_size
    elif val_size is None:
        val_size = dset_len - train_size

    if train_size + val_size > dset_len:
        if is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0, (
        f"One of training ({train_size}), validation ({val_size})"
        f" splits ended up with a negative size."
    )

    total = train_size + val_size
    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
    if total < dset_len:
        Warning(f"{dset_len - total} samples were excluded from the dataset")

    uniprot_freq_table = dset.data.uniprotID.value_counts()
    selected_val_uniprotIDs, _ = select_by_uniprot(uniprot_freq_table, val_size)

    idxs = np.arange(dset_len, dtype=int)
    idx_train = idxs[np.isin(dset.data.uniprotID, selected_val_uniprotIDs, invert=True)]
    idx_val = idxs[np.isin(dset.data.uniprotID, selected_val_uniprotIDs)]

    if order is None:
        idx_train = np.random.default_rng(seed).permutation(idx_train)
        idx_val = np.random.default_rng(seed).permutation(idx_val)
        idx_train = guarantee_no_same_protein_in_one_batch(
            idx_train, batch_size, np.array(dset.data.uniprotID.iloc[idx_train] + dset.data.ENST.iloc[idx_train])
        )
        idx_val = guarantee_no_same_protein_in_one_batch(
            idx_val, batch_size, np.array(dset.data.uniprotID.iloc[idx_val] + dset.data.ENST.iloc[idx_val])
        )
    else:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]

    return np.array(idx_train), np.array(idx_val)


def make_splits_train_val_by_uniprot_id(
        dset,
        train_size,
        val_size,
        seed,
        batch_size=48,
        filename=None,
        splits=None,
        order=None,
):
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
    else:
        idx_train, idx_val = train_val_split_by_uniprot_id(dset, train_size, val_size, seed, batch_size, order)

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
    )


def train_val_split_by_good_batch(dset, train_size, val_size, seed, batch_size=48, order=None):
    assert (train_size is None) + (
            val_size is None) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
    )
    dset_len: int = len(dset)

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size

    if train_size is None:
        train_size = dset_len - val_size
    elif val_size is None:
        val_size = dset_len - train_size

    if train_size + val_size > dset_len:
        if is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0, (
        f"One of training ({train_size}), validation ({val_size})"
        f" splits ended up with a negative size."
    )

    total = train_size + val_size
    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
    if total < dset_len:
        Warning(f"{dset_len - total} samples were excluded from the dataset")

    idxs = np.arange(dset_len, dtype=int)
    if order is None:
        idxs = np.random.default_rng(seed).permutation(idxs)

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size: total]
    
    if order is None:
        idx_train = np.random.default_rng(seed).permutation(idx_train)
        idx_val = np.random.default_rng(seed).permutation(idx_val)
        idx_train = guarantee_good_batch(idx_train, batch_size, dset)
    else:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
    
    return np.array(idx_train), np.array(idx_val)


def make_splits_train_val_by_good_batch(
        dset,
        train_size,
        val_size,
        seed,
        batch_size=48,
        filename=None,
        splits=None,
        order=None,
):
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
    else:
        idx_train, idx_val = train_val_split_by_good_batch(dset, train_size, val_size, seed, batch_size, order)

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
    )


def train_val_split_by_uniprot_id_good_batch(dset, train_size, val_size, seed, batch_size=48, order=None):
    assert (train_size is None) + (
            val_size is None) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
    )
    dset_len: int = len(dset)

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size

    if train_size is None:
        train_size = dset_len - val_size
    elif val_size is None:
        val_size = dset_len - train_size

    if train_size + val_size > dset_len:
        if is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0, (
        f"One of training ({train_size}), validation ({val_size})"
        f" splits ended up with a negative size."
    )

    total = train_size + val_size
    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
    if total < dset_len:
        Warning(f"{dset_len - total} samples were excluded from the dataset")

    uniprot_freq_table = dset.data.uniprotID.value_counts()
    selected_val_uniprotIDs, _ = select_by_uniprot(uniprot_freq_table, val_size)

    idxs = np.arange(dset_len, dtype=int)
    idx_train = idxs[np.isin(dset.data.uniprotID, selected_val_uniprotIDs, invert=True)]
    idx_val = idxs[np.isin(dset.data.uniprotID, selected_val_uniprotIDs)]

    if order is None:
        idx_train = np.random.default_rng(seed).permutation(idx_train)
        idx_val = np.random.default_rng(seed).permutation(idx_val)
        idx_train = guarantee_good_batch(idx_train, batch_size, dset)
    else:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]

    return np.array(idx_train), np.array(idx_val)


def make_splits_train_val_by_uniprot_id_good_batch(
        dset,
        train_size,
        val_size,
        seed,
        batch_size=48,
        filename=None,
        splits=None,
        order=None,
):
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
    else:
        idx_train, idx_val = train_val_split_by_uniprot_id_good_batch(dset, train_size, val_size, seed, batch_size, order)

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
    )


def train_val_test_split_by_uniprot_id(dset, train_size, val_size, test_size, seed, batch_size=48, order=None):
    assert (train_size is None) + (val_size is None) + (
            test_size is None
    ) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
        isinstance(test_size, float),
    )
    dset_len: int = len(dset)
    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size
    test_size = round(dset_len * test_size) if is_float[2] else test_size

    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size

    if train_size + val_size + test_size > dset_len:
        if is_float[2]:
            test_size -= 1
        elif is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0 and test_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) or "
        f"testing ({test_size}) splits ended up with a negative size."
    )

    total = train_size + val_size + test_size
    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
    if total < dset_len:
        Warning(f"{dset_len - total} samples were excluded from the dataset")

    uniprot_freq_table = dset.data.uniprotID.value_counts()
    selected_test_uniprotIDs, uniprot_freq_table = select_by_uniprot(uniprot_freq_table, test_size)
    selected_val_uniprotIDs, uniprot_freq_table = select_by_uniprot(uniprot_freq_table, val_size)

    idxs = np.arange(dset_len, dtype=int)

    idx_test = idxs[np.isin(dset.data.uniprotID, selected_test_uniprotIDs)]
    idx_val = idxs[np.isin(dset.data.uniprotID, selected_val_uniprotIDs)]
    idx_train = idxs[np.isin(dset.data.uniprotID, selected_test_uniprotIDs + selected_val_uniprotIDs, invert=True)]

    if order is None:
        idx_train = np.random.default_rng(seed).permutation(idx_train)
        idx_val = np.random.default_rng(seed).permutation(idx_val)
        idx_test = np.random.default_rng(seed).permutation(idx_test)
        idx_train = guarantee_no_same_protein_in_one_batch(
            idx_train, batch_size, np.array(dset.data.uniprotID.iloc[idx_train] + dset.data.ENST.iloc[idx_train])
        )
        idx_val = guarantee_no_same_protein_in_one_batch(
            idx_val, batch_size, np.array(dset.data.uniprotID.iloc[idx_val] + dset.data.ENST.iloc[idx_val])
        )
        idx_test = guarantee_no_same_protein_in_one_batch(
            idx_test, batch_size, np.array(dset.data.uniprotID.iloc[idx_test] + dset.data.ENST.iloc[idx_test])
        )
    else:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
        idx_test = [order[i] for i in idx_test]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def make_splits_train_val_test_by_uniprot_id(
        dset,
        train_size,
        val_size,
        test_size,
        seed,
        batch_size=48,
        filename=None,
        splits=None,
        order=None,
):
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:
        idx_train, idx_val, idx_test = train_val_test_split_by_uniprot_id(
            dset, train_size, val_size, test_size, seed, batch_size, order
        )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
        torch.from_numpy(idx_test),
    )


def reshuffle_train_by_uniprot_id(idx_train, batch_size, dset, seed=None):
    idx_train = guarantee_no_same_protein_in_one_batch(
            idx_train, batch_size, 
            np.array(dset.data.uniprotID.iloc[idx_train] + dset.data.ENST.iloc[idx_train]),
            seed=seed
        )
    return idx_train


def reshuffle_train_by_good_batch(idx_train, batch_size, dset, seed=None):
    idx_train = guarantee_good_batch(
            idx_train, batch_size, dset, seed=seed
        )
    return idx_train


def reshuffle_train_by_uniprot_id_good_batch(idx_train, batch_size, dset, seed=None):
    return reshuffle_train_by_good_batch(idx_train, batch_size, dset, seed)


def reshuffle_train_by_anno(idx_train, batch_size, dset, seed=None):
    if seed is not None:
        idx_train = np.random.default_rng(seed).permutation(idx_train)
    else:
        idx_train = np.random.permutation(idx_train)
    return idx_train


def reshuffle_train(idx_train, batch_size, dset, seed=None):
    if seed is not None:
        idx_train = np.random.default_rng(seed).permutation(idx_train)
    else:
        idx_train = np.random.permutation(idx_train)
    return idx_train


def select_by_uniprot(freq_table, number_to_select):
    selected = 0
    selected_uniprotIDs = []
    candidates = freq_table[freq_table <= number_to_select - selected]
    while selected < number_to_select and len(candidates) > 0:
        selected_uniprotID = np.random.choice(candidates.index)
        selected_uniprotIDs.append(selected_uniprotID)
        selected += freq_table[selected_uniprotID]
        # update freq_table and candidates
        freq_table = freq_table.drop(selected_uniprotID)
        candidates = freq_table[freq_table <= number_to_select - selected]
    return selected_uniprotIDs, freq_table


def guarantee_no_same_protein_in_one_batch(idxs, batch_size, protein_identifiers, seed=0):
    assert len(idxs) == len(protein_identifiers)
    if seed is not None:
        np.random.seed(seed)
    # assume idxs and protein_identifiers are shuffled
    result = []
    while len(protein_identifiers) >= batch_size:
        unique_protein_identifiers, first_idx, counts = np.unique(protein_identifiers,
                                                                  return_index=True, return_counts=True)
        if len(unique_protein_identifiers) < batch_size:
            break
        unique_random_selected = np.random.choice(unique_protein_identifiers,
                                                  size=batch_size, replace=False, p=counts / sum(counts))
        unique_random_selected_idx = first_idx[[np.argwhere(unique_protein_identifiers == i)[0][0]
                                                for i in unique_random_selected]]
        result += list(idxs[unique_random_selected_idx])
        # drop selected
        idxs = np.delete(idxs, unique_random_selected_idx)
        protein_identifiers = np.delete(protein_identifiers, unique_random_selected_idx)
    # if idxs is not empty, append it to result
    if len(idxs) > 0:
        result += list(idxs)
    return np.array(result)


def guarantee_good_batch_not_same_len(idxs, batch_size, dset, seed=0):
    # guarantee generate good batches during training, which at least include:
    # A positive example in gene A.
    # A negative example in gene A.
    # A positive example in gene !A.
    # A negative example in gene !A.
    # This only work for binary classification problem
    assert batch_size >= 4
    if seed is not None:
        np.random.seed(seed)
    # first get all positives and negatives
    pos_idxs = idxs[dset.data[dset._y_columns[0]].iloc[idxs] == 1]
    pos_ids = dset.data["uniprotID"].iloc[pos_idxs].to_numpy()
    neg_idxs = idxs[dset.data[dset._y_columns[0]].iloc[idxs] != 1]
    neg_ids = dset.data["uniprotID"].iloc[neg_idxs].to_numpy()
    # assume idxs and protein_identifiers are shuffled
    result = []
    # loop through all positive idexes
    for i in range(len(pos_idxs)):
        id = pos_ids[i]
        result.append(pos_idxs[i])
        # get negative example in the same protein
        neg_idx = neg_idxs[neg_ids == id]
        if len(neg_idx) > 0:
            result.append(np.random.choice(neg_idx))
        else:
            result.append(np.random.choice(neg_idxs))
        # get positive example in different protein
        pos_idx = pos_idxs[pos_ids != id]
        if len(pos_idx) == 0:
            pos_idx = pos_idxs[pos_idxs != pos_idxs[i]]
        result.append(np.random.choice(pos_idx))
        # get negative example in different protein
        neg_idx = neg_idxs[neg_ids != id]
        if len(neg_idx) == 0:
            neg_idx = neg_idxs[neg_idxs != neg_idxs[i]]
        result.append(np.random.choice(neg_idx))
        # if batch_size is larger than 4, randomly select more
        if batch_size > 4:
            result += list(np.random.choice(idxs, size=batch_size - 4, replace=False))
    # if some idxs are not used, randomly select them
    unused_idxs = np.setdiff1d(idxs, result)
    if len(unused_idxs) > 0:
        result += list(unused_idxs)
    return np.array(result)


def guarantee_good_batch(idxs, batch_size, dset, seed=0):
    # guarantee generate good batches during training, which at least include:
    # A positive example in gene A.
    # A negative example in gene A.
    # A positive example in gene !A.
    # A negative example in gene !A.
    # This only work for binary classification problem
    assert batch_size >= 4
    if seed is not None:
        np.random.seed(seed)
    if not isinstance(idxs, np.ndarray):
        idxs = np.array(idxs)
    # first get all positives and negatives
    pos_label = 3
    if sum(dset.data[dset._y_columns[0]].iloc[idxs] == 1) > 0:
        pos_label = 1
    pos_idxs = idxs[dset.data[dset._y_columns[0]].iloc[idxs] == pos_label]
    pos_ids = dset.data["uniprotID"].iloc[pos_idxs].to_numpy()
    neg_idxs = idxs[dset.data[dset._y_columns[0]].iloc[idxs] != pos_label]
    neg_ids = dset.data["uniprotID"].iloc[neg_idxs].to_numpy()
    # assume idxs and protein_identifiers are shuffled
    result = []
    # loop through all positive idexes
    while len(pos_idxs) > 0:
        this_batch_added = 0
        # get a positive example
        pos_idx = np.random.choice(pos_idxs)
        id = pos_ids[pos_idxs == pos_idx][0]
        result.append(pos_idx)
        this_batch_added += 1
        # drop selected
        pos_ids = np.delete(pos_ids, np.argwhere(pos_idxs == pos_idx))
        idxs = np.delete(idxs, np.argwhere(idxs == pos_idx))
        pos_idxs = np.delete(pos_idxs, np.argwhere(pos_idxs == pos_idx))
        # get negative example in the same protein
        neg_idx = neg_idxs[neg_ids == id]
        if len(neg_idx) > 0:
            neg_idx = np.random.choice(neg_idx)
        elif len(neg_idxs) > 0:
            neg_idx = np.random.choice(neg_idxs)
        else:
            neg_idx = None
        if neg_idx is not None:
            result.append(neg_idx)
            this_batch_added += 1
            # drop selected
            neg_ids = np.delete(neg_ids, np.argwhere(neg_idxs == neg_idx))
            idxs = np.delete(idxs, np.argwhere(idxs == neg_idx))
            neg_idxs = np.delete(neg_idxs, np.argwhere(neg_idxs == neg_idx))
        # get positive example in different protein
        pos_idx_candidate = pos_idxs[pos_ids != id]
        if len(pos_idx_candidate) == 0:
            pos_idx_candidate = pos_idxs[pos_idxs != pos_idx]
        if len(pos_idx_candidate) > 0:
            pos_idx = np.random.choice(pos_idx_candidate)
            result.append(pos_idx)
            this_batch_added += 1
            # drop selected
            pos_ids = np.delete(pos_ids, np.argwhere(pos_idxs == pos_idx))
            idxs = np.delete(idxs, np.argwhere(idxs == pos_idx))
            pos_idxs = np.delete(pos_idxs, np.argwhere(pos_idxs == pos_idx))
        # get negative example in different protein
        neg_idx_candidate = neg_idxs[neg_ids != id]
        if len(neg_idx_candidate) == 0:
            neg_idx_candidate = neg_idxs[neg_idxs != neg_idx]
        if len(neg_idx_candidate) > 0:
            neg_idx = np.random.choice(neg_idx_candidate)
            result.append(neg_idx)
            this_batch_added += 1
            # drop selected
            neg_ids = np.delete(neg_ids, np.argwhere(neg_idxs == neg_idx))
            idxs = np.delete(idxs, np.argwhere(idxs == neg_idx))
            neg_idxs = np.delete(neg_idxs, np.argwhere(neg_idxs == neg_idx))
        # if batch_size is larger than this_batch_added, randomly select more
        if batch_size > this_batch_added and len(idxs) >= batch_size - this_batch_added:
            to_add = np.random.choice(idxs, size=batch_size - this_batch_added, replace=False)
            result += list(to_add)
            # drop selected
            for idx in to_add:
                pos_ids = np.delete(pos_ids, np.argwhere(pos_idxs == idx))
                pos_idxs = np.delete(pos_idxs, np.argwhere(pos_idxs == idx))
                neg_ids = np.delete(neg_ids, np.argwhere(neg_idxs == idx))
                neg_idxs = np.delete(neg_idxs, np.argwhere(neg_idxs == idx))
                idxs = np.delete(idxs, np.argwhere(idxs == idx))
    # if some idxs are not used, randomly select them
    unused_idxs = np.setdiff1d(idxs, result)
    if len(unused_idxs) > 0:
        result += list(unused_idxs)
    return np.array(result)


def save_argparse(args, filename, exclude=None):
    import json

    if filename.endswith("yaml") or filename.endswith("yml"):
        if isinstance(exclude, str):
            exclude = [exclude]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]

        ds_arg = args.get("dataset_arg")
        if ds_arg is not None and isinstance(ds_arg, str):
            args["dataset_arg"] = json.loads(args["dataset_arg"])
        yaml.dump(args, open(filename, "w"))
    else:
        raise ValueError("Configuration file should end with yaml or yml")
