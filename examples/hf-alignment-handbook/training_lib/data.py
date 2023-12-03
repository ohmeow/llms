# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/10_data.ipynb.

# %% auto 0
__all__ = ['mix_datasets', 'get_datasets']

# %% ../nbs/10_data.ipynb 1
import re
from typing import List, Literal, Optional

from datasets import DatasetDict, concatenate_datasets, load_dataset

from .configs import DataArguments

# %% ../nbs/10_data.ipynb 2
def mix_datasets(dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training data.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets, raw_val_datasets, fracs = [], [], []
    for ds, frac in dataset_mixer.items():
        fracs.append(frac)
        for split in splits:
            if "train" in split:
                raw_train_datasets.append(load_dataset(ds, split=split))
            elif "test" in split:
                raw_val_datasets.append(load_dataset(ds, split=split))
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
            
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
            
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted.")

    return raw_datasets

# %% ../nbs/10_data.ipynb 4
def get_datasets(data_config: DataArguments | dict, splits: List[str] = ["train", "test"], shuffle: bool = True) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """

    if type(data_config) is DataArguments:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        dataset_mixer = data_config.dataset_mixer
    elif type(data_config) is dict:
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")
    
    # print(dataset_mixer)
    raw_datasets = mix_datasets(dataset_mixer, splits=splits, shuffle=shuffle)
    
    # If debugging
    if data_config.max_train_samples:
        for ds_name in raw_datasets.keys():
            if "train" in ds_name:
                raw_datasets[ds_name] = raw_datasets[ds_name].shuffle(seed=42).select(range(data_config.max_train_samples))
                
    if data_config.max_eval_samples:
        for ds_name in raw_datasets.keys():
            if "eval" in ds_name or "test" in ds_name:
                raw_datasets[ds_name] = raw_datasets[ds_name].shuffle(seed=42).select(range(data_config.max_eval_samples))
            
    return raw_datasets
