from typing import List

from datasets import DatasetDict, load_dataset

import config


def load_from_files(files: List[str]) -> DatasetDict:
    dataset = load_dataset('csv', delimiter="\t", data_files=files)
    return dataset


def load_from_dir(dataset_dir: str) -> DatasetDict:
    dataset = load_dataset('csv', delimiter="\t", data_dir=dataset_dir)
    return dataset


def train_test_split_by_label(dataset: DatasetDict, test_size: float = 0.2) -> DatasetDict:
    dataset = dataset['train'].add_column(name='label', column=dataset['train']['intent'])
    dataset = dataset.class_encode_column("label")
    dataset = dataset.train_test_split(seed=42,
                                       test_size=test_size,
                                       stratify_by_column='label')
    return dataset


def save_split(path_to_save: str, dataset: DatasetDict):
    if config.save_train_test_split:
        dataset['train'].to_csv(f'{path_to_save}/train.csv', sep='\t')
        dataset['test'].to_csv(f'{path_to_save}/test.csv', sep='\t')


def load_split_save(input, path_to_save: str):
    dataset = DatasetDict()
    if type(input) is list:
        dataset = load_from_files(input)
    if type(input) is str:
        dataset = load_from_dir(input)

    dataset = train_test_split_by_label(dataset)

    save_split(path_to_save, dataset)

    return dataset
