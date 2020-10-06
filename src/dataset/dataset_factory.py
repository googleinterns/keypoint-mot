from config.config import get_data_dir
from dataset.nuscenes_dataset import NuscenesDataset

DATASETS = {'nuscenes': NuscenesDataset}


def get_dataset(dataset_name, subset, mini_version: bool):
    return DATASETS[dataset_name](subset=subset, dataset_root=get_data_dir(dataset_name), mini_version=mini_version)
