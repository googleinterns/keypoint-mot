from config import config
from dataset import nuscenes_dataset

DATASETS = {'nuscenes': nuscenes_dataset.NuscenesDataset}


def get_dataset(dataset_name: str, subset: str, mini_version: bool):
    return DATASETS[dataset_name](subset=subset, dataset_root=config.get_data_dir(dataset_name),
                                  mini_version=mini_version)
