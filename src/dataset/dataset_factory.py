from config import config
from dataset import generic_dataset, nuscenes_dataset

DATASETS = {'nuscenes': nuscenes_dataset.NuscenesDataset}


def get_dataset(dataset_name: str, subset: str, opts: generic_dataset.DatasetOptions, mini_version: bool):
    return DATASETS[dataset_name](subset=subset, dataset_root=config.get_data_dir(dataset_name), opts=opts,
                                  mini_version=mini_version)
