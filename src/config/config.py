import collections

from config import paths

Resolution = collections.namedtuple('Resolution', ['height', 'width'])

DATASET_ROOT = {'nuscenes': paths.DIR_NUSCENES}
TRAIN_RESOLUTION = {'nuscenes': Resolution(height=448, width=800)}


def _get_dataset_param(dataset, all_params):
    if dataset in all_params:
        return all_params[dataset]

    raise ValueError(f'Unknown dataset: {dataset}')


def get_data_dir(dataset):
    return _get_dataset_param(dataset, DATASET_ROOT)


def get_train_resolution(dataset):
    return _get_dataset_param(dataset, TRAIN_RESOLUTION)
