import nuscenes
from nuscenes.utils import splits

from dataset import generic_dataset


class NuscenesDataset(generic_dataset.GenericDataset):
    """
    Handles loading and splitting of Nuscenes dataset
    """
    version = '1.0'
    sensors = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    def __init__(self, subset: str, dataset_root: str, mini_version: bool):
        super(NuscenesDataset, self).__init__()
        self.nusc = self._get_nuscenes_instance(subset, dataset_root, mini_version)
        scene_names = splits.create_splits_scenes()[f'mini_{subset}' if mini_version else subset]
        self.scenes = [scene for scene in self.nusc.scene if scene['name'] in scene_names]
        self.img_ids = [sample_data_token
                        for scene in self.scenes for sample_data_token in self._get_sample_data_list(scene)]

    def _get_sample_data_list(self, scene):
        sample_tokens = {sensor: [] for sensor in self.sensors}
        sample_token = scene['first_sample_token']

        while sample_token != '':
            sample = self.nusc.get('sample', sample_token)
            for sensor in self.sensors:
                sample_tokens[sensor].append(sample['data'][sensor])
            sample_token = sample['next']

        return [token for lst in sample_tokens.values() for token in lst]

    def _get_nuscenes_instance(self, subset, dataset_root, mini: bool):
        if subset in ['train', 'val']:
            suffix = 'trainval'
        else:
            suffix = 'test'

        if mini:
            suffix = 'mini'

        return nuscenes.NuScenes(version=f'v{NuscenesDataset.version}-{suffix}', dataroot=dataset_root)
