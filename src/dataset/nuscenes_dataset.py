import nuscenes
from nuscenes.utils import splits


class NuscenesDataset:
    """
    Handles loading and splitting of Nuscenes dataset
    """
    version = '1.0'

    def __init__(self, subset: str, dataset_root: str, mini_version: bool):
        self.nusc = self.__get_nuscenes_instance(subset, dataset_root, mini_version)
        scene_names = splits.create_splits_scenes()[f'mini_{subset}' if mini_version else subset]
        self.scenes = [scene for scene in self.nusc.scene if scene['name'] in scene_names]

    def __get_nuscenes_instance(self, subset, dataset_root, mini: bool):
        if subset in ['train', 'val']:
            suffix = 'trainval'
        else:
            suffix = 'test'

        if mini:
            suffix = 'mini'

        return nuscenes.NuScenes(version=f'v{NuscenesDataset.version}-{suffix}', dataroot=dataset_root)
