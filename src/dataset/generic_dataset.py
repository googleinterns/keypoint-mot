from dataset.dataset_utils import VideosIndexer


class GenericDataset:
    """Superclass for the used datasets"""

    def __init__(self):
        self.video_indexer: VideosIndexer

    def get_input(self, frame_idx):
        return self._pre_process_input(frame_idx)

    def _pre_process_input(self, frame_idx):
        raise NotImplementedError
