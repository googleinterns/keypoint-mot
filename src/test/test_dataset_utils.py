from unittest import TestCase

from dataset.dataset_utils import VideosIndexer


class TestVideosIndexer(TestCase):
    def setUp(self):
        self.video_list = [["a", "b", "c"], ["d"], ["e", "f"]]
        self.video_indexer = VideosIndexer(self.video_list)

    def test_get_video_OutOfBounds(self):
        self.assertRaises(ValueError, self.video_indexer.get_video_frame, -1)
        self.assertRaises(ValueError, self.video_indexer.get_video_frame, 6)

    def test_get_video_InBounds(self):
        absolute_idx = 0
        for video_idx, video_frames in enumerate(self.video_list):
            for relative_frame_idx, frame_data in enumerate(video_frames):
                self.assertEqual(self.video_indexer.get_video_frame(absolute_idx),
                                 (video_idx, relative_frame_idx, frame_data))
                absolute_idx += 1
