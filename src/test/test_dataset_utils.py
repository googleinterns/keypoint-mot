from unittest import TestCase

import numpy as np

from dataset import dataset_utils
from dataset.dataset_utils import VideosIndexer


class TestVideosIndexer(TestCase):
    def setUp(self):
        self.video_list = [["a", "b", "c"], ["d"], ["e", "f"]]
        self.video_indexer = VideosIndexer(self.video_list)

    def test_get_video_OutOfBounds(self):
        self.assertRaisesRegex(ValueError, "Absolute frame index out of bounds", self.video_indexer.get_video_frame, -1)
        self.assertRaisesRegex(ValueError, "Absolute frame index out of bounds", self.video_indexer.get_video_frame, 6)

    def test_get_video_InBounds(self):
        absolute_idx = 0
        for video_idx, video_frames in enumerate(self.video_list):
            for relative_frame_idx, frame_data in enumerate(video_frames):
                self.assertEqual(self.video_indexer.get_video_frame(absolute_idx),
                                 (video_idx, relative_frame_idx, frame_data))
                absolute_idx += 1


class TestDatasetUtilsFunctions(TestCase):
    def test_project_to_image(self):
        point_3d = np.array([[1., 2., 3.]])
        calib = np.array([[1., 0., 2., 0.],
                          [0., 3., 4., 0.],
                          [0., 0., 1., 0.]])

        point_2d = dataset_utils.project_to_image(point_3d, calib)
        np.testing.assert_equal(point_2d, np.array([[7 / 3, 18 / 3]]))

    def test_rot_y2alpha(self):
        self.assertEqual(dataset_utils.rot_y2alpha(np.pi / 4, -400, 1200, 1600), np.pi / 2)

    def test_draw_umich_gaussian(self):
        heatmap = np.zeros((5, 5))
        expected_heatmap = np.zeros((5, 5))
        expected_heatmap[2:, 2:] = dataset_utils.gaussian2D((3, 3), 0.5)
        drawn_heatmap = dataset_utils.draw_umich_gaussian(heatmap=heatmap, center=(3, 3), radius=1)
        np.testing.assert_equal(drawn_heatmap, expected_heatmap)

    def test_gaussian2d(self):
        gaussian_row = dataset_utils.gaussian2D((1, 3), sigma=1)
        gaussian_col = dataset_utils.gaussian2D((3, 1), sigma=1)
        computed_gaussian = dataset_utils.gaussian2D((3, 3), sigma=1)
        np.testing.assert_equal(gaussian_row, [[np.exp(-0.5), 1.0, np.exp(-0.5)]])
        np.testing.assert_equal(gaussian_row.T, gaussian_col)
        np.testing.assert_equal(computed_gaussian, np.dot(gaussian_col, gaussian_row))

    def test_gaussian_radius(self):
        rad = dataset_utils.gaussian_radius((10, 20), 0.5)
        self.assertAlmostEqual(rad, 5.615528, places=6)

    def test_affine_transform(self):
        point = np.array([10., 10.])
        transformed_point = dataset_utils.affine_transform(point, np.eye(2, 3))
        np.testing.assert_equal(point, transformed_point)

    def test_get_affine_transform(self):
        transform = dataset_utils.get_affine_transform(center=np.array([900, 400]), scale=np.array([1920, 1920]), rot=0,
                                                       output_size=(800, 448), shift=(0., 0.), inv=0)
        np.testing.assert_allclose(transform, [[0.41666667, -0.0, 25.0], [0., 0.41666667, 57.33333333]])
