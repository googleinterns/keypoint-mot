import bisect
from typing import Any, List, Tuple

import numpy as np


class VideosIndexer:
    """
    Handles absolute indexing for a list of videos.

    Attributes:
        video_list (List([List[Any]])) :matrix with the frame ids for each video
        video_lengths (List(int)): partial sum array with lenghts of the videos
    """
    video_list: List[List[Any]]
    video_lengths: List[int]

    def __init__(self, video_list: List[List[Any]]):
        self.video_lengths = [len(video) for video in video_list]
        for i in range(1, len(self.video_lengths)):
            self.video_lengths[i] += self.video_lengths[i - 1]
        self.video_list = video_list

    def get_video_frame(self, absolute_frame_idx: int) -> Tuple[int, int, Any]:
        """
        Considering all the videos are concatenated in a single sequence, given the absolute frame index it computes
        which is the original video and what is the frame's index inside the original video.

        :param absolute_frame_idx: the frame index in the concatenated videos sequence
        :return: video_index:int, relative_frame_index:int, frame_data
        """
        if absolute_frame_idx < 0 or absolute_frame_idx >= self.video_lengths[-1]:
            raise ValueError(f"Absolute frame index out of bounds."
                             f"Got {absolute_frame_idx}, bounds are (0,{self.video_lengths[-1]})")

        video_idx = bisect.bisect_right(self.video_lengths, absolute_frame_idx)

        relative_frame_idx = absolute_frame_idx
        if video_idx > 0:
            relative_frame_idx -= self.video_lengths[video_idx - 1]

        frame_data = self.video_list[video_idx][relative_frame_idx]

        return video_idx, relative_frame_idx, frame_data


def project_to_image(pts_3d, P):
    # pts_3d: n x 3
    # P: 3 x 4
    # return: n x 2
    pts_3d_homogen = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homogen.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d


def _rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
        alpha -= 2 * np.pi
    if alpha < -np.pi:
        alpha += 2 * np.pi
    return alpha
