import bisect
from typing import Any, List, Tuple


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
                             f"Got {absolute_frame_idx}, bounds are (0,{self.video_lengths[-1]}")

        video_idx = bisect.bisect_right(self.video_lengths, absolute_frame_idx)

        relative_frame_idx = absolute_frame_idx
        if video_idx > 0:
            relative_frame_idx -= self.video_lengths[video_idx - 1]

        frame_data = self.video_list[video_idx][relative_frame_idx]

        return video_idx, relative_frame_idx, frame_data
