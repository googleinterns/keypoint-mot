import bisect
import random
from typing import Any, List, Tuple

import cv2
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
    pts_3d_homogeneous = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homogeneous.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d


def rot_y2alpha(rot_y, x, cx, fx):
    """
    Get observation angle of object, ranging [-pi..pi]
    rot_y : Rotation around y-axis in camera coordinates [-pi..pi]
    x : x-axis object center, in pixels
    cx: x-axis optical center of camera, in pixels
    fx: x-axis camera focal length, in pixels
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
        alpha -= 2 * np.pi
    if alpha < -np.pi:
        alpha += 2 * np.pi
    return alpha


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    https://github.com/princeton-vl/CornerNet/issues/110
    Draw the gaussian activations on the heatmap for the given object's center.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.7):
    """
    https://github.com/princeton-vl/CornerNet/issues/110
    Compute the gaussian radius for a given bounding box size.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def _get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [_brightness, _contrast, _saturation]
    random.shuffle(functions)

    gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    _lighting(data_rng, image, 0.1, eig_val, eig_vec)


def _blend(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def _saturation(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    _blend(alpha, image, gs[:, :, None])


def _brightness(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def _contrast(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    _blend(alpha, image, gs_mean)


def _lighting(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)
