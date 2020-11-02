import copy
from dataclasses import dataclass, field
import math
from typing import Any, List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from config import config
from dataset import dataset_utils


@dataclass
class DatasetOptions:
    input_res: config.Resolution
    output_res: config.Resolution

    heads: dict = field(default_factory=lambda: {})

    tracking: bool = True  # True if the task is tracking
    pre_hm: bool = True  # True if previous heatmap should be computed

    not_max_crop: bool = False
    flip: float = 0.5  # probability to use flip augmentation

    same_aug_pre: bool = False  # True if the previous image should use the same augmentation
    dense_reg: int = 1

    scale: float = 0.05
    shift: float = 0.01
    aug_rot: float = 0.0
    rotate: float = 0.0
    not_rand_crop: bool = False

    velocity: bool = False
    no_color_aug: bool = False
    down_ratio: int = 4

    hm_disturb: float = 0.05
    lost_disturb: float = 0.4
    fp_disturb: float = 0.1

    max_frame_dist: int = 3  # skip at most max_frame_dist-1 frames when loading previous image


IGNORED_CLASS_ID = -99


class GenericDataset:
    """Superclass for the used datasets"""
    video_indexer: dataset_utils.VideosIndexer
    dataset_root: str
    subset: str
    videos: List[List[Any]]
    max_objs: int
    num_classes: int

    ignore_val = 1
    num_joints = 17
    flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

    # https://github.com/xingyizhou/CenterTrack/blob/master/src/lib/dataset/generic_dataset.py#L39
    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape((1, 1, 3))
    std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape((1, 1, 3))

    rest_focal_length = 1200

    _eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
    _eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)

    return_dtypes = {'image':         tf.float32, 'pre_img': tf.float32, 'pre_hm': tf.float32, 'hm': tf.float32,
                     'ind':           tf.int64, 'cat': tf.int64, 'mask': tf.float32, 'reg': tf.float32,
                     'reg_mask':      tf.float32, 'wh': tf.float32, 'wh_mask': tf.float32, 'tracking': tf.float32,
                     'tracking_mask': tf.float32, 'dep': tf.float32, 'dep_mask': tf.float32, 'dim': tf.float32,
                     'dim_mask':      tf.float32, 'amodel_offset': tf.float32, 'amodel_offset_mask': tf.float32,
                     'rotbin':        tf.int64, 'rotres': tf.float32, 'rot_mask': tf.float32}
    return_dtypes_key_list = list(return_dtypes.keys())

    def __init__(self, subset: str, dataset_root: str, opt: DatasetOptions):
        self.subset = subset
        self.dataset_root = dataset_root
        self._data_rng = np.random.RandomState(123)
        self.opt = opt

    def get_input_generator(self, shuffle=False, dataset_len: int = None):
        """Usage: Dataset.from_generator(dataset_instance.get_input_generator(args),
                                         output_types=dataset_instance.return_dtypes)"""
        if dataset_len is None:
            dataset_len = self.video_indexer.video_lengths[-1]
        dataset = tf.data.Dataset.range(dataset_len)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=dataset_len, reshuffle_each_iteration=True)

        def gen_func():
            for i in dataset:
                yield self.get_input(i)

        return gen_func

    def get_input_py_func(self, frame_idx):
        """Usage: Dataset.range(dataset_len).map(map_func=dataset_instance.get_input_py_func,
                                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)"""
        ret = tf.py_function(func=lambda i: list(self.get_input(i).values()), inp=[frame_idx],
                             Tout=list(self.return_dtypes.values()))
        return {key: ret[i] for i, key in enumerate(self.return_dtypes_key_list)}

    def get_input(self, frame_idx):
        frame_idx = frame_idx.numpy()
        video_idx, frame_idx, frame_data_token = self.video_indexer.get_video_frame(frame_idx)
        img, anns, img_info = self._pre_process_input(frame_data_token)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.not_max_crop:
            s = np.array([img.shape[1], img.shape[0]], np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
        aug_s, rot, flipped = 1, 0, 0

        if self.subset == 'train':
            c, aug_s, rot = self._get_aug_param(c, s, width, height)
            s = s * aug_s
            if np.random.random() < self.opt.flip:
                flipped = 1
                img = img[:, ::-1, :]
                anns = self._flip_anns(anns, width)

        trans_input = dataset_utils.get_affine_transform(c, s, rot,
                                                         [self.opt.input_res.width, self.opt.input_res.height])
        trans_output = dataset_utils.get_affine_transform(c, s, rot,
                                                          [self.opt.output_res.width, self.opt.output_res.height])
        inp = self._get_input(img, trans_input)
        ret = {'image': inp}
        gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}

        pre_cts, track_ids = None, None
        if self.opt.tracking:
            pre_image, pre_anns, frame_dist = self._load_pre_data(video_idx, frame_idx)
            if flipped:
                pre_image = pre_image[:, ::-1, :].copy()
                pre_anns = self._flip_anns(pre_anns, width)
            if self.opt.same_aug_pre and frame_dist != 0:
                trans_input_pre = trans_input
                trans_output_pre = trans_output
            else:
                c_pre, aug_s_pre, _ = self._get_aug_param(c, s, width, height, disturb=True)
                s_pre = s * aug_s_pre
                trans_input_pre = dataset_utils.get_affine_transform(c_pre, s_pre, rot, [self.opt.input_res.width,
                                                                                         self.opt.input_res.height])
                trans_output_pre = dataset_utils.get_affine_transform(c_pre, s_pre, rot, [self.opt.output_res.width,
                                                                                          self.opt.output_res.height])
            pre_img = self._get_input(pre_image, trans_input_pre)
            pre_hm, pre_cts, track_ids = self._get_pre_dets(pre_anns, trans_input_pre, trans_output_pre)
            ret['pre_img'] = pre_img
            if self.opt.pre_hm:
                ret['pre_hm'] = pre_hm

        self._init_ret(ret, gt_det)
        calib = self._get_calib(img_info, width, height)

        num_objs = min(len(anns), self.max_objs)
        for k in range(num_objs):
            ann = anns[k]
            cls_id = int(ann['category_id'])
            if cls_id > self.num_classes or cls_id <= IGNORED_CLASS_ID:
                continue
            bbox, bbox_amodal = self._get_bbox_output(ann['bbox'], trans_output, height, width)
            if cls_id <= 0 or ('iscrowd' in ann and ann['iscrowd'] > 0):
                self._mask_ignore_or_crowd(ret, cls_id, bbox)
                continue
            self._add_instance(
                    ret, gt_det, k, cls_id, bbox, bbox_amodal, ann,
                    trans_output, aug_s,
                    calib, pre_cts, track_ids)

        return ret

    def _get_specific_nuscenes_frame_debug(self, idx):
        if idx == 0:
            filename = 'samples/CAM_BACK/n008-2018-08-28-13-40-50-0400__CAM_BACK__1535478715187558.jpg'
        else:
            filename = 'samples/CAM_BACK/n008-2018-08-21-11-53-44-0400__CAM_BACK__1534867443937696.jpg'
        frame_abs = 0
        for vid in self.videos:
            for frame in vid:
                sample_data = self.nusc.get('sample_data', frame)
                if sample_data['filename'] == filename:
                    return frame_abs

                frame_abs += 1

        raise FileNotFoundError(f'Could not find file {filename}')

    def _add_instance(self, ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output,
                      aug_s, calib, pre_cts=None, track_ids=None):
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h <= 0 or w <= 0:
            return
        radius = dataset_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        ret['cat'][k] = cls_id - 1
        ret['mask'][k] = 1
        if 'wh' in ret:
            ret['wh'][k] = 1. * w, 1. * h
            ret['wh_mask'][k] = 1
        ret['ind'][k] = ct_int[1] * self.opt.output_res.width + ct_int[0]
        ret['reg'][k] = ct - ct_int
        ret['reg_mask'][k] = 1
        dataset_utils.draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius)

        gt_det['bboxes'].append(
                np.array([ct[0] - w / 2, ct[1] - h / 2,
                          ct[0] + w / 2, ct[1] + h / 2], dtype=np.float32))
        gt_det['scores'].append(1)
        gt_det['clses'].append(cls_id - 1)
        gt_det['cts'].append(ct)

        if 'tracking' in self.opt.heads:
            if ann['track_id'] in track_ids:
                pre_ct = pre_cts[track_ids.index(ann['track_id'])]
                ret['tracking_mask'][k] = 1
                ret['tracking'][k] = pre_ct - ct_int
                gt_det['tracking'].append(ret['tracking'][k])
            else:
                gt_det['tracking'].append(np.zeros(2, np.float32))

        if 'ltrb' in self.opt.heads:
            ret['ltrb'][k] = bbox[0] - ct_int[0], bbox[1] - ct_int[1], \
                             bbox[2] - ct_int[0], bbox[3] - ct_int[1]
            ret['ltrb_mask'][k] = 1

        if 'ltrb_amodal' in self.opt.heads:
            ret['ltrb_amodal'][k] = \
                bbox_amodal[0] - ct_int[0], bbox_amodal[1] - ct_int[1], \
                bbox_amodal[2] - ct_int[0], bbox_amodal[3] - ct_int[1]
            ret['ltrb_amodal_mask'][k] = 1
            gt_det['ltrb_amodal'].append(bbox_amodal)

        if 'velocity' in self.opt.heads:
            if ('velocity' in ann) and min(ann['velocity']) > -1000:
                ret['velocity'][k] = np.array(ann['velocity'], np.float32)[:3]
                ret['velocity_mask'][k] = 1
            gt_det['velocity'].append(ret['velocity'][k])

        if 'hps' in self.opt.heads:
            self._add_hps(ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w)

        if 'rot' in self.opt.heads:
            self._add_rot(ret, ann, k, gt_det)

        if 'dep' in self.opt.heads:
            if 'depth' in ann:
                ret['dep_mask'][k] = 1
                ret['dep'][k] = ann['depth'] * aug_s
                gt_det['dep'].append(ret['dep'][k])
            else:
                gt_det['dep'].append(2)

        if 'dim' in self.opt.heads:
            if 'dim' in ann:
                ret['dim_mask'][k] = 1
                ret['dim'][k] = ann['dim']
                gt_det['dim'].append(ret['dim'][k])
            else:
                gt_det['dim'].append([1, 1, 1])

        if 'amodel_offset' in self.opt.heads:
            if 'amodel_center' in ann:
                amodel_center = dataset_utils.affine_transform(ann['amodel_center'], trans_output)
                ret['amodel_offset_mask'][k] = 1
                ret['amodel_offset'][k] = amodel_center - ct_int
                gt_det['amodel_offset'].append(ret['amodel_offset'][k])
            else:
                gt_det['amodel_offset'].append([0, 0])

    def _alpha_to_8(self, alpha):
        ret = [0, 0, 0, 1, 0, 0, 0, 1]
        if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
            r = alpha - (-0.5 * np.pi)
            ret[1] = 1
            ret[2], ret[3] = np.sin(r), np.cos(r)
        if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
            r = alpha - (0.5 * np.pi)
            ret[5] = 1
            ret[6], ret[7] = np.sin(r), np.cos(r)
        return ret

    def _add_rot(self, ret, ann, k, gt_det):
        if 'alpha' in ann:
            ret['rot_mask'][k] = 1
            alpha = ann['alpha']
            if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
                ret['rotbin'][k, 0] = 1
                ret['rotres'][k, 0] = alpha - (-0.5 * np.pi)
            if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
                ret['rotbin'][k, 1] = 1
                ret['rotres'][k, 1] = alpha - (0.5 * np.pi)
            gt_det['rot'].append(self._alpha_to_8(ann['alpha']))
        else:
            gt_det['rot'].append(self._alpha_to_8(0))

    def _add_hps(self, ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w):
        num_joints = self.num_joints
        pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3) \
            if 'keypoints' in ann else np.zeros((self.num_joints, 3), np.float32)

        hp_radius = dataset_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        hp_radius = max(0, int(hp_radius))

        for j in range(num_joints):
            pts[j, :2] = dataset_utils.affine_transform(pts[j, :2], trans_output)
            if pts[j, 2] > 0:
                if pts[j, 0] >= 0 and pts[j, 0] < self.opt.output_res.width and pts[j, 1] >= 0 and \
                        pts[j, 1] < self.opt.output_res.height:
                    ret['hps'][k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                    ret['hps_mask'][k, j * 2: j * 2 + 2] = 1
                    pt_int = pts[j, :2].astype(np.int32)
                    ret['hp_offset'][k * num_joints + j] = pts[j, :2] - pt_int
                    ret['hp_ind'][k * num_joints + j] = pt_int[1] * self.opt.output_res.width + pt_int[0]
                    ret['hp_offset_mask'][k * num_joints + j] = 1
                    ret['hm_hp_mask'][k * num_joints + j] = 1
                    ret['joint'][k * num_joints + j] = j
                    dataset_utils.draw_umich_gaussian(ret['hm_hp'][j], pt_int, hp_radius)
                    if pts[j, 2] == 1:
                        ret['hm_hp'][j, pt_int[1], pt_int[0]] = self.ignore_val
                        ret['hp_offset_mask'][k * num_joints + j] = 0
                        ret['hm_hp_mask'][k * num_joints + j] = 0
                else:
                    pts[j, :2] *= 0
            else:
                pts[j, :2] *= 0
                self._ignore_region(ret['hm_hp'][j, int(bbox[1]): int(bbox[3]) + 1, int(bbox[0]): int(bbox[2]) + 1])
        gt_det['hps'].append(pts[:, :2].reshape(num_joints * 2))

    def _ignore_region(self, region, ignore_val=1):
        np.maximum(region, ignore_val, out=region)

    def _mask_ignore_or_crowd(self, ret, cls_id, bbox):
        # mask out crowd region, only rectangular mask is supported
        if cls_id == 0:  # ignore all classes
            self._ignore_region(ret['hm'][:, int(bbox[1]): int(bbox[3]) + 1, int(bbox[0]): int(bbox[2]) + 1])
        else:
            # mask out one specific class
            self._ignore_region(ret['hm'][abs(cls_id) - 1,
                                int(bbox[1]): int(bbox[3]) + 1,
                                int(bbox[0]): int(bbox[2]) + 1])
        if ('hm_hp' in ret) and cls_id <= 1:
            self._ignore_region(ret['hm_hp'][:, int(bbox[1]): int(bbox[3]) + 1, int(bbox[0]): int(bbox[2]) + 1])

    def _get_bbox_output(self, bbox, trans_output, height, width):
        bbox = self._coco_box_to_bbox(bbox).copy()

        rect = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]],
                         [bbox[2], bbox[3]], [bbox[2], bbox[1]]],
                        dtype=np.float32)
        for t in range(4):
            rect[t] = dataset_utils.affine_transform(rect[t], trans_output)
        bbox[:2] = rect[:, 0].min(), rect[:, 1].min()
        bbox[2:] = rect[:, 0].max(), rect[:, 1].max()

        bbox_amodal = copy.deepcopy(bbox)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_res.width - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_res.height - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        return bbox, bbox_amodal

    def _get_calib(self, img_info, width, height):
        if 'calib' in img_info:
            calib = np.array(img_info['calib'], dtype=np.float32)
        else:
            calib = np.array([[self.rest_focal_length, 0, width / 2, 0],
                              [0, self.rest_focal_length, height / 2, 0],
                              [0, 0, 1, 0]])
        return calib

    def _init_ret(self, ret, gt_det):
        max_objs = self.max_objs * self.opt.dense_reg
        ret['hm'] = np.zeros((self.num_classes, self.opt.output_res.height, self.opt.output_res.width), np.float32)
        ret['ind'] = np.zeros((max_objs), dtype=np.int64)
        ret['cat'] = np.zeros((max_objs), dtype=np.int64)
        ret['mask'] = np.zeros((max_objs), dtype=np.float32)

        regression_head_dims = {
            'reg':          2, 'wh': 2, 'tracking': 2, 'ltrb': 4,
            'ltrb_amodal':  4,
            'nuscenes_att': 8, 'velocity': 3, 'hps': self.num_joints * 2,
            'dep':          1, 'dim': 3, 'amodel_offset': 2}

        for head in regression_head_dims:
            if head in self.opt.heads:
                ret[head] = np.zeros((max_objs, regression_head_dims[head]), dtype=np.float32)
                ret[head + '_mask'] = np.zeros((max_objs, regression_head_dims[head]), dtype=np.float32)
                gt_det[head] = []

        if 'hm_hp' in self.opt.heads:
            num_joints = self.num_joints
            ret['hm_hp'] = np.zeros((num_joints, self.opt.output_res.height, self.opt.output_res.width),
                                    dtype=np.float32)
            ret['hm_hp_mask'] = np.zeros((max_objs * num_joints), dtype=np.float32)
            ret['hp_offset'] = np.zeros((max_objs * num_joints, 2), dtype=np.float32)
            ret['hp_ind'] = np.zeros((max_objs * num_joints), dtype=np.int64)
            ret['hp_offset_mask'] = np.zeros((max_objs * num_joints, 2), dtype=np.float32)
            ret['joint'] = np.zeros((max_objs * num_joints), dtype=np.int64)

        if 'rot' in self.opt.heads:
            ret['rotbin'] = np.zeros((max_objs, 2), dtype=np.int64)
            ret['rotres'] = np.zeros((max_objs, 2), dtype=np.float32)
            ret['rot_mask'] = np.zeros((max_objs), dtype=np.float32)
            gt_det.update({'rot': []})

    def _get_aug_param(self, c, s, width, height, disturb=False):
        if (not self.opt.not_rand_crop) and not disturb:
            aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(128, width)
            h_border = self._get_border(128, height)
            c[0] = np.random.randint(low=w_border, high=width - w_border)
            c[1] = np.random.randint(low=h_border, high=height - h_border)
        else:
            sf = self.opt.scale
            cf = self.opt.shift
            if type(s) == float:
                s = [s, s]
            c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            aug_s = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        if np.random.random() < self.opt.aug_rot:
            rf = self.opt.rotate
            rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        else:
            rot = 0

        return c, aug_s, rot

    def _flip_anns(self, anns, width):
        for k in range(len(anns)):
            bbox = anns[k]['bbox']
            anns[k]['bbox'] = [
                width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]

            if 'hps' in self.opt.heads and 'keypoints' in anns[k]:
                keypoints = np.array(anns[k]['keypoints'], dtype=np.float32).reshape(self.num_joints, 3)
                keypoints[:, 0] = width - keypoints[:, 0] - 1
                for e in self.flip_idx:
                    keypoints[e[0]], keypoints[e[1]] = \
                        keypoints[e[1]].copy(), keypoints[e[0]].copy()
                anns[k]['keypoints'] = keypoints.reshape(-1).tolist()

            if 'rot' in self.opt.heads and 'alpha' in anns[k]:
                anns[k]['alpha'] = np.pi - anns[k]['alpha'] if anns[k]['alpha'] > 0 else - np.pi - anns[k]['alpha']

            if 'amodel_offset' in self.opt.heads and 'amodel_center' in anns[k]:
                anns[k]['amodel_center'][0] = width - anns[k]['amodel_center'][0] - 1

            if self.opt.velocity and 'velocity' in anns[k]:
                anns[k]['velocity'] = [-10000, -10000, -10000]

        return anns

    def _get_input(self, img, trans_input):
        inp = cv2.warpAffine(img, trans_input, (self.opt.input_res.width, self.opt.input_res.height),
                             flags=cv2.INTER_LINEAR)

        inp = (inp.astype(np.float32) / 255.)
        if self.subset == 'train' and not self.opt.no_color_aug:
            dataset_utils.color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def _get_pre_dets(self, anns, trans_input, trans_output):
        hm_h, hm_w = self.opt.input_res.height, self.opt.input_res.width
        down_ratio = self.opt.down_ratio
        trans = trans_input
        reutrn_hm = self.opt.pre_hm
        pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if reutrn_hm else None
        pre_cts, track_ids = [], []
        for ann in anns:
            cls_id = ann['category_id']
            if cls_id > self.num_classes or cls_id <= IGNORED_CLASS_ID or ('iscrowd' in ann and ann['iscrowd'] > 0):
                continue
            bbox = self._coco_box_to_bbox(ann['bbox'])
            bbox[:2] = dataset_utils.affine_transform(bbox[:2], trans)
            bbox[2:] = dataset_utils.affine_transform(bbox[2:], trans)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            max_rad = 1
            if (h > 0 and w > 0):
                radius = dataset_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                max_rad = max(max_rad, radius)
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct0 = ct.copy()
                conf = 1

                ct[0] = ct[0] + np.random.randn() * self.opt.hm_disturb * w
                ct[1] = ct[1] + np.random.randn() * self.opt.hm_disturb * h
                conf = 1 if np.random.random() > self.opt.lost_disturb else 0

                ct_int = ct.astype(np.int32)
                if conf == 0:
                    pre_cts.append(ct / down_ratio)
                else:
                    pre_cts.append(ct0 / down_ratio)

                track_ids.append(ann['track_id'] if 'track_id' in ann else -1)
                if reutrn_hm:
                    dataset_utils.draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)

                if np.random.random() < self.opt.fp_disturb and reutrn_hm:
                    ct2 = ct0.copy()
                    # Hard code heatmap disturb ratio, haven't tried other
                    # numbers.
                    ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
                    ct2[1] = ct2[1] + np.random.randn() * 0.05 * h
                    ct2_int = ct2.astype(np.int32)
                    dataset_utils.draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf)

        return pre_hm, pre_cts, track_ids

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
        return bbox

    def _load_pre_data(self, video_idx: int, frame_idx: int):
        video_frames = self.videos[video_idx]
        if 'train' in self.subset:
            lo = max(0, frame_idx - self.opt.max_frame_dist + 1)
            hi = min(len(video_frames) - 1, frame_idx + self.opt.max_frame_dist - 1)
            prev_frame_idx = np.random.randint(lo, hi + 1)
        else:
            prev_frame_idx = max(0, frame_idx - 1)

        frame_dist = abs(frame_idx - prev_frame_idx)
        prev_img, prev_anns, _ = self._pre_process_input(video_frames[prev_frame_idx])
        return prev_img, prev_anns, frame_dist

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _pre_process_input(self, frame_idx) -> Tuple[Any, Any, Any]:
        raise NotImplementedError
