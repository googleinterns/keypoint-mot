import copy
import os

import cv2
import numpy as np
import nuscenes
import nuscenes.eval.detection.utils
from nuscenes.utils import geometry_utils
from nuscenes.utils import kitti
import nuscenes.utils.splits
from pyquaternion import Quaternion

from dataset import dataset_utils, generic_dataset


def _bbox_inside(box1, box2):
    return box1[0] > box2[0] and box1[0] + box1[2] < box2[0] + box2[2] and \
           box1[1] > box2[1] and box1[1] + box1[3] < box2[1] + box2[3]


class NuscenesDataset(generic_dataset.GenericDataset):
    """
    Handles loading and splitting of Nuscenes dataset
    """

    version = '1.0'
    sensors = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    class_name = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
                  'traffic_cone', 'barrier']
    num_classes = len(class_name)
    cat_ids = {v: i + 1 for i, v in enumerate(class_name)}

    max_objs = 128

    nuscenes_att_range = {0: [0, 1], 1: [0, 1], 2: [2, 3, 4], 3: [2, 3, 4], 4: [2, 3, 4], 5: [5, 6, 7], 6: [5, 6, 7],
                          7: [5, 6, 7]}

    def __init__(self, subset: str, dataset_root: str, opts: generic_dataset.DatasetOptions, mini_version: bool):
        super().__init__(subset, dataset_root, opts)
        self.nusc = self._get_nuscenes_instance(subset, dataset_root, mini_version)
        scene_names = nuscenes.utils.splits.create_splits_scenes()[f'mini_{subset}' if mini_version else subset]
        self.scenes = [scene for scene in self.nusc.scene if scene['name'] in scene_names]
        self.videos = self._get_videos(self.scenes, self.sensors)
        self.video_indexer = dataset_utils.VideosIndexer(self.videos)

    def _pre_process_input(self, sample_data_token):
        frame_sample_data = self.nusc.get('sample_data', sample_data_token)
        cal_sensor_data, pose_data, trans_matrix = self._compute_calibration_data(frame_sample_data)
        _, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_data_token,
                                                               box_vis_level=geometry_utils.BoxVisibility.ANY)
        calibration = self._get_calibration_matrix(camera_intrinsic)

        image_info = {'file_name':         frame_sample_data['filename'],
                      'calib':             calibration.tolist(),
                      'trans_matrix':      trans_matrix.tolist(),
                      'width':             frame_sample_data['width'],
                      'height':            frame_sample_data['height'],
                      'pose_record_trans': pose_data['translation'],
                      'pose_record_rot':   pose_data['rotation'],
                      'cs_record_trans':   cal_sensor_data['translation'],
                      'cs_record_rot':     cal_sensor_data['rotation']}

        annotations = self._get_annotations(boxes, calibration, camera_intrinsic, trans_matrix)

        frame_path = os.path.join(self.dataset_root, frame_sample_data['filename'])
        read_frame = cv2.imread(frame_path)

        return read_frame, annotations, image_info

    def _compute_calibration_data(self, frame_sample_data):
        cal_sensor_data = self.nusc.get('calibrated_sensor', frame_sample_data['calibrated_sensor_token'])
        pose_data = self.nusc.get('ego_pose', frame_sample_data['ego_pose_token'])
        car_to_global = geometry_utils.transform_matrix(pose_data['translation'], Quaternion(pose_data['rotation']),
                                                        inverse=False)
        sensor_to_car = geometry_utils.transform_matrix(cal_sensor_data['translation'],
                                                        Quaternion(cal_sensor_data['rotation']),
                                                        inverse=False)
        trans_matrix = np.dot(car_to_global, sensor_to_car)
        return cal_sensor_data, pose_data, trans_matrix

    def _get_calibration_matrix(self, camera_intrinsic):
        calibration = np.eye(4, dtype=np.float32)
        calibration[:3, :3] = camera_intrinsic
        calibration = calibration[:3]
        return calibration

    def _get_annotations(self, boxes, calibration, camera_intrinsic, trans_matrix):
        annotations = []
        for box in boxes:
            det_name = nuscenes.eval.detection.utils.category_to_detection_name(box.name)
            if det_name is None:
                continue

            ann = self._get_annotation(box, calibration, camera_intrinsic, det_name, trans_matrix)
            annotations.append(ann)

        annotations = self._get_visible_annotations(annotations)
        return annotations

    def _get_annotation(self, box, calibration, camera_intrinsic, det_name, trans_matrix):
        v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
        yaw = -np.arctan2(v[2], v[0])
        box.translate(np.array([0, box.wlh[2] / 2, 0]))
        category_id = self.cat_ids[det_name]
        amodel_center = dataset_utils.project_to_image(
                np.array([box.center[0], box.center[1] - box.wlh[2] / 2, box.center[2]], np.float32).reshape(1, 3),
                calibration)[0].tolist()
        sample_ann = self.nusc.get('sample_annotation', box.token)
        instance_token = sample_ann['instance_token']
        vel = self.nusc.box_velocity(box.token)
        vel = np.dot(np.linalg.inv(trans_matrix), np.array([vel[0], vel[1], vel[2], 0], np.float32)).tolist()
        ann = {
            'category_id':   category_id,
            'dim':           [box.wlh[2], box.wlh[0], box.wlh[1]],
            'location':      [box.center[0], box.center[1], box.center[2]],
            'depth':         box.center[2],
            'occluded':      0,
            'truncated':     0,
            'rotation_y':    yaw,
            'amodel_center': amodel_center,
            'iscrowd':       0,
            'track_id':      instance_token,
            'velocity':      vel
        }
        bbox = kitti.KittiDB.project_kitti_box_to_image(copy.deepcopy(box), camera_intrinsic, imsize=(1600, 900))
        alpha = dataset_utils.rot_y2alpha(yaw, (bbox[0] + bbox[2]) / 2, camera_intrinsic[0, 2], camera_intrinsic[0, 0])
        ann['bbox'] = [bbox[0], bbox[1], bbox[2] - bbox[0],
                       bbox[3] - bbox[1]]
        ann['area'] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        ann['alpha'] = alpha
        return ann

    def _get_visible_annotations(self, anns):
        visible_anns = []
        for i in range(len(anns)):
            vis = True
            for j in range(len(anns)):
                if anns[i]['depth'] - min(anns[i]['dim']) / 2 > anns[j]['depth'] + max(
                        anns[j]['dim']) / 2 and _bbox_inside(anns[i]['bbox'], anns[j]['bbox']):
                    vis = False
                    break
            if vis:
                visible_anns.append(anns[i])
            else:
                pass
        return visible_anns

    def _add_instance(self, ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output,
                      aug_s, calib, pre_cts=None, track_ids=None):
        super()._add_instance(ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output,
                              aug_s, calib, pre_cts, track_ids)

        if 'nuscenes_att' in self.opt.heads:
            if ('attributes' in ann) and ann['attributes'] > 0:
                att = int(ann['attributes'] - 1)
                ret['nuscenes_att'][k][att] = 1
                ret['nuscenes_att_mask'][k][self.nuscenes_att_range[att]] = 1
            gt_det['nuscenes_att'].append(ret['nuscenes_att'][k])

    def _get_videos(self, scenes, sensors):
        videos = []
        for scene in scenes:
            sensor_videos = self._get_scene_keyframe_lists(scene, sensors)
            for video in sensor_videos.values():
                videos.append(video)
        return videos

    def _get_scene_keyframe_lists(self, scene, sensors):
        sample_tokens = {sensor: [] for sensor in sensors}
        sample_token = scene['first_sample_token']

        while sample_token != '':
            sample = self.nusc.get('sample', sample_token)
            for sensor in self.sensors:
                sample_tokens[sensor].append(sample['data'][sensor])
            sample_token = sample['next']

        return sample_tokens

    def _get_nuscenes_instance(self, subset, dataset_root, mini: bool):
        if subset in ['train', 'val']:
            suffix = 'trainval'
        else:
            suffix = 'test'

        if mini:
            suffix = 'mini'

        return nuscenes.NuScenes(version=f'v{NuscenesDataset.version}-{suffix}', dataroot=dataset_root)
