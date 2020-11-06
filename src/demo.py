from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
import sys

import cv2
import numpy as np

from detector import Detector
from model.dla import DLASeg
from model.weights_converter import DLASegConverter
from opts import opts

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']


def demo(opt):
    opt.debug = 1
    heads = {'hm': 10, 'reg': 2, 'wh': 2, 'tracking': 2, 'dep': 1, 'rot': 8, 'dim': 3, 'amodel_offset': 2}
    head_conv = {'hm':  [256], 'reg': [256], 'wh': [256], 'tracking': [256], 'dep': [256], 'rot': [256],
                 'dim': [256], 'amodel_offset': [256]}

    detector = DLASeg(num_layers=34, heads=heads, head_convs=head_conv, opt=opt)
    converter = DLASegConverter('../resources/nuScenes_3Dtracking.pth', detector, batch_size=1, input_height=448,
                                input_width=800)()
    detector = Detector(opt, detector)

    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        is_video = True
        # demo on video stream
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        cam2 = cv2.VideoCapture('/home/vasilup_google_com/code/keypoint-mot/results/obj_det_nuscenes_mini.mp4')
    else:
        is_video = False
        # Demo on images sequences
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

    # Initialize output video
    out = None
    out_name = opt.demo[opt.demo.rfind('/') + 1:].replace('.mp4', '')
    print('out_name', out_name)
    if opt.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter('../results/{}.mp4'.format(
                opt.exp_id + '_' + out_name), fourcc, opt.save_framerate, (
            opt.video_w, opt.video_h))

    if opt.debug < 5:
        detector.pause = False
    cnt = 0
    results = {}

    while True:
        if is_video:
            _, img = cam.read()
            if img is None:
                save_and_exit(opt, out, results, out_name)
        else:
            if cnt < len(image_names):
                img = cv2.imread(image_names[cnt])
            else:
                save_and_exit(opt, out, results, out_name)
        cnt += 1

        # resize the original video for saving video results
        if opt.resize_video:
            img = cv2.resize(img, (opt.video_w, opt.video_h))

        # skip the first X frames of the video
        if cnt < opt.skip_first:
            continue

        cv2.imshow('input', img)

        # track or detect the image.
        ret = detector.run(img)

        # log run time
        time_str = 'frame {} |'.format(cnt)
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)

        # results[cnt] is a list of dicts:
        #  [{'bbox': [x1, y1, x2, y2], 'tracking_id': id, 'category_id': c, ...}]
        results[cnt] = ret['results']

        # save debug image to video
        if opt.save_video:
            _, obj_det = cam2.read()
            assert obj_det is not None
            frame_h, frame_w = ret['generic'].shape[0], ret['generic'].shape[1]
            out_frame = np.zeros((frame_h, frame_w * 3, 3), dtype=np.uint8)
            out_frame[:frame_h, :frame_w, :] = ret['previous']
            out_frame[:frame_h, frame_w:2 * frame_w, :] = ret['generic']
            out_frame[:frame_h, 2 * frame_w:, :] = obj_det
            out.write(out_frame)
            if not is_video:
                cv2.imwrite('../results/demo{}.jpg'.format(cnt), ret['generic'])

        # esc to quit and finish saving video
        if cv2.waitKey(1) == 27:
            save_and_exit(opt, out, results, out_name)
            return
    save_and_exit(opt, out, results)


def save_and_exit(opt, out=None, results=None, out_name=''):
    if opt.save_results and (results is not None):
        save_dir = '../results/{}_results.json'.format(opt.exp_id + '_' + out_name)
        print('saving results to', save_dir)
        json.dump(_to_list(copy.deepcopy(results)),
                  open(save_dir, 'w'))
    if opt.save_video and out is not None:
        z = out.release()
        pass
    sys.exit(0)


def _to_list(results):
    for img_id in results:
        for t in range(len(results[img_id])):
            for k in results[img_id][t]:
                if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
                    results[img_id][t][k] = results[img_id][t][k].tolist()
    return results


if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser('Train the network')

    parser.add_argument('--dataset', choices=['nuscenes'], default='nuscenes')
    parser.add_argument('--mini-dataset', action='store_true', default=False)

    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)

    parser.add_argument('--savedir', default='../resources/tests')

    parser.add_argument('--head_kernel', type=int, default=3, help='')
    parser.add_argument('--prior_bias', type=float, default=-4.6)
    parser.add_argument('--dla_node', default='dcn')
    parser.add_argument('--load_model', default='asdf')
    parser.add_argument('--model_output_list', action='store_true')
    parser.add_argument('--demo', default='')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--resize_video', action='store_true')
    parser.add_argument('--pre_hm', action='store_true')
    parser.add_argument('--track_thresh', type=float, default=0.3)
    parser.add_argument('--test_focal_length', type=int, default=-1)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--skip_first', type=int, default=-1, help='')
    parser.add_argument('--no_pause', action='store_true')
    parser.add_argument('--debugger_theme', default='white',choices=['white', 'black'])
    parser.add_argument('--test_scales', type=str, default='1')
    parser.add_argument('--fix_short', type=int, default=-1)
    parser.add_argument('--keep_res', action='store_true')
    parser.add_argument('--flip_test', action='store_true')
    parser.add_argument('--zero_pre_hm', action='store_true')
    parser.add_argument('--depth_scale', type=float, default=1,help='')
    parser.add_argument('--K', type=int, default=100,help='max number of output objects.')
    parser.add_argument('--zero_tracking', action='store_true')
    parser.add_argument('--public_det', action='store_true')
    parser.add_argument('--hungarian', action='store_true')
    parser.add_argument('--new_thresh', type=float, default=0.3)

    parser.add_argument('--show_track_color', action='store_true')
    parser.add_argument('--only_show_dots', action='store_true')
    parser.add_argument('--tango_color', action='store_true')
    parser.add_argument('--print_iter', type=int, default=0,
                             help='disable progress bar and print to screen.')
    parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    parser.add_argument('--vis_thresh', type=float, default=0.3,
                             help='visualization threshold.')

    parser.add_argument('--eval_val', action='store_true')
    parser.add_argument('--save_imgs', default='', help='')
    parser.add_argument('--save_img_suffix', default='', help='')


    parser.add_argument('--save_framerate', type=int, default=30)

    parser.add_argument('--video_h', type=int, default=512, help='')
    parser.add_argument('--video_w', type=int, default=512, help='')
    parser.add_argument('--transpose_video', action='store_true')

    parser.add_argument('--not_show_bbox', action='store_true')
    parser.add_argument('--not_show_number', action='store_true')
    parser.add_argument('--not_show_txt', action='store_true')
    parser.add_argument('--qualitative', action='store_true')


    parser.add_argument('--show_trace', action='store_true')
    parser.add_argument('--vis_gt_bev', default='', help='')
    parser.add_argument('--max_age', type=int, default=-1)
    args = parser.parse_args()

    # hacky, should all be attributes in argparse
    res = config.TRAIN_RESOLUTION['nuscenes']
    heads = {'hm': 10, 'reg': 2, 'wh': 2, 'tracking': 2, 'dep': 1, 'rot': 8, 'dim': 3, 'amodel_offset': 2}
    dataset_opts = generic_dataset.DatasetOptions(res, config.Resolution(height=res.height // 4, width=res.width // 4),
                                                  heads=heads)
    for key, val in dataset_opts.__dict__.items():
        args.__setattr__(key, val)
    args.pre_img = True  # should be removed
    args.fix_res = not args.keep_res
    args.input_h, args.input_w = res.height, res.width
    args.num_classes = 10
    args.out_thresh = args.track_thresh
    args.new_thresh = max(args.track_thresh, args.new_thresh)
    args.pre_thresh = args.track_thresh
    args.show_track_color = True

    args.save_video = True
    args.exp_id='demo'
    """
    opt = opts().init()
    # opt.save_video = True
    opt.video_h = 450
    opt.video_w = 800 * 3
    opt.exp_id = 'tracking'
    # opt.show_track_color = False
    opt.save_framerate = 2
    demo(opt)
