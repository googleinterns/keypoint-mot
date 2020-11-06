import os
import shutil
import time

import cv2
import numpy as np
import tensorflow as tf

from config import config
from dataset import dataset_factory, generic_dataset
from debugger import Debugger
from decode import generic_decode
from model.dla import DLASeg
from model.weights_converter import DLASegConverter
from opts import opts
from post_process import generic_post_process

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def get_data_loaders(args):
    dataset_train = dataset_factory.get_dataset(args.dataset, 'train', args, args.mini_dataset)
    dataset_val = dataset_factory.get_dataset(args.dataset, 'val', args, args.mini_dataset)
    print(f'Dataset size train: {len(dataset_train)}, val: {len(dataset_val)}')

    loader_train = tf.data.Dataset.from_generator(dataset_train.get_input_generator(shuffle=True),
                                                  output_types=dataset_train.return_dtypes).batch(args.batch_size)
    loader_val = tf.data.Dataset.from_generator(dataset_val.get_input_generator(shuffle=False),
                                                output_types=dataset_train.return_dtypes).batch(args.batch_size)

    loader_train.dataset = dataset_train
    loader_val.dataset = dataset_val
    return loader_train, loader_val


def trainer(args):
    train(args)


def train(args):
    loader_train, loader_val = get_data_loaders(args)

    start_epoch = 1
    heads = {'hm': 10, 'reg': 2, 'wh': 2, 'tracking': 2, 'dep': 1, 'rot': 8, 'dim': 3, 'amodel_offset': 2}
    head_conv = {'hm':  [256], 'reg': [256], 'wh': [256], 'tracking': [256], 'dep': [256], 'rot': [256],
                 'dim': [256], 'amodel_offset': [256]}

    # model = BaseModel(heads=heads, head_convs = head_conv, opt=args, num_stacks=1, last_channel=64)
    model = DLASeg(num_layers=34, heads=heads, head_convs=head_conv, opt=args)
    # my_code(model, batch_size=args.batch_size)
    DLASegConverter('../resources/nuScenes_3Dtracking.pth', model, args.batch_size, 448, 800)
    for epoch in range(start_epoch, args.num_epochs + 1):
        time_start_epoch = time.time()
        average_epoch_loss_train = do_epoch(loader_train, model, args)

        # average_epoch_loss_val = do_epoch(loader_val, model)
        time_epoch = time.time() - time_start_epoch
        print(f'TRAIN + VAL duration: {time_epoch}')
        break


class Dummy:
    pass


def do_epoch(loader: tf.data.Dataset, model, opt):
    debugger = Debugger(opt, loader.dataset)
    self = Dummy
    self.opt = opt
    self.debugger = debugger
    loader.dataset.get_input(tf.constant(0))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter('../results/{}.mp4'.format(
            opt.exp_id + '_data_loading'), fourcc, opt.save_framerate, (
        opt.video_w, opt.video_h))

    for i, batch in loader.enumerate():
        pre_img = batch['pre_img'] if 'pre_img' in batch else None
        pre_hm = batch['pre_hm'] if 'pre_hm' in batch else None
        outputs = model(batch['image'], pre_img, pre_hm)
        print(f'built model: {model.built}')
        #tf.keras.utils.plot_model(model, to_file='model.png', expand_nested=True)
        #model.summary()
        #break
        imgs = debug(self, batch, outputs[0], i, loader.dataset)
        imgs = {k: v.copy() for k, v in imgs.items()}

        if opt.save_video:
            frame_h = 448
            frame_w = 800
            rets = {'pred_hm':      (448, 800, 3), 'gt_hm': (448, 800, 3), 'pre_img_pred': (448, 800, 3),
                    'pre_img_gt':   (448, 800, 3), 'pre_hm': (448, 800, 3), 'out_pred': (448, 800, 3),
                    'out_gt':       (448, 800, 3),
                    'bird_pred_gt': (512, 512, 3)}
            """
            pre_img_pred pre_hm
            out_pred out_hm
            """
            #out_frame = np.zeros((frame_h, frame_w * 3, 3), dtype=np.uint8)
            #out_frame[:frame_h, :frame_w, :] = ret['previous']
            #out_frame[:frame_h, frame_w:2 * frame_w, :] = ret['generic']
            #out_frame[:frame_h, 2 * frame_w:, :] = obj_det
            #out.write(out_frame)

    if opt.save_video and out is not None:
        z = out.release()


def debug(self, batch, output, iter_id, dataset):
    opt = self.opt
    if 'pre_hm' in batch:
        output.update({'pre_hm': batch['pre_hm']})
    dets = generic_decode(output, K=opt.K, opt=opt)
    for k in dets:
        dets[k] = dets[k].numpy()
    dets_gt = batch['meta']['gt_det']
    for i in range(1):
        debugger = Debugger(opt=opt, dataset=dataset)
        img = batch['image'][i].numpy().transpose(1, 2, 0)
        img = np.clip(((
                               img * dataset.std + dataset.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][i].numpy())
        gt = debugger.gen_colormap(batch['hm'][i].numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        debugger.add_blend_img(img, gt, 'gt_hm')

        if 'pre_img' in batch:
            pre_img = batch['pre_img'][i].numpy().transpose(1, 2, 0)
            pre_img = np.clip(((
                                       pre_img * dataset.std + dataset.mean) * 255), 0, 255).astype(np.uint8)
            debugger.add_img(pre_img, 'pre_img_pred')
            debugger.add_img(pre_img, 'pre_img_gt')
            if 'pre_hm' in batch:
                pre_hm = debugger.gen_colormap(
                        batch['pre_hm'][i].numpy())
                debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')

        debugger.add_img(img, img_id='out_pred')
        if 'ltrb_amodal' in opt.heads:
            debugger.add_img(img, img_id='out_pred_amodal')
            debugger.add_img(img, img_id='out_gt_amodal')

        # Predictions
        for k in range(len(dets['scores'][i])):
            if dets['scores'][i, k] > opt.vis_thresh:
                debugger.add_coco_bbox(
                        dets['bboxes'][i, k] * opt.down_ratio, dets['clses'][i, k],
                        dets['scores'][i, k], img_id='out_pred')

                if 'ltrb_amodal' in opt.heads:
                    debugger.add_coco_bbox(
                            dets['bboxes_amodal'][i, k] * opt.down_ratio, dets['clses'][i, k],
                            dets['scores'][i, k], img_id='out_pred_amodal')

                if 'hps' in opt.heads and int(dets['clses'][i, k]) == 0:
                    debugger.add_coco_hp(
                            dets['hps'][i, k] * opt.down_ratio, img_id='out_pred')

                if 'tracking' in opt.heads:
                    debugger.add_arrow(
                            dets['cts'][i][k] * opt.down_ratio,
                            dets['tracking'][i][k] * opt.down_ratio, img_id='out_pred')
                    debugger.add_arrow(
                            dets['cts'][i][k] * opt.down_ratio,
                            dets['tracking'][i][k] * opt.down_ratio, img_id='pre_img_pred')

        # Ground truth
        debugger.add_img(img, img_id='out_gt')
        for k in range(len(dets_gt['scores'][i])):
            if dets_gt['scores'][i][k] > opt.vis_thresh:
                debugger.add_coco_bbox(
                        dets_gt['bboxes'][i][k] * opt.down_ratio, dets_gt['clses'][i][k],
                        dets_gt['scores'][i][k], img_id='out_gt')

                if 'ltrb_amodal' in opt.heads:
                    debugger.add_coco_bbox(
                            dets_gt['bboxes_amodal'][i, k] * opt.down_ratio,
                            dets_gt['clses'][i, k],
                            dets_gt['scores'][i, k], img_id='out_gt_amodal')

                if 'hps' in opt.heads and \
                        (int(dets['clses'][i, k]) == 0):
                    debugger.add_coco_hp(
                            dets_gt['hps'][i][k] * opt.down_ratio, img_id='out_gt')

                if 'tracking' in opt.heads:
                    debugger.add_arrow(
                            dets_gt['cts'][i][k] * opt.down_ratio,
                            dets_gt['tracking'][i][k] * opt.down_ratio, img_id='out_gt')
                    debugger.add_arrow(
                            dets_gt['cts'][i][k] * opt.down_ratio,
                            dets_gt['tracking'][i][k] * opt.down_ratio, img_id='pre_img_gt')

        if 'hm_hp' in opt.heads:
            pred = debugger.gen_colormap_hp(
                    output['hm_hp'][i].numpy())
            gt = debugger.gen_colormap_hp(batch['hm_hp'][i].numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')
            debugger.add_blend_img(img, gt, 'gt_hmhp')

        if 'rot' in opt.heads and 'dim' in opt.heads and 'dep' in opt.heads:
            dets_gt = {k: dets_gt[k].numpy() for k in dets_gt}
            calib = batch['meta']['calib'].numpy() \
                if 'calib' in batch['meta'] else None
            det_pred = generic_post_process(opt, dets,
                                            batch['meta']['c'].numpy(), batch['meta']['s'].numpy(),
                                            output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
                                            calib)
            det_gt = generic_post_process(opt, dets_gt,
                                          batch['meta']['c'].numpy(), batch['meta']['s'].numpy(),
                                          output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
                                          calib)

            debugger.add_3d_detection(
                    batch['meta']['img_path'][i].numpy().decode('ascii'), batch['meta']['flipped'][i],
                    det_pred[i], calib[i],
                    vis_thresh=opt.vis_thresh, img_id='add_pred')
            debugger.add_3d_detection(
                    batch['meta']['img_path'][i].numpy().decode('ascii'), batch['meta']['flipped'][i],
                    det_gt[i], calib[i],
                    vis_thresh=opt.vis_thresh, img_id='add_gt')
            debugger.add_bird_views(det_pred[i], det_gt[i],
                                    vis_thresh=opt.vis_thresh, img_id='bird_pred_gt')

        if opt.debug == 4:
            debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
        else:
            debugger.show_all_imgs(pause=True)

        return debugger.imgs


def custom_get_weights(model: tf.keras.Model):
    weights = []
    if isinstance(model, tf.keras.Model):
        for layer in model.layers:
            weights.extend(custom_get_weights(layer))
    elif isinstance(model, tf.keras.layers.Layer):
        return model.weights
    else:
        raise Exception
    return weights


def my_code(model: tf.keras.Model, batch_size=4, inp_h=448, inp_w=800):
    img = tf.zeros((batch_size, 3, inp_h, inp_w), tf.float32)
    pre_img = tf.zeros_like(img)
    pre_hm = tf.zeros((batch_size, 1, inp_h, inp_w), tf.float32)
    model(img, pre_img, pre_hm)  # construct the weights

    import torch
    import numpy as np

    w = torch.load('../resources/nuScenes_3Dtracking.pth', map_location=torch.device('cpu'))
    w = w['state_dict']
    w = list(w.items())
    i1 = 0
    i2 = 0
    weights_me = model.weights  # custom_get_weights(model)

    run_mean_w1 = []
    run_mean_w2 = []

    run_var_w1 = []
    run_var_w2 = []
    print(f'loading {len(weights_me), len(w)} weights')
    while (i1 < len(weights_me)):
        w1 = weights_me[i1]
        w2 = w[i2]

        shape_w1 = tuple(w1.shape)
        shape_w2 = tuple(w2[1].shape)

        if shape_w1 == shape_w2:
            w1.assign(w2[1].numpy())
            print(f'{w1.name}/{w2[0]}, {i1}, {i2} was assigned successfuly')
            i1 += 1
            i2 += 1
        elif len(shape_w1) == len(shape_w2) == 4 and shape_w1 == (shape_w2[2], shape_w2[3], shape_w2[1], shape_w2[0]):
            w1.assign(w2[1].numpy().transpose(2, 3, 1, 0))
            print(f'{w1.name}/{w2[0]}, {i1}, {i2} was assigned successfuly, transposed')
            i1 += 1
            i2 += 1
        elif 'conv2d_transpose' in w1.name and shape_w1[:2] == shape_w2[2:] and shape_w1[3] == shape_w2[0] and shape_w2[
            1] == 1:
            print(f'WARNING NOT SKIPPING CONVTRANSPOSED {w1.name} {w2[0]}')
            w1_weights = np.zeros(shape_w1, dtype=np.float32)
            w2_transposed = w2[1].numpy().transpose(2, 3, 1, 0)
            for ch in range(shape_w1[2]):
                w1_weights[:, :, ch, ch] = w2_transposed[:, :, 0, ch]
            w1.assign(tf.convert_to_tensor(w1_weights))
            i1 += 1
            i2 += 1
        elif len(shape_w1) == 4 and 'running_mean' in w2[0]:
            print(f'skipped {w2[0]},{i1},{i2}')
            run_mean_w2.append(w2)
            i2 += 1
        elif len(shape_w1) == 4 and 'running_var' in w2[0]:
            print(f'skipped {w2[0]},{i1},{i2}')
            run_var_w2.append(w2)
            i2 += 1
        elif 'moving_mean' in w1.name:
            print(f'skipped w1 {w1.name},{i1},{i2}')
            run_mean_w1.append(w1)
            i1 += 1
        elif 'moving_variance' in w1.name:
            print(f'skipped w1 {w1.name},{i1},{i2}')
            run_var_w1.append(w1)
            i1 += 1
        elif 'num_batches_tracked' in w2[0]:
            print(f'skipped {w2[0]},{i1},{i2}')
            i2 += 1
        # elif 'deformable_conv2d' in w1.name and 'conv.bias' in w2[0]:
        #    print(f'skipped {w2[0]},{i1},{i2}')
        #    i2 += 1
        else:
            print(f'{shape_w1}, {shape_w2}, {w1.name}, {w2[0]}')
            raise RuntimeError(f'{w1.name} NNOOOTTT assigned successfuly')

    for w1, w2 in zip(run_mean_w1, run_mean_w2):
        w1.assign(w2[1].numpy())
    for w1, w2 in zip(run_var_w1, run_var_w2):
        w1.assign(w2[1].numpy())

    pass


def main(args):
    start_training = time.time()

    savedir = args.savedir
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(os.path.join(savedir, 'opts.txt'), 'w') as opts_file:
        opts_file.write(str(args).replace(', ', '\n').replace('(', '(\n'))

    shutil.copyfile(__file__, os.path.join(savedir, os.path.basename(__file__)))

    # initialize network
    # copy_object_sourcefile

    trainer(args)

    training_duration = time.time() - start_training
    minutes, seconds = divmod(int(training_duration), 60)
    hours, minutes = divmod(minutes, 60)
    print(f'Training duration: {hours:02}:{minutes:02}:{seconds:02}')


if __name__ == '__main__':
    """parser = argparse.ArgumentParser('Train the network')

    parser.add_argument('--dataset', choices=['nuscenes'], default='nuscenes')
    parser.add_argument('--mini_dataset', action='store_true', default=True)

    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--savedir', default='../resources/tests')

    parser.add_argument('--head_kernel', type=int, default=3, help='')
    parser.add_argument('--prior_bias', type=float, default=-4.6)
    parser.add_argument('--dla_node', default='dcn')
    parser.add_argument('--load_model', default='asdf')
    parser.add_argument('--model_output_list', action='store_true')
    args = parser.parse_args()"""

    # hacky, should all be attributes in argparse
    res = config.TRAIN_RESOLUTION['nuscenes']
    heads = {'hm': 10, 'reg': 2, 'wh': 2, 'tracking': 2, 'dep': 1, 'rot': 8, 'dim': 3, 'amodel_offset': 2}
    dataset_opts = generic_dataset.DatasetOptions(res, config.Resolution(height=res.height // 4, width=res.width // 4),
                                                  heads=heads)
    # for key, val in dataset_opts.__dict__.items():
    #   args.__setattr__(key, val)
    # args.pre_img = True  # should be removed
    opt = opts().init()
    opt.savedir = '../resources/tests'
    opt.mini_dataset = True
    dataset_opts = generic_dataset.DatasetOptions(res, config.Resolution(height=res.height // 4, width=res.width // 4),
                                                  heads=opt)
    for key, val in dataset_opts.__dict__.items():
        if key == 'heads':
            continue
        opt.__setattr__(key, val)
    opt.pre_img = True  # should be removed
    opt.show_track_color = False
    opt.save_video = True
    opt.video_h = 450
    opt.video_w = 800 * 3
    opt.exp_id = 'tracking'
    opt.save_framerate = 2
    main(opt)
