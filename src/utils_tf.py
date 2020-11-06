from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def _sigmoid12(x):
    y = torch.clamp(x.sigmoid_(), 1e-12)
    return y


def _gather_feat(feat, ind):
    dim = feat.shape[2]
    ind = tf.tile(tf.expand_dims(ind,axis=2),(ind.shape[0], ind.shape[1], dim))
    feat = tf.gather(feat,ind)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)


def flip_lr(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = \
            tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def flip_lr_off(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    tmp = tmp.reshape(tmp.shape[0], 17, 2,
                      tmp.shape[2], tmp.shape[3])
    tmp[:, :, 0, :, :] *= -1
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = \
            tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = tf.nn.max_pool2d(
            heat, (kernel, kernel), strides=1, padding='SAME', data_format='NCHW')
    keep = tf.cast(hmax == heat,tf.float32)
    return heat * keep


def _topk_channel(scores, K=100):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _topk(scores, K=100):
    batch, cat, height, width = scores.shape

    topk_scores, topk_inds = tf.math.top_k(tf.reshape(scores,(batch, cat, -1)), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = tf.cast(tf.cast(topk_inds / width,tf.int32), tf.float32)
    topk_xs = tf.cast(tf.cast(topk_inds % width, tf.int32),tf.float32)

    topk_score, topk_ind = tf.math.top_k(tf.reshape(topk_scores,(batch, -1)), K)
    topk_clses = tf.cast(topk_ind / K,tf.int32)
    topk_inds = tf.reshape(_gather_feat(tf.reshape(topk_inds,(batch, -1, 1)), topk_ind),(batch, K))
    topk_ys = _gather_feat(topk_ys.reshape(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
