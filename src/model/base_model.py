from dataclasses import dataclass
from typing import Dict, List

import tensorflow as tf
from tensorflow.keras import layers


@dataclass
class BaseModelOptions:
    head_kernel: int  # kernel convolution sizes
    prior_bias: float  # bias for hm head last layer
    model_output_list: bool  # if true, the networks output is a list of lists instead of list of dicts


class BaseModel(tf.keras.Model):
    def __init__(self, heads: Dict[str, int], head_convs: Dict[str, List[int]], num_stacks: int,
                 opt: BaseModelOptions = None):
        """
        heads: Dict[str, int] - head name: corresponding number of output classes
        head_convs: Dict[str, List[int]] - head name: list with number of output channels for each convolution
        num_stacks: int - how many times the output is replicated in the output list
        opt - object with following attributes:
        """
        super().__init__()
        if opt is not None and opt.head_kernel != 3:
            print('Using head kernel:', opt.head_kernel)
            head_kernel = opt.head_kernel
        else:
            head_kernel = 3
        self.num_stacks = num_stacks
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            head_conv = head_convs[head]
            if head_conv:
                out = layers.Conv2D(classes,
                                    kernel_size=1, strides=1, padding='valid', use_bias=True, bias_initializer='zeros',
                                    data_format='channels_first')
                conv = layers.Conv2D(head_conv[0], kernel_size=head_kernel, padding='same', use_bias=True,
                                     bias_initializer='zeros', data_format='channels_first')
                convs = [conv, layers.ReLU()]
                for k in range(1, len(head_conv)):
                    convs.extend(
                            [layers.Conv2D(head_conv[k], kernel_size=1, use_bias=True, data_format='channels_first'),
                             layers.ReLU()])
                convs.append(out)
                fc = tf.keras.Sequential(convs)
                if 'hm' in head:
                    fc.layers[-1].bias_initializer = tf.keras.initializers.constant(opt.prior_bias)
            else:
                fc = layers.Conv2D(classes, kernel_size=1, strides=1, padding='valid', use_bias=True,
                                   bias_initializer='zeros')
                if 'hm' in head:
                    fc.bias_initializer = tf.keras.initializers.constant(opt.prior_bias)
            self.__setattr__(head, fc)

    def img2feats(self, x):
        raise NotImplementedError

    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
        raise NotImplementedError

    def call(self, x, pre_img=None, pre_hm=None):
        """all inputs should be given in channels_first format i.e. (batch x channels x height x width) tensors"""
        if (pre_hm is not None) or (pre_img is not None):
            feats = self.imgpre2feats(x, pre_img, pre_hm)
        else:
            feats = self.img2feats(x)
        out = []
        if self.opt.model_output_list:
            for s in range(self.num_stacks):
                z = []
                for head in sorted(self.heads):
                    z.append(self.__getattribute__(head)(feats[s]))
                out.append(z)
        else:
            for s in range(self.num_stacks):
                z = {}
                for head in self.heads:
                    z[head] = self.__getattribute__(head)(feats[s])
                out.append(z)
        return out
