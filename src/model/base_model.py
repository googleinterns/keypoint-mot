import tensorflow as tf
from tensorflow.keras import layers


def fill_fc_weights(fc):
    if isinstance(fc, tf.keras.Sequential):
        all_layers = fc.layers
    else:
        all_layers = fc

    for m in all_layers:
        if isinstance(m, layers.Conv2D):
            if m.use_bias:
                m.bias_initialier = tf.keras.initializers.constant(0)


class BaseModel(tf.keras.Model):
    def __init__(self, heads, head_convs, num_stacks, opt=None):
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
            if len(head_conv) > 0:
                out = layers.Conv2D(classes,
                                    kernel_size=1, strides=1, padding='valid', use_bias=True,
                                    data_format='channels_first')
                conv = layers.Conv2D(head_conv[0], kernel_size=head_kernel, padding='same', use_bias=True,
                                     data_format='channels_first')
                convs = [conv, layers.ReLU()]
                for k in range(1, len(head_conv)):
                    convs.extend(
                            [layers.Conv2D(head_conv[k], kernel_size=1, use_bias=True, data_format='channels_first'),
                             layers.ReLU()])
                convs.append(out)
                fc = tf.keras.Sequential(convs)
                if 'hm' in head:
                    fc.layers[-1].bias_initizlier = tf.keras.initializers.constant(opt.prior_bias)
                else:
                    fill_fc_weights(fc)
                    pass
            else:
                fc = layers.Conv2D(classes, kernel_size=1, strides=1, padding='valid', use_bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(opt.prior_bias)
                else:
                    fill_fc_weights(fc)
                    pass
            self.__setattr__(head, fc)

    def img2feats(self, x):
        raise NotImplementedError

    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
        raise NotImplementedError

    def call(self, x, pre_img=None, pre_hm=None):
        """all inputs should be given in channels_last format i.e. (batch x channels x height x width) tensors"""
        if (pre_hm is not None) or (pre_img is not None):
            feats = self.imgpre2feats(x, pre_img, pre_hm)
        else:
            feats = self.img2feats(x)
        out = []
        if self.opt.model_output_list:
            for s in range(self.num_stacks):
                z = []
                for head in sorted(self.heads):
                    z.append(self.__getattr__(head)(feats[s]))
                out.append(z)
        else:
            for s in range(self.num_stacks):
                z = {}
                for head in self.heads:
                    z[head] = self.__getattribute__(head)(feats[s])
                out.append(z)
        return out
