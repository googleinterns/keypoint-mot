import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from model import base_model

BN_MOMENTUM = 0.1


def BatchNorm2d(momentum, data_format='channels_first'):
    axis = 1 if data_format == 'channels_first' else -1
    return layers.BatchNormalization(momentum=momentum, axis=axis, epsilon=1e-05)


class CustomConv2d(layers.Conv2D):
    """
    Wrapper over layers.Conv2D that produces symmetric padding.

    Args:
        padding: int - the number of added zero rows and columns on each side of the image
                       (image with size H,W -> H + 2*padding, W + 2*padding)

    see tf.keras.layers.Conv2D
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding: int = None,
                 data_format=None,
                 dilation_rate=(1, 1),
                 groups=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        assert padding is not None, 'This layer should only be used with a given padding'
        assert isinstance(padding, int), 'The padding whould be given as an int'

        self.custom_padding = padding
        self.custom_padding_tensor = tf.constant([[0, 0], [0, 0], [padding, padding], [padding, padding]])
        if data_format == 'channels_first':
            self.custom_padding_tensor = tf.constant([[0, 0], [0, 0], [padding, padding], [padding, padding]])
        else:
            self.custom_padding_tensor = tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]])

        super().__init__(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='valid',
                         data_format=data_format,
                         dilation_rate=dilation_rate,
                         groups=groups,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs)

    def build(self, input_shape):
        self.orig_input_shape = input_shape
        super().build(tf.TensorShape(np.array(input_shape) + self.custom_padding_tensor.numpy().sum(axis=1)))

    def call(self, inputs):
        x = tf.pad(inputs, self.custom_padding_tensor)
        x = super().call(x)
        return x


class BasicBlock(tf.keras.Model):
    def __init__(self, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = CustomConv2d(planes, kernel_size=3,
                                  strides=stride, padding=dilation,
                                  use_bias=False, dilation_rate=dilation, data_format='channels_first')
        self.bn1 = BatchNorm2d(momentum=BN_MOMENTUM)
        self.relu = layers.ReLU()
        self.conv2 = CustomConv2d(planes, kernel_size=3,
                                  strides=1, padding=dilation,
                                  use_bias=False, dilation_rate=dilation, data_format='channels_first')
        self.bn2 = BatchNorm2d(momentum=BN_MOMENTUM)
        self.stride = stride

    def call(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(tf.keras.Model):
    def __init__(self, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = CustomConv2d(
                out_channels, 1,
                strides=1, use_bias=False, padding=(kernel_size - 1) // 2, data_format='channels_first')
        self.bn = BatchNorm2d(momentum=BN_MOMENTUM)
        self.relu = layers.ReLU()
        self.residual = residual

    def call(self, *x):
        children = x
        x = self.conv(tf.concat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(tf.keras.Model):
    def __init__(self, levels, block, in_channels, out_channels, stride=1, level_root=False, root_dim=0,
                 root_kernel_size=1, dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, root_dim=0,
                              root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = layers.MaxPool2D(stride, strides=stride, data_format='channels_first')
        if in_channels != out_channels:
            self.project = tf.keras.Sequential([
                layers.Conv2D(out_channels, kernel_size=1, strides=1, use_bias=False, data_format='channels_first'),
                BatchNorm2d(momentum=BN_MOMENTUM)
            ])

    def call(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(tf.keras.Model):
    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock, residual_root=False, linear_root=False,
                 opt=None):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = tf.keras.Sequential([
            CustomConv2d(channels[0], kernel_size=7, strides=1, padding=3, use_bias=False,
                         data_format='channels_first'),
            BatchNorm2d(momentum=BN_MOMENTUM),
            layers.ReLU()])

        self.level0 = self._make_conv_level(channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root)

        if opt.pre_img:
            self.pre_img_layer = tf.keras.Sequential([
                CustomConv2d(channels[0], kernel_size=7, strides=1, padding=3, use_bias=False,
                             data_format='channels_first'),
                BatchNorm2d(momentum=BN_MOMENTUM),
                layers.ReLU()])
        if opt.pre_hm:
            self.pre_hm_layer = tf.keras.Sequential([
                CustomConv2d(channels[0], kernel_size=7, strides=1, padding=3, use_bias=False,
                             data_format='channels_first'),
                BatchNorm2d(momentum=BN_MOMENTUM),
                layers.ReLU()])

    def _make_conv_level(self, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                CustomConv2d(planes, kernel_size=3,
                             strides=stride if i == 0 else 1,
                             padding=dilation, use_bias=False, dilation_rate=dilation, data_format='channels_first'),
                BatchNorm2d(momentum=BN_MOMENTUM),
                layers.ReLU()])
        return tf.keras.Sequential(modules)

    def call(self, x, pre_img=None, pre_hm=None):
        y = []
        x = self.base_layer(x)
        if pre_img is not None:
            x = x + self.pre_img_layer(pre_img)
        if pre_hm is not None:
            x = x + self.pre_hm_layer(pre_hm)
        for i in range(6):
            layers = getattr(self, 'level{}'.format(i))
            if isinstance(layers, tf.keras.Sequential) and layers.built:
                for layer in layers.layers:
                    x = layer(x)
            else:
                x = layers(x)
            y.append(x)

        return y


def dla34(**kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    return model


class DeformConvUnit(layers.Layer):
    def __init__(self, channels, kernel_size, strides, padding):
        super().__init__()
        import tensorflow_addons as tfa
        self.deform_conv = tfa.layers.DeformableConv2D(channels, kernel_size=(3, 3), use_bias=True, padding='SAME',
                                                       use_mask=True)
        channels_ = 1 * 3 * kernel_size[0] * kernel_size[1]
        self.conv_offset = layers.Conv2D(channels_, kernel_size=kernel_size, strides=strides, padding=padding,
                                         use_bias=True, data_format='channels_first')

    def call(self, inputs):
        weight_info = self.conv_offset(inputs)
        o1, o2, mask = tf.split(weight_info, 3, axis=1)
        offset = tf.concat((o1, o2), axis=1)
        mask = tf.sigmoid(mask)
        return self.deform_conv([inputs, offset, mask])


class DeformConv(tf.keras.Model):
    def __init__(self, channels_out):
        super(DeformConv, self).__init__()
        self.actf = tf.keras.Sequential([
            BatchNorm2d(momentum=BN_MOMENTUM),
            layers.ReLU()
        ])
        self.conv = DeformConvUnit(channels_out, (3, 3), 1, 'same')

    def call(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(tf.keras.Model):
    def __init__(self, o, channels, up_f, node_type=(DeformConv, DeformConv)):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = node_type[0](o)
            node = node_type[1](o)

            up = layers.Conv2DTranspose(o, f * 2, strides=f, padding='same', use_bias=False,
                                        data_format='channels_first', groups=o)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def call(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            x = project(layers[i])
            x = upsample(x)
            layers[i] = x
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(tf.keras.Model):
    def __init__(self, startp, channels, scales, in_channels=None,
                 node_type=DeformConv):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j],
                          node_type=node_type))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def call(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


DLA_NODE = {
    'dcn': (DeformConv, DeformConv)
}


class DLASeg(base_model.BaseModel):
    def __init__(self, num_layers, heads, head_convs, opt):
        super().__init__(heads, head_convs, 1, opt=opt)
        down_ratio = 4
        self.opt = opt
        self.node_type = DLA_NODE[opt.dla_node]
        print('Using node type:', self.node_type)
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        self.base = globals()['dla{}'.format(num_layers)](opt=opt)

        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales, node_type=self.node_type)
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)],
                            node_type=self.node_type)

    def img2feats(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]]

    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
        x = self.base(x, pre_img, pre_hm)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(tf.identity(x[i]))
        self.ida_up(y, 0, len(y))

        return [y[-1]]
