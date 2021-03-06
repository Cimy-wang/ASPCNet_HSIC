"""
__author__ = 'Cimy Wang'
__mtime__  = '2021/9/26'
If necessary, please contact us. e-mail: jinping_wang@foxmail.com
"""
import tensorflow as tf
import keras.layers as KL


class ASPConv(KL.Layer):
    def __init__(self, filters=128,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (1, stride, stride, 1)
        self.dilation = (1, dilation, dilation, 1)
        self.deformable_groups = 1
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        super(ASPConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=self.kernel_size + (int(input_shape[-1]), self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype='float32',
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype='float32',
            )

        self.offset_kernel = self.add_weight(
            name='offset_kernel',
            shape=self.kernel_size + (
                input_shape[-1], 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]),
            initializer='zeros',
            trainable=True,
            dtype='float32')

        self.offset_bias = self.add_weight(
            name='offset_bias',
            shape=(3 * self.kernel_size[0] * self.kernel_size[1] * self.deformable_groups,),
            initializer='zeros',
            trainable=True,
            dtype='float32',
        )
        self.ks = self.kernel_size[0] * self.kernel_size[1]
        self.ph, self.pw = (self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2
        self.phw = tf.constant([self.ph, self.pw], dtype='int32')
        self.patch_yx = tf.stack(tf.meshgrid(tf.range(-self.phw[1], self.phw[1] + 1),
                                             tf.range(-self.phw[0], self.phw[0] + 1))[::-1],
                                 axis=-1)
        self.patch_yx = tf.reshape(self.patch_yx, [-1, 2])
        super(ASP, self).build(input_shape)

    def call(self, x):
        offset = tf.nn.conv2d(x, self.offset_kernel, strides=self.stride, padding='SAME')
        offset += self.offset_bias
        bs, ih, iw, ic = [v.value for v in x.shape]
        bs = tf.shape(x)[0]
        oyox, mask = offset[..., :2 * self.ks], offset[..., 2 * self.ks:]
        mask = tf.nn.sigmoid(mask)
        grid_yx = tf.stack(tf.meshgrid(tf.range(iw), tf.range(ih))[::-1], axis=-1)
        grid_yx = tf.reshape(grid_yx, [1, ih, iw, 1, 2]) + self.phw + self.patch_yx
        grid_yx = tf.cast(grid_yx, 'float32') + tf.reshape(oyox, [bs, ih, iw, -1, 2])
        grid_iy0ix0 = tf.floor(grid_yx)
        grid_iy1ix1 = tf.clip_by_value(grid_iy0ix0 + 1, 0, tf.constant([ih + 1, iw + 1], dtype='float32'))
        grid_iy1, grid_ix1 = tf.split(grid_iy1ix1, 2, axis=4)
        grid_iy0ix0 = tf.clip_by_value(grid_iy0ix0, 0, tf.constant([ih + 1, iw + 1], dtype='float32'))
        grid_iy0, grid_ix0 = tf.split(grid_iy0ix0, 2, axis=4)
        grid_yx = tf.clip_by_value(grid_yx, 0, tf.constant([ih + 1, iw + 1], dtype='float32'))
        batch_index = tf.tile(tf.reshape(tf.range(bs), [bs, 1, 1, 1, 1, 1]), [1, ih, iw, self.ks, 4, 1])
        grid = tf.reshape(tf.concat([grid_iy1ix1, grid_iy1, grid_ix0, grid_iy0, grid_ix1, grid_iy0ix0], axis=-1),
                          [bs, ih, iw, self.ks, 4, 2])
        grid = tf.concat([batch_index, tf.cast(grid, 'int32')], axis=-1)
        delta = tf.reshape(tf.concat([grid_yx - grid_iy0ix0, grid_iy1ix1 - grid_yx], axis=-1),
                           [bs, ih, iw, self.ks, 2, 2])
        w = tf.expand_dims(delta[..., 0], axis=-1) * tf.expand_dims(delta[..., 1], axis=-2)
        x = tf.pad(x, [[0, 0], [int(self.ph), int(self.ph)], [int(self.pw), int(self.pw)], [0, 0]])
        map_sample = tf.gather_nd(x, grid)
        map_bilinear = tf.reduce_sum(tf.reshape(w, [bs, ih, iw, self.ks, 4, 1]) * map_sample, axis=-2) * tf.expand_dims(
            mask, axis=-1)
        map_all = tf.reshape(map_bilinear, [bs, ih, iw, -1])
        output = tf.nn.conv2d(map_all, tf.reshape(self.kernel, [1, 1, -1, self.filters]), strides=self.stride,
                              dilations=self.dilation, padding='SAME')
        if self.use_bias:
            output += self.bias
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)
