from functools import partial

import numpy as np
import tensorflow as tf


class Conv2DBatchNormReLU(tf.keras.Model):
    def __init__(self, num_filters, kernel_size, padding, strides=(1, 1), **kwargs):
        super(Conv2DBatchNormReLU, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.batch_norm(x, training=training)
        output = self.activation(x)
        return output


class FPNHead(tf.keras.Model):
    def __init__(self, num_filters, num_output, **kwargs):
        super(FPNHead, self).__init__(**kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(
            num_filters, kernel_size=(3, 3), padding="same", use_bias=False
        )
        self.conv_2 = tf.keras.layers.Conv2D(
            num_output, kernel_size=(3, 3), padding="same", use_bias=False
        )

    def call(self, x):
        return self.conv_2(self.conv_1(x))


class Block17(tf.keras.Model):
    def __init__(self, scale=1.0, **kwargs):
        super(Block17, self).__init__(**kwargs)

        self.scale = scale

        self.branch_0 = Conv2DBatchNormReLU(
            192, kernel_size=1, strides=1, padding="same"
        )

        self.branch_1 = tf.keras.Sequential(
            [
                Conv2DBatchNormReLU(128, kernel_size=1, strides=1, padding="same"),
                Conv2DBatchNormReLU(160, kernel_size=(1, 7), strides=1, padding="same"),
                Conv2DBatchNormReLU(192, kernel_size=(7, 1), strides=1, padding="same"),
            ]
        )

        self.concat = tf.keras.layers.Concatenate()
        self.conv = tf.keras.layers.Conv2D(
            1088, kernel_size=(1, 1), strides=(1, 1), activation=None
        )
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        out = self.concat([self.branch_0(x), self.branch_1(x)])
        out = self.conv(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block35(tf.keras.Model):
    def __init__(self, scale=1.0, **kwargs):
        super(Block35, self).__init__(**kwargs)

        self.scale = scale

        self.branch_0 = Conv2DBatchNormReLU(
            num_filters=32, kernel_size=(1, 1), padding="same"
        )

        self.branch_1 = tf.keras.Sequential(
            [
                Conv2DBatchNormReLU(num_filters=32, kernel_size=1, padding="same"),
                Conv2DBatchNormReLU(num_filters=32, kernel_size=3, padding="same"),
            ]
        )

        self.branch_2 = tf.keras.Sequential(
            [
                Conv2DBatchNormReLU(num_filters=32, kernel_size=1, padding="same"),
                Conv2DBatchNormReLU(num_filters=48, kernel_size=3, padding="same"),
                Conv2DBatchNormReLU(num_filters=64, kernel_size=3, padding="same"),
            ]
        )

        self.conv = tf.keras.layers.Conv2D(
            320, kernel_size=1, strides=(1, 1), activation=None
        )
        self.relu = tf.keras.layers.ReLU()
        self.concat = tf.keras.layers.Concatenate()

    def call(self, x, training=False):
        out = self.concat([self.branch_0(x), self.branch_1(x), self.branch_2(x)])
        out = self.conv(out)
        out = out * self.scale + x
        out = self.relu(out)

        return out


class InceptionMixed5b(tf.keras.Model):
    def __init__(self):
        super(InceptionMixed5b, self).__init__()

        self.branch_0 = Conv2DBatchNormReLU(
            num_filters=96, kernel_size=(1, 1), padding="same"
        )

        self.branch_1 = tf.keras.Sequential(
            [
                Conv2DBatchNormReLU(num_filters=48, kernel_size=1, padding="same"),
                Conv2DBatchNormReLU(num_filters=64, kernel_size=5, padding="same"),
            ]
        )

        self.branch_2 = tf.keras.Sequential(
            [
                Conv2DBatchNormReLU(num_filters=64, kernel_size=1, padding="same"),
                Conv2DBatchNormReLU(num_filters=96, kernel_size=3, padding="same"),
                Conv2DBatchNormReLU(num_filters=96, kernel_size=3, padding="same"),
            ]
        )

        self.branch_3 = tf.keras.Sequential(
            [
                tf.keras.layers.AveragePooling2D(
                    pool_size=(3, 3), strides=1, padding="same"
                ),
                Conv2DBatchNormReLU(num_filters=64, kernel_size=1, padding="same"),
            ]
        )

        self.concat = tf.keras.layers.Concatenate()

    def call(self, x, training=False):
        return self.concat(
            [self.branch_0(x), self.branch_1(x), self.branch_2(x), self.branch_3(x)]
        )


class InceptionMixed6a(tf.keras.Model):
    def __init__(self):
        super(InceptionMixed6a, self).__init__()

        self.branch_0 = Conv2DBatchNormReLU(
            num_filters=384, kernel_size=(3, 3), strides=(2, 2), padding="valid"
        )

        self.branch_1 = tf.keras.Sequential(
            [
                Conv2DBatchNormReLU(num_filters=256, kernel_size=1, padding="valid"),
                Conv2DBatchNormReLU(num_filters=256, kernel_size=3, padding="same"),
                Conv2DBatchNormReLU(
                    num_filters=384, kernel_size=3, strides=(2, 2), padding="valid"
                ),
            ]
        )

        self.branch_2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))

        self.concat = tf.keras.layers.Concatenate()

    def call(self, x, training=False):
        return self.concat([self.branch_0(x), self.branch_1(x), self.branch_2(x)])


class InceptionMixed7a(tf.keras.Model):
    def __init__(self):
        super(InceptionMixed7a, self).__init__()

        self.branch_0 = tf.keras.Sequential(
            [
                Conv2DBatchNormReLU(
                    num_filters=256, kernel_size=(1, 1), strides=(1, 1), padding="valid"
                ),
                Conv2DBatchNormReLU(
                    num_filters=384, kernel_size=(3, 3), strides=(2, 2), padding="valid"
                ),
            ]
        )
        self.branch_1 = tf.keras.Sequential(
            [
                Conv2DBatchNormReLU(
                    num_filters=256, kernel_size=(1, 1), strides=(1, 1), padding="valid"
                ),
                Conv2DBatchNormReLU(
                    num_filters=288, kernel_size=(3, 3), strides=(2, 2), padding="valid"
                ),
            ]
        )

        self.branch_2 = tf.keras.Sequential(
            [
                Conv2DBatchNormReLU(num_filters=256, kernel_size=1, padding="same"),
                Conv2DBatchNormReLU(
                    num_filters=288, kernel_size=3, strides=(1, 1), padding="same"
                ),
                Conv2DBatchNormReLU(
                    num_filters=320, kernel_size=3, strides=(2, 2), padding="valid"
                ),
            ]
        )

        self.branch_3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))

        self.concat = tf.keras.layers.Concatenate()

    def call(self, x, training=False):
        return self.concat(
            [self.branch_0(x), self.branch_1(x), self.branch_2(x), self.branch_3(x)]
        )


class FPN(tf.keras.Model):
    def __init__(self, num_filters):
        super(FPN, self).__init__()

        self.encoder_0 = Conv2DBatchNormReLU(
            32, kernel_size=(3, 3), strides=2, padding="valid", name="encoder_0"
        )
        self.encoder_1 = tf.keras.Sequential(
            [
                Conv2DBatchNormReLU(32, kernel_size=(3, 3), padding="valid"),
                Conv2DBatchNormReLU(64, kernel_size=(3, 3), padding="same"),
                tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
            ],
            name="encoder_1",
        )
        self.encoder_2 = tf.keras.Sequential(
            [
                Conv2DBatchNormReLU(80, kernel_size=(1, 1), padding="valid"),
                Conv2DBatchNormReLU(192, kernel_size=(3, 3), padding="valid"),
                tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
            ],
            name="encoder_2",
        )
        self.encoder_3 = tf.keras.Sequential(
            [InceptionMixed5b()] + 10 * [Block35(scale=0.17)] + [InceptionMixed6a()],
            name="encoder_3",
        )
        self.encoder_4 = tf.keras.Sequential(
            [Block17(scale=0.10)] * 17 + [InceptionMixed7a()]
        )

        self.td_1 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ],
            name="td_1",
        )
        self.td_2 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ],
            name="td_2",
        )
        self.td_3 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ],
            name="td_3",
        )

        self.lateral_0 = tf.keras.layers.Conv2D(
            num_filters // 2, kernel_size=1, use_bias=False, name="lateral_0"
        )
        self.lateral_1 = tf.keras.layers.Conv2D(
            num_filters, kernel_size=1, use_bias=False, name="lateral_1"
        )
        self.lateral_2 = tf.keras.layers.Conv2D(
            num_filters, kernel_size=1, use_bias=False, name="lateral_2"
        )
        self.lateral_3 = tf.keras.layers.Conv2D(
            num_filters, kernel_size=1, use_bias=False, name="lateral_3"
        )
        self.lateral_4 = tf.keras.layers.Conv2D(
            num_filters, kernel_size=1, use_bias=False, name="lateral_4"
        )

        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2))

    def call(self, x, training=False):
        encoded_0 = self.encoder_0(x)
        encoded_1 = self.encoder_1(encoded_0)
        encoded_2 = self.encoder_2(encoded_1)
        encoded_3 = self.encoder_3(encoded_2)
        encoded_4 = self.encoder_4(encoded_3)

        lateraled_0 = self.lateral_0(encoded_0)
        lateraled_1 = tf.pad(
            self.lateral_1(encoded_1),
            tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]),
            "REFLECT",
        )
        lateraled_2 = self.lateral_2(encoded_2)
        lateraled_3 = tf.pad(
            self.lateral_3(encoded_3),
            tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]),
            "REFLECT",
        )
        lateraled_4 = tf.pad(
            self.lateral_4(encoded_4),
            tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]),
            "REFLECT",
        )

        map_4 = lateraled_4
        map_3 = self.td_1(lateraled_3 + self.upsample(map_4))
        map_2 = self.td_2(
            tf.pad(
                lateraled_2, tf.constant([[0, 0], [1, 2], [1, 2], [0, 0]]), "reflect"
            )
            + self.upsample(map_3)
        )
        map_1 = self.td_3(lateraled_1 + self.upsample(map_2))
        map_0 = tf.pad(
            lateraled_0, tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]]), "reflect"
        )

        return map_0, map_1, map_2, map_3, map_4


class FPNInception(tf.keras.Model):
    def __init__(self, num_filters=128, num_filters_fpn=256):
        super(FPNInception, self).__init__()

        self.fpn = FPN(num_filters_fpn)

        self.head_1 = FPNHead(num_filters, num_filters)
        self.head_2 = FPNHead(num_filters, num_filters)
        self.head_3 = FPNHead(num_filters, num_filters)
        self.head_4 = FPNHead(num_filters, num_filters)

        self.smooth = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ]
        )

        self.smooth_2 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(num_filters // 2, kernel_size=3, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ]
        )

        self.final_conv = tf.keras.layers.Conv2D(3, kernel_size=3, padding="same")

        self.concat = tf.keras.layers.Concatenate()
        self.upsample = tf.keras.layers.UpSampling2D((2, 2))

    def upsample_iterations(self, x, scale_iterations):
        for _ in range(scale_iterations):
            x = self.upsample(x)

        return x

    def call(self, inputs):
        map_0, map_1, map_2, map_3, map_4 = self.fpn(inputs)

        map_1 = self.upsample_iterations(self.head_1(map_1), scale_iterations=0)
        map_2 = self.upsample_iterations(self.head_2(map_2), scale_iterations=1)
        map_3 = self.upsample_iterations(self.head_3(map_3), scale_iterations=2)
        map_4 = self.upsample_iterations(self.head_4(map_4), scale_iterations=3)

        x = self.upsample(self.smooth(self.concat([map_4, map_3, map_2, map_1])))
        x = self.upsample(self.smooth_2(x + map_0))

        final = self.final_conv(x)
        residual = tf.tanh(final) + inputs

        return tf.clip_by_value(residual, clip_value_min=-1, clip_value_max=1)


class NLayerDiscriminator(tf.keras.Model):
    def __init__(self, ndf, n_layers):
        super(NLayerDiscriminator, self).__init__()

        self.kernel_size = 4
        self.padding_size = int(np.ceil((self.kernel_size - 1) / 2))

        self.initial = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=ndf, kernel_size=self.kernel_size, strides=2
                ),
                tf.keras.layers.LeakyReLU(0.2),
            ]
        )

        self.filters_upscale_blocks = []
        for n in range(1, n_layers):
            nf_mult = min(2 ** n, 8)
            self.filters_upscale_blocks.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Conv2D(
                            ndf * nf_mult, kernel_size=self.kernel_size, strides=2
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(0.2),
                    ]
                )
            )

        nf_mult = min(2 ** n_layers, 8)
        self.filters_upscale_blocks.append(
            tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        ndf * nf_mult,
                        kernel_size=self.kernel_size,
                        strides=1,
                        padding="valid",
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(0.2),
                ]
            )
        )

        self.final = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    1, kernel_size=self.kernel_size, strides=1, padding="valid"
                )
            ]
        )

        self.pad = partial(
            tf.pad,
            paddings=tf.constant(
                [
                    [0, 0],
                    [self.padding_size, self.padding_size],
                    [self.padding_size, self.padding_size],
                    [0, 0],
                ]
            ),
            mode="REFLECT",
        )

    def call(self, x):

        x = self.pad(x)
        x = self.initial(x)

        for upscale_filter_block in self.filters_upscale_blocks:
            x = self.pad(x)
            x = upscale_filter_block(x)

        x = self.pad(x)
        x = self.final(x)

        return x


if __name__ == "__main__":
    model = FPNInception(num_filters=128, num_filters_fpn=256)
    x = model(tf.random.uniform((4, 256, 256, 3)))
    print(x.shape)

    model = NLayerDiscriminator(ndf=64, n_layers=5)
    x = model(tf.random.uniform((4, 70, 70, 3)))
    print(x.shape)
