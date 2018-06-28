import numpy as np
import tensorflow as tf

from tfmodule.modules import Conv2D, Conv2DTranspose, PlainBlock, BottleneckBlock


class _Genrerator(object):
    def __init__(self, z_dim, image_size, name = 'generator'):
        self.z_dim = z_dim
        self.image_size = image_size
        self.name = name

    def __call__(self, z):
        pass

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class GoodGenerator(_Genrerator):
    def __init__(self, z_dim, image_size, name = 'generator'):
        super().__init__(z_dim, image_size, name)

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            x = tf.layers.Dense(4*4*8*64)(z)
            x = tf.reshape(x, [-1, 4, 4, 8*64])

            x = PlainBlock(8*64, 8*64, (3, 3), resample = 'up', name = 'plain_0')(x)
            x = PlainBlock(8*64, 4*64, (3, 3), resample = 'up', name = 'plain_1')(x)
            x = PlainBlock(4*64, 2*64, (3, 3), resample = 'up', name = 'plain_2')(x)
            x = PlainBlock(2*64, 1*64, (3, 3), resample = 'up', name = 'plain_3')(x)

            add_up = np.log2(self.image_size/64).astype(np.uint8)
            for i in range(add_up):
                x = PlainBlock(1*64, 1*64, (3, 3), resample = 'up', name = f'plain_{4+i}')(x)

            x = tf.layers.BatchNormalization()(x)
            x = tf.nn.relu(x)
            x = Conv2D(1*64, 3, (3, 3), kernel_initializer = 'he_normal')(x)
            x = tf.nn.tanh(x)

            return x
            

class DCGANGenerator(_Genrerator):
    def __init__(self, z_dim, image_size, batch_norm = True, name = 'generator'):
        super().__init__(z_dim, image_size, name)
        self.init = tf.initializers.random_normal(stddev = 0.02)
        self.bn = batch_norm

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            x = tf.layers.Dense(4*4*8*128)(z)
            x = tf.reshape(x, [-1, 4, 4, 8*64])
            if self.bn:
                x = tf.layers.BatchNormalization()(x)
            x = tf.nn.relu(x)

            out_channels = 4*64
            num_up = np.log2(self.image_size/4).astype(np.uint8)
            for i in range(num_up-1):
                x = tf.layers.Conv2DTranspose(out_channels, (5, 5), (2, 2), 'same',
                                              kernel_initializer = self.init)(x)
                if self.bn:
                    x = tf.layers.BatchNormalization()(x)
                x = tf.nn.relu(x)
                out_channels = max(out_channels/2, 64)

            x = tf.layers.Conv2DTranspose(3, (5, 5), (2, 2), 'same',
                                          kernel_initializer = self.init)(x)
            x = tf.nn.tanh(x)
            return x


class _Dicriminator(object):
    def __init__(self, image_size, name = 'discriminator'):
        self.image_size = image_size
        self.name = name
        
    def __call__(self, x, reuse = True):
        pass

    @property    
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class GoodDiscriminator(_Dicriminator):
    def __init__(self, image_size, name = 'discriminator'):
        super().__init__(image_size, name)

    def __call__(self, x, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            x = Conv2D(3, 64, (3, 3), kernel_initializer = None)(x)
            
            add_down = np.log2(self.image_size/64).astype(np.uint8)
            for i in range(add_down):
                x = PlainBlock(1*64, 1*64, (3, 3), resample = 'down', name = f'plain_{i}')(x)
                
            x = PlainBlock(1*64, 2*64, (3, 3), resample = 'down', name = f'plain_{add_down}')(x)
            x = PlainBlock(2*64, 4*64, (3, 3), resample = 'down', name = f'plain_{add_down+1}')(x)
            x = PlainBlock(4*64, 8*64, (3, 3), resample = 'down', name = f'plain_{add_down+2}')(x)
            x = PlainBlock(8*64, 8*64, (3, 3), resample = 'down', name = f'plain_{add_down+3}')(x)

            x = tf.layers.flatten(x)
            x = tf.layers.Dense(1)(x)
            return x


class DCGANDiscriminator(_Dicriminator):
    def __init__(self, image_size, batch_norm = True, name = 'discriminator'):
        super().__init__(image_size, name)
        self.init = tf.initializers.random_normal(stddev = 0.02)
        self.bn = batch_norm

    def __call__(self, x, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            x = Conv2D(3, 64, (5, 5), strides = (2, 2),
                       kernel_initializer = self.init)(x)
            x = tf.nn.leaky_relu(x, 0.2)

            out_channels = 32
            num_down = np.log2(self.image_size/4).astype(np.uint8)
            for i in range(num_down-1):
                x = tf.layers.Conv2D(max(out_channels, 64), (5, 5), (2, 2), 'same',
                                     kernel_initializer = self.init)(x)
                x = tf.layers.BatchNormalization()(x)
                x = tf.nn.leaky_relu(x, 0.2)
                out_channels *= 2
                
            x = tf.layers.flatten(x)
            x = tf.layers.Dense(1)(x)
            return x


# Auxiliary classifier GAN by ResNet discriminator
class ACGANDiscriminator(_Dicriminator):
    def __init__(self, image_size, num_classes = None, name = 'discriminator'):
        super().__init__(image_size, name)
        assert num_classes is not None, 'num_classes must be specified'
        self.num_classes = num_classes
        
    def __call__(self, x, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            x = Conv2D(3, 64, (3, 3), kernel_initializer = None)(x)
            
            add_down = np.log2(self.image_size/64).astype(np.uint8)
            for i in range(add_down):
                x = PlainBlock(1*64, 1*64, (3, 3), resample = 'down', name = f'plain_{i}')(x)
                
            x = PlainBlock(1*64, 2*64, (3, 3), resample = 'down', name = f'plain_{add_down}')(x)
            x = PlainBlock(2*64, 4*64, (3, 3), resample = 'down', name = f'plain_{add_down+1}')(x)
            x = PlainBlock(4*64, 8*64, (3, 3), resample = 'down', name = f'plain_{add_down+2}')(x)
            x = PlainBlock(8*64, 8*64, (3, 3), resample = 'down', name = f'plain_{add_down+3}')(x)

            x = tf.layers.flatten(x)
            class_ = tf.layers.Dense(self.num_classes)(x)
            
            x = tf.layers.Dense(1)(x)
            return x, class_
