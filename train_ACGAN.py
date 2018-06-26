import numpy as np
import tensorflow as tf

from models import GoodGenerator, ACGANDiscriminator
from train_util import AbstractTrainer, get_parser
from misc import combine_images, show_progress


class ACGANTrainer(AbstractTrainer):
    def __init__(self, args):
        super().__init__(args)

    def _build_graph(self):
        self.z = tf.placeholder(tf.float32, shape = (None, self.args.z_dim),
                                name = 'z')
        self.class_ = tf.placeholder(tf.float32, shape = (None, self.d_loader.num_classes), name = 'class')
        self.x = tf.placeholder(tf.float32, shape = (None, self.args.image_size, self.args.image_size, 3),
                                name = 'x')

        input_dim = self.args.z_dim+self.d_loader.num_classes
        self.gen = GoodGenerator(input_dim, self.args.image_size)
        self.disc = ACGANDiscriminator(self.args.image_size, self.d_loader.num_classes)

        # fake sampels
        inputs = tf.concat([self.z, self.class_], axis = 1)
        self.x_ = self.gen(inputs)

        # discriminator outputs
        d_real, class_real = self.disc(self.x, reuse = False)
        d_fake, class_fake = self.disc(self.x_)

        # Wasserstein distance and gradient penalty --------------------
        self.d_real = tf.reduce_mean(d_real)
        self.d_fake = tf.reduce_mean(d_fake)

        alpha = tf.random.uniform((self.args.batch_size, 1, 1, 1), minval = 0., maxval = 1.)
        gradients = tf.gradients(self.disc(x_interp), [x_interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis = 3))
        gradient_penalty = tf.reduce_mean((slopes -1.)**2)
        
        self.d_loss = self.d_fake - self.d_real + self.args.lambda_*gradient_penalty
        self.g_loss = -self.d_fake
        # -------------------------- Wasserstein distance
        
        # auxiliary classifier loss ----------
        c_real_loss = tf.losses.softmax_cross_entropy(self.class_, class_real)
        c_fake_loss = tf.losses.softmax_cross_entropy(self.class_, class_fake)
        self.d_loss += c_real_loss + c_fake_loss
        self.g_loss += c_fake_loss
        # ----------------------- aux loss

        self.g_opt = tf.train.AdamOptimizer(learning_rate = 1e-4, beta1 = 0.5, beta2 = 0.9)\
                             .minimize(self.g_loss, var_list = self.gen.vars)
        self.d_opt = tf.train.AdamOptimizer(learning_rate = 1e-4, beta1 = 0.5, beta2 = 0.9)\
                             .minimize(self.d_loss, var_list = self.disc.vars)

        self.saver = tf.train.Saver()
        if self.args.resume is not None:
            print(f'Loading learned model from checkpoint {self.args.resume}')
            self.saver.restore(self.sess, self.args.resume)
        else:
            self.sess.run(tf.global_variables_initializer())

    def train(self):
        pass

    def pred(self, num_samples = 9):
        pass


if __name__ == '__main__':
    parser = get_parser()
    
