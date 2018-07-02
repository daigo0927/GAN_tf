import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

from models import GoodGenerator, GoodDiscriminator
from train_util import AbstractTrainer, get_parser
from misc import combine_images, show_progress


class WGANgpTrainer(AbstractTrainer):
    def __init__(self, args):
        super().__init__(args)

    def _build_graph(self):
        self.z = tf.placeholder(tf.float32, shape = (None, self.args.z_dim),
                                name = 'z')
        self.x = tf.placeholder(tf.float32, shape = (None, self.args.image_size, self.args.image_size, 3),
                                name = 'x')

        self.gen = GoodGenerator(self.args.z_dim, self.args.image_size)
        self.disc = GoodDiscriminator(self.args.image_size)

        # fake sampels
        self.x_ = self.gen(self.z)

        # Wasserstein distance -------------------
        self.d_real = tf.reduce_mean(self.disc(self.x, reuse = False))
        self.d_fake = tf.reduce_mean(self.disc(self.x_))

        # Gradient penalty ---------------------
        alpha = tf.random_uniform((self.args.batch_size, 1, 1, 1), minval = 0., maxval = 1.)
        x_interp = alpha*self.x + (1 - alpha)*self.x_
        gradients = tf.gradients(self.disc(x_interp), [x_interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis = 3))
        gradient_penalty = tf.reduce_mean((slopes -1.)**2)
        
        self.d_loss = self.d_fake - self.d_real + self.args.lambda_*gradient_penalty
        self.g_loss = -self.d_fake

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
        for e in range(self.args.n_epoch):
            for i, images in enumerate(self.d_loader):
                images = images.numpy()/127.5 - 1.

                for _ in range(self.args.n_critic):
                    randoms = np.random.uniform(-1, 1, (self.args.batch_size, self.args.z_dim))
                    _, d_real, d_fake = self.sess.run([self.d_opt, self.d_real, self.d_fake],
                                                      feed_dict = {self.z: randoms, self.x: images})

                randoms = np.random.uniform(-1, 1, (self.args.batch_size, self.args.z_dim))
                _ = self.sess.run(self.g_opt, feed_dict = {self.z: randoms})

                if i%10 == 0:
                    show_progress(e+1, i+1, self.num_batches, d_real-d_fake, None)

                if i%100 == 0:
                    images_fake = self.pred(self.args.sample_class, 9)
                    self.save_samples(images_fake, epoch = e+1, batch = i+1)

            print()
            if not os.path.exists('./model_WGANgp'):
                os.mkdir('./model_WGANgp')
            self.saver.save(self.sess, f'./model_WGANgp/model_{str(e+1).zfill(3)}.ckpt')

    def pred(self, num_samples = 9):
        randoms = np.random.uniform(-1, 1, (num_samples, self.args.z_dim))
        images_fake = self.sess.run(self.x_, feed_dict = {self.z:randoms})
        return images_fake

    def save_samples(self, images_fake, class_name = None, epoch = None, batch = None):
        assert images_fake.ndim == 4
        images_fake = combine_images(images_fake)
        images_fake = images_fake*127.5 + 127.5
        
        if not os.path.exists('./samples_WGANgp'):
            os.mkdir('./samples_WGANgp')

        filename = './samples_WGANgp/sample'
        if epoch is not None:
            filename += f'_{str(epoch).zfill(3)}'
        if batch is not None:
            filename += f'_{str(batch).zfill(4)}'
        filename += '.png'
        Image.fromarray(images_fake.astype(np.uint8)).save(filename)


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--n_critic', type = int, default = 5,
                        help = '# of critic training [5]')
    parser.add_argument('--lambda_', type = float, default = 10,
                        help = 'Coefficient for gradient penalty [10]')
    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (-1:cpu) : ')

    trainer = ACGANTrainer(args)
    trainer.train()
