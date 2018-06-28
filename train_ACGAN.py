import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

from models import GoodGenerator, ACGANDiscriminator
from train_util import AbstractTrainer, get_parser
from misc import combine_images, show_progress


class ACGANTrainer(AbstractTrainer):
    def __init__(self, args):
        super().__init__(args)

    def _build_graph(self):
        self.z = tf.placeholder(tf.float32, shape = (None, self.args.z_dim),
                                name = 'z')
        self.class_ = tf.placeholder(tf.float32, shape = (None, self.num_classes), name = 'class')
        self.x = tf.placeholder(tf.float32, shape = (None, self.args.image_size, self.args.image_size, 3),
                                name = 'x')

        input_dim = self.args.z_dim+self.num_classes
        self.gen = GoodGenerator(input_dim, self.args.image_size)
        self.disc = ACGANDiscriminator(self.args.image_size, self.num_classes)

        # fake sampels
        inputs = tf.concat([self.z, self.class_], axis = 1)
        self.x_ = self.gen(inputs)

        # discriminator outputs
        d_real, class_real = self.disc(self.x, reuse = False)
        d_fake, class_fake = self.disc(self.x_)

        # Wasserstein distance and gradient penalty --------------------
        self.d_real = tf.reduce_mean(d_real)
        self.d_fake = tf.reduce_mean(d_fake)

        alpha = tf.random_uniform((self.args.batch_size, 1, 1, 1), minval = 0., maxval = 1.)
        x_interp = alpha*self.x + (1. - alpha)*self.x_
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
        for e in range(self.args.n_epoch):
            for i, (images, labels) in enumerate(self.d_loader):

                images = images.numpy()/127.5 - 1.
                labels = labels.numpy()
                randoms = np.random.uniform(-1, 1, (self.args.batch_size, self.args.z_dim))
                _, d_real, d_fake = self.sess.run([self.d_opt, self.d_real, self.d_fake],
                                                  feed_dict = {self.z:randoms, self.x:images, self.class_:labels})

                if i%self.args.n_critic == 0:
                    randoms = np.random.uniform(-1, 1, (self.args.batch_size, self.args.z_dim))
                    _ = self.sess.run(self.g_opt, feed_dict = {self.z:randoms, self.class_:labels})

                if i%10 == 0:
                    show_progress(e+1, i+1, self.num_batches, d_real-d_fake, None)

                if i%100 == 0:
                    images_fake = self.pred(self.args.sample_class, 9)
                    self.save_samples(images_fake, self.args.sample_class, e+1, b+1)

            print()
            if not os.path.exists('./model_ACGAN'):
                os.mkdir('/model_ACGAN')
            self.saver.save(self.sess, f'/model_ACGAN/model_{str(e+1).zfill(3).ckpt}')

    def pred(self, class_name = None, num_samples = 9):
        randoms = np.random.uniform(-1, 1, (num_samples, self.args.z_dim))
        if class_name is not None:
            assert class_name in self.classes
            label = np.where(np.array(self.classes) == class_name, 1, 0)
            labels = np.array([label for _ in range(num_samples)])
        else:
            label = np.zeros(self.num_classes)
            label[0] = 1
            labels = np.array([np.random.permutation(label) for _ in range(num_samples)])

        images_fake = self.sess.run(self.x_, feed_dict = {self.z:randoms, self.class_:labels})
        return images_fake

    def save_samples(images_fake, class_name = None, epoch = None, batch = None):
        assert images_fake.ndim == 4
        images_fake = combine_images(images_fake)
        images_fake = images_fake*127.5 + 127.5
        
        if not os.path.exists('./samples_ACGAN'):
            os.mkdir('./samples_ACGAN')
        Image.fromarray(images_fake.astype(np.uint8))\
          .save(f'./samples_ACGAN/sample_')

        filename = './samples_ACGAN/sample'
        if class_name is not None:
            filename += f'_{class_name}'
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
    parser.add_argument('--sample_class', type = str, default = None,
                        help = 'Class of sampling fake images [None]')
    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (-1:cpu) : ')

    trainer = ACGANTrainer(args)
    trainer.train()
