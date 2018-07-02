import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

from models import DCGANGenerator, ACGANDiscriminator
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
        self.gen = DCGANGenerator(input_dim, self.args.image_size)
        self.disc = ACGANDiscriminator(self.args.image_size, self.num_classes)

        # fake sampels
        inputs = tf.concat([self.z, self.class_], axis = 1)
        self.x_ = self.gen(inputs)

        # discriminator outputs
        d_real, class_real = self.disc(self.x, reuse = False)
        d_fake, class_fake = self.disc(self.x_)

        d_loss_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(d_real), d_real))
        d_loss_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.zeros_like(d_fake), d_fake))
        self.d_loss = d_loss_real + d_loss_real
        self.g_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(d_fake), d_fake))

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
                _, d_loss = self.sess.run([self.d_opt, self.d_loss],
                                          feed_dict = {self.z:randoms, self.x:images, self.class_:labels})

                _ = self.sess.run(self.g_opt, feed_dict = {self.z:randoms, self.class_:labels})

                if i%10 == 0:
                    show_progress(e+1, i+1, self.num_batches, d_loss, None)

                if i%100 == 0:
                    images_fake = self.pred(self.args.sample_class, 9)
                    self.save_samples(images_fake, self.args.sample_class, e+1, i+1)

            print()
            if not os.path.exists('./model_ACGAN'):
                os.mkdir('./model_ACGAN')
            self.saver.save(self.sess, f'./model_ACGAN/model_{str(e+1).zfill(3)}.ckpt')

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

    def save_samples(self, images_fake, class_name = None, epoch = None, batch = None):
        assert images_fake.ndim == 4
        images_fake = combine_images(images_fake)
        images_fake = images_fake*127.5 + 127.5
        
        if not os.path.exists('./samples_ACGAN'):
            os.mkdir('./samples_ACGAN')

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
