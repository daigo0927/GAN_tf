import argparse
import numpy as np
import tensorflow as tf
import torch
from torch.utils import data

# from datautils.utils import get_dataset

class AbstractTrainer(object):
    def __init__(self, args):
        self.args = args
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self._build_dataloader()
        self._build_graph()

    def _build_dataloader(self):
        dataset = get_dataset(self.args.dataset)
        d_set = dataset(self.args.dataset_dir, 'train',
                        self.args.crop_type, self.args.crop_shape,
                        self.args.resize_shape, self.args.resize_scale)
        self.num_batches = int(len(d_set.samples)/self.args.batch_size)
        self.d_loader = data.DataLoader(d_set, shuffle = True,
                                        batch_size = self.args.batch_size,
                                        num_workers = self.args.num_workers,
                                        pin_memory = True, drop_last = True)

    def _build_graph(self):
        pass

    def train(self):
        pass

    def pred(self, num_samples = 9):
        pass

    def save_samples(self):
        pass


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, required = True,
                        help = 'Target dataset (like CelebA)')
    parser.add_argument('--dataset_dir', type = str, required = True,
                        help = 'Directory containing target dataset')
    parser.add_argument('--n_epoch', type = int, default = 15,
                        help = '# of epochs [15]')
    parser.add_argument('--batch_size', type = int, default = 64,
                        help = 'Batch size [64]')
    parser.add_argument('--num_workers', type = int, default = 4,
                        help = '# of workers for dataloading [4]')

    parser.add_argument('--crop_type', type = str, default = 'center',
                        help = 'Crop type for raw images [center]')
    parser.add_argument('--crop_shape', nargs = 2, type = int, default = [256, 256],
                        help = 'Crop shape for raw data [128, 128]')
    parser.add_argument('--resize_shape', nargs = 2, type = int, default = [64, 64],
                        help = 'Resize shape for raw data [64, 64]')
    parser.add_argument('--resize_scale', type = float, default = None,
                        help = 'Resize scale for raw data [None]')
    parser.add_argument('--image_size', type = int, default = 64,
                        help = 'Image size to be processed [64]')

    parser.add_argument('--z_dim', type = int, default = 128,
                        help = 'z (fake seed) dimension [128]')

    parser.add_argument('--resume', type = str, default = None,
                        help = 'Learned parameter checkpoint [None]')

    return parser
