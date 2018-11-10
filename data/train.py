from __future__ import division

import math
import numbers
import os
import os.path
import random
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Scale, CenterCrop

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img1, img2):
        w, h = img1.size
        th, tw = self.size
        if w == tw and h == th:  # ValueError: empty range for randrange() (0,0, 0)
            return img1, img2

        if w == tw:
            x1 = 0
            y1 = random.randint(0, h - th)
            return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))

        elif h == th:
            x1 = random.randint(0, w - tw)
            y1 = 0
            return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))

        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))


class RandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.9, 1.) * area
            aspect_ratio = random.uniform(7. / 8, 8. / 7)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root):
    images = []

    for _, __, fnames in sorted(os.walk(os.path.join(root, 'color'))):
        for fname in fnames:
            if is_image_file(fname):
                images.append(fname)
    return images


def color_loader(path):
    return Image.open(path).convert('RGB')


def sketch_loader(path):
    return Image.open(path).convert('L')


# class DistributedSampler(Sampler):
#     """Sampler that restricts data loading to a subset of the dataset.
#
#     It is especially useful in conjunction with
#     :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
#     process can pass a DistributedSampler instance as a DataLoader sampler,
#     and load a subset of the original dataset that is exclusive to it.
#
#     .. note::
#         Dataset is assumed to be of constant size.
#
#     Arguments:
#         dataset: Dataset used for sampling.
#         world_size (optional): Number of processes participating in
#             distributed training.
#         rank (optional): Rank of the current process within world_size.
#     """
#
#     def __init__(self, dataset, round_up=True):
#         self.dataset = dataset
#         self.round_up = round_up
#         self.epoch = 0
#
#         self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.world_size))
#         if self.round_up:
#             self.total_size = self.num_samples * self.world_size
#         else:
#             self.total_size = len(self.dataset)
#
#     def __iter__(self):
#         # deterministically shuffle based on epoch
#         g = torch.Generator()
#         g.manual_seed(self.epoch)
#         indices = list(torch.randperm(len(self.dataset), generator=g))
#
#         # add extra samples to make it evenly divisible
#         if self.round_up:
#             indices += indices[:(self.total_size - len(indices))]
#         assert len(indices) == self.total_size
#
#         # subsample
#         offset = self.num_samples * self.rank
#         indices = indices[offset:offset + self.num_samples]
#         if self.round_up or (not self.round_up and self.rank < self.world_size - 1):
#             assert len(indices) == self.num_samples
#
#         return iter(indices)
#
#     def __len__(self):
#         return self.num_samples
#
#     def set_epoch(self, epoch):
#         self.epoch = epoch


class GivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, diter, last_iter=-1):
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.diter = diter
        self.last_iter = last_iter

        self.total_size = self.total_iter * self.batch_size * (self.diter + 1)

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter + 1) * self.batch_size * (self.diter + 1):])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):
        # each process shuffle all list with same seed
        np.random.seed(0)

        indices = np.arange(len(self.dataset))
        indices = indices[:self.total_size]
        num_repeat = (self.total_size - 1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:self.total_size]

        np.random.shuffle(indices)

        assert len(indices) == self.total_size
        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        # return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, vtransform=None, stransform=None):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in folders."))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.vtransform = vtransform
        self.stransform = stransform

    def __getitem__(self, index):
        fname = self.imgs[index]
        Cimg = color_loader(os.path.join(self.root, 'color', fname))
        Simg = sketch_loader(os.path.join(self.root, str(random.randint(0, 2)), fname))
        Cimg, Simg = RandomCrop(512)(Cimg, Simg)
        if random.random() < 0.5:
            Cimg, Simg = Cimg.transpose(Image.FLIP_LEFT_RIGHT), Simg.transpose(Image.FLIP_LEFT_RIGHT)

        Cimg, Vimg, Simg = self.transform(Cimg), self.vtransform(Cimg), self.stransform(Simg)

        return Cimg, Vimg, Simg

    def __len__(self):
        return len(self.imgs)


def CreateDataLoader(config):
    random.seed(config.seed)

    # folder dataset
    CTrans = transforms.Compose([
        transforms.Scale(config.image_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    VTrans = transforms.Compose([
        RandomSizedCrop(config.image_size // 4, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def jitter(x):
        ran = random.uniform(0.7, 1)
        return x * ran + 1 - ran

    STrans = transforms.Compose([
        transforms.Scale(config.image_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Lambda(jitter),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = ImageFolder(root=config.train_root, transform=CTrans, vtransform=VTrans, stransform=STrans)

    assert train_dataset

    train_sampler = GivenIterationSampler(train_dataset, config.lr_scheduler.max_iter, config.batch_size, config.diters, last_iter=config.lr_scheduler.last_iter)

    return data.DataLoader(train_dataset, batch_size=config.batch_size,
                           shuffle=False, pin_memory=True, num_workers=int(config.workers), sampler=train_sampler)
