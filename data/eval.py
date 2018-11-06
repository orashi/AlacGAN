from __future__ import division

import os
import os.path
import random
import numbers
import numpy as np
from scipy.misc import fromimage
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root):
    images = []

    for _, __, fnames in sorted(os.walk(root)):
        for fname in fnames:
            if is_image_file(fname):
                images.append(fname)
    return images


def sketch_loader(path):
    return Image.open(path).convert('L')


def resize_by(img, side_min):
    return img.resize((int(img.size[0] / min(img.size) * side_min),
                       int(img.size[1] / min(img.size) * side_min)),
                      Image.BICUBIC)


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

    def __call__(self, img1):
        w, h = img1.size
        th, tw = self.size
        if w == tw and h == th:  # ValueError: empty range for randrange() (0,0, 0)
            return img1

        if w == tw:
            x1 = 0
            y1 = random.randint(0, h - th)
            return img1.crop((x1, y1, x1 + tw, y1 + th))

        elif h == th:
            x1 = random.randint(0, w - tw)
            y1 = 0
            return img1.crop((x1, y1, x1 + tw, y1 + th))

        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            return img1.crop((x1, y1, x1 + tw, y1 + th))


class ImageFolder(data.Dataset):
    def __init__(self, root, stransform=None):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in folders."))
        self.root = root
        self.imgs = imgs
        self.stransform = stransform

    def __getitem__(self, index):
        fname = self.imgs[index]  # random.randint(1, 3
        Simg = sketch_loader(os.path.join(self.root, fname))
        Simg = resize_by(Simg, 512.0)
        if random.random() < 0.5:
            Simg = Simg.transpose(Image.FLIP_LEFT_RIGHT)
        Simg = self.stransform(Simg)

        return Simg

    def __len__(self):
        return len(self.imgs)


def CreateDataLoader(config):
    random.seed(config.seed)

    STrans = transforms.Compose([
        RandomCrop(512),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(root=config.val_root, stransform=STrans)

    assert dataset

    return data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=10, drop_last=False)