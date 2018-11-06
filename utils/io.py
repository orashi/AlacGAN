import os
import logging
import shutil
import torch
import numpy as np


def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')


def load_state(path, netG, netD, optimizerG, optimizerD):
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)
        netG.load_state_dict(checkpoint['state_dictG'], strict=False)
        netD.load_state_dict(checkpoint['state_dictD'], strict=False)
        best_fid = checkpoint['best_fid']
        last_iter = checkpoint['step']
        print("=> also loaded optimizer from checkpoint '{}' (iter {})".format(path, last_iter))
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])

        ckpt_keys = set(checkpoint['state_dictG'].keys())
        own_keys = set(netG.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print(f'caution: missing netG keys from checkpoint {path}: {k}')

        ckpt_keys = set(checkpoint['state_dictD'].keys())
        own_keys = set(netD.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print(f'caution: missing netD keys from checkpoint {path}: {k}')

        return best_fid, last_iter
    else:
        print("=> no checkpoint found at '{}'".format(path))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count
