import argparse
import os
import random
import yaml
import time
import logging
import pprint

import scipy.stats as stats
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import grad
from easydict import EasyDict

from data.train import CreateDataLoader as train_loader
from data.eval import CreateDataLoader as val_loader
from utils import create_logger, save_checkpoint, load_state, get_scheduler, AverageMeter, calculate_fid
from models.standard import *

parser = argparse.ArgumentParser(description='PyTorch Colorization Training')

parser.add_argument('--config', default='experiments/origin/config.yaml')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')


def calc_gradient_penalty(netD, real_data, fake_data, sketch_feat):
    alpha = torch.rand(config.batch_size, 1, 1, 1, device=config.device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates.requires_grad = True

    disc_interpolates = netD(interpolates, sketch_feat)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones(disc_interpolates.size(), device=config.device), create_graph=True,
                     retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * config.gpW
    return gradient_penalty


def mask_gen():
    maskS = config.image_size // 4

    mask1 = torch.cat(
        [torch.rand(1, 1, maskS, maskS).ge(X.rvs(1)[0]).float() for _ in range(config.batch_size // 2)], 0)
    mask2 = torch.cat([torch.zeros(1, 1, maskS, maskS).float() for _ in range(config.batch_size // 2)], 0)
    mask = torch.cat([mask1, mask2], 0)

    return mask.to(config.device)


def main():
    global args, config, X

    args = parser.parse_args()
    print(args)

    with open(args.config) as f:
        config = EasyDict(yaml.load(f))

    config.save_path = os.path.dirname(args.config)

    ####### regular set up
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    config.device = device

    # random seed setup
    print("Random Seed: ", config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    cudnn.benchmark = True

    ####### regular set up end


    netG = torch.nn.DataParallel(NetG(ngf=config.ngf))
    netD = torch.nn.DataParallel(NetD(ndf=config.ndf))

    netF = torch.nn.DataParallel(NetF())
    netI = torch.nn.DataParallel(NetI()).eval()
    for param in netF.parameters():
        param.requires_grad = False

    criterion_MSE = nn.MSELoss()

    fixed_sketch = torch.tensor(0, device=device).float()
    fixed_hint = torch.tensor(0, device=device).float()
    fixed_sketch_feat = torch.tensor(0, device=device).float()

    ####################
    netD = netD.to(device)
    netG = netG.to(device)
    netF = netF.to(device)
    netI = netI.to(device)
    criterion_MSE = criterion_MSE.to(device)

    # setup optimizer

    optimizerG = optim.Adam(netG.parameters(), lr=config.lr_scheduler.base_lr, betas=(0.5, 0.9))
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr_scheduler.base_lr, betas=(0.5, 0.9))

    last_iter = -1
    best_fid = 1e6

    if args.resume:
        best_fid, last_iter = load_state(args.resume, netG, netD, optimizerG, optimizerD)

    config.lr_scheduler['last_iter'] = last_iter

    config.lr_scheduler['optimizer'] = optimizerG
    lr_schedulerG = get_scheduler(config.lr_scheduler)
    config.lr_scheduler['optimizer'] = optimizerD
    lr_schedulerD = get_scheduler(config.lr_scheduler)

    tb_logger = SummaryWriter(config.save_path + '/events')
    logger = create_logger('global_logger', config.save_path + '/log.txt')
    logger.info(f'args: {pprint.pformat(args)}')
    logger.info(f'config: {pprint.pformat(config)}')

    batch_time = AverageMeter(config.print_freq)
    data_time = AverageMeter(config.print_freq)
    flag = 1
    mu, sigma = 1, 0.005
    X = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)
    i = 0
    curr_iter = last_iter + 1

    dataloader = train_loader(config)
    data_iter = iter(dataloader)

    end = time.time()
    while i < len(dataloader):
        lr_schedulerG.step(curr_iter)
        lr_schedulerD.step(curr_iter)
        current_lr = lr_schedulerG.get_lr()[0]
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in netG.parameters():
            p.requires_grad = False  # to avoid computation ft_params

        # train the discriminator Diters times
        j = 0
        while j < config.diters:
            netD.zero_grad()

            i += 1
            j += 1

            data_end = time.time()
            real_cim, real_vim, real_sim = data_iter.next()
            data_time.update(time.time() - data_end)

            real_cim, real_vim, real_sim = real_cim.to(device), real_vim.to(device), real_sim.to(device)
            mask = mask_gen()
            hint = torch.cat((real_vim * mask, mask), 1)

            # train with fake
            with torch.no_grad():
                feat_sim = netI(real_sim).detach()
                fake_cim = netG(real_sim, hint, feat_sim).detach()

            errD_fake = netD(fake_cim, feat_sim)
            errD_fake = errD_fake.mean(0).view(1)

            errD_fake.backward(retain_graph=True)  # backward on score on real

            errD_real = netD(real_cim, feat_sim)
            errD_real = errD_real.mean(0).view(1)
            errD = errD_real - errD_fake

            errD_realer = -1 * errD_real + errD_real.pow(2) * config.drift

            errD_realer.backward(retain_graph=True)  # backward on score on real

            gradient_penalty = calc_gradient_penalty(netD, real_cim, fake_cim, feat_sim)
            gradient_penalty.backward()

            optimizerD.step()

        ############################
        # (2) Update G network
        ############################

        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        for p in netG.parameters():
            p.requires_grad = True
        netG.zero_grad()

        data = data_iter.next()
        real_cim, real_vim, real_sim = data
        i += 1

        real_cim, real_vim, real_sim = real_cim.to(device), real_vim.to(device), real_sim.to(device)

        if flag:  # fix samples
            mask = mask_gen()
            hint = torch.cat((real_vim * mask, mask), 1)
            with torch.no_grad():
                feat_sim = netI(real_sim).detach()

            tb_logger.add_image('target imgs', vutils.make_grid(real_cim.mul(0.5).add(0.5), nrow=4))
            tb_logger.add_image('sketch imgs', vutils.make_grid(real_sim.mul(0.5).add(0.5), nrow=4))
            tb_logger.add_image('hint', vutils.make_grid((real_vim * mask).mul(0.5).add(0.5), nrow=4))

            fixed_sketch.resize_as_(real_sim).copy_(real_sim)
            fixed_hint.resize_as_(hint).copy_(hint)
            fixed_sketch_feat.resize_as_(feat_sim).copy_(feat_sim)

            flag -= 1

        mask = mask_gen()
        hint = torch.cat((real_vim * mask, mask), 1)

        with torch.no_grad():
            feat_sim = netI(real_sim).detach()

        fake = netG(real_sim, hint, feat_sim)

        errd = netD(fake, feat_sim)
        errG = errd.mean() * config.advW * -1
        errG.backward(retain_graph=True)
        feat1 = netF(fake)
        with torch.no_grad():
            feat2 = netF(real_cim)

        contentLoss = criterion_MSE(feat1, feat2)
        contentLoss.backward()

        optimizerG.step()
        batch_time.update(time.time() - end)

        ############################
        # (3) Report & 100 Batch checkpoint
        ############################
        curr_iter += 1

        if curr_iter % config.print_freq == 0:
            tb_logger.add_scalar('VGG MSE Loss', contentLoss.item(), curr_iter)
            tb_logger.add_scalar('wasserstein distance', errD.item(), curr_iter)
            tb_logger.add_scalar('errD_real', errD_real.item(), curr_iter)
            tb_logger.add_scalar('errD_fake', errD_fake.item(), curr_iter)
            tb_logger.add_scalar('Gnet loss toward real', errG.item(), curr_iter)
            tb_logger.add_scalar('gradient_penalty', gradient_penalty.item(), curr_iter)
            tb_logger.add_scalar('lr', current_lr, curr_iter)
            logger.info(f'Iter: [{curr_iter}/{len(dataloader)//(config.diters+1)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        f'errG {errG.item():.4f}\t'
                        f'errD {errD.item():.4f}\t'
                        f'err_D_real {errD_real.item():.4f}\t'
                        f'err_D_fake {errD_fake.item():.4f}\t'
                        f'content loss {contentLoss.item():.4f}\t'
                        f'LR {current_lr:.4f}'
                        )

        if curr_iter % config.print_img_freq == 0:
            with torch.no_grad():
                fake = netG(fixed_sketch, fixed_hint, fixed_sketch_feat)
                tb_logger.add_image('colored imgs',
                                    vutils.make_grid(fake.detach().mul(0.5).add(0.5), nrow=4),
                                    curr_iter)

        if curr_iter % config.val_freq == 0:
            fid, var = validate(netG, netI)
            tb_logger.add_scalar('fid_val', fid, curr_iter)
            tb_logger.add_scalar('fid_variance', var, curr_iter)
            logger.info(f'fid: {fid:.3f} ({var})\t')

            # remember best fid and save checkpoint
            is_best = fid < best_fid
            best_fid = min(fid, best_fid)
            save_checkpoint({
                'step': curr_iter,
                'state_dictG': netG.state_dict(),
                'state_dictD': netD.state_dict(),
                'best_fid': best_fid,
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict(),
            }, is_best, config.save_path + '/ckpt')

        end = time.time()


def validate(netG, netI):
    fids = []
    fid_value = 0
    for _ in range(3):
        fid = calculate_fid(netG, netI, val_loader(config), config, 2048)
        print('FID: ', fid)
        fid_value += fid
        fids.append(fid)
    fid_value /= 3
    return fid_value, np.var(fids)

if __name__ == '__main__':
    main()
