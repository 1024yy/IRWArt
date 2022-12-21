
import datetime
import math
import os.path
import warnings
from collections import OrderedDict

import albumentations as albu
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.optim
from lightly.loss import NegativeCosineSimilarity
from tensorboardX import SummaryWriter
from tqdm import tqdm

import datasets
import modules.Unet_common as common
import viz
from Simsiam import SimSiam
from model import *

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transform = albu.Compose([
    albu.Resize(224, 224),
    albu.JpegCompression(quality_lower=1, quality_upper=100, p=0.5),
    albu.GaussianBlur(p=0.5),
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
    albu.GaussNoise(p=0.5),
    albu.ColorJitter(p=0.5),

    albu.Normalize()])


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)


# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def computePSNR(origin, pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


#####################
# Model initialize: #
#####################
num_ftrs = 24
out_dim = proj_hidden_dim = 24
pred_hidden_dim = 12
net = Model()
net.cuda()
init_model(net)
model = SimSiam(net, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim)
model = torch.nn.DataParallel(model, device_ids=c.device_ids)
para = get_parameter_number(model)
print(para)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))

optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


dwt = common.DWT()
iwt = common.IWT()

if c.tain_next:
    load(c.MODEL_PATH + c.suffix)

try:
    writer = SummaryWriter(comment='irwart', filename_suffix="steg")
    best_psnr = 0
    save_path = r'D:\code2.0\IRWArt\model'
    nowtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    model_save_path = os.path.join(save_path, nowtime)
    os.makedirs(model_save_path, exist_ok=True)
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('psnr_s', []),
        ('psnr_c', []),
        ('psnr_rec', [])])
    avg_meters = {'loss': AverageMeter()}
    for i_epoch in range(c.epochs):
        i_epoch = i_epoch + c.trained_epoch + 1
        loss_history = []

        #################
        #     train:    #
        #################
        pbar = tqdm(total=len(datasets.trainloader))
        for i_batch, data in enumerate(datasets.trainloader):
            cover = data[0]
            secret = data[1]
            cover = cover.to(device)
            secret = secret.to(device)
            cover_input = dwt(cover)
            secret_input = dwt(secret)

            input_img = torch.cat((cover_input, secret_input), 1)
            secret_zero = torch.zeros_like(input_img)
            input_img1 = torch.cat((cover_input, secret_zero), 1)
            #################
            #    forward:   #
            #################
            criterion_ssl = NegativeCosineSimilarity()
            z0, p0, output = model(input_img, rev=False)
            z1, p1, _ = model(input_img1, rev=False)
            loss_ssl = 0.5 * (criterion_ssl(z0, p1) + criterion_ssl(z1, p0))
            output_steg = output.narrow(1, 0, 4 * c.channels_in)
            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
            steg_img = iwt(output_steg)
            # steg_img_noise = output_steg.cpu().detach().numpy()
            # for i in range(c.batch_size):
            #     steg_img_i = steg_img_noise[i].transpose(2, 1, 0)
            #     noise = train_transform(image=steg_img_i)
            #     steg_img_noise[i] = noise['image'].transpose(2, 0, 1)
            # steg_img_noise = torch.from_numpy(steg_img_noise)
            #################
            #   backward:   #
            #################

            output_z_guass = gauss_noise(output_z.shape)

            output_rev = torch.cat((output_steg, output_z_guass), 1)
            _, _, output_image = model(output_rev, rev=True)

            secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
            secret_rev = iwt(secret_rev)
            cover_rev = output_image.narrow(1, 0, 4 * c.channels_in)
            cover_rev = iwt(cover_rev)
            resi_secret = (secret_rev - secret) * 20
            #################
            #     loss:     #
            #################
            g_loss = guide_loss(steg_img.cuda(), cover.cuda())
            r_loss = reconstruction_loss(secret_rev, secret)
            steg_low = output_steg.narrow(1, 0, c.channels_in)
            cover_low = cover_input.narrow(1, 0, c.channels_in)
            l_loss = low_frequency_loss(steg_low, cover_low)

            cover_loss = reconstruction_loss(cover, cover_rev)
            re_inconsist_loss = reconstruction_loss(secret, resi_secret)
            total_loss = c.lamda_reconstruction * r_loss + c.lamda_guide * g_loss + c.lamda_low_frequency * l_loss + \
                         loss_ssl + re_inconsist_loss + cover_loss
            total_loss.backward()
            optim.step()
            optim.zero_grad()

            loss_history.append([total_loss.item(), 0.])
            avg_meters['loss'].update(total_loss.item(), input_img.size(0))
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        epoch_losses = np.mean(np.array(loss_history), axis=0)
        epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])
        pbar.close()
        #################
        #     val:    #
        #################
        if i_epoch % c.val_freq == 0:
            with torch.no_grad():
                psnr_s = []
                psnr_c = []
                psnr_rec = []
                model.eval()
                pbar = tqdm(total=len(datasets.testloader))
                for x in datasets.testloader:
                    cover = x[0]
                    secret = x[1]
                    cover = cover.to(device)
                    secret = secret.to(device)
                    cover_input = dwt(cover)
                    secret_input = dwt(secret)

                    input_img = torch.cat((cover_input, secret_input), 1)

                    #################
                    #    forward:   #
                    #################
                    z0, p0, output = model(input_img, rev=False)
                    output_steg = output.narrow(1, 0, 4 * c.channels_in)
                    steg = iwt(output_steg)
                    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                    output_z = gauss_noise(output_z.shape)

                    #################
                    #   backward:   #
                    #################
                    output_steg = output_steg.cuda()
                    output_rev = torch.cat((output_steg, output_z), 1)
                    _, _, output_image = model(output_rev, rev=True)
                    secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
                    secret_rev = iwt(secret_rev)
                    cover_rev = output_image.narrow(1, 0, 4 * c.channels_in)
                    cover_rev = iwt(cover_rev)
                    secret_rev = secret_rev.cpu().numpy().squeeze() * 255
                    np.clip(secret_rev, 0, 255)
                    secret = secret.cpu().numpy().squeeze() * 255
                    np.clip(secret, 0, 255)
                    cover = cover.cpu().numpy().squeeze() * 255
                    np.clip(cover, 0, 255)
                    steg = steg.cpu().numpy().squeeze() * 255
                    np.clip(steg, 0, 255)
                    cover_rev = cover_rev.cpu().numpy().squeeze() * 255
                    np.clip(cover_rev, 0, 255)
                    psnr_temp = computePSNR(secret_rev, secret)
                    psnr_s.append(psnr_temp)
                    psnr_temp_c = computePSNR(cover, steg)
                    psnr_c.append(psnr_temp_c)
                    psnr_temp_rev = computePSNR(cover, cover_rev)
                    psnr_rec.append(psnr_temp_rev)
                    pbar.update(1)
                writer.add_scalars("PSNR_S", {"average psnr": np.mean(psnr_s)}, i_epoch)
                writer.add_scalars("PSNR_C", {"average psnr": np.mean(psnr_c)}, i_epoch)
                writer.add_scalars("PSNR_REC", {"average psnr": np.mean(psnr_rec)}, i_epoch)

                log['epoch'].append(i_epoch)
                log['lr'].append(epoch_losses[1])
                log['loss'].append(epoch_losses[0])
                log['psnr_s'].append(np.mean(psnr_s))
                log['psnr_c'].append(np.mean(psnr_c))
                log['psnr_rec'].append(np.mean(psnr_rec))
                pd.DataFrame(log).to_csv(os.path.join(model_save_path, 'log.csv'), index=False)
                pbar.close()
        # viz.show_loss(epoch_losses)
        print("==========================================================")
        print('Eopch %d - Loss %.4f - lr %.4f' % (i_epoch, epoch_losses[0], epoch_losses[1]))
        print('Evaluation PSNR_S %.4f - PSNR_C %.4f - PSNR_REC %.4f' % (np.mean(psnr_s), np.mean(psnr_c), np.mean(psnr_rec)))
        writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)
        if (np.mean(psnr_s) + np.mean(psnr_c) + np.mean(psnr_rec)) / 3 > best_psnr:
            print("=> saved best model")
            # print('Best PSNR_S %.4f - Best PSNR_C %.4f' % (np.mean(psnr_s), np.mean(psnr_c)))
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, os.path.join(model_save_path, 'model_best_psnr.pt'))
            best_psnr = (np.mean(psnr_s) + np.mean(psnr_c) + np.mean(psnr_rec)) / 3
            print('Best PSNR_S %.4f - PSNR_C %.4f - PSNR_REC %.4f' % (
                np.mean(psnr_s), np.mean(psnr_c), np.mean(psnr_rec)))
        if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i' % i_epoch + '.pt')
        weight_scheduler.step()

    # torch.save({'opt': optim.state_dict(),
    #             'net': net.state_dict()}, c.MODEL_PATH + 'model' + '.pt')
    writer.close()

except:
    if c.checkpoint_on_error:
        torch.save({'opt': optim.state_dict(),
                    'net': net.state_dict()}, c.MODEL_PATH + 'model_ABORT' + '.pt')
    raise

finally:
    viz.signal_stop()
