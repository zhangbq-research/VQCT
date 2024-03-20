import sys
sys.path.append(".")
import argparse
import os
from skimage import metrics
# also disable grad to save memory
import torch
import yaml
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel as VQModel 
from taming.models.vqct import VQModel as vqct
from taming.models.vqgan_cvq import VQModel as cvqgan
from taming.models.vqct import GumbelVQ as topkgumbel
import torch.nn as nn
from PIL import Image
from PIL import ImageDraw, ImageFont
import PIL
import numpy as np
from taming.modules.losses import IS
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from taming.data.custom import CustomTest
from taming.dataloader import flowers
from torch.utils.data import random_split, DataLoader, Dataset
from pytorch_fid.fid_score import calculate_fid_given_paths
import shutil
import torchvision
from tqdm import tqdm
import math
from taming.dataloader import celehq
os.environ['TORCH_HOME'] = 'pretrained_weights'

def count_PSNR_SSIM(targets,outputs):
    pic_names = os.listdir(targets)
    psnr = 0.0
    ssim = 0.0
    for pic in pic_names:
        img1 = Image.open(os.path.join(targets,pic))
        img2 = Image.open(os.path.join(outputs,pic))
        img1array = np.array(img1)
        img2array = np.array(img2)
        mse = np.mean((img1array/1.0-img2array/1.0)**2)
        psnr += 20*math.log10(255.0/math.sqrt(mse))
        img1_L = img1.convert('L')
        img2_L = img2.convert('L')
        img1_L_array = np.array(img1_L)
        img2_L_array = np.array(img2_L)
        ssim += metrics.structural_similarity(img1_L_array,img2_L_array)
    psnr = psnr/len(pic_names)
    ssim = ssim/len(pic_names)
    return psnr,ssim

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path=None, model='VQModel'):
    if model == 'VQModel':
        model = VQModel(**config.model.params)
    elif model == 'vqct':
        model = vqct(**config.model.params)
    elif model == 'cvq':
        model = cvqgan(**config.model.params)
    elif model == 'topkgumbel':
        model = topkgumbel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def reconstruct_with_vqgan(x, model):
    with torch.no_grad():
        # could also use model(x) for reconstruction but use explicit encoding and decoding here
        z, _, mes = model.encode(x)
        # print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
        xrec = model.decode(z)
    return xrec.detach(),mes

def reconstruction_pipeline(images, size=320,vqctmodel=None,save_dir=None, idx=None):
    images = images.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).to(DEVICE)
    topkgcn_x,topk_mes = reconstruct_with_vqgan(images, vqctmodel)
    loss1 = nn.L1Loss()
    loss2 = nn.MSELoss()
    topkgcnloss = loss1(images,topkgcn_x)
    topkgcnloss2 = loss2(images,topkgcn_x)
    for j in range(topkgcn_x.shape[0]):
        # save target image
        save_image_tensor2pillow(images[j:j+1], os.path.join(save_dir, 'target', '{}_{}.jpg'.format(idx, j)))

        save_image_tensor2pillow(topkgcn_x[j:j+1], os.path.join(save_dir, 'vqct', '{}_{}.jpg'.format(idx, j)))
    return topkgcnloss,topkgcnloss2


def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为pillow
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    input_tensor = torch.clamp(input_tensor, -1., 1.)
    input_tensor = input_tensor.cpu()
    input_tensor = input_tensor.squeeze()
    input_tensor = input_tensor.add_(1).mul_(0.5).mul_(255).add_(0.5).permute(1, 2, 0).type(torch.uint8).numpy()
    
    im = Image.fromarray(input_tensor)
    im.save(filename)

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-vqctc', '--vqctconfig', type=str, default='configs/cub_vqct.yaml')
    parser.add_argument('-vqctp', '--vqctckpt', type=str, default='/home/ices/taming-transformers-master/checkpoint/cub_and_coco/rec_loss=0.39.ckpt')


    parser.add_argument('-s', '--save_path', type=str, default='result')
    args = parser.parse_args()


    dataset = CustomTest(size=256, test_images_list_file='data/cub/test.txt')
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)

    torch.set_grad_enabled(False)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vqctconfig = load_config(args.vqctconfig, display=False)
    vqctmodel = load_vqgan(vqctconfig, ckpt_path=args.vqctckpt, model='vqct').to(DEVICE)


    if os.path.exists(os.path.join(args.save_path, 'target')):
        shutil.rmtree(os.path.join(args.save_path, 'target')) 
        os.mkdir(os.path.join(args.save_path, 'target'))
    else:
        os.mkdir(os.path.join(args.save_path, 'target'))


    if os.path.exists(os.path.join(args.save_path, 'vqct')):
        shutil.rmtree(os.path.join(args.save_path, 'vqct')) 
        os.mkdir(os.path.join(args.save_path, 'vqct'))
    else:
        os.mkdir(os.path.join(args.save_path, 'vqct'))
    topkgcnlossl1 = 0.0
    topkgcnlossl2 = 0.0
    for i, batch in tqdm(enumerate(dataloader)):
        _topkgcnloss,_topkgcnloss2 = reconstruction_pipeline(batch["image"], size=256, vqctmodel=vqctmodel,save_dir=args.save_path, idx=i)
        topkgcnlossl1 += _topkgcnloss
        topkgcnlossl2 += _topkgcnloss2
    topkgcnlossl1 = topkgcnlossl1/(len(dataloader))
    topkgcnlossl2 = topkgcnlossl2/(len(dataloader))
    fid_value = calculate_fid_given_paths([os.path.join(args.save_path, 'target'), os.path.join(args.save_path, 'vqct')], 
                            128, 
                            device = torch.device('cuda:4' if (torch.cuda.is_available()) else 'cpu'), dims=2048, num_workers=8)
    print('vqct FID: ', fid_value)
    psnr,ssim = count_PSNR_SSIM(os.path.join(args.save_path, 'target'),os.path.join(args.save_path, 'vqct'))
    print('vqct L1loss: ',topkgcnlossl1)
    print('vqct L2loss: ',topkgcnlossl2)
    print('vqct PSNR',psnr)
    print('vqct SSIM',ssim)





