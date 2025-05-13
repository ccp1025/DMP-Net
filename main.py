import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import time
import torch
import argparse
from func import *
from numpy import *
import scipy.io as sio
from model import admm_denoise
from torchmetrics import SpectralAngleMapper
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random.seed(5)

parser = argparse.ArgumentParser()
parser.add_argument('--method', default='ADMM', help="Select GAP or ADMM")
parser.add_argument('--lambda_', default=1, help="_lambda is the regularization factor")
parser.add_argument('--denoiser', default='TV', help="Select which denoiser: Total Variation (TV)")
parser.add_argument('--accelerate', default=True, help="Acclearted version of GAP or not")
parser.add_argument('--iter_max', default=50, help="Maximum number of iterations")
parser.add_argument('--tv_weight', default=24, help="TV denoising weight (larger for smoother but slower)")
parser.add_argument('--tv_iter_max', default=25, help="TV denoising maximum number of iterations each")
parser.add_argument('--x0', default=None, help="The initialization data point")
parser.add_argument('--sigma', default=[0, 10, 30, 50, 70, 100], help="The noise levels")
args = parser.parse_args()
#------------------------------------------------------------------#

#----------------------- Data Configuration -----------------------#

data_truth = torch.from_numpy((sio.loadmat(r'.\dataset\org_dataset\scene22.mat')['scene1']))
step, beta = 2, 0.9
h, w, nC = data_truth.size()

data_truth_shift = torch.zeros((h, w + step*(nC - 1), nC))
for i in range(nC):
    data_truth_shift[:, i*step:i*step+w, i] = data_truth[:, :, i]
data_truth_shift = data_truth_shift * beta

dataset_dir3  = r'.\dataset\CNN_PRM\scene22.mat'
ref_img  = torch.from_numpy((sio.loadmat(dataset_dir3)['scene1'])/255)

ref_img = ref_img * (1 - beta)

#----------------------- Mask Configuration -----------------------#
mask = torch.zeros((h, w + step*(nC - 1)))
mask_3d = torch.unsqueeze(mask, 2).repeat(1, 1, nC)
mask_256 = torch.from_numpy((sio.loadmat(r'.\dataset\mask\mask22.mat')['mask'])/255)
for i in range(nC):
    mask_3d[:, i*step:i*step+w, i] = mask_256
Phi = mask_3d
meas = torch.sum(Phi * data_truth_shift, 2)

#------------------------------------------------------------------#

begin_time = time.time()
if args.method == 'GAP':
    pass
        
elif args.method == 'ADMM':
    recon, psnr_all = admm_denoise(meas.to(device), Phi.to(device), data_truth.to(device), ref_img.to(device), args)
    end_time = time.time()
    recon = shift_back(recon, step=2)
    
    sam = SpectralAngleMapper()
    vrecon = recon.double().cpu()
    sam = sam(torch.unsqueeze(vrecon.permute(2, 0, 1), 0).double(), torch.unsqueeze(data_truth.permute(2, 0, 1), 0).double())
    print('ADMM, SAM {:2.3f}, running time {:.1f} seconds.'.format(sam, end_time - begin_time))

sio.savemat(r'.\result\result1.mat', {'img':recon.cpu().numpy()})

