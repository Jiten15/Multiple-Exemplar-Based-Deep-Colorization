from __future__ import print_function

import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform_lib
from PIL import Image
from tqdm import tqdm

import lib.TestTransforms as transforms
from models.ColorVidNet import ColorVidNet
from models.FrameColor import frame_colorization
from models.NonlocalNet import VGG19_pytorch, WarpNet
from utils.util import (batch_lab2rgb_transpose_mc, folder2vid, mkdir_if_not,
                        save_frames, tensor_lab2rgb, uncenter_l)
from utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=[216 * 2, 384 * 2], help="the image size, eg. [216,384]")
    parser.add_argument("--cuda", action="store_false")
    parser.add_argument("--gpu_ids", type=str, default="0", help="separate by comma")
    parser.add_argument("--clip_path", type=str, default="./exp_sample/target", help="path of input clips")
    parser.add_argument("--ref_path", type=str, default="./exp_sample/references", help="path of refernce images")
    parser.add_argument("--output_path", type=str, default="./exp_sample/output", help="path of output clips")
    opt = parser.parse_args()
    opt.gpu_ids = [int(x) for x in opt.gpu_ids.split(",")]
    cudnn.benchmark = True
    print("running on GPU", opt.gpu_ids)

    nonlocal_net = WarpNet(1)
    colornet = ColorVidNet(7)
    vggnet = VGG19_pytorch()
    vggnet.load_state_dict(torch.load("data/vgg19_conv.pth"))
    for param in vggnet.parameters():
        param.requires_grad = False

    nonlocal_test_path = os.path.join("checkpoints/", "video_moredata_l1/nonlocal_net_iter_76000.pth")
    color_test_path = os.path.join("checkpoints/", "video_moredata_l1/colornet_iter_76000.pth")
    print("succesfully load nonlocal model: ", nonlocal_test_path)
    print("succesfully load color model: ", color_test_path)
    nonlocal_net.load_state_dict(torch.load(nonlocal_test_path))
    colornet.load_state_dict(torch.load(color_test_path))

    nonlocal_net.eval()
    colornet.eval()
    vggnet.eval()
    nonlocal_net.cuda()
    colornet.cuda()
    vggnet.cuda()

    targets = os.listdir(opt.clip_path)
    targets.sort()
    targets.remove(".ipynb_checkpoints")

    references = os.listdir(opt.ref_path)
    references.sort()
    references.remove(".ipynb_checkpoints")

    wls_filter_on = True
    lambda_value = 500
    sigma_color = 4

    transform = transforms.Compose([CenterPad(opt.image_size), transform_lib.CenterCrop(opt.image_size), RGB2Lab(), ToTensor(), Normalize()])

    for target in targets:
        frame1 = Image.open(os.path.join(opt.clip_path, target))
        IA_lab_large = transform(frame1).unsqueeze(0).cuda()
        IA_lab = torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear")

        IA_l = IA_lab[:, 0:1, :, :]
        IA_ab = IA_lab[:, 1:3, :, :]
        I_last_lab_predict = None

        for reference in references:
           IB = Image.open(os.path.join(opt.ref_path, reference)) 
           IB_lab_large = transform(IB).unsqueeze(0).cuda()
           IB_lab = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear")
           IB_l = IB_lab[:, 0:1, :, :]
           IB_ab = IB_lab[:, 1:3, :, :]
           with torch.no_grad():
            I_reference_lab = IB_lab
            I_reference_l = I_reference_lab[:, 0:1, :, :]
            I_reference_ab = I_reference_lab[:, 1:3, :, :]
            I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))
            features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)

            if I_last_lab_predict is None:
                I_last_lab_predict = IB_lab

            with torch.no_grad():
              I_current_lab = IA_lab
            I_current_ab_predict, I_current_nonlocal_lab_predict, features_current_gray = frame_colorization(
                I_current_lab,
                I_reference_lab,
                I_last_lab_predict,
                features_B,
                vggnet,
                nonlocal_net,
                colornet,
                feature_noise=0,
                temperature=1e-10,
            )
            I_last_lab_predict = torch.cat((IA_l, I_current_ab_predict), dim=1)

            # upsampling
            curr_bs_l = IA_lab_large[:, 0:1, :, :]
            curr_predict = (torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear") * 1.25)

            # filtering
            if wls_filter_on:
                guide_image = uncenter_l(curr_bs_l) * 255 / 100
                wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
                    guide_image[0, 0, :, :].cpu().numpy().astype(np.uint8), lambda_value, sigma_color
                )
                curr_predict_a = wls_filter.filter(curr_predict[0, 0, :, :].cpu().numpy())
                curr_predict_b = wls_filter.filter(curr_predict[0, 1, :, :].cpu().numpy())
                curr_predict_a = torch.from_numpy(curr_predict_a).unsqueeze(0).unsqueeze(0)
                curr_predict_b = torch.from_numpy(curr_predict_b).unsqueeze(0).unsqueeze(0)
                curr_predict_filter = torch.cat((curr_predict_a, curr_predict_b), dim=1)
                IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict_filter[:32, ...])
            else:
                IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict[:32, ...])

            save_frames(IA_predict_rgb, opt.output_path, reference)
        






