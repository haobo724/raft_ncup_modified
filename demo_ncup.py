import math
import sys

import tqdm

sys.path.append('core')
import cv2
import argparse
# import ipdb
import os
import glob
import numpy as np
import torch
from PIL import Image
from core.raft import RAFT

from core.raft_nc_dbl import RAFT as RAFT_NC
from core.utils import flow_viz
from core.utils.utils import InputPadder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import re


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

    def _get_kwargs(self):
        return self.__dir__()

def dictToObj(dictObj):
        if not isinstance(dictObj, dict):
            return dictObj
        d = Dict()
        for k, v in dictObj.items():
            d[k] = dictToObj(v)
        return d

def parse_data_cfg(path):
    """Parses the data configuration file"""
    print('data_cfg ： ',path)
    options = dict()
    # options['gpus'] = '0,1,2,3'
    # options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def str2int(v_str):
    temp = [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]
    return temp


def sort_humanly(v_list):
    # temp = str2int(v_list)
    # print(temp)

    return sorted(v_list, key=str2int)


def load_image(imfile):
    # img = np.array(Image.open(imfile)).astype(np.uint8)
    img = cv2.imread(imfile)
    h,w = img.shape[:2]
    # img = cv2.resize(img,(768,1024))
    if h/w <1:
        img = np.ascontiguousarray(img)
    else:
        img = np.ascontiguousarray(np.rot90(img))
    img = torch.FloatTensor(img).permute(2, 0, 1)
    return img


def load_image_list(image_files):
    images = []
    for imfile in image_files:
        images.append(load_image(imfile))

    images = torch.stack(images, dim=0)
    images = images.to(device)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]


def viz(img, flo):
    # img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    flo_norm = np.array(np.sqrt(np.power(flo[..., 0], 2) + np.power(flo[..., 1], 2))).astype(np.uint8)
    ret, flo_norm = cv2.threshold(flo_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值
    # print(np.max(flo_norm),np.min(flo_norm))
    # threshhold = 2
    # flo_norm = np.array(flo_norm > threshhold, dtype=int) * 255
    # flo_norm_rgb = np.stack([flo_norm for _ in range(3)], axis=-1)
    return flo_norm

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo_norm_rgb], axis=0)
    # cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    # cv2.waitKey(1)


def demo_ncup(args):
    # if torch.cuda.is_available():
    #     map_location = lambda storage, loc: storage.cuda()
    # else:
    #     map_location = 'cpu'
    map_location = "cuda:0"
    # map_location = 'cpu'
    model = torch.nn.DataParallel(RAFT_NC(args))
    # model = RAFT(args)
    # print(model)
    # ipdb.set_trace()
    # a=torch.load(path, map_location=map_location)
    state_dict = torch.load(args.restore_ckpt)

    model.load_state_dict(state_dict)
    # ipdb.set_trace()
    # print(torch.cuda.is_available)
    # model.load_state_dict(torch.load(args.restore_ckpt, map_location=map_location))
    # for root , path ,file in os.walk(args.datapath):


    model = model.module
    model.to(device)
    model.eval()

    dirs = os.listdir(args.datapath)
    for dir in dirs:
        output_folder = os.path.join('output',dir)


        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            print(output_folder, 'is exist')
            continue
        input_folder = os.path.join(args.datapath,dir)
        images_name = glob.glob(os.path.join(input_folder, '*.png')) + \
                      glob.glob(os.path.join(input_folder, '*.jpg'))
        images_name = sort_humanly(images_name)
        images = load_image_list(images_name)
        print(input_folder)
        with torch.no_grad():
            # ipdb.set_trace()


            for i, name in tqdm.tqdm(zip ( range(images.shape[0] - 1),images_name[1:]),total=len(images_name)-1):
                image1 = images[i, None]
                image2 = images[i + 1, None]
                name = 'mask_' + name.split('\\')[-1]

                save_name = os.path.join(output_folder, name)

                flow_low, flow_up = model(image1, image2, iters=10, test_mode=True)

                mask = viz(image1, flow_up)
                cv2.imwrite(save_name, mask)
                del image1
                del image2
                del mask
        del images

if __name__ == '__main__':

    args = parse_data_cfg('config.data')#返回训练配置参数，类型：字典
    for k,v in args.items():
        if v == 'True':
            args[k] = True
            continue
        elif v == 'False':
            args[k] = False
            continue
        result=re.findall('([0-9])', v)
        if len(result)==1:
            args[k]=int(result[0])

        result=re.findall(']', v)
        if len(result) :
            temp =re.findall('([0-9]+)', v)
            int_list = [int(x) for x in temp]
            args[k]=int_list

    args= dictToObj(args)

    demo_ncup(args)

    # print(args.final_upsampling)
    # input()



    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', help="model name")
    # parser.add_argument('--restore_ckpt', help="restore checkpoint")
    # parser.add_argument('--datapath', help="data used for inference")
    # parser.add_argument('--dataset', help="dataset for evaluation")
    # parser.add_argument('--small', action='store_true', help='use small model')
    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--upsampler_bi', action='store_true', help='use bilinear upsampling')
    # parser.add_argument('--align_corners', action='store_true', help='align_corners for bilinear upsampling')
    # parser.add_argument('--load_pretrained', default=None, help='freeze the optical flow network and train only nc')
    # parser.add_argument('--freeze_raft', action='store_true', help='freeze the optical flow network and train only nc')
    # parser.add_argument('--compressed_ft', action='store_true', help='load the compressed version of FlyingThings3D')
    #
    # from core.utils.args import _add_arguments_for_module, str2bool, str2intlist
    # from core import upsampler
    #
    # _add_arguments_for_module(
    #     parser,
    #     upsampler,
    #     name="final_upsampling",
    #     default_class=None,
    #     exclude_classes=["_*"],
    #     exclude_params=["self", "args", "interpolation_net", "weights_est_net", "size"],
    #     forced_default_types={"scale": int,
    #                           "use_data_for_guidance": str2bool,
    #                           "channels_to_batch": str2bool,
    #                           "use_residuals": str2bool,
    #                           "est_on_high_res": str2bool},
    # )
    #
    # from core import nconv_modules
    #
    # _add_arguments_for_module(
    #     parser,
    #     nconv_modules,
    #     name="interp_net",
    #     default_class=None,
    #     exclude_classes=["_*"],
    #     exclude_params=["self", "args"],
    #     forced_default_types={"encoder_fiter_sz": int,
    #                           "decoder_fiter_sz": int,
    #                           "out_filter_size": int,
    #                           "use_double_conv": str2bool,
    #                           "use_bias": str2bool}
    # )
    #
    # import interp_weights_est
    #
    # _add_arguments_for_module(
    #     parser,
    #     interp_weights_est,
    #     name="weights_est_net",
    #     default_class=None,
    #     exclude_classes=["_*"],
    #     exclude_params=["self", "args", "out_ch", "final_act"],
    #     unknown_default_types={"num_ch": str2intlist,
    #                            "filter_sz": str2intlist},
    #     forced_default_types={"dilation": str2intlist,
    #                           }
    # )
    #
    # args = parser.parse_args()


