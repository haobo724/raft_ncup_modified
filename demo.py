# import sys
# sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from core.raft_nc_dbl import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def load_image_list(image_files):
    images = []
    for imfile in sorted(image_files):
        images.append(load_image(imfile))
 
    images = torch.stack(images, dim=0)
    images = images.to(device)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]
        

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def demo(args):
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    model = torch.nn.DataParallel(RAFT(args))
    checkpoint = torch.load(args.model, map_location=map_location)
    # model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint)

    model = model.module
    model.to(device)
    model.eval()

    with torch.no_grad():
        print(os.path.join)
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = load_image_list(images)
        for i in range(images.shape[0]-1):
            image1 = images[i, None]
            image2 = images[i+1, None]

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()


    # parser.add_argument('--model', help="restore checkpoint")
    # parser.add_argument('--path', help="dataset for evaluation")
    # parser.add_argument('--small', action='store_true', help='use small model')
    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # args = parser.parse_args()

    demo(args)
