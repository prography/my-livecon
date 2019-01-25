import numpy as np
import torch
import os, sys, argparse
import cv2

from hairstyle_classifier.networks import get_network
import torchvision.transforms as T

def str2bool(s):
    return s.lower() in ('t', 'true', 1)

def hair_segmentation_main(frame, result_filename):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', help='path to ckpt file',type=str,
            default='hairstyle_classifier/models/pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth')
    parser.add_argument('--networks', help='name of neural network', type=str, default='pspnet_resnet101')
    parser.add_argument('--use_gpu', type=str2bool, default=False,
            help='True if using gpu during inference')

    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    network = args.networks.lower()
    device = 'cuda' if args.use_gpu else 'cpu'

    assert os.path.exists(ckpt_dir)

    # prepare network with trained parameters
    net = get_network(network).to(device)
    state = torch.load(ckpt_dir, map_location=lambda storage, loc: storage)
    net.load_state_dict(state['weight'])
    net.eval()

    # image transform
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # generate tensor from image frame
    data = transforms(frame).to(device).unsqueeze(0)

    # inference (forwarding)
    logit = net(data)

    # prepare mask
    pred = torch.sigmoid(logit.cpu())[0][0].data.numpy()

    mh, mw = data.size(2), data.size(3)
    mask = pred < 0.5

    mask_n = np.zeros((mh, mw, 3))
    mask_n[:, :, :] = 255
    mask_n[:, :, 0] *= mask
    mask_n[:, :, 1] *= mask
    mask_n[:, :, 2] *= mask

    # discard padded area
    ih, iw, _ = frame.shape

    delta_h = mh - ih
    delta_w = mw - iw

    top = delta_h // 2
    bottom = mh - (delta_h - top)
    left = delta_w // 2
    right = mw - (delta_w - left)

    mask_n = mask_n[top:bottom, left:right, :]

    # addWeighted
    image_n = frame + mask_n
    cv2.imwrite(result_filename, image_n)
    return result_filename

if __name__ == '__main__':
    frame = cv2.imread('../assets/images/SmallEyes.jpeg')
    hair_segmentation_main(frame)