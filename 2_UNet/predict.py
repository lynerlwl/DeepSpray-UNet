import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import glob
from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                file_name,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():

        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        probs = probs.cpu().numpy().transpose((1, 2, 0))
        mask = np.argmax(probs, axis=2)
        Image.fromarray((mask).astype(np.uint8)).save(f"1_Data/predicted/{file_name}")
        print(f"Printed image {file_name}")

        if file_name == 560: 
          sys.exit("end of image")

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=5, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    file_list = [f"{args.input[0]}/{name}" for name in os.listdir(args.input[0]) if not name.startswith('.')] 

    for i, filename in enumerate(file_list):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename).convert('RGB')
        img_name = filename.split('/')[-1]
        # print(img_name)

        mask = predict_img(net=net,
                           full_img=img,
                           file_name=img_name,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)


