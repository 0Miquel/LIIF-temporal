import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./input')
    parser.add_argument('--model', default='pths/temporal-1d-3346.pth')
    #parser.add_argument('--resolution')
    parser.add_argument('--output', default='./output')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    ####### VIDEO ########
    root_path = args.input
    filenames = sorted(os.listdir(root_path))
    frames = []
    for filename in filenames:
        file = os.path.join(root_path, filename)
        frames.append(transforms.ToTensor()(Image.open(file).convert('RGB')))
    video = torch.stack(frames).permute(1, 0, 2, 3)

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    #t, h, w = list(map(int, args.resolution.split(',')))
    t = video.shape[1] * 2
    h = video.shape[2]
    w = video.shape[3]
    coord = make_coord((t, h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / t
    cell[:, 1] *= 2 / h
    cell[:, 2] *= 2 / w
    #coord = torch.stack(sorted(coord, key=lambda x: (x[1], x[2])))
    #cell = torch.stack(sorted(cell, key=lambda x: (x[1], x[2])))
    pred = \
    batched_predict(model, ((video - 0.5) / 0.5).cuda().unsqueeze(0), coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(t, h, w, 3).permute(0,3,1,2).cpu()

    output_path = args.output
    for i, img in enumerate(pred):
        transforms.ToPILImage()(img).save(output_path+str(i).zfill(3)+".png")
