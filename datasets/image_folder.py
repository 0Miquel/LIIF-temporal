import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from datasets import register

@register('video-image-folder')
class VideoImageFolder(Dataset):
    def __init__(self, root_path):
        self.files = []
        folders = sorted(os.listdir(root_path))
        for folder in folders:
            filenames = sorted(os.listdir(os.path.join(root_path, folder)))
            frames = []
            for filename in filenames:
                file = os.path.join(root_path, folder, filename)
                frames.append(transforms.ToTensor()(Image.open(file).convert('RGB')))
            self.files.append(torch.stack(frames).permute(1,0,2,3))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx]

@register('vimeo-image-folder')
class VimeoImageFolder(Dataset):
    def __init__(self, root_path):
        self.files = []
        folders = sorted(os.listdir(root_path))
        for folder in folders:
            sequences = sorted(os.listdir(os.path.join(root_path, folder)))
            for sequence in sequences:
                path = os.path.join(root_path, folder, sequence)
                self.files.append(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sequence_path = self.files[idx]
        frames = sorted(os.listdir(sequence_path))
        sequence = []
        for frame in frames:
            file = os.path.join(sequence_path, frame)
            sequence.append(transforms.ToTensor()(Image.open(file).convert('RGB')))

        return torch.stack(sequence).permute(1,0,2,3)


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]

"""if __name__ == "__main__":
    data = VideoImageFolder("../load/train-temporal/train")
    loader = DataLoader(data, batch_size=2)
    print(loader)
    print(data)"""