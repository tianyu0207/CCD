import os
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image
from skimage import io
import torch.nn as nn
from os import listdir
from torch.utils.data import Dataset
from os import listdir
# import transformations as ts
# import abc
# import itertools
import numpy as np
# from imgaug import augmenters as iaa
# from keras.preprocessing.image import apply_affine_transform
# import cv2
# import albumentations as A
from skimage.util import random_noise

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class Cutout(nn.Module):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, length):
        super(Cutout, self).__init__()
        self.n_holes= [0, 1, 2, 3]
        self.length = length

    def forward(self, img, aug_index=None):
        n_holes = self.n_holes[aug_index]
        output = self._cutout(img, n_holes)
        return output

    def _cutout(self, img, n_holes):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        img = torch.squeeze(img ,dim=0)

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(n_holes):
            y = np.random.randint(h-self.length)
            x = np.random.randint(w-self.length)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return torch.unsqueeze(img ,dim=0)

class Gaussian_noise(nn.Module):
    def __init__(self):
        super(Gaussian_noise, self).__init__()


    def forward(self, img,  aug_index=None):

        if aug_index is None:
            print('please provide aug_index')
        img = img.squeeze(dim=0)
        img = img.detach().cpu().numpy()
        img = NormalizeData(img)
        img = torch.from_numpy(img)

        # noise = self.noise_options[aug_index]
        if aug_index ==0:
            output = (img*255).unsqueeze(dim=0)
        elif aug_index ==1:
            output = self._gaussian_noise(img)
        elif aug_index ==2:
            output = self._speckle_noise(img)
        elif aug_index ==3:
            output = self._salt_pepper_noise(img)
        return output

    def _gaussian_noise(self, img):

        gauss_img = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.05, clip=True))
        # output = noise_aug(input)
        return (gauss_img*255).unsqueeze(dim=0)

    def _speckle_noise(self, img):

        speckle_noise = torch.tensor(random_noise(img, mode='speckle', mean=0, var=0.05, clip=True))
        return (speckle_noise*255).unsqueeze(dim=0)

    def _salt_pepper_noise(self, img):
        s_and_p = torch.tensor(random_noise(img, mode='s&p', salt_vs_pepper=0.5, clip=True))
        return (s_and_p*255).unsqueeze(dim=0)

class Rotation(nn.Module):
    def __init__(self, max_range = 4):
        super(Rotation, self).__init__()
        self.max_range = max_range
        self.prob = 0.5

    def forward(self, input, aug_index=None):
        _device = input.device

        _, _, H, W = input.size()

        if aug_index is None:
            aug_index = np.random.randint(4)

            output = torch.rot90(input, aug_index, (2, 3))

            _prob = input.new_full((input.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            output = _mask * input + (1-_mask) * output

        else:
            aug_index = aug_index % self.max_range
            output = torch.rot90(input, aug_index, (2, 3))

        return output

class CutPerm(nn.Module):
    def __init__(self, max_range = 4):
        super(CutPerm, self).__init__()
        self.max_range = max_range
        self.prob = 0.5

    def forward(self, input, aug_index=None):
        _device = input.device

        _, _, H, W = input.size()

        if aug_index is None:
            aug_index = np.random.randint(4)

            output = self._cutperm(input, aug_index)

            _prob = input.new_full((input.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            output = _mask * input + (1 - _mask) * output

        else:
            aug_index = aug_index % self.max_range
            output = self._cutperm(input, aug_index)

        return output

    def _cutperm(self, inputs, aug_index):

        _, _, H, W = inputs.size()
        h_mid = int(H / 2)
        w_mid = int(W / 2)

        jigsaw_h = aug_index // 2
        jigsaw_v = aug_index % 2

        if jigsaw_h == 1:
            inputs = torch.cat((inputs[:, :, h_mid:, :], inputs[:, :, 0:h_mid, :]), dim=2)
        if jigsaw_v == 1:
            inputs = torch.cat((inputs[:, :, :, w_mid:], inputs[:, :, :, 0:w_mid]), dim=3)

        return inputs


shift_trans = CutPerm()

class lag_Dataset(Dataset):
    def __init__(self, root_dir=None, transform=None, train=True, simple_aug=False):
        self.transform = transform
        self.root_dir = root_dir
        self.img_folder_path = None

        self.K = 32
        self.repeat = 100
        # self.mode = mode
        self.simple_aug = simple_aug

        self.data = []
        # self.label = []
        if train:
            self.img_folder_path = os.path.join(self.root_dir, 'train_set')

            self.imgs = np.asarray([f for f in listdir(self.img_folder_path)])

            for index in range(0, len(self.imgs)):
                img = io.imread(os.path.join(self.img_folder_path, self.imgs[index]))
                img = Image.fromarray(img).convert('RGB')
                img = img.resize((512, 512), Image.ANTIALIAS)
                img = img.crop((45, 45, 475, 475))   # cut off the margin of hyper-kvasir
                img = img.resize((224, 224), Image.ANTIALIAS)
                img = np.array(img)
                # print(img.shape)
                self.data.append(img)
            self.data = np.array(self.data)

        self.n_rot = 4
        self.k_shift = 4

        self.augdata = []
        self.labels = []
        print(self.data.shape)
        # exit(1)
        for i in range(0, self.data.shape[0]):

            tmp = np.expand_dims(self.data[i], axis=0)
            tmp = tmp.transpose((0, 3, 1, 2))
            tmp = torch.from_numpy((tmp))
            images = torch.cat([shift_trans(tmp, k) for k in range(self.k_shift)])
            images = images.detach().cpu().numpy()
            shift_labels = torch.cat([torch.ones_like(self.label) * k for k in range(self.n_rot)], 0)  # B -> 4B
            shift_labels = shift_labels.detach().cpu().numpy()

            self.labels.append(shift_labels)

            self.augdata.append(images)

        self.data = np.concatenate(self.augdata, axis=0)
        self.data = self.data.transpose((0, 2, 3, 1))
        self.labels = np.concatenate(self.labels, axis=0)


    def __getitem__(self, item):

        img = self.data[item]
        label = self.labels[item]
        img_size = img.shape

        img = img.astype(np.uint8)
        img = Image.fromarray(img).convert('RGB')
        label = torch.as_tensor(label).long()

        if self.transform is not None:
            img = self.transform(img)

        class_name = 0
        print('get item {}'.format(item))
        out = {'image': img, 'target': label, 'meta': {'im_size': img_size, 'index': item, 'class_name': class_name}}
        return out

    def get_n_rot(self):
        return self.n_rot

    def __len__(self):
        return len(self.data)

#
